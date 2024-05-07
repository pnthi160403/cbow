from .prepare_train import read_dataset, read_tokenizer, get_dataloader, get_Adam_optimizer, get_model_cbow, get_nll_loss
from .util import set_seed, draw_loss_plot
from tqdm import tqdm
import torch
from .test import test_model
from .config.config import create_dirs

# train
def train(config):
    # create dirs
    create_dirs(config=config)

    # set seed
    set_seed()
    
    # hyperparameters
    batch_size = config["TRAIN"]["batch_size"]
    epochs = config["TRAIN"]["epochs"]
    learning_rate = config["TRAIN"]["learning_rate"]
    device = config["TRAIN"]["device"]
    checkpoint_path = config["CHECKPOINT"]["path"] + "/model.pth"

    # get dataset
    train_data, val_data, test_data = read_dataset(config=config)

    # get tokenizer
    tokenizer = read_tokenizer(config=config)

    # get dataloader
    train_dataloader = get_dataloader(
        config=config,
        tokenizer=tokenizer,
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )

    # get model
    model = get_model_cbow(
        config=config,
        tokenizer=tokenizer
    )

    # get optimizer
    optimizer = get_Adam_optimizer(
        model=model,
        learning_rate=learning_rate
    )

    # get loss function
    loss_fn = get_nll_loss()
    losses = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")        
        for target, context in batch_iterator:
            if context.size()[0] != batch_size:
                continue

            target = target.squeeze(1).to(device)
            context = context.to(device)

            optimizer.zero_grad()
            log_probs = model(context)
            loss = loss_fn(log_probs, target)

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    
    draw_loss_plot(config=config, losses=losses)

    # save model
    torch.save(model.state_dict(), checkpoint_path)

    # test model
    test_dataloader = get_dataloader(
        config=config,
        tokenizer=tokenizer,
        dataset=test_data,
        batch_size=1,
        shuffle=False
    )
    test_model(config=config, dataloader=test_dataloader)