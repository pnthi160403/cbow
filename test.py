from .prepare_train import get_model_cbow, read_tokenizer
import torch
from tqdm import tqdm
from .util import calc_accuracy, calc_recall, calc_precision, calc_f_beta, set_seed
import pandas as pd

def test_model(config, dataloader):
    # set seed
    set_seed()

    device = config["TRAIN"]["device"]
    checkpoint_path = config["CHECKPOINT"]["path"] + "/" + config["CHECKPOINT"]["model_name"]

    # get tokenizer
    tokenizer = read_tokenizer(config=config)
    
    # load model
    model = get_model_cbow(
        config=config,
        tokenizer=tokenizer
    )
    if device == "cuda":
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    labels = []
    predictions = []
    print_step = len(dataloader) // 10
    step = 0
    with torch.no_grad():
        model.eval()
        batch_iterator = tqdm(dataloader, desc=f"Testing Model")
        for target, context in batch_iterator:
            target = target.to(device)
            context = context.to(device)
            probs = model(context)

            # get predicted word
            _, predicted = torch.max(probs, 1)

            labels.append(target.item())
            predictions.append(predicted.item())
            step += 1
            if step % print_step == 0:
                decode_context = tokenizer.decode(context[0].detach().cpu().numpy())
                decode_target = tokenizer.decode(target[0].detach().cpu().numpy())
                decode_predicted = tokenizer.decode(predicted.detach().cpu().numpy())
                print(f"Context: {decode_context}")
                print(f"Target: {decode_target} - Predicted: {decode_predicted}")

    labels = torch.tensor(labels).clone().detach().to(device)
    predictions = torch.tensor(predictions).clone().detach().to(device)

    # calculate metrics
    accuracy = calc_accuracy(
        preds=predictions,
        target=labels,
        num_classes=tokenizer.get_vocab_size(),
        device=device
    )

    recall = calc_recall(
        preds=predictions,
        target=labels,
        num_classes=tokenizer.get_vocab_size(),
        device=device
    )

    precision = calc_precision(
        preds=predictions,
        target=labels,
        num_classes=tokenizer.get_vocab_size(),
        device=device
    )

    f_beta = calc_f_beta(
        preds=predictions,
        target=labels,
        beta=1,
        num_classes=tokenizer.get_vocab_size(),
        device=device
    )

    results = {
        "accuracy": accuracy.item(),
        "recall": recall.item(),
        "precision": precision.item(),
        "f_beta": f_beta.item()
    }

    print(pd.DataFrame(results, index=[0]))