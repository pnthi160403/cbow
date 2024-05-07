from .prepare_train import get_model_cbow, read_tokenizer
import torch
from tqdm import tqdm
from .util import calc_accuracy

def test_model(config, dataloader):
    device = config["TRAIN"]["device"]
    checkpoint_path = config["CHECKPOINT"]["path"] + "/model.pth"

    # get tokenizer
    tokenizer = read_tokenizer(config=config)
    
    # load model
    model = get_model_cbow(
        config=config,
        tokenizer=tokenizer
    )
    model.load_state_dict(torch.load(checkpoint_path))

    labels = []
    predictions = []
    with torch.no_grad():
        model.eval()
        batch_iterator = tqdm(dataloader, desc=f"Testing Model")
        for target, context in batch_iterator:
            target = target.squeeze(1).to(device)
            context = context.to(device)
            probs = model(context)
            
            # get predicted word
            _, predicted = torch.max(probs, 1)
            predicted = predicted.cpu().numpy()
            target = target.cpu().numpy()
            labels += target.tolist()
            predictions += predicted.tolist()

    labels = torch.tensor(labels).clone().detach().to(device)
    predictions = torch.tensor(predictions).clone().detach().to(device)

    accuracy = calc_accuracy(
        preds=predictions,
        target=labels,
        num_classes=tokenizer.get_vocab_size(),
        device=device
    )

    print(f"Accuracy: {accuracy}")