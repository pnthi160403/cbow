from .prepare_train import get_model_cbow, read_tokenizer
import torch
from tqdm import tqdm
from .util import calc_accuracy, set_seed

def test_model(config, dataloader):
    # set seed
    set_seed()

    device = config["TRAIN"]["device"]
    checkpoint_path = config["CHECKPOINT"]["path"] + "/model.pth"

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
            # print("probs shape: ", probs.shape)
            # print(f"probs: {probs}")
            # print("target", target)
            # print("context", context)
            
            # get predicted word
            _, predicted = torch.max(probs, 1)

            # print("predicted: ", predicted)
            # print("target: ", target)
            labels.append(target.item())
            predictions.append(predicted.item())
            step += 1
            if step % print_step == 0:
                print("target: ", target)
                print("predicted: ", predicted)
                decode_target = tokenizer.decode(target[0])
                decode_predicted = tokenizer.decode(predicted[0])
                print(f"Target: {decode_target} - Predicted: {decode_predicted}")

    labels = torch.tensor(labels).clone().detach().to(device)
    predictions = torch.tensor(predictions).clone().detach().to(device)

    # print("label shape: ", labels.shape)
    # print("prediction shape: ", predictions.shape)

    accuracy = calc_accuracy(
        preds=predictions,
        target=labels,
        num_classes=tokenizer.get_vocab_size(),
        device=device
    )

    print(f"Accuracy: {accuracy}")