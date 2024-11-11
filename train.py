import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import models as mds
from tqdm import tqdm
import data
import datetime
import utils
import wandb

def get_best_gpu():
    u0 = torch.cuda.utilization("cuda:0")
    u1 = torch.cuda.utilization("cuda:1")
    if u0 < u1:
        return "cuda:0"
    else:
        return "cuda:1"


def main(args):
    time = datetime.datetime.now().strftime("%d-%m@%H:%M")
    if torch.cuda.is_available():
        device = torch.device(get_best_gpu())
    else:
        device = torch.device("cpu")
    print("Training with following config:", args, "on device:", device)
    train_loader, val_loader, test_loader = data.get_data(args['data'], validate=args['validate'], batch_size=args['bs'])

    num_classes = args['num_classes']

    if args['unc'] == 'ensemble':
        num_members = args['num_members']
        ensemble = mds.Ensemble(num_classes=num_classes)
    else:
        num_members = 1

    for _ in tqdm(range(num_members), desc='Members'):
        wandb.init(project='credal-uncertainty', config=args, reinit=True, mode=args['wandb'])
        model = mds.get_model(args['unc'], args['base'], num_classes).to(device)

        model.compile()
        optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["wd"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["epochs"])
        print("Starting training")
        for epoch in tqdm(range(args["epochs"]), desc="Epochs"):
            model.train()
            running_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() / len(train_loader)
            wandb.log({"train_loss": loss.item()})
            print(f"Training loss: {running_loss}")
            scheduler.step()
            if args['validate'] and False:
                with torch.no_grad():
                    model.eval()
                    loss = 0
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss += F.cross_entropy(outputs, targets)
                    loss /= len(val_loader)
                    print(f"Validation loss: {loss.item()}")
                    acc = utils.accuracy(model, val_loader, device)
                    print(f"Validation accuracy: {acc}")
                    wandb.log({"val_loss": loss.item(), "val_acc": acc})
        print("Finished training")
        model.eval()
        acc = utils.accuracy(model, test_loader, device)
        wandb.log({"test_acc": acc})
        print(f"Test accuracy: {acc}")
        if args['unc'] == 'ensemble':
            ensemble.members.append(model)
        wandb.finish()

    torch.save(ensemble if args['unc'] == 'ensemble' else model, f"./checkpoints/{args['data']}_{args['base']}_{args['unc']}_{time}.pt")


if __name__ == "__main__":
    with open("trainconfig.yml", "r") as file:
        args = yaml.safe_load(file)
    main(args)
