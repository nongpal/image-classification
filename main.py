import os
import glob
import argparse
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src import utils
from src.processing import AerialData, get_dataloader
from src.model import ResNet
from src.train import fit, test

def main(path: str, epochs: int):

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(F"Using {device} device.")

    if not glob.glob(os.path.join(path, "*.csv")):
        utils.make_file(path, is_split=True, output_dir=path)

    train_transform = A.Compose([
        #A.RandomCrop(height=256, width=256, p=1.0),
        #A.HorizontalFlip(),
        #A.CoarseDropout(),
        #A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.7),
        A.SquareSymmetry(),
        A.Normalize(),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

    train_dataloader = get_dataloader(AerialData(f"{path}/train.csv", train_transform), device=device)
    valid_dataloader = get_dataloader(AerialData(f"{path}/val.csv", test_transform), device=device)
    test_dataloader = get_dataloader(AerialData(f"{path}/test.csv", test_transform), device=device)

    model = ResNet(n_classes=15).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params}")

    fit(int(epochs), train_dataloader, valid_dataloader, model, loss_fn, optimizer, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to data directory.")
    parser.add_argument("--epochs", default=1)
    args = parser.parse_args()
    main(args.path, args.epochs)
