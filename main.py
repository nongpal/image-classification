import os
import glob
import argparse
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src import utils
from src.processing import AerialData, get_dataloader
from src.model import Model
from src.train import fit, test

def main(path: str):

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    torch.set_default_device(device)
    print(F"Using {torch.get_default_device()} device.")

    if not glob.glob(os.path.join(path, "*.csv")):
        utils.make_file(path, is_split=True, output_dir=path)

    train_transform = A.Compose([
        A.ToFloat(),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.ToFloat(),
        ToTensorV2(),
    ])

    train_dataloader = get_dataloader(AerialData(f"{path}/train.csv", train_transform), device=device)
    valid_dataloader = get_dataloader(AerialData(f"{path}/val.csv", test_transform), device=device)
    test_dataloader = get_dataloader(AerialData(f"{path}/test.csv", test_transform), device=device)

    model = Model(3, 15)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    fit(10, train_dataloader, valid_dataloader, model, loss_fn, optimizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to data directory.")
    args = parser.parse_args()
    main(args.path)
