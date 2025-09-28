import os
import glob
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T

from src import utils
from src.processing import AerialData, get_dataloader
from src.model import ResNet
from src.train import fit, test
from visualize import metrics_plotting, visualize_predictions

def main(path: str, epochs: int, num_workers: int) -> None:
    """
    Train, validate, and test a ResNet model on the Aerial Landscapes dataset.

    This function handles the full training pipeline:
    - Prepares the dataset (splits into train/val/test if CSV files are missing).
    - Applies preprocessing transforms (resize, tensor conversion, normalization).
    - Builds dataloaders for training, validation, and testing.
    - Initializes the ResNet model, loss function, and optimizer.
    - Trains the model for a given number of epochs.
    - Evaluates the trained model on the test set.
    - Generates and saves training curves (loss and accuracy).
    - Visualizes and saves prediction results on test samples.

    Args:
        path (str): Path to the dataset directory (must contain train/val/test CSV files or raw images).
        epochs (int): Number of training epochs.
        num_workers (int): Number of worker processes for dataloading.

    Returns:
        None
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(F"Using {device} device.")

    if not glob.glob(os.path.join(path, "*.csv")):
        utils.make_file(path, is_split=True, output_dir=path)

    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataloader, classes = get_dataloader(AerialData(f"{path}/train.csv", train_transform), num_workers=num_workers)
    valid_dataloader, _ = get_dataloader(AerialData(f"{path}/val.csv", test_transform), shuffle=False, num_workers=num_workers)
    test_dataloader, _ = get_dataloader(AerialData(f"{path}/test.csv", test_transform), shuffle=False, num_workers=num_workers)

    model = ResNet(num_classes=len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params}")

    results = fit(epochs, train_dataloader, valid_dataloader, model, loss_fn, optimizer, device)

    testing_acc, testing_loss = test(test_dataloader, model, loss_fn, device)
    print(F"Testing score: \nAccuracy: {(testing_acc*100):>0.1f}% | Avg. loss: {testing_loss:8f} \n")

    if not os.path.isdir("assets"):
        print("Create folder 'assets' for store visualization")
        os.mkdir("assets")

    metrics_plotting(results)
    
    visualize_predictions(model, test_dataloader, classes, device)

if __name__ == "__main__":
    # Command-line interface (CLI) entry point.
    # Allows running the training pipeline directly via terminal.
    #
    # Arguments:
    #   --path (str): Path to the dataset directory (must contain train/val/test splits).
    #   --epochs (int, default=1): Number of training epochs.
    #   --num_workers (int, default=1): Number of workers for dataloading.
    #
    # Example:
    #   python main.py --path "/path/to/data" --epochs 10 --num_workers 4
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to data directory.", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    args = parser.parse_args()
    main(args.path, args.epochs, args.num_workers)
