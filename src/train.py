from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def train(dataloader, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Iterates over the training dataloader, performs forward and backward passes,
    updates model parameters using the optimizer, and computes average loss
    and accuracy for the epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for training dataset.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (torch.nn.Module): Loss function to minimize.
        device (str or torch.device): Device to run computations on.

    Returns:
        tuple[float, float]: (accuracy, average_loss) for the training epoch.
    """
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    losses, correct = 0, 0
    model.train()
    for X, y in tqdm(dataloader, desc="Training", unit="Batch"):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        losses += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    losses /= n_batches
    correct /= size


    return correct, losses

def test(dataloader, model: nn.Module, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    """
    Evaluate the model on validation or test data.

    Disables gradient computation, runs the model on the dataloader,
    and calculates average loss and accuracy.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for validation or test dataset.
        model (torch.nn.Module): Trained model to evaluate.
        loss_fn (torch.nn.Module): Loss function for evaluation.
        device (str or torch.device): Device to run computations on.

    Returns:
        tuple[float, float]: (accuracy, average_loss) for the evaluation.
    """
    model.eval()
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    losses, correct = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Testing", unit="batch"):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            losses += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    losses /= n_batches
    correct /= size

    return correct, losses

def fit(epochs: int, train_dataloader, valid_dataloader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
    """
    Train and validate the model over multiple epochs.

    For each epoch, trains the model on the training set and evaluates it
    on the validation set. Tracks metrics (loss and accuracy) for both sets.

    Args:
        epochs (int): Number of training epochs.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training dataset.
        valid_dataloader (torch.utils.data.DataLoader): DataLoader for validation dataset.
        model (torch.nn.Module): Model to train and validate.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        device (str or torch.device): Device to run computations on.

    Returns:
        dict: Training history with keys:
            - "train_loss": list of training losses per epoch
            - "train_accuracy": list of training accuracies per epoch
            - "val_loss": list of validation losses per epoch
            - "val_accuracy": list of validation accuracies per epoch
    """
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for t in range(epochs):
        print(F"Epoch {t+1}\n")
        train_acc, train_loss = train(train_dataloader, model, optimizer, loss_fn, device)
        print(f"Train: \n Accuracy: {(100*train_acc):>0.1f}%, Avg. loss: {train_loss:>8f} \n")
        val_acc, val_loss = test(valid_dataloader, model, loss_fn, device)
        print(f"Validation: \n Accuracy: {(100*val_acc):>0.1f}%, Avg. loss: {val_loss:>8f} \n")

        results["train_accuracy"].append(train_acc)
        results["train_loss"].append(train_loss)
        results["val_accuracy"].append(val_acc)
        results["val_loss"].append(val_loss)

    return results
