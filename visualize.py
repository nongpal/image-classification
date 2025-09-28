import os
import torch
import matplotlib.pyplot as plt

def metrics_plotting(results: dict) -> None:
    """
    Plot and save training/validation loss and accuracy curves.

    This function takes the training history (loss and accuracy values for both
    training and validation) and generates two plots:
    - Loss curve (train vs validation)
    - Accuracy curve (train vs validation)

    Each plot will be saved into the "assets/" directory as PNG images.

    Args:
        results (dict): A dictionary containing training history with keys:
            - 'train_loss' (list[float]): Training loss per epoch.
            - 'val_loss' (list[float]): Validation loss per epoch.
            - 'train_accuracy' (list[float]): Training accuracy per epoch.
            - 'val_accuracy' (list[float]): Validation accuracy per epoch.

    Returns:
        None
    """
    training_loss = results['train_loss']
    training_accuracy = results['train_accuracy']
    valid_loss = results['val_loss']
    valid_accuracy = results['val_accuracy']
    
    # Loss curve
    plt.figure()
    plt.plot(training_loss, label="Train Loss")
    plt.plot(valid_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("assets/loss_curve.png")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(training_accuracy, label="Train Acc")
    plt.plot(valid_accuracy, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("assets/accuracy_curve.png")
    plt.close()

def visualize_predictions(model, dataloader, classes: list[str], device, max_images: int = 8, filename: str = "predictions.png"):
    """
    Visualize and save model predictions on a batch of images.

    This function takes a batch of images from the given dataloader, performs
    inference with the provided model, and plots a grid of images with their
    predicted and true labels. The resulting figure is saved as a PNG file.

    Args:
        model (torch.nn.Module): The trained model used for prediction.
        dataloader (torch.utils.data.DataLoader): DataLoader to sample a batch of images.
        classes (list[str]): List of class names corresponding to dataset labels.
        device (torch.device or str): Device to perform inference on ("cpu" or "cuda").
        max_images (int, optional): Maximum number of images to visualize. Default is 8.
        filename (str, optional): Name of the saved PNG file (inside "assets/"). Default is "predictions.png".

    Returns:
        None
    """

    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Plot
    plt.figure(figsize=(15, 6))
    for idx in range(min(max_images, len(images))):
        img = images[idx].cpu().permute(1, 2, 0)  # C,H,W → H,W,C
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # unnormalize
        img = img.clamp(0, 1)

        plt.subplot(2, max_images//2, idx+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Pred: {classes[preds[idx]]}\nTrue: {classes[labels[idx]]}")

    plt.tight_layout()
    plt.savefig(f"assets/{filename}")
    plt.close()
