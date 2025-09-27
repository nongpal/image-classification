import torch
import matplotlib.pyplot as plt

def metrics_plotting(results: dict) -> None:
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

def visualize_predictions(model, dataloader, classes, device, max_images=8, filename="predictions.png"):
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
