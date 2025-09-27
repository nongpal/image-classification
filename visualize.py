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
    plt.savefig("loss_curve.png")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(training_accuracy, label="Train Acc")
    plt.plot(valid_accuracy, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("accuracy_curve.png")
    plt.close()
