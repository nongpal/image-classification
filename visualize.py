import matplotlib.pyplot as plt

def metrics_plotting(results: dict, save_path: str | None = None) -> None:
    training_loss = results['train_loss']
    training_accuracy = results['train_accuracy']
    valid_loss = results['val_loss']
    valid_accuracy = results['val_accuracy']
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # --- Loss ---
    ax[0].plot(training_loss, color="red", label="Train")
    ax[0].plot(valid_loss, color="blue", label="Valid")
    ax[0].set_title("CrossEntropyLoss", fontsize=12, fontweight="bold")
    ax[0].set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax[0].set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax[0].legend()

    # --- Accuracy ---
    ax[1].plot(training_accuracy, color="red", label="Train")
    ax[1].plot(valid_accuracy, color="blue", label="Valid")
    ax[1].set_title("Accuracy", fontsize=12, fontweight="bold")
    ax[1].set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax[1].set_ylabel("Score", fontsize=10, fontweight="bold")
    ax[1].legend()

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Saved metrics plot to {save_path}")

    plt.show()
