import matplotlib.pyplot as plt

def metrics_plotting(results: dict) -> None:
    
    training_loss = results['train_loss']
    training_accuracy = results['train_accuracy']
    
    valid_loss = results['val_loss']
    valid_accuracy = results['val_accuracy']
    
    fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4))
    ax = ax.flat
    
    ax[0].plot(training_loss, color = "red", label = "Train")
    ax[0].plot(valid_loss, color = "blue", label = "Valid")
    ax[0].set_title("CrossEntropyLoss", fontsize = 12, fontweight = "bold", color = "black")
    ax[0].set_xlabel("Epoch", fontsize = 10, fontweight = "bold", color = "black")
    ax[0].set_ylabel("loss", fontsize = 10, fontweight = "bold", color = "black")
    
    ax[1].plot(training_accuracy, color = "red", label = "Train")
    ax[1].plot(valid_accuracy, color = "blue", label = "Valid")
    ax[1].set_title("Accuracy", fontsize = 12, fontweight = "bold", color = "black")
    ax[1].set_xlabel("Epoch", fontsize = 10, fontweight = "bold", color = "black")
    ax[1].set_ylabel("score", fontsize = 10, fontweight = "bold", color = "black")
    
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.show()
