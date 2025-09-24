from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

def train(dataloader, model, optimizer, loss_fn, device) -> tuple[float, float]:
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    losses, correct = 0, 0
    model.train()
    for X, y in dataloader:
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

def test(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    losses, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            losses += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    losses /= n_batches
    correct /= size

    return correct, losses

def fit(epochs, train_dataloader, valid_dataloader, model, loss_fn, optimizer, device):
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for t in tqdm(range(epochs)):
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

def prediction(model, dataloader, classes: list[str], device, n_samples: int = 5):
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(1)

            for i in range(len(X)):
                img = X[i].cpu().permute(1, 2, 0).numpy()  # ubah ke HWC
                true_label = classes[y[i].item()]
                pred_label = classes[preds[i].item()]

                plt.imshow(img)
                plt.title(f"Pred: {pred_label}, Correct: {true_label}", fontsize=12, color="green" if pred_label == true_label else "red")
                plt.axis("off")
                plt.show()

                images_shown += 1
                if images_shown >= n_samples:
                    return
