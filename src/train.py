from tqdm import tqdm
import torch

def train(dataloader, model, optimizer, loss_fn, device) -> None:
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()
    for batch, (X, y) in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            tqdm.write(f"Loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Testing", unit="batch"):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= n_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg. loss: {test_loss:>8f} \n")

def fit(epochs, train_dataloader, valid_dataloader, model, loss_fn, optimizer, device):
    for t in range(epochs):
        print(F"Epoch {t+1}\n")
        train(train_dataloader, model, optimizer, loss_fn, device)
        test(valid_dataloader, model, loss_fn, device)

