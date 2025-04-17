import torch
from torch import Tensor

from nn_helper import get_dataset
from nn_models import NNYuEtAl

def train_epoch(model: torch.nn.Module, dataset: torch.utils.data.Dataset, loss_fn: torch.nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,
                device: int | str = "cpu", validation_split: float = 0.2, batch_size: int = 8,
                lambda1: float = 0.0, lambda2: float = 0.0, debug: bool = False):
    met = {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "l1_loss": 0.0,
        "l2_loss": 0.0
    }

    model.train()
    train_loss = 0
    train_dl, val_dl = get_dataset(dataset, validation_split, batch_size)

    l1_loss, l2_loss = 0.0, 0.0

    for i, (x_batch, y_batch) in enumerate(train_dl):
        x_batch: Tensor = x_batch.to(device, dtype=torch.float32)
        y_batch: Tensor = y_batch.to(device, dtype=torch.float32)

        # CNN model needs reshaping of input tensoro
        if isinstance(model, NNYuEtAl):
            x_batch = x_batch.reshape(x_batch.shape[0], 1, x_batch.shape[1])

        y_pred = model(x_batch)
        loss: Tensor = loss_fn(y_pred, y_batch)

        l1_reg, l2_reg = 0.0, 0.0
        param_vec = torch.cat([param.view(-1) for param in model.parameters()])
        l1_reg = lambda1 * torch.linalg.norm(param_vec, ord=1)
        l2_reg = lambda2 * torch.linalg.norm(param_vec, ord=2)**2

        #l1_reg = lambda1 * sum(p.abs().sum() for p in model.parameters())
        #l2_reg = lambda2 * sum(p.pow(2).sum() for p in model.parameters())

        loss = loss + l1_reg + l2_reg
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        l1_loss += l1_reg
        l2_loss += l2_reg
                
        if debug and i % 10 == 0:
            print(f"Loss after mini-batch {i}: {loss.item():5f} (of which {l1_reg:5f} L1 loss; {l2_reg:5f} L2 loss)")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_dl:
            x_val: Tensor = x_val.to(device, dtype=torch.float32)
            y_val: Tensor = y_val.to(device, dtype=torch.float32)
            
            if isinstance(model, NNYuEtAl):
                x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])

            y_val_pred = model(x_val)
            val_loss += loss_fn(y_val_pred, y_val).item()

    met["train_loss"] = train_loss / len(train_dl)
    met["val_loss"] = val_loss / len(val_dl)
    met["l1_loss"] = l1_loss / len(train_dl)
    met["l2_loss"] = l2_loss / len(train_dl)

    return met

def train_model(model: torch.nn.Module, dataset: torch.utils.data.Dataset, loss_fn: torch.nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,
                device: int | str = "cpu", epochs: int = 150, validation_split: float = 0.2, batch_size: int = 8,
                lambda1: float = 0.0, lambda2: float = 0.0, debug: bool = False):

    model.to(device)

    met = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(epochs):
        epoch_met = train_epoch(model, dataset, loss_fn, optimizer, device, validation_split, batch_size, lambda1, lambda2, debug)

        if epoch % 10 == 0:
            print(f"Loss at epoch {epoch:03d}: train_loss = {epoch_met["train_loss"]:5f} / val_loss = {epoch_met["val_loss"]:5f}")

        met["train_loss"].append(epoch_met["train_loss"])
        met["val_loss"].append(epoch_met["val_loss"])

    return met