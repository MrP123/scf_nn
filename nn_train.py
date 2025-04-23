import torch
from torch import Tensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt
from typing import Callable

from nn_helper import LocationsContainer, get_dataset
from nn_models import NNYuEtAl

def train_epoch(model: torch.nn.Module, dataset: torch.utils.data.Dataset, loss_fn: torch.nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,
                device: int | str = "cpu", validation_split: float = 0.2, batch_size: int = 8,
                lambda1: float = 0.0, lambda2: float = 0.0, debug: bool = False) -> dict[str, float]:
    """
    Train a model on the provided dataset for just one epoch.
    The desired optimizer, loss function and regularization parameters (L1 & L2) can be passed as arguments.

    This function is used both in the context of normal training where it is simply called in a loop, and with automatic tuning of hyper parameters with raytune.

    Returns the loss (training & validation) as well as the regularization loss as dict, for accumulation and later plotting in the calling functions.
    """

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
        param_vec = torch.cat([param.view(-1) for param in model.parameters()]) #params need to be flattend for some model architectures. Not sure why
        l1_reg = lambda1 * torch.linalg.norm(param_vec, ord=1)
        l2_reg = lambda2 * torch.linalg.norm(param_vec, ord=2)**2

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
                lambda1: float = 0.0, lambda2: float = 0.0, debug: bool = False) -> dict[str, list[float]]:
    """
    Training function for a model that runs for the set number of epochs. Other parameters are identical to `train_epoch(...)`, as it just wraps the function in a loop.
    The returned metrics from each epoch are accumulated in `met` and returned as a history of the training process.
    """

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

def test_model(model: torch.nn.Module, dataset: torch.utils.data.Dataset, locations: LocationsContainer,
               error_fun: Callable[[npt.ArrayLike, npt.ArrayLike], float] = None, device: int | str = "cpu") -> tuple[plt.Figure, plt.Axes, dict[str, list[float]]]:
    """
    Test the trained model with the so far unseen test dataset.
    An error function `err_fun` can be provided which will be used to evaluate the performance. If none is provided the norm of (target - prediction) will be used.

    The function automatically plots the result and returns the matplotlib `Figure` and `Axes` objects of the plot, so that it is immediately shown in the Jupyter notebook.
    Additionally a dict with the errors for each impact location is returned for further analysis.
    """

    model.eval()
    model = model.to(device)

    fig, ax = plt.subplots()
    errors = {label: [] for label in locations.get_labels()}

    with torch.no_grad():
        for x, y in DataLoader(dataset, shuffle=True):
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y_np = y.reshape((2,)).numpy()

            if isinstance(model, NNYuEtAl):
                x = x.reshape(x.shape[0], 1, x.shape[1])

            y_predict: Tensor = model(x)
            y_pred_np = y_predict.detach().reshape((2,)).numpy()
            location = locations.from_xy(y_np[0], y_np[1])

            ax.plot(y_np[0], y_np[1], "o", mfc="none", mew=2, markersize=8, color=location.color, label="target - " + location.label)
            ax.plot(y_pred_np[0], y_pred_np[1], "x", markersize=8, color=location.color, label="prediction - " + location.label, alpha=0.5)

            if error_fun is None:
                def my_error_fun(prediction: npt.ArrayLike, target: npt.ArrayLike) -> float:
                    return np.linalg.norm(target - prediction)
                error_fun = my_error_fun

            error = error_fun(y_pred_np, y_np)
            errors[location.label].append(error)

            #ax.yaxis.set_inverted(True)
            # TODO: make this into parameters!
            ax.set_xlim((-50, 50))
            ax.set_ylim((70, -70)) #inverted!

            ax.set_xlabel("X coordinate (mm)")
            ax.set_ylabel("Y coordinate (mm)")

            # force unique labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

    return fig, ax, errors

def __plot_violins_no_outliers(errors: list[list[float]], labels: list[str]) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots only a basic violin plot and returns the created matplotlib `Figure` and `Axes`
    """
    fig, ax = plt.subplots()

    ax.violinplot(errors)
    ax.set_xticks([i + 1 for i in range(len(labels))], labels=labels)
    ax.set_ylabel("Error (mm)")

    return fig, ax

def __plot_violins_with_outliers(errors: list[list[float]], outliers: list[list[float]], labels: list[str]) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plots a figure with broken axis. One axis (`ax_bot`) contains the violin plots of the data and the other (`ax_top`) contains the outliers.
    """
    fig = plt.figure()
    #Dummy axes in the background of the whole plot, spanning over the other 2 axes
    #This is used for getting a centered label on the y-axis, as is desired for plots with broken axes
    ax = fig.add_subplot(111)

    # Actual axes for plotting
    ax_top = fig.add_subplot(211)
    ax_bot = fig.add_subplot(212, sharex=ax_top)
    fig.subplots_adjust(hspace=0.05)  
    
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel("Error (mm)")

    #plot the actual violin plots
    ax_bot.violinplot(errors)

    #iterate through the outliers and plot them
    for i, vals in enumerate(outliers):
        if len(vals) == 0: #current impact location has no outliers --> skip plotting
            continue

        ax_top.plot(np.ones(len(vals)) * (i+1), vals, 'x') #the violin plots are plotted at all integers starting from 1 to how many plots there are

    ax_bot.spines.top.set_visible(False)
    ax_top.spines.bottom.set_visible(False)

    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False) # don't put tick labels at the top
    ax_bot.xaxis.tick_bottom()

    ax_bot.set_xticks([i + 1 for i in range(len(labels))], labels=labels)

    # Code for plotting symbol for broken axis taken from: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)

    return fig, (ax_top, ax_bot)

def plot_violins_from_error(errors: dict[str, list[float]], threshold: float = 100) -> tuple[plt.Figure, plt.Axes]:
    """
    This function can be used to create violin plots for each impact location from the errors returned by the `test_model()` function.
    
    As violin plots are IMHO not particularly great for visualizing significant outliers the threshold parameter can be used to exclude them from the violin calculation.
    These points are then specifically ploted as outlier points on a separate broken axis.
    
    If the data does not contain any outliers then only normal violin plots are created.
    """
    #TODO: clean up by making the data structure of `all_data` a np.ndarray instead of nested loops

    errors = dict(sorted(errors.items())) # sort so that keys (names of the impact locations) are in alphabetical order
    labels = list(errors.keys()) # list of labels is now in sorted order

    all_data = [v for v in errors.values()]

    removed_data = [[val for val in cat if val > threshold] for cat in all_data] #extract all points that are greater than the threshold
    all_data = [[val for val in cat if val <= threshold] for cat in all_data] #actually remove the data by keeping only values less then or equal to the threshold

    #Depending on if there are outliers or not plot accordingly
    if any(removed_data):
        fig, ax = __plot_violins_with_outliers(all_data, removed_data, labels)
    else:
        fig, ax = __plot_violins_no_outliers(all_data, labels)
    
    return fig, ax

if __name__ == "__main__":
    # quick test case
    errors = {'front': [np.float32(1.3877047), np.float32(0.4650986), np.float32(0.5736587), np.float32(0.6306847), np.float32(0.72050554), np.float32(0.49555248), np.float32(0.42456514), np.float32(2.5520551), np.float32(0.69600177), np.float32(0.50885177), np.float32(0.38532776), np.float32(0.9196282), np.float32(0.37682468), np.float32(0.91961294), np.float32(1.7318237), np.float32(0.8981945), np.float32(0.9165314), np.float32(2.5529287), np.float32(0.6140457), np.float32(0.7211996), np.float32(1.6954455), np.float32(0.39814422), np.float32(0.63076097), np.float32(1.3603944), np.float32(0.57313263), np.float32(0.57383794), np.float32(0.4244051), np.float32(0.9190333), np.float32(1.6927868), np.float32(2.5511892)], 'center': [np.float32(0.23155807), np.float32(0.24845217), np.float32(0.48596364), np.float32(0.20586996), np.float32(0.23126088), np.float32(0.19678019), np.float32(0.21014419), np.float32(0.20953189), np.float32(0.1778304), np.float32(0.21403803), np.float32(0.20629588), np.float32(0.20531254), np.float32(0.19573452), np.float32(0.18281722), np.float32(40.86105), np.float32(0.20643395), np.float32(0.20651706), np.float32(0.22505696), np.float32(0.48609045), np.float32(0.2191482), np.float32(0.20545661), np.float32(0.20125332), np.float32(0.48613265), np.float32(0.17709832), np.float32(0.21758877), np.float32(0.23174119), np.float32(40.86105), np.float32(0.20206489), np.float32(0.19248514), np.float32(0.23102918)], 'back': [np.float32(0.63613987), np.float32(0.5825399), np.float32(0.5244399), np.float32(0.78231835), np.float32(0.67898387), np.float32(0.63545346), np.float32(0.78707504), np.float32(0.91539264), np.float32(0.8008457), np.float32(0.51913035), np.float32(0.9988633), np.float32(0.52868134), np.float32(0.76627827), np.float32(0.17164242), np.float32(0.9130925), np.float32(0.6248225), np.float32(0.87640774), np.float32(0.7689523), np.float32(0.8530169), np.float32(0.6279541), np.float32(0.6252306), np.float32(0.48309714), np.float32(0.8187625), np.float32(0.5082445), np.float32(0.1708688), np.float32(0.9740913), np.float32(0.17391764), np.float32(0.5754909), np.float32(0.88909495), np.float32(0.6270349)]}
    fig, ax = plot_violins_from_error(errors, threshold=40)
    plt.show()