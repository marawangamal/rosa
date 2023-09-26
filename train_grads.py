import copy
import argparse

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import numpy as np

from utils.utils import set_seeds, refactorize, get_experiment_name
import peftnet as pn
from peftnet.peft_module.loralinear import LoraLinear


class LinearModel(nn.Module):
    """A simple linear model."""

    def __init__(self, in_features=768, out_features=32, bias=False):
        super().__init__()
        self.l1 = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """ Forward pass.

        Args:
            x: [batch_size, in_features]

        Returns:
            y: [batch_size, out_features]

        """
        return self.l1(x)


class MLP2Layer(nn.Module):
    """A 2-layer multi-layer perceptron."""

    def __init__(self, in_features=768, out_features=32, hidden=64, bias=False):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden, bias=bias)
        self.l2 = nn.Linear(hidden, out_features, bias=bias)

    def forward(self, x):
        """ Forward pass.

        Args:
            x: [batch_size, in_features]

        Returns:
            y: [batch_size, out_features]

        """
        return self.l2(nn.functional.relu(self.l1(x)))


def build_synthetic_dataset(model, n_samples=1000, n_dims=768):
    """Build a synthetic dataset from a given model."""
    with torch.no_grad():
        x = torch.randn(n_samples, n_dims)
        y = model(x)
        return x, y


def evaluate_model(model, dataloader):
    with torch.no_grad():
        loss_fn = torch.nn.MSELoss()
        losses = []
        for i, batch in enumerate(dataloader):
            x_train, y_train = batch
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            losses.append(loss.item())
        return sum(losses) / len(losses)


def main():
    # *** Configure HPs ***

    # Experiment parameters
    n_trials = 10
    n_grad_pts_in_plot = 20  # number of points to plot in the gradient plot

    # Baseline model parameters
    model_name = "mlp2"
    true_rank = 24
    in_f = 512
    out_f = 32

    # Dataset parameters
    n_train_samples = 5000

    # Train parameters
    bs = 64

    # *** End of Config ***

    # Get experiment name
    filename = "grads_synthetic"

    # Set seeds
    set_seeds(42)

    models = {"linear": LinearModel, "mlp2": MLP2Layer}
    init_model = models[model_name](in_features=in_f, out_features=out_f, bias=False)
    true_model = pn.LoraNet(copy.deepcopy(init_model), rank=true_rank, init_method="random")
    x_train, y_train = build_synthetic_dataset(true_model, n_samples=n_train_samples, n_dims=in_f)

    # Dataloader
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=False)

    test_model = pn.LoraNet(copy.deepcopy(init_model), rank=1, debug=True)
    layers_of_interest = [name for name, module in test_model.named_modules() if isinstance(module, LoraLinear)]

    layer_grads = {
        "rosa": {name: [] for name in layers_of_interest},
        "lora": {name: [] for name in layers_of_interest}
    }

    for trial in range(n_trials):

        peft_models = {
            "lora": pn.LoraNet(copy.deepcopy(init_model), rank=1, debug=True),
            "rosa": pn.RosaNet(copy.deepcopy(init_model), rank=1, debug=True)
        }

        for name, model in peft_models.items():

            # Make loss function
            loss_fn = torch.nn.MSELoss()

            test_batch = next(iter(train_dataloader))
            x_train, y_train = test_batch
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()

            if name in ["lora", "rosa"]:
                for layername, module in model.named_modules():
                    if layername in layers_of_interest:
                        print(f"{name}: {layername} grad: {module.ab.grad.flatten()[:5]}")
                        layer_grads[name][layername].append(module.ab.grad.flatten()[:n_grad_pts_in_plot])
            else:
                raise NotImplementedError

    # loop over layers to plot in same figure
    layernames2colors = {
        k: v for k, v in zip(
            layers_of_interest,
            # Generate len(layers_of_interest) colors
            cm.viridis(np.linspace(0, 1, len(layers_of_interest)))
        )

    }
    plt.figure(figsize=(8, 6))
    for name, grads in layer_grads.items():
        for layername, layer_grad in grads.items():
            # if "l1" not in layername:
            #     continue
            grads_mean = torch.stack(layer_grad).mean(dim=0)
            grads_std = torch.stack(layer_grad).std(dim=0)
            plt.errorbar(
                range(len(grads_mean)),
                grads_mean,
                yerr=grads_std,
                fmt="-." if name == "lora" else "--x",
                facecolors='none',
                label=f"{name}_{layername}",
                color=layernames2colors[layername]
            )
    plt.xlabel("Dimension")
    plt.ylabel("Gradient")
    plt.legend()
    plt.savefig(f"figures/grads_all_{filename}.png")


if __name__ == '__main__':
    main()
