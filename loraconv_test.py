import torch
import torch.nn as nn

from peftnet import LoraNetConv

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        Conv2d = nn.Conv2d
        self.layers = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.layers(x)


def main():

    # Define parameters
    params = {
        "batch_size": 32,
        "in_channels": 3,
        "out_channels": 32,
        "kernel_size": 3,
        "rank": 1,
        "device": "cuda:0"
    }

    # Print parameters
    print("Experiment Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("-" * 50)

    device = torch.device(params["device"])

    model = CNN(in_channels=params["in_channels"], out_channels=params["out_channels"], kernel_size=params["kernel_size"])
    model = LoraNetConv(model, rank=params["rank"])
    model = model.to(device)

    print(model)

    x = torch.randn(params["batch_size"], params["in_channels"], 224, 224, device=device)
    y = model(x)

    print(y.shape)


if __name__ == '__main__':
    main()