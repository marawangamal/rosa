import torch
import torch.nn as nn

from peftnet import PEFTNet


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()

        Conv2d = nn.Conv2d
        self.layers = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
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
        "stride": 1,
        "padding": 1,
        "rank": 4,
        "device": "cuda:0"
    }

    # Print parameters
    print("Experiment Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("-" * 50)

    device = torch.device(params["device"])

    model = CNN(
        in_channels=params["in_channels"], out_channels=params["out_channels"], kernel_size=params["kernel_size"],
        stride=params["stride"], padding=params["padding"]
    )
    model = PEFTNet(
        model,
        method="loraconv2d",
        factorize_list=[nn.Conv2d.__name__],
        rank=params["rank"]
    )
    model = model.to(device)

    print(model)

    x = torch.randn(params["batch_size"], params["in_channels"], 224, 224, device=device)
    y = model(x)

    print(y.shape)


if __name__ == '__main__':
    main()