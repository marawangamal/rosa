import torch
import torch.nn as nn

# todo: make batch_size larger
# todo: try pip install torchviz


class AETLinear(nn.Module):
    def __init__(self, aet=True, in_features=10, out_features=10, rank=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_trainable = aet
        self.w = nn.Parameter(torch.randn(in_features, out_features), requires_grad=False)
        self.a = torch.randn(in_features, rank, requires_grad=False)
        self.b = nn.Parameter(torch.randn(rank, out_features), requires_grad=True)
        # if not self.b_trainable else nn.Parameter(torch.randn(rank, out_features), requires_grad=self.b_trainable)

    def __repr__(self):
        # Show which parameters are trainable
        trainable_params = {
            'w': self.w.requires_grad,
            'a': self.a.requires_grad,
            'b': self.b.requires_grad
        }
        return f"{self.__class__.__name__}({', '.join([f'{k} trainable={v}' for k, v in trainable_params.items()])})"

    def forward(self, x):
        """ Forward pass of AET layer

        Args:
            x: [batch_size, in_features]

        Returns:

        """
        self.a = self.a.to(x.device)
        self.b = self.b.to(x.device)
        return x @ self.w + (x @ self.a) @ self.b


class Model(nn.Module):
    def __init__(self,  in_features=10, hidden_features=10, out_features=10, rank=1, aet=True):
        super().__init__()
        self.layers = nn.Sequential(
            AETLinear(aet=aet, in_features=in_features, out_features=hidden_features, rank=rank),
            AETLinear(aet=aet, in_features=hidden_features, out_features=hidden_features, rank=rank),
            AETLinear(aet=aet, in_features=hidden_features, out_features=hidden_features, rank=rank),
            AETLinear(aet=aet, in_features=hidden_features, out_features=hidden_features, rank=rank),
            AETLinear(aet=aet, in_features=hidden_features, out_features=out_features, rank=rank)
        )

    def forward(self, x):
        return self.layers(x)


def report_memory_usage(message="", width=30):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # in MBs
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # in MBs
    print(f"{message.ljust(width)} --> Allocated memory: {allocated:.10f}MB, Reserved memory: {reserved:.10f}MB")


def main():
    # Define parameters
    params = {
        "batch_size": 512*32,
        "in_features": 512,
        "out_features": 512,
        "rank": 1,
        "aet": False,
        "hidden_features": 512,
        "device": "cuda:0"
    }

    # Print parameters
    print("Experiment Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("-" * 50)

    device = torch.device(params["device"])

    # report_memory_usage("Initial memory")
    # w1 = torch.randn(params["in_features"], params["in_features"], device=device, requires_grad=False)
    # a1 = torch.randn(params["in_features"], params["rank"], device=device, requires_grad=False)
    # b1 = torch.randn(params["rank"], params["in_features"], device=device, requires_grad=False)
    # w2 = torch.randn(params["in_features"], params["out_features"], device=device, requires_grad=False)
    # a2 = torch.randn(params["in_features"], params["rank"], device=device, requires_grad=True)
    # b2 = torch.randn(params["rank"], params["out_features"], device=device, requires_grad=False)
    report_memory_usage("After model creation on GPU")

    # Model
    model = Model(in_features=params["in_features"], hidden_features=params["hidden_features"], out_features=params["out_features"], rank=params["rank"], aet=params["aet"])
    model.to(device)
    report_memory_usage("After model creation on GPU")

    x_true = torch.randn(params["batch_size"], params["in_features"], device=device, requires_grad=False)
    y_true = torch.randn(params["batch_size"], params["out_features"], device=device, requires_grad=False)
    report_memory_usage("After data creation")

    # Forward pass
    loss = torch.nn.functional.mse_loss(model(x_true), y_true)
    # x2 = x_true @ w1 + (x_true @ a1) @ b1
    # loss = torch.nn.functional.mse_loss(x2 @ w2 + (x2 @ a2) @ b2, y_true)
    print(f"Loss: {loss}")
    report_memory_usage("After forward pass")
    # loss.backward()
    # report_memory_usage("After backward pass")

if __name__ == "__main__":
    main()