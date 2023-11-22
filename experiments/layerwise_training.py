import torch
import torch.nn as nn
import copy

def forward_layerwise(layers, x, device='cpu', verbose=True):
    """ Forward pass through a list of layers.

    Description:
        For n = 0, 1, ..., N-1:
            Load layer n, forward pass using activation n, save activation n+1, unload layer n, load layer n+1,

    """
    activations = []
    x = x.to(device)
    for i, layer in enumerate(layers):
        if verbose:
            print(f"Forward pass through layer {i}...")
        layer = layer.to(device)
        x = layer(x)
        activations.append(x)
        layer.to('cpu')  # Unload layer from GPU
        del layer
    return activations


def backward_layerwise(layers, activations, loss, optimizers, device='cpu', verbose=True):
    """ Backward pass through a list of layers.

    Description:
        For n = N-1, N-2, ..., 1, 0:
            Load layer n, backward pass using activation n, save activation n-1, unload layer n, load layer n-1,

    """
    pass


def cuda_time_operation(func, func_kwargs, device='cuda:0', verbose=False):
    """ Time an operation on the GPU. """
    if verbose:
        print(f"Running operation on {device}...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(**func_kwargs)
    end.record()
    torch.cuda.synchronize()
    if verbose:
        print(f"Time elapsed: {start.elapsed_time(end)}ms")
    return start.elapsed_time(end)


if __name__ == '__main__':
    # Config
    input_size = 512
    hidden_sizes = (512, 512, 512)
    batch_size = 32


    # Create random TensorDataset
    dataset = torch.rand(1000, input_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create MLP
    input_layer = nn.Linear(input_size, hidden_sizes[0])
    hidden_layers = [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)]
    layers = [input_layer, *hidden_layers]

    # Create nn.Sequential
    mlp = nn.Sequential(*copy.deepcopy(layers))

    # Train Layer-wise
    loss_fn = nn.CrossEntropyLoss()
    optimizers = [torch.optim.Adam(layers[i].parameters(), lr=0.001) for i in range(len(layers))]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Forward pass (layer-wise)
    x = next(iter(dataloader))
    elapsed_layer_wise = cuda_time_operation(
        forward_layerwise, {'layers': layers, 'x': x, 'device': device}, device=device
    )
    # Forward pass (nn.Sequential)
    x = next(iter(dataloader))
    mlp_func = lambda x: mlp(x)
    mlp.to(device)
    x = x.to(device)
    elapsed_sequential = cuda_time_operation(
        mlp_func, {'x': x}, device=device
    )

    print(f"Layer-wise: {elapsed_layer_wise:0.2f}ms")
    print(f"Sequential: {elapsed_sequential:0.2f}ms")

