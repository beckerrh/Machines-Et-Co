import numpy as np
import torch
from torch import nn
import pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import mlp

#-------------------------------------------------------------
if __name__ == '__main__':
    # Set fixed random number seed
    # torch.manual_seed(42)
    #data
    x = np.array([0, 1, 3])
    y = np.array([3, 0, 1])
    # Initialize the MLP
    mlp = mlp.MLP(1,1,3,1)
    mlp.train()
    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    # assert mlp.layers.parameters()== mlp.parameters()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.1)
    inputs, targets = torch.from_numpy(x).float().reshape(-1, 1), torch.from_numpy(y).float().reshape(-1, 1)
    for it in range(50):
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        mlp.print_parameters()
        optimizer.step()
        print(f"Loss after{it} {loss:.3e}")
    mlp.eval()
    dtype = mlp.layers[0].weight.dtype
    # y_nn = mlp(torch.from_numpy(x).reshape(-1,1).to(dtype)).detach().numpy()
    y_nn = mlp.fromnptonp(x)
    import matplotlib.pyplot as plt
    plt.plot(x, y, label='true')
    plt.plot(x, y_nn, label='nn')
    plt.legend()
    plt.show()