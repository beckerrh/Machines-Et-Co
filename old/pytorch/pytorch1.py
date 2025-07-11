import numpy as np
import torch
from torch import nn
import pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import mlp, trainer

#-------------------------------------------------------------
if __name__ == '__main__':
    # torch.manual_seed(42)
    x = np.array([0, 1, 3])
    y = np.array([3, 0, 1])
    mlp = mlp.MLP(1,1,3,1)
    tr = trainer.Trainer(mlp, x, y, optimizer="bfgs")
    tr.train()
    y_nn = mlp.fromnptonp(x)

    import matplotlib.pyplot as plt
    plt.plot(x, y, label='true')
    plt.plot(x, y_nn, label='nn')
    plt.legend()
    plt.show()