# Multi-layer Perceptron

import torch

#-------------------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, depth, actfct=torch.nn.Tanh()):
        super().__init__()
        layers = [torch.nn.Linear(input_size, hidden_size)]
        layers.append(actfct)
        for i in range(depth):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(actfct)
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.nn = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.nn(x)
