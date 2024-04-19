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
        self.layers = torch.nn.Sequential(*layers)
        self.dtype = self.layers[0].weight.dtype
    def forward(self, x):
        return self.layers(x)
    def print_parameters(self):
        for p in self.layers.parameters():
            print(f"{p.data=} {p.grad=}")
    def fromnptonp(self, x):
        print(f"{x.shape=}")
        if x.ndim==1: x = x.reshape(-1,1)
        return self(torch.from_numpy(x).to(self.dtype)).detach().numpy().reshape(-1)
