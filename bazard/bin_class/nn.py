import torch

#-------------------------------------------------------------
class Classifier_NN(torch.nn.Module):
    def __init__(self, in_features, out_features, n_neurons, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        # print(f"{self.factory_kwargs=}")
        super().__init__()
        actfct = kwargs.pop('actfct', torch.nn.ReLU())
        n_layers = kwargs.pop('n_layers', 0)
        layers = [torch.nn.Linear(in_features, n_neurons, **self.factory_kwargs)]
        layers.append(actfct)
        for i in range(n_layers):
            layers.append(torch.nn.Linear(n_neurons, n_neurons, **self.factory_kwargs))
            layers.append(actfct)
        layers.append(torch.nn.Linear(n_neurons, out_features, **self.factory_kwargs))
        layers.append(torch.nn.Tanh())
        self.network = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)