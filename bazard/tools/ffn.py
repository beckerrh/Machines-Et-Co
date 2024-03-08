import torch
import matplotlib.pyplot as plt
#-------------------------------------------------------------
class Normalize(torch.nn.Module):
    def __init__(self, n_neurons, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        # print(f"{self.factory_kwargs=}")
        super().__init__()
    def forward(self, x):
        return x / x.sum(dim=1, keepdim=True)
#-------------------------------------------------------------
class FFN_1d(torch.nn.Module):
    def __init__(self, a, b, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        # print(f"{self.factory_kwargs=}")
        super().__init__()
        self.in_features = kwargs.pop('n_dim', 1)
        self.out_features = kwargs.pop('n_neurons', 10)
        n_dim  = self.in_features
        n_neurons = self.out_features
        self.a, self.b, self.n_neurons = a, b, n_neurons
        actfct = kwargs.pop('act', torch.nn.ReLU())
        layers = [torch.nn.Linear(n_dim, n_neurons, **self.factory_kwargs)]
        layers.append(actfct)
        # layers.append(torch.nn.Linear(n_neurons, n_neurons, **self.factory_kwargs, bias=False))
        # layers.append(Normalize(n_neurons))
        self.network = torch.nn.Sequential(*layers)
        # init
        init = True
        if init:
            x = torch.linspace(a, b, n_neurons, **self.factory_kwargs, requires_grad=False)
            self.network[0].bias.data[0] = 1
            self.network[0].weight.data[0, 0] = -1
            for i in range(n_neurons-1):
                self.network[0].bias.data[i+1] = -x[i]
                self.network[0].weight.data[i+1,0] = 1
        # self.network[2].weight.data = torch.eye(n_neurons, dtype=dtype)
    def compute_mesh(self):
        from itertools import combinations
        W, b = self.network[0].weight, self.network[0].bias
        m, n = W.shape
        points = []
        for combination in combinations(range(m), n):
            Wsmall, bsmall = W[list(combination), :], b[list(combination)]
            try:
                points.append(-torch.linalg.solve(Wsmall,bsmall).detach().numpy())
            except:
                pass
        # print(f"{m=} {n=} {points=} {b.shape=} {W.shape=}")
        return points
    def regularize(self):
        return 1e-10*torch.sum(self.network[0].weight**2)
    def recombine_for_plotting(self):
        return
        x = torch.linspace(self.a, self.b, self.n_neurons, **self.factory_kwargs, requires_grad=False)
        M = self.network(x.reshape(-1,1))
        self.combine = torch.linalg.inv(M)
    def forward(self, x):
        x = x.reshape(-1,1)
        return self.network(x)
#-------------------------------------------------------------
if __name__ == "__main__":
    import plot
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fem = FFN_1d(-1, 1, n_neurons=10, device=DEVICE)
    fem.recombine_for_plotting()
    plot.plot_basis(fem)
    plt.show()
