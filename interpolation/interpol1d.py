import numpy as np
import torch, tools

#-------------------------------------------------------------
class Interpolater(torch.nn.Module):
    def __init__(self, data, model, **kwargs):
        super().__init__()
        self.print_parameters = kwargs.pop("print_parameters", False)
        self.data = data
        self.model = model
        self.algo = kwargs.pop('algo', 'def')
        if self.algo == "def":
            ncomp = 1
            n = model[0].out_features
            self.out = torch.nn.Linear(n, ncomp, bias=False, **self.model[0].factory_kwargs)
        if self.print_parameters:
            for parameter in self.parameters():
                print(f"{parameter=}")
            print(f"\n----------------------\n")
            for parameter in self.model.parameters():
                print(f"model {parameter=}")
            print(f"\n----------------------\n")

    def forward(self, x):
        phis = self.model(x.reshape(-1,1))
        if self.algo != "def":
            return phis@self.lssolution
        else:
            return self.out(phis)
    def loss_fct(self):
        if self.print_parameters:
            for parameter in self.parameters():
                print(f"{parameter=}")
        M = self.model(self.data['x'].reshape(-1,1))
        # D = torch.einsum('ki,kj->ij', M, M)
        if self.algo == 'def':
            residual = self.out(M).flatten() - self.data['y']
            # print(f"loss = {torch.mean( (M.t()@residual))**2 }")
            return torch.mean(residual ** 2)
        self.lssolution, res, rank, singular_values = torch.linalg.lstsq(M, self.data['y'])
        residuals = (M@self.lssolution).flatten() - self.data['y']
        return torch.mean(residuals ** 2) + self.model[0].regularize()
    def closure(self):
        self.optimiser.zero_grad()
        self.loss = self.loss_fct()
        self.loss.backward(retain_graph=True)
        return self.loss
    def train(self, n_iter=130, learning_rate=0.1, rtol=1e-6, gtol=1e-9, out=20, dtol=1e-12):
        torch.autograd.set_detect_anomaly(True)
        if self.algo == "def":
            parameters =  self.parameters()
        else:
            parameters = self.model.parameters()
        self.optimiser = torch.optim.Adam(parameters, lr=learning_rate)
        # self.optimiser = torch.optim.SGD(parameters, lr=learning_rate/10, momentum=0.9)
        # self.optimiser = torch.optim.LBFGS(parameters)
        res_history = []
        if out > n_iter: filter = 1
        elif out == 0: filter = n_iter
        else: filter = n_iter//out
        for i in range(n_iter):
            self.optimiser.step(self.closure)
            loss = float(self.loss)
            diffres = torch.nan
            if i==0: tol = max(rtol*loss, gtol)
            else: diffres = abs(loss-res_history[-1])
            if out and i%filter==0:
                print(f"Iter {i:6d} loss {loss:12.8e} {diffres:12.8e}")
            res_history.append(float(loss))
            if loss < tol:
                return res_history
            if i and loss < tol and diffres < dtol:
                return res_history
        return res_history


#-------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = lambda x: np.sin( np.exp(1.5*x))
    # f = lambda x: np.sin(2*np.pi*x)
    f = lambda x: np.maximum((x-0.5)*(x-1.5), 0.25-(x-1)**2)
    x = torch.linspace(0, 2, 33)
    y = f(x)
    #
    n_neurons = 5
    #
    n_layers = 3
    algo = 'def'
    n_iter = 1000
    out=0
    algo = 'ls'
    n_iter = 10
    out=10
    actfct = torch.nn.ReLU()

    interpols = {}
    for n_layer in range(n_layers):
        layers = [tools.p1_1d.P1_1d(torch.min(x), torch.max(x), n_neurons=n_neurons, device=DEVICE)]
        for i in range(n_layer):
            layers.append(torch.nn.Linear(n_neurons, n_neurons, device=DEVICE))
            layers.append(actfct)
        name = 'p1'
        if n_layer: name += f'{n_layer}'
        interpols[name] = Interpolater(data={'x':x, 'y':y}, model=torch.nn.Sequential(*layers), algo=algo)

    tools.plot.plot_basis({n:ip.model for n,ip in interpols.items()}, add_title=" start")
    plt.show()
    hist = {}
    for n, ip in interpols.items():
        hist[n] = ip.train(n_iter=n_iter, out=out, rtol=1e-3, dtol=1e-7)
        print(f"n={len(hist[n])} res={hist[n][-1]}")
    for n, h in hist.items():
        plt.loglog(h, label=n)
    plt.legend()
    plt.grid()
    plt.show()
    plt.plot(x, y, 'xr', label='data')
    for n, ip in interpols.items():
        plt.plot(x, ip(x).detach().numpy(), '--', label=n)
    plt.legend()
    plt.grid()
    plt.show()
    tools.plot.plot_basis({n:ip.model for n,ip in interpols.items()}, add_title=" final")
    plt.show()
