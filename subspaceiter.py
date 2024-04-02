import pathlib, sys
localpath = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0,localpath)

from functools import partial
import numpy as np
import torch
import p1_1d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NNFull(torch.nn.Module):
    def __init__(self, nneurons, nlayers=1, actfct=None, **kwargs):
        super().__init__()
        if actfct is None: actfct = torch.nn.ReLU()
        layers = [torch.nn.Linear(1, nneurons, device=DEVICE)]
        layers.append(actfct)
        for i in range(nlayers):
            layers.append(torch.nn.Linear(nneurons, nneurons, device=DEVICE))
            layers.append(actfct)
        self.basis = torch.nn.Sequential(*layers)
        # self.basis[0].weight.data[0] = 0
        # self.basis[0].weight.data[1] = 1
        # self.basis[0].bias.data[0] = 1
        # self.basis[0].bias.data[1] = 0
    def forward(self, x):
        return self.basis(x)
    def regularize(self, eps):
        return eps*torch.mean(self.basis[0].weight.data**2)
        return eps*torch.mean(torch.abs(self.basis[0].weight.data))


#-------------------------------------------------------------
class Machine(torch.nn.Module):
    def __init__(self, data, nneurons, actfct=None, **kwargs):
        super().__init__()
        self.data_x, self.data_y = data[0].reshape(-1, 1), data[1].reshape(-1, 1)
        type = kwargs.pop("type", "nn")
        if type=="fem":
            self.basis = p1_1d.P1_1d(a=0, b=2, n_neurons=nneurons)
        else:
            self.basis = NNFull(nneurons=nneurons)

        # nin = self.basis[0].in_features
        # nbasis = self.basis[-1].out_features
        self.coeffs = torch.zeros(nneurons, requires_grad=False)
        # self.out = torch.nn.Linear(nin, nbasis, bias=False, device=DEVICE)
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, x):
        phis = self.basis(x.reshape(-1,1))
        print(f"{self.coeffs.requires_grad=} {x.shape=} {phis.shape=} {self.coeffs.shape=}")
        return phis@self.coeffs
    def solve(self):
        with torch.no_grad():
            M = self.basis(self.data_x.reshape(-1,1))
            coeffs, res, rank, singular_values = torch.linalg.lstsq(M, self.data_y)
            coeffs = coeffs.flatten()
            if not coeffs.shape==self.coeffs.shape:
                raise ValueError(f"{coeffs.shape=} {self.coeffs.shape=}")
            self.coeffs = coeffs
            residuals = (M@self.coeffs).flatten() - self.data_y
        return torch.mean(residuals ** 2)
    def closure(self, lossfct, optimiser):
        optimiser.zero_grad()
        loss = lossfct()
        loss.backward(retain_graph=True)
        # loss.backward()
        return loss
    def optimize(self, lossfct, optimiser, niter=100, rtol=1e-6, gtol=1e-9, out=0, dtol=1e-12):
        trn_loss_history = []
        if out > niter: filter = 1
        elif out == 0: filter = niter
        else: filter = niter//out
        if(out): print(f"{'Iter':^6s}  {'loss':^12s} {'diffres':^12s}")
        for i in range(niter):
            ret = optimiser.step(partial(self.closure,lossfct=lossfct,optimiser=optimiser))
            # print(f"{ret=}")
            loss = abs(float(ret))
            diffres = torch.nan
            if i == 0:
                tol = max(rtol * loss, gtol)
            else:
                diffres = abs(loss - trn_loss_history[-1])
            if out and i % filter == 0:
                print(f"{i:6d} {loss:14.6e} {diffres:12.6e}")
            trn_loss_history.append(loss)
            if loss < tol:
                return trn_loss_history
            if i and loss < tol and diffres < dtol:
                return trn_loss_history
        return trn_loss_history
        # print(f"Iter {i:6d} loss {loss:12.8e}")
            # for name, parameter in self.named_parameters():
            #     print(f"{name=} {parameter=} {parameter.requires_grad=}")
            # for parameter in self.parameters():
            #     print(f"{parameter=}")
    def init(self, machine):
        oldsolution = machine(self.data_x)
        optimiser = torch.optim.Adam(self.parameters(), lr=0.01)
        # optimiser = torch.optim.LBFGS(self.parameters())
        loss = partial(self.loss_interpolate, oldsolution=oldsolution)
        hist = self.optimize(lossfct=loss, optimiser=optimiser, niter= 100, out=10)
    def loss_interpolate(self, oldsolution):
        M = self.basis(self.data_x.reshape(-1, 1))
        self.coeffs, res, rank, singular_values = torch.linalg.lstsq(M, oldsolution)
        residuals = (M@self.coeffs).flatten() - oldsolution
        return torch.mean(residuals ** 2)
    def loss_train(self, residual_old):
        M = self.basis(self.data_x.reshape(-1, 1))
        # print(f"train {M=}")
        self.coeffs, res, rank, sv = torch.linalg.lstsq(M, residual_old)
        # print(f"train {rank=} {res=} {sv=}")
        v = (M@self.coeffs).flatten()
        print(f"{v=} {M=}")
        return -torch.sqrt(torch.mean(v*v))
        # return -torch.sqrt(torch.mean(v * v)) - self.basis.regularize(0.001)

    def train(self, machine):
        residual_old = machine(self.data_x) - self.data_y
        optimiser = torch.optim.Adam(self.parameters(), lr=0.01)
        # optimiser = torch.optim.LBFGS(self.parameters())
        loss = partial(self.loss_train, residual_old=residual_old)
        hist = self.optimize(lossfct=loss, optimiser=optimiser, niter= 100, out=10)
        # normalize !!!!
        norm = np.sqrt(np.mean(self.forward(self.data_x).detach().numpy()**2))
        self.coeffs /= norm
        return hist[-1]

#-------------------------------------------------------------
class BigMachine(torch.nn.Module):
    def __init__(self, machines, data):
        super().__init__()
        self.machines = machines
        self.data_x, self.data_y = data[0], data[1]
        self.coeffs = torch.zeros(len(machines))
        for machine in machines: machine.freeze()
    def forward(self, x):
        fx = torch.zeros_like(x.flatten())
        for i,machine in enumerate(self.machines):
            print(f"{fx.shape=} {machine(x).flatten().shape=} {self.coeffs.shape=}")
            fx += self.coeffs[i]*machine(x).flatten()
        return fx
    def add(self, machine):
        machine.freeze()
        self.machines.append(machine)
    def solve(self):
        with torch.no_grad():
            M = torch.zeros(len(self.data_x), len(self.machines))
            for i,machine in enumerate(self.machines):
                print(f"{M.shape=} {machine(self.data_x).shape=}")
                M[:,i] = machine(self.data_x).flatten()
            self.coeffs, res, rank, singular_values = torch.linalg.lstsq(M, self.data_y)
            residuals = (M@self.coeffs).flatten() - self.data_y
        return torch.mean(residuals ** 2)
    def plot_basis(self, ax, **kwargs):
        npoints = kwargs.get("npoints",100)
        x = torch.linspace(self.data_x.min(), self.data_x.max(), steps=npoints)
        for i, machine in enumerate(self.machines):
            ax.plot(x, machine(x).detach().numpy(), label=r"$\Phi\_$"+f"{i:03d}")
        ax.legend()

#-------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    f = lambda x: np.sin( np.exp(1.5*x))
    f = lambda x: np.maximum((x-0.3)*(x-1.6), 0.25-(x-1)**2)
    x = torch.linspace(0, 2, 150, requires_grad=False)
    y = f(x)

    machine = Machine(data=(x,y), type="fem", nneurons=5)
    basis = machine.basis(x.reshape(-1,1)).detach().numpy()
    fig, axs = plt.subplots(1,2, sharex=True)
    for inbase in range(basis.shape[1]):
        axs[0].plot(x, basis[:,inbase], label=f"{inbase}")
    axs[0].legend()
    machine.solve()
    machine.freeze()
    axs[1].plot(x, y, label="u_ex")
    axs[1].plot(x, machine(x).detach().numpy(), label="f1")
    axs[1].legend()
    plt.show()

    niter = 10
    bigmachine = BigMachine(machines=[machine], data=(x,y))
    torch.autograd.set_detect_anomaly(True)
    for iter in range(niter):
        bigmachine.solve()
        yiter = bigmachine(x).detach().numpy()
        err = np.sqrt(np.mean((yiter-y.numpy())**2))
        fig, axs = plt.subplots(2,1, sharex=True)
        fig.suptitle(f'This is iteration no.{iter}', fontsize=16)
        p1, =axs[0].plot(x, yiter, label="u_"+f"{iter}")
        p2, = axs[0].plot(x, y, '--y', label="u_ex")
        ax2 = axs[0].twinx()
        ax2.set_ylabel('err', color='tab:orange')
        # print(f"{u_plot.shape=} {u_ex.shape=}")
        p3, = ax2.plot(x, np.abs(yiter-y.numpy()), color='tab:orange', label='err')
        axs[0].legend(handles=[p1,p2,p3])
        bigmachine.plot_basis(axs[1])
        plt.show()
        if iter==niter-1: break
        machine = Machine(data=(x, y), nlayers=1, nneurons=2**(iter+1), actfct=torch.nn.ReLU())
        machine.init(bigmachine)
        eta = machine.train(bigmachine)
        bigmachine.add(machine)
        print(f"{iter=} {err=} {eta=}")