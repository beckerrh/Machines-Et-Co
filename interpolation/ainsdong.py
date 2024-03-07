from functools import partial
import numpy as np
import torch
from tools import p1_1d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------
class Machine(torch.nn.Module):
    def __init__(self, data, nlayers, nneurons, actfct=None, **kwargs):
        super().__init__()
        self.data_x, self.data_y = data[0], data[1]
        # layers=[torch.nn.Linear(1, nneurons, device=DEVICE)]
        # for i in range(nlayers):
        #     layers.append(actfct)
        #     layers.append(torch.nn.Linear(nneurons, nneurons, device=DEVICE))
        # self.basis = torch.nn.Sequential(*layers)
        self.basis = p1_1d.P1_1d(a=0, b=2, n_neurons=nneurons)
        # nin = self.basis[0].in_features
        # nbasis = self.basis[-1].out_features
        self.coeffs = torch.zeros(nneurons, requires_grad=False)
        # self.out = torch.nn.Linear(nin, nbasis, bias=False, device=DEVICE)
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, x):
        # print(f"{self.coeffs.requires_grad=}")
        phis = self.basis(x.reshape(-1,1))
        return phis@self.coeffs
    def solve(self):
        with torch.no_grad():
            M = self.basis(self.data_x.reshape(-1,1))
            self.coeffs, res, rank, singular_values = torch.linalg.lstsq(M, self.data_y)
            residuals = (M@self.coeffs).flatten() - self.data_y
        return torch.mean(residuals ** 2)

    def loss_train(self, residual_old):
        M = self.basis(self.data_x.reshape(-1, 1))
        # print(f"{M.shape=}")
        # other = self.other_machine(self.data_x)
        # residual_old = other - self.data_y
        self.coeffs, res, rank, singular_values = torch.linalg.lstsq(M, residual_old)
        v = (M@self.coeffs).flatten()
        return -torch.sqrt(torch.mean(v*v))
        vn = -torch.linalg.norm(v)
        print(f"{residual_old=} {v=} {vn=}")
        # return -vn
        # v /= vn
        # return 1/vn
        return torch.sum(v*residual_old) + 10000*torch.maximum(vn-1,torch.zeros(1))
        # return torch.mean(v*v)
        rayl = torch.mean(v*residual_old)/torch.mean(v*v)
        return rayl
    def loss_interpolate(self):
        M = self.basis(self.data_x.reshape(-1, 1))
        other = self.other_machine(self.data_x)
        self.coeffs, res, rank, singular_values = torch.linalg.lstsq(M, other)
        residuals = (M@self.coeffs).flatten() - other
        # print(f"{residuals=} {torch.mean(residuals ** 2)=}")
        return torch.mean(residuals ** 2)
    def closure(self, lossfct, optimiser):
        optimiser.zero_grad()
        loss = lossfct()
        loss.backward(retain_graph=True)
        return loss
    def optimize(self, lossfct, lr=0.01, niter=100):
        self.optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        # self.optimiser = torch.optim.LBFGS(self.parameters())
        for i in range(niter):
            ret = self.optimiser.step(partial(self.closure,lossfct=lossfct, optimiser=self.optimiser))
            # print(f"{ret=}")
            loss = float(ret)
            print(f"Iter {i:6d} loss {loss:12.8e}")
            # for name, parameter in self.named_parameters():
            #     print(f"{name=} {parameter=} {parameter.requires_grad=}")
            # for parameter in self.parameters():
            #     print(f"{parameter=}")
    def init(self, machine):
        self.other_machine = machine
        self.optimize(lossfct=self.loss_interpolate, lr=0.1, niter=5)
    def train(self, machine):
        self.other_machine = machine
        other = self.other_machine(self.data_x)
        residual_old = other - self.data_y
        self.optimize(lossfct=partial(self.loss_train, residual_old=residual_old), lr=0.1, niter= 50)

#-------------------------------------------------------------
class BigMachine(torch.nn.Module):
    def __init__(self, machines, data):
        super().__init__()
        self.machines = machines
        self.data_x, self.data_y = data[0], data[1]
        self.coeffs = torch.zeros(len(machines))
        for machine in machines: machine.freeze()
    def forward(self, x):
        fx = torch.zeros_like(x)
        for i,machine in enumerate(self.machines):
            fx += self.coeffs[i]*machine(x)
        return fx
    def add(self, machine):
        machine.freeze()
        self.machines.append(machine)
    def solve(self):
        with torch.no_grad():
            M = torch.zeros(len(self.data_x), len(self.machines))
            for i,machine in enumerate(self.machines):
                M[:,i] = machine(self.data_x.reshape(-1,1))
            self.coeffs, res, rank, singular_values = torch.linalg.lstsq(M, self.data_y)
            residuals = (M@self.coeffs).flatten() - self.data_y
        return torch.mean(residuals ** 2)
    def plot_basis(self, ax, **kwargs):
        npoints = kwargs.get("npoints",100)
        x = torch.linspace(self.data_x.min(), self.data_x.max(), steps=npoints)
        for i, machine in enumerate(self.machines):
            ax.plot(x, machine(x).detach().numpy(), label=r"\Phi_"+f"{i:03d}")
        ax.legend()

#-------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    f = lambda x: np.sin( np.exp(1.5*x))
    f = lambda x: np.maximum((x-0.5)*(x-1.5), 0.25-(x-1)**2)
    x = torch.linspace(0, 2, 15, requires_grad=False)
    y = f(x)
    print(f"{x.requires_grad=} {y.requires_grad=}")

    machine = Machine(data=(x,y), nlayers=0, nneurons=2)
    # print(f"{machine.basis[0].weight.data}=")
    # machine.basis[0].weight.data[0] = 0
    # machine.basis[0].weight.data[1] = 1
    # machine.basis[0].bias.data[0] = 1
    # machine.basis[0].bias.data[1] = 0
    basis = machine.basis(x.reshape(-1,1)).detach().numpy()
    fig, axs = plt.subplots(3,1, sharex=True)
    for inbase in range(basis.shape[1]):
        axs[0].plot(x, basis[:,inbase], label=f"{inbase}")
    axs[0].legend()
    machine.solve()
    machine.freeze()
    axs[1].plot(x, y, label="f")
    axs[1].plot(x, machine(x).detach().numpy(), label="f1")
    machine2 = Machine(data=(x,y), nlayers=1, nneurons=5, actfct=torch.nn.ReLU())
    # machine2.init(machine)
    # axs[1].plot(x, machine2(x).detach().numpy(), '--xy', label="f2_0")
    machine2.train(machine)
    # machine2.solve()
    axs[1].plot(x, machine2(x).detach().numpy(), label="f2")
    bigmachine = BigMachine(machines=[machine, machine2], data=(x,y))
    bigmachine.solve()
    axs[1].plot(x, bigmachine(x).detach().numpy(), label="u2")
    axs[1].legend()
    bigmachine.plot_basis(axs[2])
    plt.show()
