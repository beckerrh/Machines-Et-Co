import pathlib, sys
localpath = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0,localpath)

from functools import partial
import numpy as np
import torch
import p1_1d
from sklearn import svm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE=}")
#-------------------------------------------------------------
class Kernel_P1Basis():
    def __repr__(self):
        if self.x is None: return f"P{self.k}"
        return f"P1_"+f"{len(self.x)}"
    def __init__(self, x):
        self.x = x
        self.nnodes = len(x)
        xpos = np.empty(self.nnodes+2)
        xpos[1:-1] = self.x
        xpos[0] = 2*self.x[0] -self.x[1]
        xpos[-1] = 2*self.x[-1] -self.x[-2]
        self.h = xpos[1:] - xpos[:-1]
        self.weight1 = np.empty(shape=(self.nnodes, 1))
        self.bias1 = np.empty(self.nnodes)
        self.weight2 = np.empty(shape=(self.nnodes, 1))
        self.bias2 = np.empty(self.nnodes)
        self.bias1[:] = xpos[2:]/self.h[1:]
        self.weight1[:,0] = -1/self.h[1:]
        self.bias2[:] = -xpos[:-2]/self.h[:-1]
        self.weight2[:,0] = 1/self.h[:-1]
    def phi(self, x):
        print(f"{x.shape=} {self.weight1.shape=} {self.bias1.shape=}")
        vl = np.dot(x, self.weight1.T)+ self.bias1
        vr = np.dot(x, self.weight2.T) + self.bias2
        return np.maximum(0, vl-np.maximum(0,vl-vr))
    def forward(self, x):
        return self.phi(x)
    def __call__(self, X,Y):
        # return np.eye(self.nnodes)
        print(f"{X.shape=}")
        phix = self.forward(X)
        phiy = self.forward(Y)
        print(f"{phix.shape=}")
        print(f"{np.linalg.norm(np.dot(phix,phiy.T)-np.eye(self.nnodes))=}")
        return np.dot(phix.T,phiy)
        return np.dot(phix,phiy.T)


#-------------------------------------------------------------
class Kernel_P1():
    def __repr__(self):
        if self.x is None: return f"P1"
        return f"P1_"+f"{len(self.x)}"
    def __init__(self, x=None):
        self.x = x
    def __call__(self, X,Y):
        if self.x is None:
            xmin = np.minimum(X,Y.T)
            xabs = np.abs(X-Y.T)
            # print(f"{X.shape=} {Y.shape=} {xmin.shape=} {xabs.shape=} {np.dot(X,Y.T).shape=}")
            # return 1 + X*Y.T + X*Y.T*xmin - 0.5*xmin*(X+Y.T)*xmin + xmin**3/3
            #Vapnik
            return 1 + np.dot(X,Y.T) + 0.5 * xabs * xmin ** 2 + xmin ** 3 / 3
        print(f"{np.linalg.norm(X-Y)=}")
        # return np.eye(len(X))
        X1 = np.maximum(0, X-self.x)
        print(f"{X.shape=} {self.x.shape=} {X1.shape=} {np.dot(X,Y.T).shape=}")
        Y1 = np.maximum(0, Y-self.x)
        return 1 + np.dot(X,Y.T)  + np.dot(X1,Y1.T)

#-------------------------------------------------------------
class FullLinear(torch.nn.Module):
    def __init__(self, nneurons, nlayers=0, actfct=None, **kwargs):
        super().__init__()
        layers = [torch.nn.Linear(1, nneurons, device=DEVICE)]
        layers.append(actfct)
        for i in range(nlayers):
            layers.append(torch.nn.Linear(nneurons, nneurons, device=DEVICE))
            layers.append(actfct)
        self.nlayers = nlayers
        self.nn = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.nn(x)
    def regularize(self, eps=0.01):
        reg = 0
        for i in range(self.nlayers+1):
            reg += eps*torch.mean(self.nn[2*i].weight.data**2)
            # reg += eps * torch.mean(torch.abs(self.nn[2 * i].weight.data))
        return reg


#-------------------------------------------------------------
class NNFull(torch.nn.Module):
    def __init__(self, nneurons, nlayers=0, actfct=None, type='full', **kwargs):
        super().__init__()
        if actfct is None: actfct = torch.nn.ReLU()
        if type=='full':
            self.basis = FullLinear(nneurons=nneurons, nlayers=nlayers, actfct=actfct)
            # layers = [torch.nn.Linear(1, nneurons, device=DEVICE)]
            # layers.append(actfct)
            # for i in range(nlayers):
            #     layers.append(torch.nn.Linear(nneurons, nneurons, device=DEVICE))
            #     layers.append(actfct)
        elif type =='fem':
            a = kwargs.pop('a')
            b = kwargs.pop('b')
            self.basis = p1_1d.P1_1d(a,b, nneurons, device=DEVICE)
        else:
            raise ValueError(f"unknown {type=}")
        self.coeff = torch.nn.Linear(nneurons, 1, device=DEVICE, bias=False)
        # layers.append(torch.nn.Linear(nneurons, 1, device=DEVICE, bias=False))
        # self.nlayers = nlayers
        # self.nn = torch.nn.Sequential(*layers)
    def forward(self, x):
        # print(f"{x.shape=} {self.nn(x.reshape(-1, 1)).shape=}")
        return self.coeff(self.basis(x))

    def regularize(self, eps=0.02):
        return self.basis.regularize(eps=eps)

#-------------------------------------------------------------
class Trainer():
    def __init__(self, x, y, machine):
        self.x = x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        assert self.x.shape == self.y.shape
        self.machine = machine
        self.optimizer = torch.optim.LBFGS(machine.parameters(), max_iter=1)
        # self.optimizer = torch.optim.Adam(machine.parameters(), lr=0.1)

    def closure(self):
        self.optimizer.zero_grad()
        res = self.machine(self.x) -self.y
        # print(f"{res=}")
        loss = torch.mean(res**2) + self.machine.regularize(0.01)
        loss.backward(retain_graph=True)
        print(f'loss: {loss}')
        return loss
    def train(self):
        for i in range(30):
            self.optimizer.step(self.closure)
            print(f'{i}')


#-------------------------------------------------------------
def plot_data(data):
    fig, axs = plt.subplots(1,1, sharex=True)
    if not isinstance(axs, list) : axs = [axs]
    for k,v in data.items():
        if len(v)==2:
            axs[0].plot(v[0], v[1], label=k)
        else:
            axs[0].plot(v[0], v[1], v[2], label=k)
    axs[0].legend()
    plt.show()

#-------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    f = lambda x: np.sin( np.exp(1.5*x))
    f = lambda x: np.maximum((x-0.3)*(x-1.6), 0.25-(x-1)**2) + 0.2*(x>1)
    x = torch.linspace(0, 2, 130, requires_grad=False)
    f = lambda x: 0.3+1.2*x
    x = torch.linspace(0, 2, 10, requires_grad=False)
    y = f(x)
    pltdata = {'y_ex': (x.numpy(),y.numpy())}

    regr = svm.SVR(C=2, epsilon=0.1, tol=1e-5, verbose=True)
    regr.fit(x.numpy().reshape(-1, 1), y.numpy())
    svi = regr.support_
    sv = x[svi]
    pltdata['y_svn_rbf'] = x, regr.predict(x.numpy().reshape(-1, 1))
    pltdata[f'x_svn_rbf {len(sv)}'] = sv, 1*y.min()*np.ones(len(sv)), "gX"

    kernel = Kernel_P1(x=x.numpy()[::1])
    # kernel = Kernel_P1()
    regr = svm.SVR(kernel=kernel, C=2, epsilon=0.1, tol=1e-5, verbose=True)
    regr.fit(x.numpy().reshape(-1, 1), y.numpy())
    svi = regr.support_
    sv = x[svi]
    pltdata['y_svn'] = x, regr.predict(x.numpy().reshape(-1, 1))
    pltdata[f'x_svn {len(sv)}'] = sv, 1.2*y.min()*np.ones(len(sv)), "yX"

    kernel = Kernel_P1Basis(x=x.numpy()[::1])
    # kernel = Kernel_P1()
    regr = svm.SVR(kernel=kernel, C=2, epsilon=0.1, tol=1e-5, verbose=True)
    regr.fit(x.numpy().reshape(-1, 1), y.numpy())
    svi = regr.support_
    sv = x[svi]
    pltdata['y_svn2'] = x, regr.predict(x.numpy().reshape(-1, 1))
    pltdata[f'x_svn2 {len(sv)}'] = sv, 1.4*y.min()*np.ones(len(sv)), "rX"

    # nneurons = 12
    # machine = NNFull(nneurons=nneurons, type='fem', a=0, b=2)
    # # machine = NNFull(nneurons=20, nlayers=3)
    # trainer = Trainer(x, y, machine)
    # trainer.train()
    # pltdata['y_nn'] = x, machine(x.reshape(-1, 1)).detach().numpy()
    # xmesh = machine.basis.compute_mesh()
    # pltdata[f'x_nn {nneurons}'] = xmesh, y.min()*np.ones(len(xmesh)), "rX"

    plot_data(pltdata)

