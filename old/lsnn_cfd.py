import torch
from sklearn import svm, gaussian_process
import mlp, trainer

# CUDA support
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

#-------------------------------------------------------------
def train_svm(x,y):
    # machine = svm.SVR(kernel='poly', C=100, degree=2, epsilon=0.1, verbose=True)
    machine = svm.SVR(kernel='rbf', C=100, gamma='auto', epsilon=1., verbose=True)
    machine.fit(x, y)
    return machine

#-------------------------------------------------------------
def train_nn(model, x,y):
    tr = trainer.Trainer(model, x, y, optimizer="bfgs")
    tr.train(niter=600, out=100)
    return

#-------------------------------------------------------------
class Pinn():
    def __init__(self, model, points, v, T):
        self.model = model
        self.points, self.T = points, T
        self.x, self.y = torch.tensor(points[:,0], requires_grad=True), torch.tensor(points[:,1], requires_grad=True)
        self.v = torch.tensor(v[0]), torch.tensor(v[1])
    def parameters(self):
        return self.model.parameters()
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    def loss_function(self, inputs, targets):
        # u = self.model(inputs)
        inputs2 = torch.stack([self.x, self.y]).to(dtype=self.model.dtype).T
        # print(f"{inputs.shape=} {inputs2.shape=}")
        u = self.model(inputs2)
        u_x = torch.autograd.grad(u, self.x, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True)[0]
        # print(f"{u_x.shape=}")
        u_xx = torch.autograd.grad(u_x.reshape(-1,1), self.x, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, self.y, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.reshape(-1,1), self.y, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True)[0]
        # print(f"{u_x.shape=} {torch.linalg.norm(u_x)} {self.v[0].shape=} {torch.linalg.norm(self.v[0])}")
        pderes = self.v[0]*u_x + self.v[1]*u_y# - 0.001*(u_xx+u_yy)
        pdeloss = torch.mean(pderes**2)
        # return torch.mean((u-targets)**2)
        return pdeloss
        return torch.mean((u-targets)**2) + pdeloss


#-------------------------------------------------------------
def train_pinn(model, data):
    points, T, v1, v2 = data['points'][:,:-1], data['T'], data['V1'], data['V2']
    pinns = Pinn(model, points, v=[v1,v2], T=T)
    tr = trainer.Trainer(pinns, points, T, optimizer="bfgs")
    tr.train(niter=300, out=10)
    return

#-------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import pathlib
    import get_fem_data
    import copy
    import matplotlib.pyplot as plt

    datadir = pathlib.Path("data")
    if datadir.is_dir():
        data = {}
        for p in datadir.glob("*.npy"):
            name = p.name.split('.npy')[0]
            data[name] = np.load(p)
    else:
        datadir.mkdir()
        data = get_fem_data.get_fem_data(plotting=True, verbose=1)
        for k,v in data.items(): np.save(datadir / k, v)

    # oublie z
    points, T, simp, bdry = data['points'][:,:-1], data['T'], data['simp'], data['bdry']
    print(f"{points.shape=} {T.shape=}")

    machine = train_svm(points, T)
    svi = machine.support_
    sv = points[svi]
    Tsvm = machine.predict(points)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = mlp.MLP(input_size=2, output_size=1, hidden_size=60, depth=8, actfct=torch.nn.ReLU()).to(device)
    # train_nn(model, points, T)
    train_pinn(model, data)

    # Tnn = model(torch.from_numpy(points).to(dtype=dtype)).detach().numpy().reshape(-1)
    Tnn = model.fromnptonp(points)
    print(f"{T.mean()=} {T.min()=} {T.max()=}")
    print(f"{Tnn.mean()=} {Tnn.min()=} {Tnn.max()=}")
    print(f"{Tsvm.mean()=} {Tsvm.min()=} {Tsvm.max()=}")
    print(f"{T.shape=} {svi.shape=}")
    fig, axs = plt.subplots(1, 3, sharex=True)
    axs[0].set_title(f"fem")
    axs[0].tricontourf(points[:,0], points[:,1], simp, T)
    axs[1].set_title(f"nn")
    axs[1].tricontourf(points[:,0], points[:,1], simp, Tnn)
    axs[2].set_title(f"svm")
    axs[2].tricontourf(points[:,0], points[:,1], simp, Tsvm)
    axs[2].scatter(points[svi,0], points[svi,1], c='r', marker='x', alpha=0.25)
    plt.show()
