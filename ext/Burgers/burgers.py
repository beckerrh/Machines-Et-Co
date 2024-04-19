#https://github.com/jayroxis/PINNs/blob/master/Burgers%20Equation/Burgers.ipynb
import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import mlp
sns.set_style("white")

np.random.seed(123456)
torch.manual_seed(0)

#-------------------------------------------------------------
class Net:
    def __init__(self, model, device):
        self.model = model

        self.h = 0.1
        self.k = 0.1
        x = torch.arange(-1, 1 + self.h, self.h)
        t = torch.arange(0, 1 + self.k, self.k)

        # self.X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
        X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T

        # training data
        bc1 = torch.stack(torch.meshgrid(x[0], t)).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(x[-1], t)).reshape(2, -1).T
        ic = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T
        self.X_train = torch.cat([bc1, bc2, ic])
        y_bc1 = torch.zeros(len(bc1))
        y_bc2 = torch.zeros(len(bc2))
        y_ic = -torch.sin(math.pi * ic[:, 0])
        self.y_train = torch.cat([y_bc1, y_bc2, y_ic])
        self.y_train = self.y_train.unsqueeze(1)

        # self.X = self.X.to(device)
        self.X_train = self.X_train.to(device)
        self.y_train = self.y_train.to(device)
        # self.X.requires_grad = True

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.criterion = torch.nn.MSELoss()
        self.iter = 1

        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # better numerical stability
        )

        self.adam = torch.optim.Adam(self.model.parameters())

    def loss_func(self):
        # this is more like a not so elegant hack to zero grad both optimizers
        self.adam.zero_grad()
        self.optimizer.zero_grad()

        y_pred = self.model(self.X_train)
        loss_data = self.criterion(y_pred, self.y_train)
        # u = self.model(self.X)
        t, x = self.t, self.x
        u = self.model(torch.cat([x,t], dim=1))

        # x = torch.tensor(self.X_train[:, 0:1], requires_grad=True).float().to(device)
        # t = torch.tensor(self.X_train[:, 1:2], requires_grad=True).float().to(device)
        # du_dX = torch.autograd.grad(
        #     inputs=self.X,
        #     outputs=u,
        #     grad_outputs=torch.ones_like(u),
        #     retain_graph=True,
        #     create_graph=True
        # )[0]
        # du_dt = du_dX[:, 1]
        # du_dx = du_dX[:, 0]
        # du_dxx = torch.autograd.grad(
        #     inputs=self.X,
        #     outputs=du_dX,
        #     grad_outputs=torch.ones_like(du_dX),
        #     retain_graph=True,
        #     create_graph=True
        # )[0][:, 0]
        # loss_pde = self.criterion(du_dt + u.squeeze() * du_dx, 0.01 / math.pi * du_dxx)

        # u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + u * u_x - 0.01 / math.pi * u_xx
        loss_pde = torch.mean(f**2)

        loss = loss_pde + loss_data
        loss.backward()
        if self.iter % 100 == 0:
            print(self.iter, loss.item())
        self.iter = self.iter + 1
        return loss

    def train(self):
        self.model.train()
        for i in range(1000):
            self.adam.step(self.loss_func)
        self.optimizer.step(self.loss_func)

    # def eval_(self):
    #     self.model.eval()

#-------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = mlp.MLP(
        input_size=2,
        hidden_size=20,
        output_size=1,
        depth=4,
        actfct=torch.nn.Tanh()
    ).to(device)
    net = Net(model, device)
    net.train()
    model.eval()
    h = 0.01
    k = 0.01
    x = torch.arange(-1, 1, h)
    t = torch.arange(0, 1, k)

    X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
    X = X.to(device)
    net.model.eval()
    with torch.no_grad():
        y_pred = net.model(X).reshape(len(x), len(t)).cpu().numpy()
    plt.figure(figsize=(5, 3), dpi=150)
    sns.heatmap(y_pred, cmap='jet')
    plt.show()