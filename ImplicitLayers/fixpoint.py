import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TanhFixedPointLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        # initialize output z to be zero
        z = torch.zeros_like(x)
        self.iterations = 0

        # iterate until convergence
        self.err = []
        while self.iterations < self.max_iter:
            z_next = torch.tanh(self.linear(z) + x)
            err = torch.norm(z - z_next)
            self.err.append(err)
            z = z_next
            self.iterations += 1
            if err < self.tol:
                break

        return z


class TanhNewtonImplicitLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        # Run Newton's method outside of the autograd framework
        with torch.no_grad():
            z = torch.tanh(x)
            self.iterations = 0
            while self.iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break

                # newton step
                J = torch.eye(z.shape[1])[None, :, :] - (1 / torch.cosh(z_linear) ** 2)[:, :,
                                                        None] * self.linear.weight[None, :, :]
                z = z - torch.linalg.solve(g[:, :, None], J)[0][:, :, 0]
                self.iterations += 1

        # reengage autograd and add the gradient hook
        z = torch.tanh(self.linear(z) + x)
        z.register_hook(lambda grad: torch.solve(grad[:, :, None], J.transpose(1, 2))[0][:, :, 0])
        return z


def simple_test():
    layer = TanhFixedPointLayer(50)
    X = torch.randn(10,50)
    Z = layer(X)
    print(f"Terminated after {layer.iterations} iterations with error {layer.err[-1]}")
    err = np.array([float(e) for e in layer.err])
    plt.plot(err)
    plt.show()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.optim as optim

torch.manual_seed(0)
model = nn.Sequential(nn.Flatten(),
                      nn.Linear(784, 100),
                      # nn.ReLU(),
                      # TanhFixedPointLayer(100, max_iter=200),
                      TanhNewtonImplicitLayer(100, max_iter=200),
                      nn.Linear(100, 10)
                      # ,nn.Softmax(dim=1)
                      ).to(device)
opt = optim.SGD(model.parameters(), lr=1e-1)

from tqdm.notebook import tqdm


def epoch(loader, model, opt=None, monitor=None):
    total_loss, total_err, total_monitor = 0., 0., 0.
    model.eval() if opt is None else model.train()
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            if sum(torch.sum(torch.isnan(p.grad)) for p in model.parameters()) == 0:
                opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if monitor is not None:
            total_monitor += monitor(model)
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), total_monitor / len(loader)


for i in range(10):
    if i == 5:
        opt.param_groups[0]["lr"] = 1e-2

    train_err, train_loss, train_fpiter = epoch(train_loader, model, opt, lambda x : x[2].iterations)
    test_err, test_loss, test_fpiter = epoch(test_loader, model, monitor = lambda x : x[2].iterations)
    print(f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f}, FP Iters: {train_fpiter:.2f} | " +
          f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}, FP Iters: {test_fpiter:.2f}")

    # train_err, train_loss, train_fpiter = epoch(train_loader, model, opt)
    # test_err, test_loss, train_fpiter = epoch(test_loader, model, monitor = None)
    # print(f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f} | " +
    #       f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}")