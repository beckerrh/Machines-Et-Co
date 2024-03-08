import numpy as np
import torch
import matplotlib.pyplot as plt

#-------------------------------------------------------------
class Integrator1d():
    def __init__(self, n_gauss):
        self.n_gauss = n_gauss
        # sur ]-1;+1[
        self.ref_x, self.ref_w = np.polynomial.legendre.leggauss(self.n_gauss)

#-------------------------------------------------------------
class P1_1d(torch.nn.Module):
    """
    phi_i(x) = max(0, min( (x-x_{i-1})/(x_{i}-x_{i-1}), (x_{i+1}-x)/(x_{i+1}-x_{i})))
    u_i := (x_{i+1}-x)/(x_{i+1}-x_i)
    v_i := (x-x_{i})/(x_{i+1}-x_{i})
    phi_i(x) = max(0, min( u_i, v_{i-1}))
    """
    def __init__(self, a, b, n_neurons, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        n = n_neurons
        self.a, self.b, self.n_neurons = a, b, n
        self.in_features = kwargs.pop('n_dim', 1)
        # self.out_features = kwargs.pop('n_neurons', 10)
        self.out_features = n_neurons
        self.dim = 1
        self.nnodes = n_neurons
        self.dp = torch.nn.Parameter(torch.empty(self.nnodes-1, **self.factory_kwargs))
        self.dp.data[:] = 0
        self.mesh_points = torch.empty(self.nnodes, **self.factory_kwargs)
        self.mesh_points[0] = self.a
        self.mesh_points[-1] = self.b
        self.actfct = kwargs.pop('act', torch.nn.ReLU())
        self.compute_phis()
    def regularize(self):
        return 0.00001*torch.sum(self.dp**2)
    def compute_mesh(self):
        return self.mesh_points.detach().numpy()
    def compute_phis(self):
        delta = torch.exp(self.dp)
        cs = torch.cumsum(delta, dim=0)
        self.mesh_points[1:] = self.a + (self.b-self.a)*cs/cs[-1]
        xpos = torch.empty(len(self.mesh_points)+2, **self.factory_kwargs)
        # print(f"{xin.dtype=} {xpos.dtype=}")
        xpos[1:-1] = self.mesh_points
        xpos[0] = 2*self.mesh_points[0] -self.mesh_points[1]
        xpos[-1] = 2*self.mesh_points[-1] -self.mesh_points[-2]
        self.h = xpos[1:] - xpos[:-1]
        self.weight1 = torch.empty((self.nnodes, self.dim), **self.factory_kwargs)
        self.bias1 = torch.empty(self.nnodes, **self.factory_kwargs)
        self.weight2 = torch.empty((self.nnodes, self.dim), **self.factory_kwargs)
        self.bias2 = torch.empty(self.nnodes, **self.factory_kwargs)
        self.bias1[:] = xpos[2:]/self.h[1:]
        self.weight1[:,0] = -1/self.h[1:]
        self.bias2[:] = -xpos[:-2]/self.h[:-1]
        self.weight2[:,0] = 1/self.h[:-1]
    def phi(self, x):
        vl = torch.nn.functional.linear(x, self.weight1, self.bias1)
        vr = torch.nn.functional.linear(x, self.weight2, self.bias2)
        return self.actfct(vl-self.actfct(vl-vr))
    def forward(self, x):
        self.compute_phis()
        return self.phi(x)

#-------------------------------------------------------------
if __name__ == "__main__":
    fem = P1_1d(-1, 1, 10)
    fem.plot_basis()
    plt.show()
