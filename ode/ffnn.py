import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Integrator1d():
    def __init__(self, n_gauss):
        self.n_gauss = n_gauss
        # sur ]-1;+1[
        self.ref_x, self.ref_w = np.polynomial.legendre.leggauss(self.n_gauss)

#-------------------------------------------------------------
class P1Basis(torch.nn.Module):
    def __init__(self, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        n_dim  = kwargs.pop('n_dim', 1)
        self.n_neurons = kwargs.pop('n_neurons', 2)
        self.n_basis = self.n_neurons
        self.a = kwargs.pop('a', 0)
        self.b = kwargs.pop('b', 1)
        # self.in_layer = torch.nn.Sequential(torch.nn.Linear(n_dim, self.n_neurons), torch.nn.ReLU())
        x0 = torch.linspace(self.a, self.b, self.n_neurons)
        self.act = torch.nn.ReLU()
        # for i in range(self.n_neurons):
        #     self.in_layer[0].bias.data[i] = -x0[i]
        #     self.in_layer[0].weight.data[i,0] = 1
        # self.pos = torch.nn.Parameter(torch.empty(self.n_neurons, **self.factory_kwargs))
        self.pos = torch.empty(self.n_neurons, **self.factory_kwargs)
        for i in range(self.n_neurons):
            self.pos.data[i] = x0[i]
        self.xpos = torch.empty(self.n_neurons+2, **self.factory_kwargs)
        # print(f"{xin.dtype=} {self.xpos.dtype=}")
        self.xpos[1:-1] = self.pos
        self.xpos[0] = 2*self.pos[0] -self.pos[1]
        self.xpos[-1] = 2*self.pos[-1] -self.pos[-2]
        self.h = self.xpos[1:] - self.xpos[:-1]
        self.weight1 = torch.empty((self.n_basis, n_dim), **self.factory_kwargs)
        self.bias1 = torch.empty(self.n_basis, **self.factory_kwargs)
        self.weight2 = torch.empty((self.n_basis, n_dim), **self.factory_kwargs)
        self.bias2 = torch.empty(self.n_basis, **self.factory_kwargs)
        self.bias1[:] = self.xpos[2:]/self.h[1:]
        self.weight1[:,0] = -1/self.h[1:]
        self.bias2[:] = -self.xpos[:-2]/self.h[:-1]
        self.weight2[:,0] = 1/self.h[:-1]
        assert(np.allclose(self.pos[1:]-self.pos[:-1], self.h[1:-1]))
        self.n_gauss = 2
        self.int_x, self.int_w = np.polynomial.legendre.leggauss(self.n_gauss)
        print(f"{self.int_x=} {self.int_w=} {0.5/np.sqrt(3)=}")
        self.x_int = torch.empty((self.n_gauss,self.n_basis-1), **self.factory_kwargs)
        self.w_int = torch.empty((self.n_gauss,self.n_basis-1), **self.factory_kwargs)
        for k in range(self.n_gauss):
            self.x_int[k] = 0.5*(self.pos[:-1]+self.pos[1:]) - 0.5*self.int_x[k]*self.h[1:-1]
            self.w_int[k] = 0.5*self.int_w[k]*self.h[1:-1]

        self.x_int = self.x_int.reshape(-1,1).requires_grad_()
        self.phi_int = self.forward(self.x_int)
        self.phi_0 = self.forward(torch.tensor([self.pos[0]], **self.factory_kwargs).reshape(-1,1))
        # print(f"{self.phi_int.shape=} {self.x_int.shape=}")
        self.phi_x_int = torch.empty_like(self.phi_int)
        for i in range(self.n_basis):
            v = torch.zeros(self.n_basis, **self.factory_kwargs)
            v[i] = 1
            u = self.phi_int@v
            self.phi_x_int[:,i] = torch.autograd.grad(u, self.x_int, (torch.ones_like(u)), create_graph=True)[0][:,0]
        if kwargs.pop('plot', False):
            import matplotlib.pyplot as plt
            import sys
            # self.plot_basis(plt.gca())
            # plt.show()
            x = np.linspace(self.a, self.b)
            xt = torch.from_numpy(x).to(torch.float32).reshape(-1, 1)
            phis = self.forward(xt).detach().numpy()
            # print(f"{self.x_int.shape=} {self.phi_int.shape=}")
            x_int = self.x_int.detach().numpy().flatten()
            phi_int = self.phi_int.detach().numpy()
            phi_x_int = self.phi_x_int.detach().numpy()
            xi = self.pos.detach().numpy()
            for i in range(self.n_neurons):
                p = plt.plot(x, phis[:, i], label=f"$\phi_{{{i}}}$")
                plt.plot(xi[i], 0, 'X', color=p[0].get_color())
                plt.plot(x_int, phi_int[:,i], marker='x', linestyle= '--', color=p[0].get_color(), label=f"$\phi_{{{i}}}$")
            plt.legend()
            plt.show()
            for i in range(self.n_neurons):
                plt.plot(x_int, phi_x_int[:,i], marker='x', label=f"$\phi'_{{{i}}}$")
            plt.legend()
            plt.show()
            sys.exit(1)

    def plot_basis(self, ax=None, x=None, add_title=""):
        import matplotlib.pyplot as plt
        if x is None:
            x = np.linspace(self.a, self.b)
        if ax is None:
            ax = plt.gca()
        ax.set(title="Basis"+add_title, xlabel="t", ylabel="phi")
        xt = torch.from_numpy(x).to(torch.float32).reshape(-1, 1)
        phis = self.forward(xt).detach().numpy()
        xi = self.pos.detach().numpy()
        # xi = -self.in_layer[0].bias[:].detach().numpy() / self.in_layer[0].weight[:, 0].detach().numpy()
        # print(f"{xi=}")
        x = np.linspace(min(xt[0], xi.min()), max(xt[-1], xi.max()), len(xt))
        for i in range(self.n_neurons):
            p = ax.plot(x, phis[:,i], label=f"$\phi_{{{i}}}$")
            ax.plot(xi[i], 0, 'X', color = p[0].get_color())
        ax.legend()

    def forward(self, x):
        vl = torch.nn.functional.linear(x, self.weight1, self.bias1)
        # print(f"{u.shape=}")
        vr = torch.nn.functional.linear(x, self.weight2, self.bias2)
        return self.act(vl-self.act(vl-vr))
        # return self.act(x-self.pos)
        # return self.in_layer(x)
    def u(self, coef, x=None):
        assert coef.shape[1] == self.n_basis
        phis = self.phi_int.T if x is None else self.phi(x.reshape(-1,1)).T
        # phis = self.phi(x.reshape(-1,1)).T
        return coef@phis
    def u_0(self, coef):
        assert coef.shape[1] == self.n_basis
        phis = self.phi_0.T
        # print(f"{phis.shape=} {coef.shape=}")
        # print(f"{phis.dtype=} {coef.dtype=}")
        return coef@phis
    def u_t(self, coef):
        assert coef.shape[1] == self.n_basis
        phis_x = self.phi_x_int.T
        return coef@phis_x

#-------------------------------------------------------------
class PINN(torch.nn.Module):
    def __init__(self, app, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        torch.manual_seed(42)
        # self.flatten = torch.nn.Flatten()
        self.app = app
        self.ncomp = len(app.u0)
        n_dim  = kwargs.pop('n_dim', 1)
        n_layers = kwargs.pop('n_layers', 2)
        n_neurons = kwargs.pop('n_neurons', 2)
        self.n_neurons = n_neurons
        self.x_boundary = torch.tensor([app.domain[0]]).reshape(-1,1)
        ncomp = len(app.u0) if isinstance(app.u0, (list, tuple, np.ndarray)) else 1
        self.random_colloc = kwargs.pop('random_colloc', False)
        act = kwargs.pop('act', 'tanh')
        if act == "tanh":
            actfct = torch.nn.Tanh()
        elif act == "relu":
            actfct = torch.nn.ReLU()
        elif act == "sigmoid":
            actfct = torch.nn.Sigmoid()
        elif act == "softmax":
            actfct = torch.nn.Softmax()
        else:
            raise ValueError(f"unknown activation fct {act=}")
        self.in_layer = P1Basis(n_dim=n_dim, n_neurons=n_neurons, a=app.domain[0], b=app.domain[1], plot=kwargs.pop('plot', False))
        layers = []
        layers.extend(n_layers*[torch.nn.Linear(n_neurons, n_neurons), actfct])
        self.network = torch.nn.Sequential(*layers)
        self.out_layer = torch.nn.Linear(n_neurons, ncomp, bias=False)
        self.u0 = torch.tensor(np.array([app.u0]))
    def forward(self, x):
        phis = self.in_layer(x)
        phis_trans = self.network(phis)
        return self.out_layer(phis_trans)
    def residual_unprec(self, coef):
        res = torch.empty(self.ncomp, self.in_layer.n_basis, **self.factory_kwargs)
        # coef = self.out_layer.weight
        u0 = self.in_layer.u_0(coef).squeeze()
        # print(f"{u0.shape=} {self.u0.shape=}")
        res[:, 0] = u0-self.u0
        res[:,1:] = coef[:,1:]-coef[:,:-1]
        if hasattr(self.app, 'f'):
            u = self.in_layer.u(coef)
            fu = torch.stack(self.app.f(u)).reshape(self.ncomp,self.in_layer.n_gauss,-1)
            for k in range(self.in_layer.n_gauss):
                res[:, 1:] -= fu[:, k, :] * self.in_layer.w_int[k]
        if hasattr(self.app, 'l'):
            lu = torch.stack(self.app.l(self.in_layer.x_int)).reshape(self.ncomp,self.in_layer.n_gauss, -1)
            for k in range(self.in_layer.n_gauss):
                res[:,1:] -= lu[:,k,:]*self.in_layer.w_int[k]
        return res

    def residual(self):
        ncomp, nb = self.ncomp, self.in_layer.n_basis
        coef = self.out_layer.weight
        J = torch.autograd.functional.jacobian(self.residual_unprec, coef)
        J = J.reshape(ncomp * nb, ncomp * nb)
        # print(f"{J=}")
        res = self.residual_unprec(coef).reshape(ncomp * nb)
        return torch.linalg.solve(J, res).reshape(ncomp, nb)

    def loss_fct(self):
        r = self.residual()
        return torch.sum(r*r)

    def closure(self):
        self.optimiser.zero_grad()
        self.loss = self.loss_fct()
        self.loss.backward(retain_graph=True)
        return self.loss

    def train_pinn(self, n_iter, learning_rate, rtol=1e-6, gtol=1e-10, filter=10):
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # self.optimiser = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        self.optimiser = torch.optim.LBFGS(self.parameters())
        res_history = []
        for i in range(n_iter):
            if self.random_colloc:
                self.x_colloc = self.app.domain[0] + (self.app.domain[1]-self.app.domain[0])*torch.rand(self.n_colloc, requires_grad=True)
            self.optimiser.step(self.closure)
            loss = float(self.loss)
            if i==0: rtol *= loss
            if i%filter==0:
                print(f"Iter {i} loss {float(loss)}")
            res_history.append(float(loss))
            if loss < rtol or loss < gtol: return res_history
        return res_history
    def u(self, x):
        xt = torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(-1,1)
        return self.forward(xt).detach().numpy().T
#-------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import applications, applications_analytical_solution
    import ffnn
    # original
    n_layers, n_neurons = 5, 5
    n_layers, n_neurons = 0, 10
    n_iter, rtol, learning_rate = 10000, 1e-6, 1

    # n_layers, n_neurons, n_colloc, n_iter = 1, 3, 10, 3

    # app = app.Logistic(R=0.2, T=30, u0=0.1)
    # app = app.Logistic(k=2, T=4, G=1., u0=0.5)

    app = applications_analytical_solution.SinusIntegration()
    app = applications_analytical_solution.Exponential()
    app = applications_analytical_solution.Oscillator()
    app = applications.Pendulum(goL=2, is_linear=False)

    # app = Lorenz()

    # act = "relu"
    act = "tanh"
    # act = "sigmoid"
    # act = "softmax"
    # act = "abs"
    # model = pinn_pytorch.PINN(app=app, n_dim=1, n_layers=n_layers, n_neurons=n_neurons, n_colloc=n_colloc, act=act)
    kwargs = {"app": app, "n_layers":n_layers, "n_neurons":n_neurons, "act":act}
    model = ffnn.PINN(**kwargs, plot=False)

    model.in_layer.plot_basis(add_title=' --initial--')
    plt.show()
    res_history = model.train_pinn(n_iter, learning_rate, rtol)
    plt.semilogy(res_history, label="perte")
    plt.gca().set(title="Loss evolution", xlabel="# epochs", ylabel="")
    t_plot = np.linspace(app.domain[0], app.domain[1], 100)
    u_plot = model.u(t_plot)
    print(f"{u_plot.shape=}")
    if hasattr(app, 'plot'):
        app.plot(t=t_plot, u=u_plot)
    app.plot_solution(t=t_plot, u=u_plot, show=True)

    # an_sol = hasattr(app, 'sol_analytic')
    # fig, axs = plt.subplots(4 + an_sol, 1, constrained_layout=True, figsize=(10, 10))
    # axs[0].set(title="ColNN", xlabel="t", ylabel="u(t)")
    # x_plot = np.linspace(app.domain[0], app.domain[1], 100)
    # u_plot = model.u(x_plot)
    # model.in_layer.plot_basis(axs[3 + an_sol], x_plot, add_title=' --initial--')
    #
    #
    # # plotting
    # u_plot = model.u(x_plot)
    # x_colloc = model.in_layer.x_int.detach().numpy()
    #
    # if u_plot.ndim > 1:
    #     for i in range(u_plot.shape[1]):
    #         axs[0].plot(x_plot, u_plot[:, i], label="NN_" + str(i))
    # else:
    #     axs[0].plot(x_plot, u_plot, label="NN")
    # p0, = axs[0].plot(x_colloc, np.ones_like(x_colloc) * u_plot.mean(), 'x', color="tab:red", label="colloc")
    # if an_sol:
    #     u_ex = np.stack(app.sol_analytic(x_plot)).reshape(u_plot.shape)
    #     axs[0].plot(x_plot, u_ex, label=f"solution", color="tab:green", alpha=0.75)
    #     axs[1].tick_params(axis='y', labelcolor='tab:orange')
    #     axs[1].set_ylabel('err', color='tab:orange')
    #     p3, = axs[1].plot(x_plot, u_plot - u_ex, color='tab:orange', label='err')
    # axs[1 + an_sol].semilogy(res_history, label="perte")
    # axs[1 + an_sol].set(title="Loss evolution", xlabel="# epochs", ylabel="")
    # model.in_layer.plot_basis(axs[2 + an_sol], x_plot)
    # for ax in axs:
    #     ax.grid()
    #     ax.legend()
    # plt.show()
