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
class FEM1DNew(torch.nn.Module):
    """
    phi_i(x) = max(0, min( (x-x_{i-1})/(x_{i}-x_{i-1}), (x_{i+1}-x)/(x_{i+1}-x_{i})))
    u_i := (x_{i+1}-x)/(x_{i+1}-x_i)
    v_i := (x-x_{i})/(x_{i+1}-x_{i})
    phi_i(x) = max(0, min( u_i, v_{i-1}))
    """
    def __init__(self, a, b, n, device=None, dtype=None):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.a, self.b = a,b
        self.dim = 1
        self.nnodes = n+1
        self.dp = torch.nn.Parameter(torch.empty(n-1, **self.factory_kwargs))
        x = (torch.linspace(a,b,n+1)[1:-1]-a)/(b-a)
        # inverse de la sigmoide
        self.dp.data[:] = -torch.log((1-x)/x)
        self.mesh_points = torch.empty(self.nnodes, **self.factory_kwargs)
        self.mesh_points[0] = self.a
        self.mesh_points[-1] = self.b
        self.relu = torch.nn.ReLU()
        self.compute_phis()
    def compute_phis(self):
        # print(f"{self.dp=}")
        dps, ind = torch.sort(self.dp)
        # print(f"{dps=}")
        m = torch.nn.Sigmoid()
        ps = m(dps)
        print(f"{ps=}")
        self.mesh_points[1:-1] = self.a + (self.b-self.a)*ps
        print(f"{self.mesh_points=}")
        xin = self.mesh_points
        xpos = torch.empty(len(xin)+2, **self.factory_kwargs)
        # print(f"{xin.dtype=} {xpos.dtype=}")
        xpos[1:-1] = xin
        xpos[0] = 2*xin[0] -xin[1]
        xpos[-1] = 2*xin[-1] -xin[-2]
        self.h = xpos[1:] - xpos[:-1]
        self.weight1 = torch.empty((self.nnodes, self.dim), **self.factory_kwargs)
        self.bias1 = torch.empty(self.nnodes, **self.factory_kwargs)
        self.weight2 = torch.empty((self.nnodes, self.dim), **self.factory_kwargs)
        self.bias2 = torch.empty(self.nnodes, **self.factory_kwargs)
        self.bias1[:] = xpos[2:]/self.h[1:]
        self.weight1[:,0] = -1/self.h[1:]
        self.bias2[:] = -xpos[:-2]/self.h[:-1]
        self.weight2[:,0] = 1/self.h[:-1]
        # print(f"{self.weight1.dtype=} {self.bias1.dtype=}")
        # midpoint
        self.x_int = 0.5*(xin[:-1]+xin[1:])
        self.w_int = xin[1:]-xin[:-1]
        # gauss 2
        # self.x_int = torch.empty(2*(self.n_basis-1), **factory_kwargs)
        # self.x_int[::2] = 0.5*(xin[:-1]+xin[1:]) - 0.5/np.sqrt(3)*(xin[1:]-xin[:-1])
        # self.x_int[1::2] = 0.5*(xin[:-1]+xin[1:]) + 0.5/np.sqrt(3)*(xin[1:]-xin[:-1])
        # self.w_int = torch.empty(2*(self.n_basis-1), **factory_kwargs)
        # self.w_int[::2] = 0.5*(xin[1:]-xin[:-1])
        # self.w_int[1::2] = 0.5*(xin[1:]-xin[:-1])

        self.x_int = self.x_int.reshape(-1,1).requires_grad_()
        self.phi_int = self.phi(self.x_int)
        self.phi_0 = self.phi(torch.tensor([xin[0]], **self.factory_kwargs).reshape(-1,1))
        # print(f"{self.phi_int.shape=} {self.x_int.shape=}")
        self.phi_x_int = torch.empty_like(self.phi_int)
        # print(f"{self.phi_x_int.shape=}")
        for i in range(self.nnodes):
            v = torch.zeros(self.nnodes, **self.factory_kwargs)
            v[i] = 1
            u = self.phi_int@v
            self.phi_x_int[:,i] = torch.autograd.grad(u, self.x_int, (torch.ones_like(u)), create_graph=True)[0][:,0]
        # self.phi_x_int = torch.autograd.functional.jacobian(self.phi, self.x_int.reshape(-1,1))
        # print(f"{self.phi_x_int=}")
        # print(f"{self.phi_x_int.shape=}")
        # self.relu = torch.nn.SiLU()
        # self.relu = ReLuSoft(alpha = 1.)
    def phi(self, x):
        vl = torch.nn.functional.linear(x, self.weight1, self.bias1)
        vr = torch.nn.functional.linear(x, self.weight2, self.bias2)
        return self.relu(vl-self.relu(vl-vr))
    def u(self, coef, x=None):
        assert coef.shape[1] == self.nnodes
        phis = self.phi_int.T if x is None else self.phi(x.reshape(-1,1)).T
        # phis = self.phi(x.reshape(-1,1)).T
        return coef@phis
    def u_0(self, coef):
        assert coef.shape[1] == self.nnodes
        phis = self.phi_0.T
        # print(f"{phis.shape=} {coef.shape=}")
        # print(f"{phis.dtype=} {coef.dtype=}")
        return coef@phis
    def u_t(self, coef):
        assert coef.shape[1] == self.nnodes
        phis_x = self.phi_x_int.T
        return coef@phis_x
    def plot_basis(self, ax=None, add_title=""):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.set(title="Basis"+add_title, xlabel="t", ylabel="phi")
        xt = self.mesh_points.reshape(-1,1)
        phis = self.phi(xt).detach().numpy()
        x = self.mesh_points.detach().numpy()
        for i in range(self.nnodes):
            p = ax.plot(x, phis[:,i], label=f"$\phi_{{{i}}}$")
            ax.plot(x[i], 0, 'X', color = p[0].get_color())
        ax.legend()

#-------------------------------------------------------------
class SolveODE():
    def __init__(self, app, n=10, device=None, dtype=torch.float64):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        # print(f"{self.factory_kwargs=}")
        self.app = app
        self.fem = FEM1DNew(app.domain[0], app.domain[1], n, **self.factory_kwargs)
        self.ncomp = len(app.u0)
        self.u0 = torch.tensor(app.u0, **self.factory_kwargs)
    def residual(self, coef):
        res = torch.empty(self.ncomp, self.fem.nnodes, **self.factory_kwargs)
        # print(f"{self.fem.u_0(coef).shape=} {self.fem.n_basis.shape=}")
        u0 = self.fem.u_0(coef).squeeze()
        # print(f"{u0.shape=} {self.u0.shape=}")
        # print(f"{u0=} {self.u0=}")
        res[:, 0] = u0-self.u0
        u = self.fem.u(coef)
        u_t = self.fem.u_t(coef)
        fu = torch.stack(self.app.f(u))
        res[:,1:] = ((u_t-fu)*self.fem.w_int)
        # gauss2
        # res[:,1:] = ((u_t-fu)*self.fem.w_int)[:,::2]
        # res[:,1:] += ((u_t-fu)*self.fem.w_int)[:,1::2]
        return res
    def jacobian(self, u):
        assert u.shape[1] == self.ncomp
        print(f"{u.shape=}")
        Ju = torch.empty(u.shape[0], u.shape[1], u.shape[1])
        for i in range(u.shape[0]):
            tJ = torch.autograd.functional.jacobian(self.app, u[i, :].reshape(1, -1))
            print(f"{Ju.shape=} {tJ.shape=}")
            Ju[i, :, :] = tJ[:, 0, 0, :]
        return Ju
    def solve(self, niter=100, rtol=1e-5, iplot=1):
        ncomp, nb = self.ncomp, self.fem.nnodes
        coef = torch.tensor(self.app.u0, **self.factory_kwargs).reshape(-1, 1) * torch.ones(1, nb, **self.factory_kwargs)
        # print(f"{coef.dtype=}")
        # coef = torch.tensor(self.app.u0).reshape(-1, 1) * torch.zeros(1, nb)
        n_plt = max(nb*3,40)
        t_plt = torch.linspace(*self.app.domain, n_plt, **self.factory_kwargs)
        fig, axs = plt.subplots(ncomp, 1, constrained_layout=True, figsize=(10, 10))
        if ncomp==1: axs = [axs]
        if hasattr(self.app, 'analytical_solution'):
            u_ex = self.app.analytical_solution(t_plt)
            u_ex = u_ex.reshape(ncomp, -1)
        else:
            u_ex = None
        for i,ax in enumerate(axs):
            ax.set(title=f"u_{i}", xlabel="t", ylabel="")
            if u_ex is not None: axs[i].plot(t_plt.numpy(), u_ex[i, :], '-x', label=f"sol")
        for k in range(niter):
            res = self.residual(coef).reshape(ncomp*nb)
            resf = torch.linalg.norm(res)
            if k==0: rtol*resf
            if k%((niter-1)//iplot) ==0 or resf<rtol:
                u = self.fem.u(coef, x=t_plt)
                u_plt = u.detach().numpy()
                for i in range(self.ncomp):
                    axs[i].plot(t_plt.numpy(), u_plt[i, :], label=f"{k=}")
            if resf < rtol:
                print(f"***** {k=} {resf}")
                break
            J = torch.autograd.functional.jacobian(self.residual, coef)
            J = J.reshape(ncomp*nb,ncomp*nb)
            dres = torch.linalg.solve(J,res).reshape(ncomp,nb)
            omega, inres = 1, 1
            for i_inner in range(40):
                coef -= omega*dres
                resf_in = torch.linalg.norm(self.residual(coef))
                if resf_in < resf: break
                coef += omega*dres
                omega *= 0.5
            print(f"***** {k=} {resf} {omega}")
        for ax in axs:
            ax.grid()
            ax.legend()
        plt.show()
        if hasattr(self.app,'plot'):
            self.app.plot(t_plt.numpy(), u_plt)
        return self.fem.u(coef, x=self.fem.mesh_points).detach().numpy()


#-------------------------------------------------------------
if __name__ == "__main__":
    import applications, applications_analytical_solution, cgk
    app = applications.Pendulum(goL=2, is_linear=False)
    # app = applications.Pendulum(goL=2, is_linear=True)
    # app = applications_analytical_solution.Oscillator()
    # app = applications.Logistic(k=20, u0=0.3)
    # app = applications.Lorenz(t_end=10)
    solver = SolveODE(app, n=50)
    solver.fem.plot_basis()
    plt.show()
    u_fem = solver.solve(niter=30, rtol=1e-5, iplot=10)
    cgp = cgk.CgK(k=1)
    t = solver.fem.mesh_points.detach().numpy()
    u_node, u_coef = cgp.run_forward(t, app)
    plt.show()
    fig = plt.figure()
    app.plotax(t=t, u=u_fem, ax=fig.gca(), label='fem')
    app.plotax(t=t, u=u_node, ax=fig.gca(), label='cgk')
    plt.show()
