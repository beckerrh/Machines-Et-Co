import numpy as np
import matplotlib.pyplot as plt

#==================================================================
class Application():
# ==================================================================
    def __init__(self, u0=None, t_begin=0.0, t_end=1.0):
        self.dtype = np.float64
        self.domain = (t_begin, t_end)
        if u0 is not None:
            if isinstance(u0, list) or isinstance(u0, np.ndarray):
                self.u0 = u0
            else:
                self.u0 = [u0]
            self.u0 = np.asarray(self.u0, dtype=self.dtype)
            self.dim = len(self.u0)
        self.name = self.__class__.__name__
        self.type = 'torch'
    def plotax(self, t, u, fig=None, ax=None, label="u", title=""):
        if ax is None:
            if fig is None: fig = plt.figure()
            if hasattr(self,'sol_analytic'):
                ax = fig.add_subplot(2,1,1)
            else:
                ax = fig.add_subplot(1,1,1)
        ax.set_title(title)
        for j in range(u.shape[0]):
            ax.plot(t, u[j], '-', label=label+f"{j}")
            # ax.plot(t_ex, np.asarray(self.sol_analytic(t_ex)).T -, 'k--', label='exact')
        ax.legend()
        ax.grid()
        if hasattr(self,'sol_analytic'):
            ax = fig.add_subplot(2,1,2)
            ax.set_title(title+"_error")
            # t_ex = np.linspace(0, t[-1], 100)
            err = np.fabs(u -np.asarray(self.sol_analytic(t)))
            for j in range(u.shape[0]):
                ax.plot(t, err[j], '-', label=label + f"{j}")
            ax.legend()
            ax.grid()
        return ax
    def plot_solution(self, **kwargs):
        u = kwargs.pop('u', None)
        t = kwargs.pop('t', None)
        fig = kwargs.pop('fig', None)
        title = kwargs.pop('title', self.__class__.__name__)
        label = kwargs.pop('label', '')
        show = kwargs.pop('show', False)
        if fig is None:
            fig = plt.figure(constrained_layout=True)
        res =  self.plotax(t, u, fig=fig, label=label, title=title)
        if show: plt.show()
        return res

#==================================================================
class Functional():
# ==================================================================
    def __init__(self):
        self.name = type(self).__name__

#==================================================================
class FunctionalEndTime(Functional):
# ==================================================================
    def __init__(self, k=0):
        super().__init__()
        self.k = k
    def lT(self, u):
        return u[self.k]
    def lT_prime(self, u):
        v = np.zeros_like(u)
        v[self.k] = 1
        return v

#==================================================================
class FunctionalMean(Functional):
# ==================================================================
    def __init__(self, k=0, t0=-np.inf, t1=np.inf):
        super().__init__()
        self.k, self.t0, self.t1 = k, t0, t1
    def l(self, t, u):
        # if t<self.t0 or t>self.t1: return np.zeros_like(u)
        return u[self.k]
    def l_prime(self, t, u):
        # if t<self.t0 or t>self.t1: return np.zeros_like(u)
        v = np.zeros_like(u)
        v[self.k] = 1
        return v


#==================================================================
class FunctionalTimePoint(Functional):
# ==================================================================
    def __init__(self, method, t0, k=0):
        super().__init__()
        self.k, self.t0, self.method = k, t0, method
    def ldelta(self, t, u_app):
        i = np.searchsorted(t, self.t0)
        print(f"{self.t0} {t[i]=}")
        # if t<self.t0 or t>self.t1: return np.zeros_like(u)
        return self.method.evaluate(u_app, self.k)
    def ldelta_prime(self, b, t, u):
        # if t<self.t0 or t>self.t1: return np.zeros_like(u)
        v = np.zeros_like(u)
        v[self.k] = 1
        return v
