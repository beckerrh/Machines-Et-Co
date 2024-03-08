import numpy as np
import torch
import classes

#-------------------------------------------------------------
class Logistic(classes.Application):
    def __init__(self, k=1.5, G=1, u0=0.5, T=2):
        super().__init__(t_end=T, u0=u0)
        self.k, self.G = k, G
    def f(self, u):
        return self.k * u * (self.G - u)
    def df(self, u):
        return self.k * (self.G - 2*u)
    def sol_analytic(self, t):
        return self.G / (1.0 + np.exp(-self.k*self.G*t)*(self.G/self.u0[0] - 1.0))

#------------------------------------------------------------------
class Exponential(classes.Application):
    def __init__(self, rate=0.2, T=5, u0=1):
        super().__init__(t_end=T, u0=u0)
        self.rate = rate
        self.f = lambda u: [self.rate*u]
        self.df = lambda u: [self.rate]
        self.sol_analytic = lambda t: [self.u0*np.exp(self.rate*t)]
        self.dsol_analytic = lambda t: [self.u0*self.rate*np.exp(self.rate*t)]
        self.f_p0 = lambda u: [u]
        self.l_p0 = lambda t: np.array([0 * t])
        self.u_zero_p0 = lambda : [0]
        self.f_p1 = lambda u: [0*u]
        self.l_p1 = lambda t: np.array([0 * t])
        self.u_zero_p1 = lambda : [1]
        self.nparam = 2
    def setParameter(self, p):
        assert p.ndim ==1
        self.rate = p[0]
        self.u0 = p[1]
#------------------------------------------------------------------
class Stiff1(classes.Application):
    def __init__(self, a=2, T=10, u0=[2,3]):
        super().__init__(t_end=T, u0=u0)
        self.a = a
        self.f = lambda u: [-2*u[0]+u[1],(self.a-1)*u[0]-self.a*u[1]]
        self.l = lambda t: [2*np.sin(t),self.a*(np.cos(t)-np.sin(t))]
        # self.df = lambda u: [[self.alpha, self.gamma], [0, self.beta]]
        self.df = lambda u: [[-2, 1], [self.a-1, -self.a]]
        self.sol_analytic = lambda t: [2*np.exp(-t)+np.sin(t), 2*np.exp(-t)+np.cos(t)]
        self.dsol_analytic = lambda t: [-2*np.exp(-t)+np.cos(t), -2*np.exp(-t)-np.sin(t)]
        # self.f_p0 = lambda u: [u]
        # self.l_p0 = lambda t: np.array([0 * t])
        # self.u_zero_p0 = lambda : [0]
        # self.f_p1 = lambda u: [0*u]
        # self.l_p1 = lambda t: np.array([0 * t])
        # self.u_zero_p1 = lambda : [1]
        # self.nparam = 2
    # def setParameter(self, p):
    #     assert p.ndim ==1
    #     self.rate = p[0]
    #     self.u0 = p[1]
#------------------------------------------------------------------
class Oscillator(classes.Application):
    def __init__(self, freq=2):
        super().__init__(t_end=4*np.pi, u0=[1, 0])
        self.freq = freq
        self.nparam = 1
        self.f = lambda u: [self.freq*u[1], -self.freq*u[0]]
        self.df = lambda u: [[0, self.freq], [-self.freq, 0]]
        self.sol_analytic = lambda t: [np.cos(self.freq*t), -np.sin(self.freq*t)]
        self.dsol_analytic = lambda t: [-self.freq*np.sin(self.freq*t), -self.freq*np.cos(self.freq*t)]
        self.f_p0 = lambda u: [0*u[0], -u[0]]
        self.l_p0 = lambda t: [0 * t, 0 * t]
        self.u_zero_p0 = lambda : [0,0]
    def setParameter(self, p): self.freq = float(p)
#------------------------------------------------------------------
class Linear(classes.Application):
    def __init__(self, u0=1, slope=0.2):
        super().__init__(t_end=5, u0=u0)
        self.slope = slope
        self.f = lambda u: [slope]
        self.df = lambda u: [0]
        self.sol_analytic = lambda t: [u0+slope*t]
        self.dsol_analytic = lambda t: [slope+0*t]
#------------------------------------------------------------------
class Quadratic(classes.Application):
    def __init__(self, u0=1):
        super().__init__(t_end=0.9/u0, u0=u0)
        self.f = lambda u: [u**2]
        self.df = lambda u: [2*u]
        self.sol_analytic = lambda t: [u0/(1-u0*t)]
        self.dsol_analytic = lambda t: [u0**2/(1-u0*t)**2]
#------------------------------------------------------------------
class SinusIntegration(classes.Application):
    def __init__(self, freq=[1,2], T=10):
        super().__init__(t_end=2, u0=[0,1])
        self.nparam = 2
        self.freq0, self.freq1 = freq[0], freq[1]
        # self.f = lambda u: [0.0, 0.0]
        # self.df = lambda u: [[0,0],[0,0]]
        self.sol_analytic = lambda t: [np.sin(self.freq0*t), np.cos(self.freq1*t)]
        self.l = lambda t: self.dsol_analytic(t)
        # self.u_zero_p0 = lambda : [0,0]
        # self.u_zero_p1 = lambda : [0,0]
        # self.f_p0 = lambda u: [0*u[0], 0*u[0]]
        # self.f_p1 = lambda u: [0*u[0], 0*u[0]]
        self.l_p0 = lambda t: [np.cos(self.freq0*t)-self.freq0*t*np.sin(self.freq0*t),0*t]
        self.l_p1 = lambda t: [0*t, -np.sin(self.freq1*t) -self.freq1*t*np.cos(self.freq1*t)]
    def setParameter(self, p):
        self.freq0 = p[0]
        self.freq1 = p[1]
    def dsol_analytic(self, t):
        if self.type=='numpy':
            return [self.freq0*np.cos(self.freq0*t), -self.freq1*np.sin(self.freq1*t)]
        return [self.freq0*torch.cos(self.freq0*t), -self.freq1*torch.sin(self.freq1*t)]


#------------------------------------------------------------------
class QuadraticIntegration(classes.Application):
    def __init__(self, u0=1):
        super().__init__(t_end=2, u0=u0)
        # self.f = lambda u: [0]
        # self.df = lambda u: [0]
        self.sol_analytic = lambda t: [u0+t*t]
        self.dsol_analytic = lambda t: [2*t]
    def l(self, t):
        return self.dsol_analytic(t)
#------------------------------------------------------------------
class LinearIntegration(classes.Application):
    def __init__(self, u0=1, slope=0.5):
        super().__init__(t_end=2, u0=u0)
        self.slope = slope
        self.nparam = 1
        # self.f = lambda u: np.zeros_like(u)
        # self.df = lambda u: np.zeros_like(u)
        self.sol_analytic = lambda t: [self.u0+self.slope*t]
        self.dsol_analytic = lambda t: [self.slope+0*t]
        self.u_zero_p0 = lambda: [0]
    def l(self, t):
        return self.dsol_analytic(t)
    def f_p0(self, u):
        # print(f"{u=} {self.rate=} {j=}")
        return [0*u]
    def l_p0(self, t):
        return [1+0*t]
    def setParameter(self, p): self.slope = p

