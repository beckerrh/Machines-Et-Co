import numpy as np
import torch
import classes


#-------------------------------------------------------------
class Pendulum(classes.Application):
    def __init__(self, alpha = 0.05, goL = 5.0, t_end=10, is_linear=False):
        super().__init__(t_end=t_end, u0=[0.8 * np.pi, 0.0])
        self.alpha, self.goL = alpha, goL
        self.f = self.f_linear if is_linear else self.f_nonlinear
        self.df = self.df_linear if is_linear else self.df_nonlinear

    def f_linear(self, u):
        theta, omega = u[0], u[1]
        alpha, goL = self.alpha, self.goL
        return [omega, -alpha * omega - goL * theta]
    def df_linear(self, u):
        alpha, goL = self.alpha, self.goL
        return [[0, 1], [-goL, -alpha]]

    def f_nonlinear(self, u):
        theta, omega = u[0], u[1]
        alpha, goL = self.alpha, self.goL
        if self.type =="torch":
            return [omega, -alpha * omega - goL * torch.sin(theta)]
        return [omega, -alpha * omega - goL * np.sin(theta)]

    def df_nonlinear(self, u):
        alpha, goL = self.alpha, self.goL
        return [[0, 1], [-goL* np.cos(u[0]), -alpha]]

#-------------------------------------------------------------
class Lorenz(classes.Application):
    def __init__(self, sigma=10, rho=28, beta=8/3, t_end=20):
        super().__init__(t_end=t_end, u0=[-10, -4.45, 35.1])
        self.FP1 =  [np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)),rho-1]
        self.FP2 =  [-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)),rho-1]
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.f = lambda u: [self.sigma*(u[1]-u[0]), self.rho*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-self.beta*u[2]]
        self.df = lambda u: [[-self.sigma, self.sigma,0], [self.rho-u[2],-1,-u[0]], [u[1],u[0],-self.beta]]
    def plot(self, t, u):
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        x,y,z = u[0], u[1], u[2]
        # x,y,z = u[:,0], u[:,1], u[:,2]
        ax.plot(x, y, z, label='u', lw=0.5)
        ax.plot(x[-1], y[-1], z[-1], 'X', label="u(T)")
        ax.plot(x[0], y[0], z[0], 'X', label="u(0)")
        ax.plot(*self.FP1, color='k', marker="8", ls='')
        ax.plot(*self.FP2, color='k', marker="8", ls='')
        ax.view_init(26, 130)
        ax.legend()
        plt.show()

#-------------------------------------------------------------
if __name__ == "__main__":
    print(f"ya no hay nada aqui")