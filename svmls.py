import numpy as np
from sklearn import svm

class Kernel_Pk():
    def __repr__(self):
        if self.x is None: return f"P{self.k}"
        return f"P{self.k}_"+f"{len(self.x)}"
    def __init__(self, k=1, x=None):
        self.k, self.x = k, x
        print(f"{self.k=} {self.x=}")
    def __call__(self, X,Y):
        if self.x is None:
            xmin = np.minimum(X,Y.T)
            xabs = np.abs(X-Y.T)
            print(f"{X.shape=} {xmin.shape=} {xabs.shape=} {np.dot(X,Y.T).shape=}")
            return 1+np.dot(X,Y.T) + 0.5*xabs*xmin**2 + xmin**3/3
        X1 = np.maximum(0, X-self.x)
        print(f"{X.shape=} {self.x.shape=} {X1.shape=} {np.dot(X,Y.T).shape=}")
        Y1 = np.maximum(0, Y-self.x)
        if self.k==1:
            return 1 + np.dot(X,Y.T) + np.dot(X1, Y1.T)
        elif self.k==2:
            return 1 + np.dot(X,Y.T) + np.dot(X**2,Y.T**2) + np.dot(X1**2, Y1.T**2)
        else:
            raise ValueError(f"not written {self.k=}")


#-------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    f = lambda x: np.sin( np.exp(1.5*x))
    f = lambda x: np.maximum((x-0.3)*(x-1.6), 0.25-(x-1)**2)
    x = np.linspace(0, 2, 150)
    y = f(x)
    kernels=['rbf']
    kernels.append(Kernel_Pk(k=1, x=x[::5]))
    kernels.append(Kernel_Pk(k=1))
    # kernels.append(Kernel_Pk(k=1, x=x[::5]))
    fig, axs = plt.subplots(len(kernels), 1, sharex=True)
    fig.suptitle(f'Comparison', fontsize=16)
    for i,kernel in enumerate(kernels):
        regr = svm.SVR(kernel=kernel, C=40, epsilon=0.01)
        regr.fit(x.reshape(-1, 1), y)
        svi = regr.support_
        sv = x[svi]
        axs[i].set_title(f"n={len(svi)}")
        yapprox = regr.predict(x.reshape(-1, 1))
        p1, = axs[i].plot(x, yapprox, label=f"{kernel}")
        axs[i].plot(sv, f(sv), 'rX')
        p2, = axs[i].plot(x, y, '--y', label="u_ex")
        ax2 = axs[i].twinx()
        ax2.set_ylabel('err', color='tab:orange')
        p3, = ax2.plot(x, np.abs(yapprox - y), color='tab:orange', label='err')
        axs[i].legend(handles=[p1, p2, p3])
    plt.show()
