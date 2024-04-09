import numpy as np
import opt.algo_data

#-------------------------------------------------------------
class SVM():
    def __init__(self, K, x, y, lam, algoin = opt.algo_data.AlgoIn()):
        self.x, self.y, self.eps = x, y, 1e-6
        print(f"{K=} {type(K)=} {hasattr(K,'__class__')=}")
        if hasattr(K,'__class__'):
            self.K = K(x,x)
            self.kernel = K
        else:
            self.K = K
        print(f"{self.K.shape=} {self.x.shape=} {self.y.shape=}")
        self.algoin = algoin
        self.idx = np.arange(len(y))
        self.e = lam*np.ones_like(y)
        self.gamma = algoin.method if hasattr(algoin,'gamma') else lam*0.01
    def predict(self, x):
        yapp =  np.dot(self.kernel(x,self.x), self.coeff)
        return yapp
    def _solve(self, A, B, IA, IB, coeffp, coeffm, xi, eta=None):
        K, y, idx, e, gamma = self.K, self.y, self.idx, self.e, self.gamma
        assert len(np.intersect1d(IA, IB, assume_unique=True))==0
        M = np.block([[K[IA[:, np.newaxis], IA]+gamma*np.eye(len(IA)),K[IA[:,np.newaxis], IB]],
                      [K[IB[:, np.newaxis], IA],K[IB[:, np.newaxis], IB]+gamma*np.eye(len(IB))]])
        m = np.block([y[IA]-e[IA], y[IB]+e[IB]])
        # z, residuals, RANK, sing = np.linalg.lstsq(M,m)
        z = np.linalg.solve(M,m)
        coeffp[IA] = z[:len(IA)]
        coeffm[IB] = z[len(IA):]
        coeffp[A] = 0
        coeffm[B] = 0
        xi[IA] = 0
        xi[A] = (-y+e)[A] + (K@(coeffp+coeffm))[A]
        eta[IB] = 0
        eta[B] = (-y-e)[B] + (K@(coeffp+coeffm))[B]
        return len(IA)+len(IB)

    def solve(self, coeff=None):
        if coeff is None:
            coeffp = np.zeros_like(self.y)
            coeffm = np.zeros_like(self.y)
        else:
            coeffp = np.maximum(0, coeff)
            coeffm = np.minimum(0, coeff)
        idx, K, y, algoin, eps, e = self.idx, self.K, self.y, self.algoin, self.eps, self.e
        xi = np.zeros_like(coeffp)
        xi[:] = -y[:] + e[:]
        eta = np.zeros_like(coeffp)
        eta[:] = -y[:] - e[:]
        for iter in range(algoin.niter):
            A = idx[coeffp<=xi]
            B = idx[coeffm>=eta]
            IA = np.setdiff1d(idx, A, assume_unique=True)
            IB = np.setdiff1d(idx, B, assume_unique=True)
            if algoin.verbose>=2: print(f"{A=} {B=}")
            n_ac = self._solve(A, B, IA, IB, coeffp, coeffm, xi, eta)
            coeffp_if, coeffm_if = np.count_nonzero(coeffp < -eps), np.count_nonzero(coeffm > eps)
            xi_if, eta_if = np.count_nonzero(xi <-eps), np.count_nonzero(eta > eps)
            print(f"{np.linalg.norm(xi-eta-2*e)=}")
            kkt = np.sum(coeffp * xi) < eps and np.sum(coeffm * eta) < eps
            loss = 0.5*np.dot(K@(coeffp+coeffm),coeffp+coeffm) - np.dot(y,coeffp+coeffm)
            pen = np.dot(e, coeffp-coeffm)
            if algoin.verbose:
                print(f"{iter:5d} {loss:12.5e} {pen:12.5e} {n_ac:5d} ({len(coeffp)}) {coeffp_if=:5d} {coeffm_if=:5d} {xi_if=:5d} {eta_if=:5d}")
            if iter and coeffp_if == 0 and coeffm_if == 0  and xi_if == 0 and eta_if == 0 and kkt:
                self.coeff = coeffp + coeffm
                return opt.algo_data.AlgoOut(niter=iter, x=coeffp+coeffm)
        return opt.algo_data.AlgoOut(niter=iter, x=coeffp+coeffm, failure=True)


#-------------------------------------------------------------
def test(type='mmatrix'):
    import matplotlib.pyplot as plt
    ns = [4**k for k in range(1,7)]
    lams = [0.0001, 0.001, 0.01, 0.1, 0.2]
    nr = 5
    data = {lam:np.zeros(len(ns)) for lam in lams}
    data_fail = {lam:np.zeros(len(ns)) for lam in lams}
    algoin = opt.algo_data.AlgoIn(niter=50, verbose=1)
    # algoin.method = 'no_eta'
    for i,n in enumerate(ns):
        for ir in range(nr):
            if type == 'ls':
                M = np.random.rand(2*n,n)
                # K = M.T@M
                # y = np.random.rand(n)-0.5
                K = M@M.T
                y = np.random.rand(2*n)-0.5
            for lam in lams:
                algoin.lam = lam
                pdas = SVM(K, y, algoin)
                algoout = pdas.solve()
                data[lam][i] += algoout.niter/nr
                if algoout.failure:
                    data_fail[lam][i] += 1
                    print(f"{np.all(np.linalg.inv(Q)>0)=}")
    for lam in lams:
        plt.plot(ns, data[lam], '-X', label=f"{lam=}")
    plt.legend()
    plt.show()
    print(f"{data_fail=}")


#-------------------------------------------------------------
if __name__ == "__main__":
    test(type='ls')
