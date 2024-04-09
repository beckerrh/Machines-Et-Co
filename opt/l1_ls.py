import numpy as np
import algo_data

#-------------------------------------------------------------
class PDAS_L1():
    def __init__(self, Q, q, algoin = algo_data.AlgoIn()):
        self.Q, self.q, self.eps = Q, q, 1e-10
        self.algoin = algoin
        self.idx = np.arange(len(q))
        self.e = algoin.lam*np.ones_like(q)
        self.method = algoin.method if hasattr(algoin,'method') else "eta"
        self.gamma = algoin.method if hasattr(algoin,'gamma') else 0.001

    def _solve(self, A, B, xp, xm, xi, eta=None):
        Q, q, idx, e, gamma = self.Q, self.q, self.idx, self.e, self.gamma
        IA = np.setdiff1d(idx, A, assume_unique=True)
        IB = np.setdiff1d(idx, B, assume_unique=True)
        assert len(np.intersect1d(IA, IB, assume_unique=True))==0
        M = np.block([[Q[IA[:, np.newaxis], IA]+gamma*np.eye(len(IA)), Q[IA[:, np.newaxis], IB]],
                      [Q[IB[:, np.newaxis], IA], Q[IB[:, np.newaxis], IB]+gamma*np.eye(len(IB))]])
        m = np.block([-q[IA]-e[IA], -q[IB]+e[IB]])
        z = np.linalg.solve(M,m)
        xp[IA] = z[:len(IA)]
        xm[IB] = z[len(IA):]
        xp[A] = 0
        xm[B] = 0
        xi[IA] = 0
        xi[A] = (q+e)[A] + (Q@(xp+xm))[A]
        if self.method == "eta":
            eta[IB] = 0
            eta[B] = (q-e)[B] + (Q@(xp+xm))[B]
        return len(IA)+len(IB)

    def solve(self, x=None):
        idx, Q, q, algoin, eps, e = self.idx, self.Q, self.q, self.algoin, self.eps, self.e
        xi = np.zeros_like(q)
        if x is None:
            xp = np.zeros_like(q)
            xm = np.zeros_like(xp)
            xi[:] = (q+e)[:]
        else:
            xp = np.maximum(0, x)
            xm = np.minimum(0, x)
            xi[:] = (Q@x + q + e)[:]
        if self.method == "eta":
            eta = np.zeros_like(q)
            if x is None:
                eta[:] = (q-e)[:]
            else:
                eta[:] = (Q@x + q - e)[:]
        else:
            eta = None
        for iter in range(algoin.niter):
            A = idx[xp<=xi]
            if self.method == "eta":
                B = idx[xm>=eta]
            else:
                B = idx[xm>xi-2*e]
            if algoin.verbose>=2: print(f"{A=} {B=}")
            n_ac = self._solve(A, B, xp, xm, xi, eta)
            xp_if, xm_if = np.count_nonzero(xp < -eps), np.count_nonzero(xm > eps)
            if self.method == "eta":
                xi_if, eta_if = np.count_nonzero(xi <-eps), np.count_nonzero(eta > eps)
                print(f"{np.linalg.norm(xi-eta-2*e)=}")
                kkt = np.sum(xp * xi) < eps and np.sum(xm * eta) < eps
            else:
                xi_if, eta_if = np.count_nonzero(xi <-eps), np.count_nonzero(xi-2*e > eps)
                kkt = np.sum(xp * xi) < eps and np.sum(xm * (xi-2*e)) < eps
            loss = 0.5*np.dot(Q@(xp+xm),xp+xm) + np.dot(q,xp+xm)
            pen = np.dot(e, xp-xm)
            if algoin.verbose:
                print(f"{iter:5d} {loss:12.5e} {pen:12.5e} {n_ac:5d} ({len(xp)}) {np.sum(xp * xm):10.3e} {xp_if=:5d} {xm_if=:5d} {xi_if=:5d} {eta_if=:5d}")
            if iter and xp_if == 0 and xm_if == 0  and xi_if == 0 and eta_if == 0 and kkt:
                return algo_data.AlgoOut(niter=iter, x=xp+xm)
        return algo_data.AlgoOut(niter=iter, x=x, failure=True)


#-------------------------------------------------------------
def test(type='mmatrix'):
    import matplotlib.pyplot as plt
    ns = [3**k for k in range(1,7)]
    lams = [0.0001, 0.001, 0.01, 0.1, 0.2]
    nr = 5
    data = {lam:np.zeros(len(ns)) for lam in lams}
    data_fail = {lam:np.zeros(len(ns)) for lam in lams}
    algoin = algo_data.AlgoIn(niter=30, verbose=1)
    # algoin.method = 'no_eta'
    for i,n in enumerate(ns):
        for ir in range(nr):
            if type == 'mmatrix':
                # M += -2*np.sum(M, axis=0).min() * np.eye(n)
                Q = - np.random.rand(n,n)
                Q = 0.5*(Q+Q.T)
                Q += np.diag(-1.1*np.sum(Q, axis=0))
                print(f"{np.linalg.eigvals(Q).min()=}")
                q = 2*(np.random.rand(n)-0.5)
            elif type == 'ls':
                M = np.random.rand(2 * n, n)
                Q = M.T @ M
                q = np.random.rand(n) - 0.5
            elif type == 'ls-dual':
                M = np.random.rand(n,n//2)
                Q = M@M.T
                q = np.random.rand(n)-0.5
            else: raise ValueError(f"{type=}")
            for lam in lams:
                algoin.lam = lam
                pdas = PDAS_L1(Q, q, algoin)
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
    test(type='ls-dual')
