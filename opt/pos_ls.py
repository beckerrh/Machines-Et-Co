import numpy as np
import algo_data

#-------------------------------------------------------------
class PDAS_pos():
    def __init__(self, M, q, algoin = algo_data.AlgoIn()):
        self.M, self.q, self.eps = M, q, 1e-14
        self.algoin = algoin
        self.idx = np.arange(len(q))
    def _solve(self, x, y, A):
        M, q, idx = self.M, self.q, self.idx
        I = np.setdiff1d(idx, A, assume_unique=True)
        x[I] = np.linalg.solve(M[I[:, np.newaxis], I], (y - q)[I])
        x[A] = 0
        y[I] = 0
        y[A] = (M@x)[A] + q[A]
    def solve(self, x=None):
        if x is None: x = np.zeros_like(self.q)
        idx, M, q, algoin, eps = self.idx, self.M, self.q, self.algoin, self.eps
        y = np.zeros_like(x)

        for iter in range(algoin.niter):
            A = idx[x<algoin.c*y]
            self._solve(x, y, A)
            xi, yi = np.count_nonzero(x < -eps), np.count_nonzero(y < -eps)
            loss = 0.5*np.dot(M@x,x) + np.dot(q,x)
            # loss = 0.5*np.dot(M@x,x)
            if algoin.verbose:
                print(f"{iter:5d} {loss:8.3e} {len(A) =:5d} {np.sum(x * y)} #x<0: {xi:5d} #y<0: {yi:5d}")
            if iter and xi == 0 and yi == 0 and np.sum(x * y) < eps:
                return algo_data.AlgoOut(niter=iter, x=x)
        return algo_data.AlgoOut(niter=iter, x=x, failure=True)


#-------------------------------------------------------------
def testMMatrix():
    import matplotlib.pyplot as plt
    ns = [4**k for k in range(1,6)]
    cs = [1]
    nr = 10
    data = {c:np.zeros(len(ns)) for c in cs}
    data_fail = {c:np.zeros(len(ns)) for c in cs}
    algoin = algo_data.AlgoIn(niter=40, verbose=True)
    for i,n in enumerate(ns):
        for ir in range(nr):
            M = - np.random.rand(n,n)
            M = 0.5*(M+M.T)
            # M += 2*n*np.eye(n)
            M += np.diag(-1.1*np.sum(M, axis=0))
            print(f"{np.linalg.eigvals(M).min()=}")
            # M += -2*np.sum(M, axis=0).min() * np.eye(n)
            q = 2*(np.random.rand(n)-0.5)
            for c in cs:
                algoin.c = c
                pdas = PDAS_pos(M, q, algoin)
                algoout = pdas.solve()
                data[c][i] += algoout.niter/nr
                if algoout.failure:
                    data_fail[c][i] += 1
                    print(f"{np.all(np.linalg.inv(M)>0)=}")
    for c in cs:
        plt.plot(ns, data[c], '-X', label=f"{c=}")
    plt.legend()
    plt.show()
    print(f"{data_fail=}")

#-------------------------------------------------------------
if __name__ == "__main__":
    testMMatrix()
    # import matplotlib.pyplot as plt
    # n = 30
    # M = - np.random.rand(n,n)
    # M = 0.5*(M+M.T)
    # # M += 2*n*np.eye(n)
    # # M += np.diag(-2*np.sum(M, axis=0))
    # M += -2*np.sum(M, axis=0).min() * np.eye(n)
    # q = n*(np.random.rand(n)-0.5)
    # algoin = algo_data.AlgoIn(niter=40, verbose=1)
    # algoin.c = 1
    # algoout = pdas_hki(M, q, algoin=algoin)
    # if algoout.failure:
    #     print(f"{np.all(np.linalg.inv(M)>0)=}")

