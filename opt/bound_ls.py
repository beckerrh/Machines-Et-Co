import numpy as np
import algo_data

#-------------------------------------------------------------
class PDAS_bound():
    def __init__(self, M, q, algoin = algo_data.AlgoIn()):
        if hasattr(algoin, 'a'):
            a = algoin.a
        else:
            a = -np.finfo(q.dtype).max
        if hasattr(algoin, 'b'):
            b = algoin.b
        else:
            b = np.finfo(q.dtype).max
        print(f"{a=} {b=}")
        assert (a < b)
        self.M, self.q, self.a, self.b = M, q, a, b
        self.idx = np.arange(len(q))
    def _solve(self, x, ya, yb, A, B):
        M, q, a, b, idx = self.M, self.q, self.a, self.b, self.idx
        IA = np.setdiff1d(idx, A, assume_unique=True)
        IB = np.setdiff1d(idx, B, assume_unique=True)
        I = np.setdiff1d(IA, B, assume_unique=True)
        x[I] = np.linalg.solve(M[I[:, np.newaxis], I], (ya - yb - q)[I])
        x[A] = a
        x[B] = b
        ya[IA] = 0
        yb[IB] = 0
        # ya[A] = (M@x)[A] - (ya - yb - q)[A]
        # yb[B] = -(M@x)[B] + (ya - yb - q)[B]

        ya[A] = (M @ x)[A] + q[A] + yb[A]
        yb[B] = -(M @ x)[B] -q[B] - ya[B]

    def solve(self, x=None):
        if x is None: x = np.zeros_like(self.q)
        eps=1e-12
        idx, a, b = self.idx, self.a, self.b
        ya = np.zeros_like(q)
        yb = np.zeros_like(q)

        for iter in range(algoin.niter):
            B = idx[x>=b-yb+eps]
            A = idx[x<=a+ya-eps]
            self._solve(x, ya, yb, A, B)
            if algoin.verbose:
                xia, xib  = np.count_nonzero(x < a-eps), np.count_nonzero(x > b+eps)
                yai, ybi  = np.count_nonzero(ya < -eps), np.count_nonzero(yb < -eps)
                print(f"{iter:3d} {len(A)=:5d} {len(B)=:5d} {xia=:4d} {xib=:4d} {yai=:4d} {ybi=:4d}")
                # print(f"{iter=:3d} {A=} {B=} {x=} {ya=} {yb=}")
            if iter and xia == 0  and xib == 0 and yai == 0 and ybi == 0 and np.sum((x-a) * ya) < eps and np.sum((x-b) * yb) < eps:
                return algo_data.AlgoOut(niter=iter, x=x)
        return algo_data.AlgoOut(niter=iter, x=x, failure=True)

#-------------------------------------------------------------
def testMMatrix():
    import matplotlib.pyplot as plt
    ns = [4**k for k in range(1,6)]
    nr = 10
    data = np.zeros(len(ns))
    data_fail = np.zeros(len(ns))
    algoin = algo_data.AlgoIn(niter=400, verbose=0)
    algoin.c = 1
    for i,n in enumerate(ns):
        for ir in range(nr):
            M = - np.random.rand(n,n)
            M = 0.5*(M+M.T)
            # M += 2*n*np.eye(n)
            M += np.diag(-2*np.sum(M, axis=0))
            # M += -2*np.sum(M, axis=0).min() * np.eye(n)
            q = n*(np.random.rand(n)-0.5)
            pdas = PDAS_bound(M, q, algoin=algoin)
            algoout = pdas.solve()
            data[i] += algoout.niter/nr
            if algoout.failure:
                data_fail[i] += 1
                print(f"{np.all(np.linalg.inv(M)>0)=}")
    plt.plot(ns, data, '-X', label=f"{algoin.c=}")
    plt.legend()
    plt.show()
    print(f"{data_fail=}")

#-------------------------------------------------------------
if __name__ == "__main__":
    # testMMatrix()
    import matplotlib.pyplot as plt
    n = 10
    M = - np.random.rand(n,n)
    M = 0.5*(M+M.T)
    # M += 2*n*np.eye(n)
    M += np.diag(-2*np.sum(M, axis=0))
    # M += -2*np.sum(M, axis=0).min() * np.eye(n)
    q = n*(np.random.rand(n)-0.5)
    x = np.linalg.solve(M,-q)
    print(f"{x.min()=} {x.max()=}")
    algoin = algo_data.AlgoIn(niter=20, verbose=1)
    algoin.a = 0.75*x.min() + 0.25*x.max()
    algoin.b = 0.25*x.min() + 0.75*x.max()
    pdas = PDAS_bound(M, q, algoin=algoin)
    algoout = pdas.solve(x)
    print(f"{x=}")
    if algoout.failure:
        print(f"{np.all(np.linalg.inv(M)>0)=}")

