import numpy as np
import algo_data

#-------------------------------------------------------------
class PDAS_bound():
    def __init__(self, Q, q, algoin = algo_data.AlgoIn()):
        self.eps=1e-16
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
        self.Q, self.q, self.a, self.b, self.algoin = Q, q, a, b, algoin
        self.idx = np.arange(len(q))
    def _solve(self, x, ya, yb, A, B):
        Q, q, a, b, idx = self.Q, self.q, self.a, self.b, self.idx
        IA = np.setdiff1d(idx, A, assume_unique=True)
        IB = np.setdiff1d(idx, B, assume_unique=True)
        I = np.setdiff1d(IA, B, assume_unique=True)
        x[A] = a
        x[B] = b
        x[I] = np.linalg.solve(Q[I[:, np.newaxis], I], -q[I] - Q[I[:, np.newaxis], A]@x[A]- Q[I[:, np.newaxis], B]@x[B])
        ya[IA] = 0
        yb[IB] = 0
        # ya[A] = (Q@x)[A] - (ya - yb - q)[A]
        # yb[B] = -(Q@x)[B] + (ya - yb - q)[B]

        ya[A] = (Q @ x)[A] + q[A]
        yb[B] = -(Q @ x)[B] -q[B]

    def solve(self, x=None):
        idx, Q, q, algoin, eps, a, b = self.idx, self.Q, self.q, self.algoin, self.eps, self.a, self.b
        if x is None: x = np.zeros_like(self.q)
        ya = np.zeros_like(x)
        yb = np.zeros_like(x)

        for iter in range(algoin.niter):
            B = idx[x>b-yb+eps]
            A = idx[x<a+ya-eps]
            self._solve(x, ya, yb, A, B)
            if algoin.verbose:
                loss = 0.5 * np.dot(Q @ x, x) + np.dot(q, x)
                xia, xib  = np.count_nonzero(x < a-eps), np.count_nonzero(x > b+eps)
                yai, ybi  = np.count_nonzero(ya < -eps), np.count_nonzero(yb < -eps)
                print(f"{iter:3d} {loss:12.5e} {len(A)=:5d} {len(B)=:5d} {xia=:4d} {xib=:4d} {yai=:4d} {ybi=:4d}")
                # print(f"{iter=:3d} {A=} {B=} {x=} {ya=} {yb=}")
            if iter and xia == 0  and xib == 0 and yai == 0 and ybi == 0 and np.sum((x-a) * ya) < eps and np.sum((x-b) * yb) < eps:
                return algo_data.AlgoOut(niter=iter, x=x)
        return algo_data.AlgoOut(niter=iter, x=x, failure=True)


#-------------------------------------------------------------
def test(type='mmatrix'):
    import matplotlib.pyplot as plt
    ns = [4**k for k in range(1,7)]
    cs = [1]
    nr = 10
    data = {c:np.zeros(len(ns)) for c in cs}
    data_fail = {c:np.zeros(len(ns)) for c in cs}
    algoin = algo_data.AlgoIn(niter=40, verbose=True)
    for i,n in enumerate(ns):
        for ir in range(nr):
            if type == 'mmatrix':
                # M += -2*np.sum(M, axis=0).min() * np.eye(n)
                Q = - np.random.rand(n,n)
                Q = 0.5*(Q+Q.T)
                Q += np.diag(-1.1*np.sum(Q, axis=0))
                print(f"{np.linalg.eigvals(Q).min()=}")
                q = 2*(np.random.rand(n)-0.5)
            else:
                M = np.random.rand(2*n,n)
                Q = M.T@M
                q = np.random.rand(n)-0.5
            x = np.linalg.solve(Q,-q)
            algoin = algo_data.AlgoIn(niter=20, verbose=1)
            algoin.a = 0.75*x.min() + 0.25*x.max()
            algoin.b = 0.25*x.min() + 0.75*x.max()
            for c in cs:
                algoin.c = c
                pdas = PDAS_bound(Q, q, algoin)
                algoout = pdas.solve()
                data[c][i] += algoout.niter/nr
                if algoout.failure:
                    data_fail[c][i] += 1
                    print(f"{np.all(np.linalg.inv(Q)>0)=}")
    for c in cs:
        plt.plot(ns, data[c], '-X', label=f"{c=}")
    plt.legend()
    plt.show()
    print(f"{data_fail=}")


#-------------------------------------------------------------
if __name__ == "__main__":
    test(type='ls')
    # import matplotlib.pyplot as plt
    # n = 640
    # Q = - np.random.rand(n,n)
    # Q = 0.5*(Q+Q.T)
    # Q += np.diag(-2*np.sum(Q, axis=0))
    # q = n*(np.random.rand(n)-0.5)
    # x = np.linalg.solve(Q,-q)
    # print(f"{x.min()=} {x.max()=}")
    # algoin = algo_data.AlgoIn(niter=20, verbose=1)
    # algoin.a = 0.75*x.min() + 0.25*x.max()
    # algoin.b = 0.25*x.min() + 0.75*x.max()
    # # algoin.a = 0.1
    # pdas = PDAS_bound(Q, q, algoin=algoin)
    # algoout = pdas.solve(x)
    # # print(f"{x=}")
    # if algoout.failure:
    #     print(f"{np.all(np.linalg.inv(Q)>0)=}")

