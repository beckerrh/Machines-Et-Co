import numpy as np
import algo_data

#-------------------------------------------------------------
class PDAS_pos():
    def __init__(self, Q, q, algoin = algo_data.AlgoIn()):
        self.Q, self.q, self.eps = Q, q, 1e-14
        self.algoin = algoin
        self.idx = np.arange(len(q))
    def _solve(self, x, y, A):
        Q, q, idx, method, kappa = self.Q, self.q, self.idx, self.algoin.method, self.algoin.kappa
        I = np.setdiff1d(idx, A, assume_unique=True)
        QI = Q[I[:, np.newaxis], I]
        if method =="abs":
            kappa = self.kappa
            # print(f"{self.kappa=}")
            param =  (1-kappa)/(1+kappa)
            # print(f"{param=}")
            QIA = Q[I[:, np.newaxis], A]
            QAI = Q[A[:, np.newaxis], I]
            QAA = Q[A[:, np.newaxis], A]
            M = np.block([[QI+param*np.eye(len(I)),-param*QIA],[QAI,-(param*QAA+np.eye(len(A)))]])
            z = np.linalg.solve(M, -np.block([q[I],q[A]]))
            x[I] = z[:len(I)]
            y[A] = z[len(I):]
            # x[I] = np.linalg.solve(QI+param*np.eye(len(I)), -q[I])
            # y[A] = np.linalg.solve(-(param*QAA+np.eye(len(A))), -QAI@x[I]-q[A])
            # print(f"{z=} {x[I]=} {y[A]=}")
            # print(f"{np.linalg.norm(z[:len(I)]-x[I])=} {np.linalg.norm(z[len(I):]-y[A])=}")
            y[I] = -param*x[I]
            x[A] = -param*y[A]
            return
        if method in ["sft"]:
            onem = kappa * np.ones(len(I))
            # print(f"{} {onem.shape=} {(x > 0).shape=}")
            onem[x[I] > 0] = 0
            QI += np.diag(onem)
        try:
            x[I] = np.linalg.solve(QI, -q[I])
        except:
            print(f"************* {method=} {len(I)=} {np.linalg.eigvals(QI).min()=}")
            x[I] = np.linalg.lstsq(QI, -q[I])
        x[A] = 0
        y[I] = 0
        y[A] = (Q@x)[A] + q[A]
    def solve(self, x=None):
        if x is None: x = np.zeros_like(self.q)
        idx, Q, q, algoin, eps = self.idx, self.Q, self.q, self.algoin, self.eps
        y = np.zeros_like(x)
        y[:] = q[:]
        for iter in range(algoin.niter):
            self.kappa = 1 - 0.0001**(iter+1)
            A = idx[x <= algoin.c * y]
            # if algoin.method=="hik":
            #     A = idx[x<=algoin.c*y]
            # elif algoin.method in ["sft","sft2"]:
            #     A = idx[x<=algoin.c*y]
            #     # A = idx[x <= y-algoin.kappa*np.minimum(0,x)-algoin.kappa*np.minimum(0,y)]
            # else:
            #     raise ValueError(f"unknown {algoin.method=}")
            self._solve(x, y, A)
            if algoin.verbose>=2: print(f"{A=} {x=}")
            xi, yi = np.count_nonzero(x < -eps), np.count_nonzero(y < -eps)
            loss = 0.5*np.dot(Q@x,x) + np.dot(q,x)
            if algoin.verbose:
                print(f"{algoin.method:5s} {iter:5d} {loss:12.5e} {len(A):5d}(#A) {np.sum(x * y):10.3e}(x*y) {xi:5d}(#x<0) {yi:5d}(#y<0)")
            if iter and xi == 0 and yi == 0 and np.sum(x * y) < eps:
                return algo_data.AlgoOut(niter=iter, x=x)
        return algo_data.AlgoOut(niter=iter, x=x, failure=True)


#-------------------------------------------------------------
def test(type='mmatrix', niter=5, p=3):
    import matplotlib.pyplot as plt
    ns = [p**k for k in range(2,2+niter)]
    methods = ['hik', 'sft', 'abs']
    nr = 5
    data = {c:np.zeros(len(ns)) for c in methods}
    data_fail = {c:np.zeros(len(ns)) for c in methods}
    algoin = algo_data.AlgoIn(niter=50, verbose=True)
    algoin.c = 1
    algoin.kappa = 0.9
    for i,n in enumerate(ns):
        for ir in range(nr):
            if type == 'mmatrix':
                # M += -2*np.sum(M, axis=0).min() * np.eye(n)
                Q = - np.random.rand(n,n)
                Q = 0.5*(Q+Q.T)
                Q += np.diag(-1.1*np.sum(Q, axis=0))
                # print(f"{np.linalg.eigvals(Q).min()=}")
                q = 2*(np.random.rand(n)-0.5)
            elif type == 'lsl1':
                M = np.random.rand(2*n,n)
                # MTM = M.T@M
                MTM = M@M.T
                Q = np.block([[MTM+0.01*np.eye(2*n), -MTM],[-MTM, MTM+0.01*np.eye(2*n)]])
                m = np.random.rand(2*n)-0.5
                e = 0.025*np.ones(2*n)
                q = np.block([m+e,-m+e])
            elif type == 'ls':
                M = np.random.rand(2*n,n)
                Q = M.T@M
                q = np.random.rand(n)-0.5
            elif type == 'ls_trans':
                M = np.random.rand(2 * n, n)
                Q = M @ M.T + 0.01*np.ones(2 * n)
                q = np.random.rand(2*n) - 0.5
            else: raise ValueError(f"unknown {type=}")
            for m in methods:
                algoin.method = m
                pdas = PDAS_pos(Q, q, algoin)
                algoout = pdas.solve()
                data[m][i] += algoout.niter/nr
                if algoout.failure:
                    data_fail[m][i] += 1
                    print(f"{np.all(np.linalg.inv(Q)>0)=}")
    for m in methods:
        plt.plot(ns, data[m], '-X', label=f"{m}")
    plt.legend()
    plt.show()
    print(f"{data_fail=}")

def test_cycle():
    Q = np.array([[4,5,-5], [5,9,-5],[-5,-5,7]])
    q = np.array([2,1,-3])
    algoin = algo_data.AlgoIn(niter=40, verbose=True)
    algoin.c=1
    pdas = PDAS_pos(Q, q, algoin)
    algoout = pdas.solve(x=np.array([1,-1,-1]))

#-------------------------------------------------------------
if __name__ == "__main__":
    test(type="mmatrix", niter=6)
    # test(type="lsl1", niter=5)
    # test(type="ls")
    # test(type="ls_trans", niter=5)
    # test_cycle()
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
    # algoout = pdas_hik(M, q, algoin=algoin)
    # if algoout.failure:
    #     print(f"{np.all(np.linalg.inv(M)>0)=}")

