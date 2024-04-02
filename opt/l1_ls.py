import numpy as np
import algo_data
#-------------------------------------------------------------
def pdas_hki(A, b, lam, x=None, algoin = algo_data.AlgoIn()):
    eps=1e-12
    # print(f"{c=}")
    if x is None: x = np.zeros(shape=A.shape[1])
    p = np.zeros_like(x)
    e = lam*np.ones_like(p)
    ATA = A.T@A
    ATb = A.T@b
    idx = np.arange(len(x))
    for iter in range(algoin.niter):
        xm, xp = -np.minimum(0,x), np.maximum(0,x)
        ik = idx[p > np.maximum(xp,2*e - xm)]
        ak = idx[p <= np.minimum(xp,2*e - xm)]
        iak = idx[np.logical_and(p > xp,p <= 2*e - xm)]
        aik = idx[np.logical_and(p > 2*e - xm, p >= xp)]
        # print(f"{ak=}")
        if algoin.verbose:
            q = 2*e-p
            pneg = np.count_nonzero(p < -eps)
            qneg = np.count_nonzero(q < -eps)
            res = 0.5*np.sum((A@x - b)**2)
            pen = np.sum(np.abs(x))
            print(f"{iter=} {res=} {pen=}")
            print(f"{iter=} {len(ik)=} {len(ak)=} {len(aik)=} {len(iak)=} {pneg=} {qneg=} {np.sum(p * xp)=} {np.sum(q * xm)=}")
        if iter and pneg == 0 and qneg == 0 and np.sum(p * xp) + np.sum(q*xm) < eps:
            return algo_data.AlgoOut(niter=iter, x=x)
        if len(ak):
            x[ak] = np.linalg.solve(ATA[ak[:, np.newaxis], ak], (p  + ATb -e)[ak])
        if len(ik):
            x[ik] = 0
        if len(aik):
            p[aik] = 0
        if len(iak):
            p[iak] = xp + 2*e
    return algo_data.AlgoOut(niter=iter, x=x, failure=True)

#-------------------------------------------------------------
def pdas_hki_schrott(A, b, lam, x=None, algoin = algo_data.AlgoIn()):
    c = algoin.c
    eps=1e-12
    # print(f"{c=}")
    u = np.zeros(shape=A.shape[1])
    v = np.zeros_like(u)
    p = np.zeros_like(u)
    q = np.zeros_like(u)
    e = lam*np.ones_like(u)
    ATA = A.T@A
    ATb = A.T@b
    idx = np.arange(len(u))
    for iter in range(algoin.niter):
        apk, ipk = idx[u < c * p], idx[u >= c * p]
        aqk, iqk = idx[v < c * q], idx[v >= c * q]
        # print(f"{ak=}")
        if algoin.verbose:
            uneg = np.count_nonzero(u < -eps)
            vneg = np.count_nonzero(v < -eps)
            pneg = np.count_nonzero(p < -eps)
            qneg = np.count_nonzero(q < -eps)
            res = 0.5*np.sum((A@u - A@v - b)**2)
            pen = np.sum(np.abs(u)+np.abs(v))
            print(f"{iter=} {res=} {pen=}")
            print(f"{iter=} {len(apk)=} {len(ipk)=} {np.sum(p * u)} #u<0: {uneg} #p<0: {pneg}")
            print(f"{iter=} {len(aqk)=} {len(iqk)=} {np.sum(q * v)} #v<0: {vneg} #q<0: {qneg}")
        if iter and uneg == 0 and vneg == 0  and pneg == 0 and qneg == 0 and np.sum(p * u) < eps and np.sum(q * v) < eps:
            return algo_data.AlgoOut(niter=iter, x=u-v)
        # if len(ipk) and len(iqk):
        #     line1 = [ ATA[ipk[:, np.newaxis], ipk],-ATA[ipk[:, np.newaxis], iqk] ]
        #     line2 = [-ATA[iqk[:, np.newaxis], ipk], ATA[iqk[:, np.newaxis], iqk] ]
        #     Abig = np.block([line1,line2])
        #     bbig = np.hstack([(p + ATA@v + ATb -e)[ipk],(p + ATA@u- ATb - e)[iqk]])
        #     print(f"{Abig.shape=} {bbig.shape=} {np.linalg.eigvals(ATA)=} {np.linalg.eigvals(Abig)=}")
        #     xbig = np.linalg.solve(Abig, bbig)
        #     print(f"{xbig.shape=}")
        #     u[ipk] = xbig[:len(ipk)]
        #     v[iqk] = xbig[len(ipk):]
        #     p[ipk] = 0
        #     q[iqk] = 0
        if len(ipk):
            u[ipk] = np.linalg.solve(ATA[ipk[:, np.newaxis], ipk], (p + ATA@v + ATb -e)[ipk])
            p[ipk] = 0
        if len(iqk):
            v[iqk] = np.linalg.solve(ATA[iqk[:, np.newaxis], iqk], (p + ATA@u- ATb - e)[iqk])
            q[iqk] = 0
        if len(apk):
            u[apk] = 0
            p[apk] = (ATA@u)[apk] - (ATA@v)[apk] - (p +ATb - e)[apk]
        if len(aqk):
            v[aqk] = 0
            q[aqk] = (ATA @ v)[aqk] - (ATA @ u)[aqk] - (p - ATb - e)[aqk]
    return algo_data.AlgoOut(niter=iter, x=x, failure=True)

#-------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N,n = 10, 3
    A = np.random.rand(N,n)
    b = np.random.rand(N)
    algoin = algo_data.AlgoIn(niter=40, verbose=1)
    algoin.c = 10
    lam = 5
    algoout = pdas_hki(A, b, lam, algoin=algoin)
    print(f"{algoout.x=}")
    # if algoout.failure:
    #     print(f"{np.all(np.linalg.inv(M)>0)=}")

