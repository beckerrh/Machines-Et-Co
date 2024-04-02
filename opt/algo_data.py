#-------------------------------------------------------------
import numpy as np


class AlgoIn():
    def __init__(self, **kwargs):
        self.verbose = kwargs.pop('verbose',1)
        self.niter = kwargs.pop('niter',100)

#-------------------------------------------------------------
class AlgoOut():
    def __init__(self, **kwargs):
        self.niter = kwargs.pop('niter',-1)
        self.x = kwargs.pop('x')
        self.failure = kwargs.pop('failure', False)
    def __repr__(self):
        return f"{self.niter=} {self.failure=} {self.x=}"
    def __eq__(self, other):
        return self.niter==other.niter and self.failure==other.failure and np.allclose(self.x,other.x)



