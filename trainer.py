import torch

#-------------------------------------------------------------
class Trainer():
    def __init__(self, model, x, y, optimizer='bfgs'):
        self.model = model
        self.inputs, self.targets = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        if self.inputs.ndim==1: self.inputs=self.inputs.reshape(-1,1)
        if self.targets.ndim==1: self.targets=self.targets.reshape(-1,1)
        if optimizer=="adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        elif optimizer=="bfgs":
            self.optimizer = torch.optim.LBFGS(
                self.model.parameters(), lr=1.0, max_iter=3, tolerance_grad=1e-6,
                line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"*** unknown {optimizer=}")
    def loss_function(self):
        outputs = self.model(self.inputs)
        return torch.mean((outputs-self.targets)**2)+self.model.regularize()
    def closure(self):
        self.optimizer.zero_grad()
        self.loss = self.loss_function()
        self.loss.backward()
        return self.loss
    def train(self, niter=150, rtol=1e-6, gtol=1e-9, out=None):
        self.model.train()
        if out==None: out = niter
        res_history = []
        if out > niter: filter = 1
        elif out == 0: filter = niter
        else: filter = niter//out
        for it in range(niter):
            self.optimizer.step(self.closure)
            loss = float(self.loss)
            if it==0: tol = max(rtol*loss, gtol)
            if out and it%filter==0:
                print(f"Iter {it:6d} loss {loss:12.8e}")
            res_history.append(float(loss))
            if loss < tol:
                self.model.eval()
                return res_history
        print(f"*** No Convergence ***")
        self.model.eval()
        return res_history


