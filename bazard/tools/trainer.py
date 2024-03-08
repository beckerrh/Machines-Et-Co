import torch
import numpy as np

#-------------------------------------------------------------
class Trainer(torch.nn.Module):
    def __init__(self, data, model, **kwargs):
        super().__init__()
        self.print_parameters = kwargs.pop("print_parameters", False)
        self.data = data
        self.model = model
        lr = kwargs.pop("lr", 0.01)
        self.optimiser = kwargs.pop("optimiser", torch.optim.Adam(self.model.parameters(), lr=lr))
        self.loss = kwargs.pop("loss", torch.nn.MSELoss())
    def loss_fct(self):
        if self.print_parameters:
            for parameter in self.parameters():
                print(f"{parameter=}")
        yhat = self.model(self.data['x'])
        return self.loss(yhat.flatten(), self.data['y'])
        # residual = yhat.flatten() - self.data['y']
        # return torch.mean(residual ** 2)
    def closure(self):
        self.optimiser.zero_grad()
        loss_value = self.loss_fct()
        loss_value.backward(retain_graph=True)
        return loss_value
    def train(self, n_iter=130, rtol=1e-6, gtol=1e-9, out=20, dtol=1e-12):
        # torch.autograd.set_detect_anomaly(True)
        trn_loss_history = []
        if out > n_iter: filter = 1
        elif out == 0: filter = n_iter
        else: filter = n_iter//out
        print(f"{'Iter':^6s}  {'loss':^12s} {'diffres':^12s}")
        for i in range(n_iter):
            loss_value = self.optimiser.step(self.closure)
            loss = abs(float(loss_value))
            diffres = torch.nan
            if i==0: tol = max(rtol*loss, gtol)
            else: diffres = abs(loss-trn_loss_history[-1])
            if out and i%filter==0:
                print(f"{i:6d} {loss:14.6e} {diffres:12.6e}")
            trn_loss_history.append(loss)
            if loss < tol:
                return trn_loss_history
            if i and loss < tol and diffres < dtol:
                return trn_loss_history
        return trn_loss_history
    def train2(self, n_iter=130, rtol=1e-6, gtol=1e-9, out=20, dtol=1e-12):
        # torch.autograd.set_detect_anomaly(True)
        n_all = self.data['x'].shape[0]
        n_training = 3*n_all//4
        n_validation = n_all - n_training
        indices = np.random.permutation(n_all)
        trn_indices, val_indices = indices[:n_training], indices[n_training:]
        Tdataset = {'x':self.data['x'][trn_indices], 'y': self.data['y'][trn_indices]}
        Vdataset = {'x':self.data['x'][val_indices], 'y': self.data['y'][val_indices]}

        batch_size = n_training//4
        trn_losses = []
        val_losses = []
        if out > n_iter: filter = 1
        elif out == 0: filter = n_iter
        else: filter = n_iter//out
        print(f"{n_all=} {n_training=} {n_validation=} {batch_size=}")
        print(f"{'Iter':^6s}  {'loss':^12s} {'diffres':^12s}")
        for i in range(n_iter):
            indices = np.random.permutation(n_training)
            batch_idx = 0
            epoch_trn_losses = []
            epoch_val_losses = []
            while batch_idx < n_training:
                idxs = indices[batch_idx:batch_idx + batch_size]
                self.optimiser.zero_grad()
                yhat = self.model(Tdataset['x'][idxs])
                loss_value = self.loss(yhat.flatten(), Tdataset['y'][idxs])
                loss_value.backward(retain_graph=True)
                self.optimiser.step()
                loss = abs(float(loss_value.item()))
                epoch_trn_losses.append(loss)
                batch_idx += batch_size
                with torch.no_grad():
                    # Compute validation
                    yhat = self.model(Vdataset['x'])
                    loss_value = self.loss(yhat.flatten(), Vdataset['y'])
                    loss = abs(float(loss_value.item()))
                    epoch_val_losses.append(loss)
            trn_losses.append(np.mean(epoch_trn_losses))
            val_losses.append(np.mean(epoch_val_losses))
            trn_loss = trn_losses[-1]
            val_loss = val_losses[-1]
            if i==0: tol = max(rtol*val_loss, gtol)
            if out and i%filter==0:
                print(f"{i:6d} {trn_loss:14.6e} {val_loss:12.6e}")
            if val_loss < tol:
                return trn_losses, val_losses
        return trn_losses, val_losses


#-------------------------------------------------------------
if __name__ == "__main__":
    print("so far not written")