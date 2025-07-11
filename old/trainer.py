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
                self.model.parameters(), lr=1.0, max_iter=5, tolerance_grad=1e-6,
                line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"*** unknown {optimizer=}")
    def loss_function(self, inputs, targets):
        outputs = self.model(inputs)
        return torch.mean((outputs-targets)**2)+self.model.regularize()
    def closure(self):
        self.optimizer.zero_grad()
        if hasattr(self.model,"loss_function"):
            self.loss = self.model.loss_function(self.inputs, self.targets)
        else:
            self.loss = self.loss_function(self.inputs, self.targets)
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




    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)
    #
    # dtype = model.dtype
    # X_train = torch.tensor(X_train, dtype=dtype)
    # y_train = torch.tensor(y_train, dtype=dtype).reshape(-1, 1)
    # X_test = torch.tensor(X_test, dtype=dtype)
    # y_test = torch.tensor(y_test, dtype=dtype).reshape(-1, 1)
    #
    # loss_fn = torch.nn.MSELoss()  # mean square error
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #
    # # training parameters
    # n_epochs = 250  # number of epochs to run
    # batch_size = 10  # size of each batch
    # batch_start = torch.arange(0, len(X_train), batch_size)
    # # Hold the best model
    # best_mse = np.inf  # init to infinity
    # best_weights = None
    # history = []
    #
    # # training loop
    # for epoch in range(n_epochs):
    #     model.train()
    #     with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
    #         bar.set_description(f"Epoch {epoch}")
    #         for start in bar:
    #             # take a batch
    #             X_batch = X_train[start:start + batch_size]
    #             y_batch = y_train[start:start + batch_size]
    #             # forward pass
    #             y_pred = model(X_batch)
    #             loss = loss_fn(y_pred, y_batch)
    #             # backward pass
    #             optimizer.zero_grad()
    #             loss.backward()
    #             # update weights
    #             optimizer.step()
    #             # print progress
    #             bar.set_postfix(mse=float(loss))
    #     # evaluate accuracy at end of each epoch
    #     model.eval()
    #     y_pred = model(X_test)
    #     mse = loss_fn(y_pred, y_test)
    #     mse = float(mse)
    #     history.append(mse)
    #     if mse < best_mse:
    #         best_mse = mse
    #         best_weights = copy.deepcopy(model.state_dict())
    #
    # # restore model and return best accuracy
    # model.load_state_dict(best_weights)
    # print(f"MSE: {np.sqrt(best_mse):.2f}")
    # plt.plot(history)
    # plt.show()

