import torch

import mlp

# CUDA support
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

#-------------------------------------------------------------
class FullLinear(torch.nn.Module):
    def __init__(self, DEVICE, nneurons, nlayers=0, actfct=torch.nn.Tanh()):
        super().__init__()
        layers = [torch.nn.Linear(2, nneurons, device=DEVICE)]
        layers.append(actfct)
        for i in range(nlayers):
            layers.append(torch.nn.Linear(nneurons, nneurons, device=DEVICE))
            layers.append(actfct)
        layers.append(torch.nn.Linear(nneurons, 1, device=DEVICE))
        self.nn = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.nn(x)


#-------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import pathlib
    import get_fem_data
    import tqdm
    import copy
    import matplotlib.pyplot as plt

    datadir = pathlib.Path("data")
    if datadir.is_dir():
        data = {}
        for p in datadir.glob("*.npy"):
            name = p.name.split('.npy')[0]
            data[name] = np.load(p)
    else:
        datadir.mkdir()
        data = get_fem_data.get_fem_data(plotting=True, verbose=1)
        for k,v in data.items(): np.save(datadir / k, v)

    # oublie z
    points, T, simp, bdry = data['p'][:,:-1], data['T'], data['s'], data['b']
    print(f"{points.shape=} {T.shape=}")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(points, T, train_size=0.7, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = mlp.MLP(
        input_size=2,
        hidden_size=400,
        output_size=1,
        depth=4,
        # actfct=torch.nn.Tanh()
        actfct = torch.nn.ReLU()
    ).to(device)

    loss_fn = torch.nn.MSELoss()  # mean square error
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # optimizer = torch.optim.LBFGS(
    #     model.parameters(),
    #     lr=1.0,
    #     max_iter=50000,
    #     max_eval=50000,
    #     history_size=50,
    #     tolerance_grad=1e-7,
    #     tolerance_change=1.0 * np.finfo(float).eps,
    #     line_search_fn="strong_wolfe",  # better numerical stability
    # )

    # training parameters
    n_epochs = 500  # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    # Hold the best model
    best_mse = np.inf  # init to infinity
    best_weights = None
    history = []

# training loop
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print(f"MSE: {np.sqrt(best_mse):.2f}")
    plt.plot(history)
    plt.show()

    y = model(torch.from_numpy(points).to(dtype=X_test.dtype)).detach().numpy().reshape(-1)
    print(f"{y.shape=} {y.min()=} {y.max()=}")
    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[0].tricontourf(points[:,0], points[:,1], simp, y)
    axs[1].tricontourf(points[:,0], points[:,1], simp, T)
    plt.show()
