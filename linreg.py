import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, LBFGS, SGD
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class LinearModel(nn.Module):
    def __init__(self, device, nneurons=15, nlayers=1, actfct=None):
        super().__init__()
        layers = [torch.nn.Linear(1, nneurons, device=device)]
        if actfct is None: actfct = torch.nn.ReLU()
        layers.append(actfct)
        for i in range(nlayers):
            layers.append(torch.nn.Linear(nneurons, nneurons, device=device))
            layers.append(actfct)
        layers.append(torch.nn.Linear(nneurons, 1, device=device, bias=False))
        self.nlayers = nlayers
        self.nn = torch.nn.Sequential(*layers)
    def regularize(self, eps=0.01):
        reg=0
        for i in range(self.nlayers+1):
            reg += eps*torch.mean(self.nn[2*i].weight.data**2)
            # reg += eps * torch.mean(torch.abs(self.nn[2 * i].weight.data))
        return reg
    def forward(self, x):
        return self.nn(x) + self.regularize()

x = torch.linspace(-5, 5, steps=20)
y = 1 + 2*x + 0.3*x**2
x, y = x.reshape(-1, 1), y.reshape(-1, 1)
print(f"{x.shape=}")
class DummyData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Determine if it is a cpu or a gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DataSet
dummy_data = DummyData(x.to(device), y.to(device))

# Training parameters
criterion = nn.MSELoss()
epochs = 20

lm_lbfgs = LinearModel(device)
lm_lbfgs.to(device)
optimizer = LBFGS(lm_lbfgs.parameters(), history_size=10, max_iter=4)

for epoch in range(epochs):
    running_loss = 0.0

    for i, (x, y) in enumerate(dummy_data):

        x_ = Variable(x, requires_grad=True)
        y_ = Variable(y)
        def closure():
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = lm_lbfgs(x_)

            # Compute loss
            loss = criterion(y_pred, y_)

            # Backward pass
            loss.backward()

            return loss

        # Update weights
        optimizer.step(closure)

        # Update the running loss
        loss = closure()
        running_loss += loss.item()

    print(f"Epoch: {epoch + 1:02}/{epochs} Loss: {running_loss:.5e}")


x_for_pred = torch.linspace(-5, 5, steps=60).reshape(-1, 1).to(device)
with torch.no_grad():
    y_pred_lbfgs = lm_lbfgs(Variable(x_for_pred)).cpu().data.numpy()


# Training data
print(f"{x.shape=}")
num_training_points = dummy_data.x.shape[0]
x_plot = dummy_data.x.reshape(num_training_points,).cpu().data.numpy()
y_plot = dummy_data.y.reshape(num_training_points,).cpu().data.numpy()

# Prediction data
num_pred_points = x_for_pred.shape[0]
x_for_pred = x_for_pred.reshape(num_pred_points,).cpu().data.numpy()
y_pred_lbfgs = y_pred_lbfgs.reshape(num_pred_points,)

print(f"{num_pred_points=} {num_training_points=} {x_plot=}")

fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(x_for_pred, y_pred_lbfgs, "b", label="Predictions with LBFGS", alpha=0.4, lw=2)
ax.plot(x_plot, y_plot, "ko", label="True data")
ax.set_xlabel(r"x", fontsize="xx-large")
ax.set_ylabel(r"y", fontsize="xx-large")
ax.legend(fontsize="xx-large")
plt.show()