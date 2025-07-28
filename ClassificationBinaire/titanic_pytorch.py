import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
import jax.numpy as jnp
from jax.tree_util import tree_map
from tqdm import tqdm
import titanic

train_dataset, eval_dataset = titanic.get_titanic_data_as_dataset()
train_dataloader_torch = DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_dataloader_torch = DataLoader(eval_dataset, batch_size=64, shuffle=False)


class TitanicNeuralNet(nn.Module):
  def __init__(self, num_hidden_1, num_hidden_2):
    super().__init__()
    self.linear1 = nn.Linear(8, num_hidden_1)
    self.dropout = nn.Dropout(0.01)
    self.linear2 = nn.Linear(num_hidden_1, num_hidden_2)
    self.relu = nn.LeakyReLU()
    self.linear3 = nn.Linear(num_hidden_2, 1, bias=False)
  def forward(self, x):
    x = self.linear1(x)
    x = self.dropout(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.dropout(x)
    x = self.relu(x)
    out = self.linear3(x)
    return out

def train(model, train_dataloader, eval_dataloader, num_epochs):
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  criterion = torch.nn.BCEWithLogitsLoss()
  for epoch in (pbar := tqdm(range(num_epochs))):
    pbar.set_description(f"Epoch {epoch}")
    model.train()
    for batch, labels in train_dataloader:
      # zero your gradients
      optimizer.zero_grad()

      # forward pass
      logits = model(batch)

      # compute loss
      loss = criterion(logits.squeeze(), labels)

      # backward pass
      loss.backward()

      # take an optimizer step
      optimizer.step()

    pbar.set_postfix(train_accuracy=eval(model, train_dataloader), eval_accuracy=eval(model, eval_dataloader))

def eval(model, eval_dataloader):
  model.eval()
  num_correct = 0
  num_samples = 0
  for batch, labels in eval_dataloader:
    logits = model(batch)
    preds = torch.round(torch.sigmoid(logits))
    num_correct += (preds.squeeze() == labels).sum().item()
    num_samples += labels.shape[0]
  return num_correct / num_samples

n1, n2, neopch = 32, 16, 500
# n1, n2, neopch = 16, 8, 100
model = TitanicNeuralNet(num_hidden_1=n1, num_hidden_2=n2)
train(model, train_dataloader_torch, eval_dataloader_torch, num_epochs=neopch)
