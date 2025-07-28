import flax
from flax import nnx
from torch.utils.data import DataLoader, default_collate
import jax.numpy as jnp
from jax.tree_util import tree_map
from tqdm import tqdm
import titanic

# This collate function is taken from the JAX tutorial with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
  return tree_map(jnp.asarray, default_collate(batch))

train_dataset, eval_dataset = titanic.get_titanic_data_as_dataset()
train_dataloader_jax = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=numpy_collate)
eval_dataloader_jax = DataLoader(eval_dataset, batch_size=64, shuffle=False, collate_fn=numpy_collate)


class TitanicNNX(nnx.Module):
  def __init__(self, num_hidden_1, num_hidden_2, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(8, num_hidden_1, rngs=rngs)
    self.dropout = nnx.Dropout(0.01, rngs=rngs)
    self.relu = nnx.leaky_relu
    self.linear2 = nnx.Linear(num_hidden_1, num_hidden_2, rngs=rngs)
    self.linear3 = nnx.Linear(num_hidden_2, 1, use_bias=False, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = self.dropout(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.dropout(x)
    x = self.relu(x)
    out = self.linear3(x)
    return out

model = TitanicNNX(32, 16, rngs=nnx.Rngs(0))
nnx.display(model)

import optax
import jax.numpy as jnp

def train(model, train_dataloader, eval_dataloader, num_epochs):
  optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.01))

  for epoch in (pbar := tqdm(range(num_epochs))):
    pbar.set_description(f"Epoch {epoch}")
    model.train()
    for batch in train_dataloader:
      train_step(model, optimizer, batch)

    pbar.set_postfix(train_accuracy=eval(model, train_dataloader), eval_accuracy=eval(model, eval_dataloader))

@nnx.jit
def train_step(model, optimizer, batch):
  def loss_fn(model):
    logits = model(batch[0])
    loss = optax.sigmoid_binary_cross_entropy(logits.squeeze(), batch[1]).mean()
    return loss
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(model)
  optimizer.update(grads)

def eval(model, eval_dataloader):
  total = 0
  num_correct = 0
  model.eval()
  for batch in eval_dataloader:
    res = eval_step(model, batch)
    total += res.shape[0]
    num_correct += jnp.sum(res)
  return num_correct / total

@nnx.jit
def eval_step(model, batch):
  logits = model(batch[0])
  logits = logits.squeeze()
  preds = jnp.round(nnx.sigmoid(logits))
  return preds == batch[1]

train(model, train_dataloader_jax, eval_dataloader_jax, num_epochs=500)