import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt
import numpy as np

#----------------------------
# Plot helper
#----------------------------
def plot_solutions(t_plot, t_colloc, u1, u2):
    plt.plot(t_plot, u1, label='MLP approx')
    plt.plot(t_plot, u2, '--', label='Exact solution')
    plt.plot(t_colloc, np.zeros_like(t_colloc), 'Xr', label='Collocation points')
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.legend()
    plt.grid(True)
    plt.show()

#----------------------------
# Flax nnx MLP
#----------------------------
class MLP(nnx.Module):
    def __init__(self, layers, key):
        super().__init__()
        keys = jax.random.split(key, len(layers))
        self.layers = []
        in_dim = layers[0]
        for i, out_dim in enumerate(layers[1:]):
            layer = nnx.Linear(in_dim, out_dim, rngs=nnx.Rngs(keys[i]))
            in_dim = out_dim
            self.layers.append(layer)

    # Forward pass
    def forward(self, t):
        t = jnp.atleast_1d(jnp.array(t))
        y = t.reshape(-1,1)  # shape (batch,1)
        for layer in self.layers[:-1]:
            y = jnp.tanh(layer(y))
        y = self.layers[-1](y)
        return y.squeeze()   # scalar output

#----------------------------
# Physics + BC loss
#----------------------------
def residual(mlp, t):
    u_fn = lambda s: mlp.forward(jnp.array(s))
    u_tt = jax.grad(jax.grad(u_fn))(t)
    return u_tt + (jnp.pi**2) * jnp.sin(jnp.pi*t)

def loss_fn(mlp, t_colloc):
    res = jax.vmap(lambda t: residual(mlp, t))(t_colloc)
    physics_loss = jnp.mean(res**2)
    bc_loss = mlp.forward(0.0)**2 + mlp.forward(3.0)**2
    return physics_loss + bc_loss

#----------------------------
# Problem setup
#----------------------------
t0, t1, n_colloc = 0, 3, 10
t_colloc = jnp.linspace(t0, t1, n_colloc)
layers = [1, 8, 8, 1]
key = jax.random.PRNGKey(0)

mlp = MLP(layers, key)
graph_def, params, batch_stats = nnx.split(mlp, nnx.Param, nnx.BatchStat)

optimizer = optax.lbfgs(learning_rate=1.0)
opt_state = optimizer.init(params)

#----------------------------
# Training step
#----------------------------
@jax.jit
def train_step(params, opt_state):
    def loss_for_params(p):
        m = nnx.merge(graph_def, p, batch_stats)
        return loss_fn(m, t_colloc)
    loss_val, grads = jax.value_and_grad(loss_for_params)(params)
    updates, opt_state = optimizer.update(
        grads, opt_state, params, value=loss_val, grad=grads, value_fn=loss_for_params
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

#----------------------------
# Training loop
#----------------------------
n_epochs = 400
for epoch in range(n_epochs):
    params, opt_state, loss_val = train_step(params, opt_state)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:7d}, Loss: {loss_val:.3e}")

trained_mlp = nnx.merge(graph_def, params, batch_stats)

#----------------------------
# Visualization
#----------------------------
t_plot = jnp.linspace(t0, t1, 200)
u_pred = jax.vmap(lambda t: trained_mlp.forward(t))(t_plot)
u_true = jnp.sin(jnp.pi*t_plot)

plot_solutions(t_plot, t_colloc, u_pred, u_true)
