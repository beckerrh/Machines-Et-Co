import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

# Commençons par la fonction de visu
def plot_solutions(t_plot, t_colloc, u1, u2, t1='Approximation', t2='Solution', ls1='', ls2='--'):
    plt.plot(t_plot, u1, ls1, label=t1)
    plt.plot(t_plot, u2, ls2, label=t2)
    plt.plot(t_colloc, np.zeros_like(t_colloc), 'Xr', label='t_colloc')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(r"$u'' = -\pi^2 \sin(\pi t)$")
    plt.grid(True)
    plt.show()

#-----------------------------------------------------------
class MLP(nnx.Module):
    def __init__(self, layers):
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        self.layers = []
        for l in range(1,len(layers)):
            in_dim, out_dim =  layers[l-1], layers[l]
            key, subkey = jax.random.split(key)
            layer = nnx.Linear(in_dim, out_dim, rngs=nnx.Rngs(key))
            self.layers.append(layer)
            in_dim = out_dim
    def forward(self, t):
        t = jnp.array([t])
        for layer in self.layers[:-1]:
            t = jnp.tanh(layer(t))
        last_layer = self.layers[-1]
        return last_layer(t)[0]
# Calcul de u''(x) par JAX autodiff
    def dudt(self, t):
        return jax.grad(lambda x: self.forward(x))(t)
    def d2udt2(self, t):
        return jax.grad(lambda x: self.dudt(x))(t)
    def residual_ode(self, t):
        return self.d2udt2(t) + (jnp.pi ** 2) * jnp.sin(jnp.pi * t)
    def residual_bc(self, t0, t1):
        return self.forward(t0) **2 + self.forward(t1)**2

# Points de collocation
t0, t1, n_colloc = 0, 3, 10
t_colloc = jnp.linspace(t0, t1, n_colloc)
# Machine
layers = [1, 8, 8, 1]
machine = MLP(layers)
graphdef, params, batch_stats = nnx.split(machine, nnx.Param, nnx.BatchStat)

def loss(params):
    # ode loss
    machine_tmp = nnx.merge(graphdef, params, batch_stats)
    res = jax.vmap(machine_tmp.residual_ode)(t_colloc)
    ode_loss = jnp.mean(res ** 2)
    # Boundary conditions
    bc_loss = machine_tmp.residual_bc(t_colloc[0], t_colloc[-1])
    return ode_loss + bc_loss




optimizer = optax.lbfgs(learning_rate=0.001)
opt_state = optimizer.init(params)
@jax.jit
def train_step(params, opt_state):
    loss_value, grads = jax.value_and_grad(loss)(params)
    updates, opt_state = optimizer.update(
        grads, opt_state, params, value=loss_value, grad=grads, value_fn=loss
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value, grads, updates

n_epochs = 400
for epoch in range(n_epochs):
    params, opt_state, loss_value, grads, updates = train_step(params, opt_state)
    if epoch % 100== 0:
        print(f"Epoch {epoch:7d}, Loss: {loss_value:.3e}")

trained_machine = nnx.merge(graphdef, params, batch_stats)
# Visu
t_plot = jnp.linspace(t0, t1, 200)
u_pred = jax.vmap(trained_machine.forward)(t_plot)
u_true = jnp.sin(jnp.pi * t_plot)

plot_solutions(t_plot, t_colloc, u_pred, u_true)
