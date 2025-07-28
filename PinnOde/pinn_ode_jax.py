import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Define the neural network: a simple 2-layer MLP
def init_mlp_params(key, layers):
    params = []
    for in_dim, out_dim in zip(layers[:-1], layers[1:]):
        key, subkey = jax.random.split(key)
        w = jax.random.normal(subkey, (out_dim, in_dim)) * jnp.sqrt(2 / in_dim)
        b = jnp.zeros(out_dim)
        params.append((w, b))
    return params

def mlp(params, x):
    activ = x
    for w, b in params[:-1]:
        activ = jnp.tanh(w @ activ + b)
    w, b = params[-1]
    return (w @ activ + b)[0]

# Vectorize for batch input
def mlp_batch(params, x_batch):
    return jax.vmap(lambda x: mlp(params, x))(x_batch)

# Define u(x) = NN(x)
def u(params, x):
    return mlp(params, jnp.array([x]))

# Compute u''(x) using JAX autodiff
def dudx(params, x):
    return jax.grad(u, argnums=1)(params, x)

def d2udx2(params, x):
    return jax.grad(dudx, argnums=1)(params, x)

# Physics loss: residual of the ODE
def residual(params, x):
    return d2udx2(params, x) + (jnp.pi ** 2) * jnp.sin(jnp.pi * x)

# Total loss = physics + boundary conditions
def loss(params, x_colloc):
    # Physics loss
    res = jax.vmap(lambda x: residual(params, x))(x_colloc)
    physics_loss = jnp.mean(res ** 2)
    # Boundary conditions: u(0)=0, u(1)=0
    bc_loss = u(params, 0.0)**2 + u(params, 1.0)**2
    return physics_loss + bc_loss

# Training step
@jax.jit
def update(params, x_colloc, lr):
    grads = jax.grad(loss)(params, x_colloc)
    return [(w - lr * dw, b - lr * db)
            for (w, b), (dw, db) in zip(params, grads)]

# Training setup
key = jax.random.PRNGKey(0)
layers = [1, 32, 32, 1]
params = init_mlp_params(key, layers)

# Collocation points (x in [0,1])
t0, t1 = 0, 3
x_colloc = jnp.linspace(t0, t1, 100)

# Training loop
for epoch in range(2000):
    params = update(params, x_colloc, lr=1e-3)
    if epoch % 100 == 0:
        l = loss(params, x_colloc)
        print(f"Epoch {epoch}, Loss: {l:.6f}")

# Plot results
x_plot = jnp.linspace(t0, t1, 200)
u_pred = jax.vmap(lambda x: u(params, x))(x_plot)
u_true = jnp.sin(jnp.pi * x_plot)

plt.plot(x_plot, u_pred, label='PINN')
plt.plot(x_plot, u_true, '--', label='Exact solution')
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(r"PINN Solution to $u'' = -\pi^2 \sin(\pi x)$")
plt.grid(True)
plt.show()