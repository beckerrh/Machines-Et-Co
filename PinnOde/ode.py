import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax


#==================================================================
def plot_solutions(t_plot, u_pred, u_true, u_cg, t_colloc, base_dict=None):
    if base_dict is not None:
        plt.subplot(211)
    plt.plot(t_plot, u_pred, '-', label='app')
    if u_true is not None:
        plt.plot(t_plot, u_true, ':', label='sol')
    if u_cg is not None:
        plt.plot(t_colloc, u_cg, ':', label='cg')
    plt.plot(t_colloc, np.zeros_like(t_colloc), 'Xr', label='t_colloc')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(r"$u' = f(u)$")
    plt.grid()
    if base_dict is not None:
        plt.subplot(212)
        plt.title(r"bases")
        plt.plot(t_plot, base_dict.T)
        plt.xlabel("t")
        plt.grid()
    plt.show()

#==================================================================
def swish(x):
    return x * jax.nn.sigmoid(x)
#==================================================================
class MachineEdoO2:
    def __init__(self, layers, n_colloc, app):
        self.layers = (len(layers)+2)*[1]
        self.layers[1:-1] = layers
        self.layers[-1] = 1 if type(app.u0)==float else len(app.u0)
        self.t_colloc =  jnp.linspace(app.t_begin, app.t_end, n_colloc)
        self.app = app
    def init_params(self):
        key = jax.random.PRNGKey(0)
        params = []
        for l in range(1, len(self.layers)):
            key, subkey = jax.random.split(key)
            W = jax.random.normal(subkey, (self.layers[l], self.layers[l-1])) * jnp.sqrt(2 / self.layers[l-1])
            b = jnp.zeros(self.layers[l])
            params.append((W, b))
        return params
    def bases(self, params, t):
        t = jnp.atleast_1d(t)   # shape (1,)
        x = t[None, :]          # shape (1, 1)
        for W, b in params[:-1]:
            x = swish(W @ x + b[:, None])
            # x = jnp.tanh(W @ x + b[:, None])
        return x
    def forward(self, params, t):
        t = jnp.atleast_1d(t)   # shape (1,)
        x = t[None, :]          # shape (1, 1)
        for W, b in params[:-1]:
            # x = swish(W @ x + b[:, None])
            x = jnp.tanh(W @ x + b[:, None])
        W, b = params[-1]
        out = W @ x + b[:, None]
        return out.squeeze()
    def dudt(self, params, t):
        return jax.jacrev(self.forward, argnums=1)(params, t)
        # return jax.grad(self.forward, argnums=1)(params, t)
    # def d2udt2(self, params, t):
    #     return jax.grad(self.dudt, argnums=1)(params, t)
    def residual(self, params, t):
        u = self.forward(params, t)
        return self.dudt(params, t) - self.app.f(u)
    # Total loss = physics + boundary conditions
    def loss(self, params):
        # Physics loss
        # u_pred = self.forward(params, self.t_colloc)
        # return jnp.mean((u_pred - self.u_true)**2)
        res_dom = jax.vmap(lambda t: self.residual(params, t))(self.t_colloc)
        res_bdry = self.forward(params, self.t_colloc[0])-self.app.u0
        physics_loss = jnp.mean(res_dom ** 2)
        bc_loss = jnp.mean(res_bdry ** 2)
        return physics_loss + bc_loss

def train_by_hand(params, model, lr=0.01, n_epochs=1000):
    # Training step
    @jax.jit
    def update(params, lr):
        grads = jax.grad(model.loss)(params)
        return  [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
    #Training loop
    for epoch in range(n_epochs):
        params = update(params, lr)
        if epoch % 100 == 0:
            loss_val = model.loss(params)
            print(f"Epoch {epoch:7d} loss {loss_val:.3e}")
    return params

def train_with_optax(params, model, lr=0.01, n_epochs=1000):
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(model.loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in range(n_epochs):
        params, opt_state, l = train_step(params, opt_state)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:7d}, Loss: {l:.3e}")
    return params


if __name__ == '__main__':
    # Application
    import ode_examples
    # app, layers, n_colloc = ode_examples.Exponential(), [8,8], 12
    app, layers, n_colloc = ode_examples.Pendulum(t_end=3.25), [12,12], 25
    # Machine setup
    # Collocation points
    model = MachineEdoO2(layers, n_colloc, app)
    params = model.init_params()

    # params = train_by_hand(params, model, 0.001, n_epochs=12000)
    params = train_with_optax(params, model, 0.001, n_epochs=12000)

    # Plot results
    t_plot = jnp.linspace(model.t_colloc[0], model.t_colloc[-1], 200)
    u_pred = jax.vmap(lambda t: model.forward(params, t))(t_plot)
    if hasattr(app, 'solution'):
        u_true = app.solution(t_plot)
        u_cg = None
    else:
        u_true = None
        import ode_solver
        cgp = ode_solver.CgK(k=2)
        u_node, u_coef = cgp.run_forward(model.t_colloc, app)
        u_cg = u_node.T

    base_dict = model.bases(params, t_plot)

    plot_solutions(t_plot, u_pred, u_true, u_cg, model.t_colloc, base_dict=base_dict)