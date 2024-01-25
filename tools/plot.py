import numpy as np
import torch
import matplotlib.pyplot as plt
#-------------------------------------------------------------
def plot_basis(machines, add_title=""):
    if not isinstance(machines, (dict)):
        if not isinstance(machines, (list,tuple)) : machines = [machines]
        else: machines = {machine.__class__.__name__:machine for machine in machines}
    n = len(machines)
    fig, axs = plt.subplots(nrows=n+1, ncols=1, layout='constrained')
    for i,name in enumerate(machines.keys()):
        machine = machines[name]
        ax = axs[i]
        x = torch.linspace(machine[0].a, machine[0].b, 301, dtype=machine[0].factory_kwargs['dtype']).reshape(-1, 1)
        if hasattr(machine,'recombine_for_plotting'): machine.recombine_for_plotting()
        phis = machine(x).detach().numpy()
        x = x.detach().numpy()
        for i_neuron in range(machine[0].n_neurons):
            ax.plot(x, phis[:, i_neuron], label=f"{i_neuron}")
        # ax.legend()
        x = np.asarray(machine[0].compute_mesh())
        y = np.ones_like(x)*i/n
        axs[-1].plot(x, y, 'X', label=name)
        ax.set(title="Basis " + name + add_title, xlabel="t", ylabel="phi")
    axs[-1].legend()
