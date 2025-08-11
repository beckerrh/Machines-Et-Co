import matplotlib.pyplot as plt

#==================================================================
def plot_solutions(plot_dict):
    for i, (k,v) in enumerate(plot_dict.items()):
        ax = plt.subplot(len(plot_dict), 1, i+1)
        ax.set_title(k)
        if isinstance(v[1], dict):
            for k2, v2 in v[1].items():
                ax.plot(v[0], v2, label=k2)
            ax.legend()
        else:
            ax.plot(v[0], v[1])
        ax.grid()
    plt.show()
