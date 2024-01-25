import numpy as np
import torch, tools
import interpol1d
#-------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = lambda x: np.sin( 1 + np.exp(x))
    f = lambda x: np.sin(2*np.pi*x)
    # f = lambda x: np.maximum(0.5*(x-0.5)*(x-1.5), 0)
    dtype = torch.float64
    x = torch.tensor([0, 0.2, 0.6, 1], dtype=dtype)
    y = torch.tensor([0, 1, 0.5, 0], dtype=dtype)
    #
    n_neurons = 3
    #
    model = tools.ffn.FFN_1d(torch.min(x), torch.max(x), n_neurons=n_neurons, dtype=dtype)
    intp = interpol1d.Interpolater(data={'x':x, 'y':y}, model=model, dtype=dtype)
    model2 =  tools.p1_1d.P1_1d(torch.min(x), torch.max(x), n_neurons=n_neurons, dtype=dtype)
    intp2 = interpol1d.Interpolater(data={'x':x, 'y':y}, model=model2, dtype=dtype)

    intp.model.plot_basis(add_title=" ffn  BEFORE")
    plt.show()
    intp2.model.plot_basis(add_title=" fem  BEFORE")
    plt.show()


    intp.train()
    intp2.train()
    xp = torch.linspace(torch.min(x), torch.max(x), 100, dtype=dtype)
    plt.plot(x, y, 'xr', label='data')
    plt.plot(xp, intp(xp).detach().numpy(), '--b', label='ffn')
    plt.plot(xp, intp2(xp).detach().numpy(), '-.g', label='p1')
    plt.legend()
    plt.grid()
    plt.show()
    intp.model.plot_basis(add_title=" ffn")
    plt.show()
    intp2.model.plot_basis(add_title=" fem")
    plt.show()
