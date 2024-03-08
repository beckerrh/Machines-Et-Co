import numpy as np
import torch
import matplotlib.pyplot as plt

#-------------------------------------------------------------
def mapping(x):
    x1, x2 = 2*(x[:,0]-0.5), 2*(x[:,1]-0.5)
    x[:,0] = np.exp(x1)*np.cos(x2)
    x[:,1] = np.exp(x1)*np.sin(x2)
    return x
def create_chats_et_chiens(n = 60):
    np.random.seed(12345)
    chats = np.random.rand(n, 2)*1
    chiens = np.random.rand(n, 2)*1
    chats[:,0] *= 0.5
    chiens[:,0] *= 0.5
    chiens[:,0] += 0.5
    chats += 0.1*np.random.randn(n, 2)
    chiens += 0.1*np.random.randn(n, 2)
    chats = mapping(chats)
    chiens = mapping(chiens)
    return chats, chiens
#-------------------------------------------------------------
if __name__ == "__main__":
    import nn

    m=22
    chats, chiens = create_chats_et_chiens(m)
    fig, ax = plt.subplots(layout='constrained')
    # ax.plot(chats[:,0], chats[:,1], 'rx', label="chats")
    # ax.plot(chiens[:,0], chiens[:,1], 'bo', label="chiens")
    ax.scatter(chats[:,0], chats[:,1], c='r', label="chats")
    ax.scatter(chiens[:,0], chiens[:,1], c='b', label="chiens")
    ax.legend()
    classifier = nn.Classifier_NN(in_features=2, out_features=1, n_neurons=6, n_layers=0, actfct=torch.nn.ReLU())
    data = {}
    data['x'] = torch.empty(2*m, 2)
    data['x'][:m,:] = torch.from_numpy(chats)
    data['x'][m:,:] = torch.from_numpy(chiens)
    data['y'] = torch.empty(2*m)
    data['y'][:m] = -1.0
    data['y'][m:] = 1.0
    x1min, x1max = torch.min(data['x'][:,0]), torch.max(data['x'][:,0])
    x2min, x2max = torch.min(data['x'][:,1]), torch.max(data['x'][:,1])
    # optimiser = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.01)
    optimiser = torch.optim.Adam(classifier.parameters(), lr=0.01)
    loss = torch.nn.HingeEmbeddingLoss()
    # loss = torch.nn.HuberLoss()
    loss = torch.nn.MSELoss()
    trainer = bazard.tools.trainer.Trainer(data, classifier, optimiser=optimiser, loss=loss)
    res_hist, val_hist = trainer.train2(n_iter=1000)
    nplot = 20
    x1, x2 = np.meshgrid(np.linspace(x1min, x1max,nplot), np.linspace(x2min, x2max,nplot))
    x = torch.empty(nplot**2, 2)
    x[:,0] = torch.from_numpy(x1.flatten())
    x[:,1] = torch.from_numpy(x2.flatten())
    z = classifier(x).reshape(nplot, nplot)
    import matplotlib.cm as cm
    cs = ax.contourf(x1, x2, z.detach().numpy(), cmap=cm.Pastel1, alpha=0.5)
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('values')
    cs2 = ax.contour(cs, colors='k', levels=[-0.5, 0.0, 0.5])
    cbar.add_lines(cs2)
    plt.show()
    plt.plot(res_hist)
    plt.plot(val_hist)
    plt.show()

