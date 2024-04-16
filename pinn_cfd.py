import torch

#-------------------------------------------------------------
class FullLinear(torch.nn.Module):
    def __init__(self, DEVICE, nneurons, nlayers=0, actfct=torch.nn.Tanh()):
        super().__init__()
        layers = [torch.nn.Linear(2, nneurons, device=DEVICE)]
        layers.append(actfct)
        for i in range(nlayers):
            layers.append(torch.nn.Linear(nneurons, nneurons, device=DEVICE))
            layers.append(actfct)
        layers.append(torch.nn.Linear(nneurons, 1, device=DEVICE))
        self.nn = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.nn(x)
#-------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import pathlib
    import get_fem_data

    datadir = pathlib.Path("data")
    if datadir.is_dir():
        T =  np.load(datadir / "T.npy")
        points = np.load(datadir / "p.npy")
        bdry = np.load(datadir / "b.npy")
    else:
        datadir.mkdir()
        T, points, bdry = get_fem_data.get_fem_data(plotting=True, verbose=1)
        np.save(datadir / "T", T)
        np.save(datadir / "p", points)
        np.save(datadir / "b", bdry)

