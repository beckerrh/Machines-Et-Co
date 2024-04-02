import numpy as np
import skfem as fem
from skfem.helpers import dot, grad
# helpers make forms look nice

@fem.BilinearForm
def a(u, v, _):
    return dot(grad(u), grad(v))

@fem.LinearForm
def L(v, w):
    x, y = w.x  # global coordinates
    f = np.sin(np.pi * x) * np.sin(np.pi * y)
    return f * v

mesh = fem.MeshTri().refined(3)
Vh = fem.Basis(mesh, fem.ElementTriP1())
A = a.assemble(Vh)
l = L.assemble(Vh)
D = Vh.get_dofs()
x = fem.solve(*fem.condense(A, l, D=D))

@fem.Functional
def error(w):
    x, y = w.x
    uh = w['uh']
    u = np.sin(np.pi * x) * np.sin(np.pi * y) / (2. * np.pi ** 2)
    return (uh - u) ** 2
print(round(error.assemble(Vh, uh=Vh.interpolate(x)), 9))

import matplotlib.pyplot as plt
from skfem.visuals.matplotlib import plot
plot(mesh, x, shading='gouraud', colorbar=True)
plt.show()