import os
import deepxde as dde 
from deepxde.backend import tf
from functools import reduce
import argparse
dde.config.disable_xla_jit()

parser = argparse.ArgumentParser(description='Set parameters')

parser.add_argument("--k", default=2, type=int)
parser.add_argument("--epochs", default=10000, type=int)
args = parser.parse_args()

k = args.k
epochs = args.epochs
seed = 1234

dde.config.set_random_seed(seed)

#class TensorGeometryXTime:
import numpy as np

class GeometryXGeometry:
    def __init__(self, geometry, k):
        self.geometry = geometry
        self.dim = geometry.dim ** k
        self.k = k
    
    def random_points(self, n, random="pseudo"):
        xx = []
        for i in range(self.k):
            xx.append(self.geometry.random_points(n, random=random))
        return np.hstack(xx)
    
    def uniform_points(self, n, boundary=False):
        print('no uniform points')
        return self.random_points(n)
    

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np

n = 1

N_f = 5000
N_f_test = 5000

parameters = [1e-2, 2, 350, "sin"] 

# Define sine function
if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
else:
    from deepxde.backend import tf
    sin = tf.sin

learning_rate, num_dense_layers, num_dense_nodes, activation = parameters


def pde(x, y):
    dy1_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy1_yy = dde.grad.hessian(y, x, i=1, j=1)

    dy1_zz = dde.grad.hessian(y, x, i=2, j=2)
    dy1_tt = dde.grad.hessian(y, x, i=3, j=3)

    D1 = - dy1_xx - dy1_yy
    D2 = - dy1_zz - dy1_tt
    
    D1D2 = dde.grad.hessian(D2, x, i=0, j=0) + dde.grad.hessian(D2, x, i=1, j=1)
    
    A = D1D2 + k0 ** 2* D1 + k0 ** 2 * D2 + k0 ** 4 * y
    F = k0 ** 4 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2]) * sin(k0 * x[:, 2:3]) * sin(k0 * x[:, 3:4])

    res = A - F
    return res


def func(x):
    f1 = np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])
    f2 = np.sin(k0 * x[:, 2:3]) * np.sin(k0 * x[:, 3:4])
    return f1 * f2

def transform(x, y):
    res1 = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
    res2 = x[:, 2:3] * (1 - x[:, 2:3]) * x[:, 3:4] * (1 - x[:, 3:4])
    return res1 * res2 * y
    #return (res1 * y) * (res2 * y)

def boundary(_, on_boundary):
    return on_boundary

geom = dde.geometry.Rectangle([0, 0], [1, 1])
Geom = GeometryXGeometry(geom,2)    

k0 = 2 * np.pi * n
wave_len = 1 / n

data = dde.data.PDE(
    Geom,
    pde,
    train_distribution = 'pseudo',
    num_domain=N_f,
    solution=func,
    bcs = [],
    num_test=N_f_test,
)

net = dde.nn.FNN(
    [4] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
)

net.apply_output_transform(transform)

model = dde.Model(data, net)

model.compile("adam", lr=learning_rate, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=epochs)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


Nx = 100
Ny = Nx

# Grid points
xmin, xmax, ymin, ymax = [0,1,0,1]
plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
points = np.vstack((plot_grid[0].ravel(),
                    plot_grid[1].ravel(),
                    np.zeros(plot_grid[0].size)))

points_2d = points[:2, :]
points_4d = np.hstack([points_2d.T, points_2d.T])

u_exact = func(points_4d)
u_sca = model.predict(points_4d)
u_exact = u_exact.reshape((Nx, Ny))
u_sca = u_sca.reshape((Nx, Ny))

result = np.sum(
            [np.prod(v.get_shape().as_list()) for v in model.net.trainable_variables]
        )
print('Theta:= ', result)



'''                  Compare against analytical solution                    '''
# Uncomment to perform comparison, takes a few seconds to run
diff = u_exact-u_sca
error = np.linalg.norm(diff)/np.linalg.norm(u_exact)
print('Relative error = ', error)

name = 'results/diag_pred' + str(k) 
test = np.save(name,u_sca)

name = 'results/diag_exact' + str(k) 
test = np.save(name,u_exact)




