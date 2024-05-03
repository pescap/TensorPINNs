import os
import deepxde as dde 
dde.config.disable_xla_jit()
from deepxde.backend import tf
from functools import reduce
import argparse

parser = argparse.ArgumentParser(description='Set parameters')
parser.add_argument("--epochs", default=10000, type=int)
parser.add_argument("--seed", default=1234, type=int)
parser.add_argument("--k", default=2, type=int)

args = parser.parse_args()
k = args.k
epochs = args.epochs
seed = args.seed
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

sigma = 1
N = 4000
N_test = 1000
lambd = 1
kappa = 0

geom = dde.geometry.Interval(0, 1)
Geom = GeometryXGeometry(geom, k)

def func(x):
    u_list = [-1/6 * x[:, i:i+1] * (x[:, i:i+1] ** 2 - 1) for i in range(k)]
    u = reduce(lambda x1, x2: x1 * x2, u_list)
    c = np.exp(1/2 * k ** 2 * sigma ** 2)
    return c * u

def hatu(x):
    return -1/6 * x * (x ** 2 - 1)

def pde(x, y):
    
    c = tf.exp(1/2 * k ** 2 * sigma ** 2)     
    
    if k == 2:
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        dy_xxyy = dde.grad.hessian(dy_yy, x, i=0, j=0)
        dy3_xx = dde.grad.hessian(y ** 3, x, i=0, j=0)

        A = dy_xxyy + dy3_xx + dy_yy ** 3 + y ** 6 
        

        xx = x[:, 0:1]
        yy = x[:, 1:2]

        vv = hatu(xx)
        ww = hatu(yy)

        F = c * xx * yy + 3 * c ** 3* (-xx * vv + (-1/2 * xx ** 2 + 1/6)**2) * ww **3 - c ** 3 * vv ** 3 * yy ** 3 + c ** 6 * vv ** 3 * ww ** 3    
    elif k == 1:
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        A = -dy_xx - lambd * y ** 3 - kappa ** 2 * y
        u = func(x)
        F_list = [c * x[:, 0:1] - lambd * u ** 3 - kappa ** 2 * u]
        F = reduce(lambda x1, x2: x1 * x2, F_list)
    else:
        print('Set k=1 or k=2')
        
    return A - F

data = dde.data.PDE(Geom, pde, [], train_distribution="pseudo", num_domain=N, solution=func, num_test=N_test)


layer_size = [k] + [50] * 4 + [1]
activation = 'tanh'
initializer = 'Glorot uniform'
net = dde.nn.FNN(layer_size, activation, initializer)

def output_transform(x, y):
    transf_list = [x[:, i:i+1] * (x[:, i:i+1] - 1) for i in range(k)]
    transf = reduce(lambda x1, x2: x1 * x2, transf_list)
    return transf * y

net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=1e-4, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=epochs)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
 
result = np.sum(
            [np.prod(v.get_shape().as_list()) for v in model.net.trainable_variables]
        )
print('Theta:= ', result)

x = np.array([np.arange(101)/100]).T
x_list = [x.T for i in range(k)]
xx = np.concatenate(x_list).T
sol = func(xx)
pred = model.predict(xx)

name = 'results/diag_pred' + str(k) 
test = np.save(name,pred)

name = 'results/diag_exact' + str(k) 
test = np.save(name,sol)

name = 'results/diag_x' + str(k) 
test = np.save(name,xx[:, 0:1])
