import os
import deepxde as dde 
dde.config.disable_xla_jit()
from deepxde.backend import tf
from functools import reduce
import argparse
import numpy 

parser = argparse.ArgumentParser(description='Set parameters')
parser.add_argument("--k", default=2, type=int)
parser.add_argument("--epochs", default=10000, type=int)
parser.add_argument("--seed", default=1234, type=int)

args = parser.parse_args()
k = args.k
epochs = args.epochs
seed = args.seed
#seed = 1234

dde.config.set_random_seed(seed)

weights = [1000 for i in range(k)]
weights[0] = 1

print(weights)

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
N = 1000
N_test = 4000
lambd = 1
kappa = 0

#geom = dde.geometry.Interval(0, np.pi)
geom = dde.geometry.Interval(0, 1)
Geom = GeometryXGeometry(geom, k)

def func(x):
    u_list = [-1/6 * x[:, i:i+1] * (x[:, i:i+1] ** 2 - 1) for i in range(k)]
    u = reduce(lambda x1, x2: x1 * x2, u_list)
    c = np.exp(1/2 * k ** 2 * sigma ** 2)
    
    res_list = k * [None]
    for i in range(k-1):
        res_list[i] = 0 * u
    res_list[-1] = c * u
    return np.hstack(res_list)

def hatu(x):
    return -1/6 * x * (x ** 2 - 1)

def pde(x, y):
    V0, V1 = y[:, 0:1], y[:, 1:2]
    dV0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dV1_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    
    c = tf.exp(1/2 * k ** 2 * sigma ** 2)         
    #F_list = [c * x[:, 0:1] - lambd * u ** 3 - kappa ** 2 * u]
    xx = x[:, 0:1]
    yy = x[:, 1:2]
    
    vv = hatu(xx)
    ww = hatu(yy)
    
     
    F = c * xx * yy + 3 * c ** 3* (-xx * vv + (-1/2 * xx ** 2 + 1/6)**2) * ww **3 - c ** 3 * vv ** 3 * yy ** 3 + c ** 6 * vv ** 3 * ww ** 3    
    L0 =  -dV0_xx - lambd * V0 ** 3 - F
    L1 = -dV1_yy - lambd * V1 ** 3 - V0
    return [L0, L1]

data = dde.data.PDE(Geom, pde, [], train_distribution="pseudo", num_domain=N, solution=func, num_test=N_test)


layer_size = [k] + [50] * 4 + [k]
activation = 'tanh'
initializer = 'Glorot uniform'
net = dde.nn.FNN(layer_size, activation, initializer)

def output_transform(x, y):
    transf_list = [x[:, i:i+1] * (x[:, i:i+1] - 1) for i in range(k)]
    transf = reduce(lambda x1, x2: x1 * x2, transf_list)
    y_list = [y[:, i:i+1] for i in range(k)]
    ys_list = k * [None]
    for i in range(k-1):
        ys_list[i] = y_list[i]
    ys_list[-1] = transf * y_list[-1]
    ys = tf.concat(ys_list, axis=1)
    return ys

net.apply_output_transform(output_transform)

def l2_error(y_true, y_pred):
    pred = y_true - y_pred
    return np.linalg.norm(pred[:, k-1:k])/np.linalg.norm(y_true[:, k-1:k]) 


model = dde.Model(data, net)
model.compile("adam", lr=1e-4, metrics=[l2_error])
#model.compile("adam", lr=1e-4, decay = ("inverse time", 5000, 0.5), metrics=["l2 relative error"])

losshistory, train_state = model.train(epochs=epochs)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
 
result = np.sum(
            [np.prod(v.get_shape().as_list()) for v in model.net.trainable_variables]
        )
print('Theta:= ', result)

x = np.array([np.arange(101)/100]).T
x_list = [x.T for i in range(k)]
xx = np.concatenate(x_list).T
sol = func(xx)[:, k-1:k]
pred = model.predict(xx)[:, k-1:k]

name = 'results/mo_diag_pred' + str(k) 
test = np.save(name,pred)

name = 'results/mo_diag_exact' + str(k) 
test = np.save(name,sol)

name = 'results/mo_diag_x' + str(k) 
test = np.save(name,xx[:, 0:1])


#print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
