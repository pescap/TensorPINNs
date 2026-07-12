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
        self.dim = geometry.dim * k
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
#N= 4000 * k
#N_test = 1000 * k

geom = dde.geometry.Interval(0, 1)
Geom = GeometryXGeometry(geom, k)


def pde(x, y):
    y_list = [y[:, i:i+1] for i in range(k)]
    if k>1:
        dy_list = [dde.grad.hessian(y, x, component=i, i=i, j=i) for i in range(k)]
    else:
        dy_list = [dde.grad.hessian(y, x, i=i, j=i) for i in range(k)]
    F_list = [x[:, i:i+1] for i in range(k)]
    c = tf.exp(1/2 * k ** 2 * sigma ** 2) 
    C = reduce(lambda x1, x2: x1 * x2, F_list)
    
    loss_list = k * [None]    
    loss_list[0] = - dy_list[0] - c * C
    for i in range(1, k):
        loss_list[i] = - dy_list[i] - y_list[i-1]
    return loss_list


def func(x):
    u_list = [-1/6 * x[:, i:i+1] * (x[:, i:i+1] ** 2 - 1) for i in range(k)]
    u = reduce(lambda x1, x2: x1 * x2, u_list)
    c = np.exp(1/2 * k ** 2 * sigma ** 2)
    
    res_list = k * [None]
    
    for i in range(k-1):
        res_list[i] = 0 * u
    res_list[-1] = c * u
    return np.hstack(res_list)



data = dde.data.PDE(Geom, pde, [], train_distribution="pseudo", num_domain=N, solution=func, num_test=N_test)

layer_size = [k]+ [50] * 4 + [k]

activation = 'tanh'
initializer = 'Glorot uniform'
net = dde.nn.FNN(layer_size, activation, initializer)

def  output_transform(x, y):
    transf_list = [x[:, i:i+1] * (x[:, i:i+1] - 1) for i in range(k)]
    transf = reduce(lambda x1, x2: x1 * x2, transf_list)
    y_list = [y[:, i:i+1] for i in range(k)]
    
    ys_list = k * [None]
    for i in range(k-1):
        ys_list[i] = y_list[i]
    ys_list[-1] = transf * y_list[-1]
    ys = tf.concat(ys_list, axis=1)
    return ys

def l2_error(y_true, y_pred):
    pred = y_true - y_pred
    print('---')
    print(np.linalg.norm(pred, axis=0))
    print('---')
    return np.linalg.norm(pred[:, k-1:k])/np.linalg.norm(y_true[:, k-1:k]) 

net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=1e-4, metrics=[l2_error], loss_weights=weights)

losshistory, train_state = model.train(epochs=epochs)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
 
print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in __import__('tensorflow').compat.v1.trainable_variables()]))

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


#print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in __import__('tensorflow').compat.v1.trainable_variables()]))
