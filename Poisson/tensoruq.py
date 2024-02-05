"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""

import deepxde as dde
import numpy as np
from dde.backend import tf
from functools import reduce
import argparse

dde.config.disable_xla_jit()
parser = argparse.ArgumentParser(description="Set parameters")

k = 2
epochs = 10000
seed = 1234

dde.config.set_random_seed(seed)


class GeometryXGeometry:
    def __init__(self, geometry, k):
        self.geometry = geometry
        self.dim = geometry.dim**k
        self.k = k

    def random_points(self, n, random="pseudo"):
        xx = []
        for i in range(self.k):
            xx.append(self.geometry.random_points(n, random=random))
        return np.hstack(xx)

    def uniform_points(self, n, boundary=False):
        print("no uniform points")
        return self.random_points(n)


sigma = 1
N = 4000
N_test = 1000

geom = dde.geometry.Interval(0, 1)
Geom = GeometryXGeometry(geom, k)


def pde(x, y):
    dy1 = -dde.grad.hessian(y, x, i=0, j=0)
    for ii in range(1, k):
        dy1 = -dde.grad.hessian(dy1, x, i=ii, j=ii)

    F_list = [x[:, i : i + 1] for i in range(k)]
    c = tf.exp(1 / 2 * k**2 * sigma**2)

    A = dy1
    C = reduce(lambda x1, x2: x1 * x2, F_list)
    return A - c * C


def func(x):
    u_list = [-1 / 6 * x[:, i : i + 1] * (x[:, i : i + 1] ** 2 - 1) for i in range(k)]
    u = reduce(lambda x1, x2: x1 * x2, u_list)
    c = np.exp(1 / 2 * k**2 * sigma**2)
    return c * u


data = dde.data.PDE(
    Geom,
    pde,
    [],
    train_distribution="pseudo",
    num_domain=N,
    solution=func,
    num_test=N_test,
)


layer_size = [k] + [50] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)


def output_transform(x, y):
    transf_list = [x[:, i : i + 1] * (x[:, i : i + 1] - 1) for i in range(k)]
    transf = reduce(lambda x1, x2: x1 * x2, transf_list)
    return transf * y


net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=epochs)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
