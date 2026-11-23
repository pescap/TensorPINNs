import jax
import jax.numpy as jnp
import numpy as np


class Interval:
    d = 1

    def gap(self, c2):
        return float(np.min(np.abs((np.pi * np.arange(1, 60)) ** 2 + c2)))

    def mask(self, x):
        return 4 * x[0] * (1 - x[0])


    def inside(self, x):
        return (x[0] > 0) & (x[0] < 1)

    def interior(self, key, n):
        return jax.random.uniform(key, (n, 1))

    def boundary(self, key, n):
        return jax.random.bernoulli(key, 0.5, (n, 1)) * 1.0


class Square:
    d = 2

    def gap(self, c2):
        j = np.arange(1, 60) ** 2
        return float(np.min(np.abs(np.pi**2 * (j[:, None] + j[None, :]) + c2)))

    def mask(self, x):
        return 16 * x[0] * (1 - x[0]) * x[1] * (1 - x[1])


    def inside(self, x):
        return jnp.all((x > 0) & (x < 1))

    def interior(self, key, n):
        return jax.random.uniform(key, (n, 2))

    def boundary(self, key, n):
        ka, kb, kc = jax.random.split(key, 3)
        t = jax.random.uniform(ka, (n,))
        side = jax.random.bernoulli(kb, 0.5, (n,)) * 1.0
        flip = jax.random.bernoulli(kc, 0.5, (n, 1))
        return jnp.where(flip, jnp.stack([t, side], 1), jnp.stack([side, t], 1))


class Box:
    def __init__(self, d):
        self.d = d

    def gap(self, c2):
        j = np.arange(1, 30) ** 2
        s = j
        for _ in range(self.d - 1):
            s = np.unique(s[:, None] + j)
        return float(np.min(np.abs(np.pi**2 * s + c2)))

    def mask(self, x):
        return jnp.prod(4 * x * (1 - x))

    def inside(self, x):
        return jnp.all((x > 0) & (x < 1))

    def interior(self, key, n):
        return jax.random.uniform(key, (n, self.d))

    def boundary(self, key, n):
        ka, kb, kc = jax.random.split(key, 3)
        x = jax.random.uniform(ka, (n, self.d))
        face = jax.random.randint(kb, (n,), 0, self.d)
        side = jax.random.bernoulli(kc, 0.5, (n,)) * 1.0
        return jnp.where(jnp.arange(self.d) == face[:, None], side[:, None], x)


class SpaceTime:
    d = 2

    def __init__(self, a=0.4, T=1.0):
        self.a, self.T = a, T

    def gap(self, c2):
        return max(self.a * np.pi**2, 1 / self.T)

    def mask(self, x):
        return 4 * x[0] * (1 - x[0]) * x[1] / self.T


    def inside(self, x):
        return (x[0] > 0) & (x[0] < 1) & (x[1] > 0) & (x[1] < self.T)

    def interior(self, key, n):
        return jax.random.uniform(key, (n, 2)) * jnp.array([1.0, self.T])

    def boundary(self, key, n):
        ka, kb, kc = jax.random.split(key, 3)
        u = jax.random.uniform(ka, (n, 2)) * jnp.array([1.0, self.T])
        side = jax.random.bernoulli(kb, 0.5, (n,)) * 1.0
        initial = jax.random.bernoulli(kc, 1 / 3, (n,))
        return jnp.where(initial[:, None], u.at[:, 1].set(0.0), u.at[:, 0].set(side))


class HoledSquare:
    d = 2

    def gap(self, c2):
        return self.gamma + c2

    def __init__(self, r0=0.3, width=0.3, pool=400000):
        from tpinn import fem

        rng = np.random.default_rng(0)
        self.r0, self.width = r0, width
        self.inner = jnp.asarray(fem.sample(rng, pool, r0))
        self.edge = jnp.asarray(fem.boundary(rng, pool // 4, r0))
        self.gamma = fem.lowest_eigenvalue(fem.holed_square(r0, 12, 48))

    def mask(self, x):
        r2 = x[0] ** 2 + x[1] ** 2
        return (1 - x[0] ** 2) * (1 - x[1] ** 2) * jnp.tanh((r2 - self.r0**2) / self.width)

    def inside(self, x):
        return jnp.all(jnp.abs(x) < 1) & (x[0] ** 2 + x[1] ** 2 > self.r0**2)

    def interior(self, key, n):
        return self.inner[jax.random.randint(key, (n,), 0, len(self.inner))]

    def boundary(self, key, n):
        return self.edge[jax.random.randint(key, (n,), 0, len(self.edge))]


def ball_points(rng, n, rmax=1.0):
    x = rng.normal(size=(n, 3))
    return x * (rmax * rng.random((n, 1)) ** (1 / 3)) / np.linalg.norm(x, axis=1, keepdims=True)


class Ball:
    d = 3

    def __init__(self, pool=400000):
        rng = np.random.default_rng(0)
        self.inner = jnp.asarray(ball_points(rng, pool))
        e = rng.normal(size=(pool // 4, 3))
        self.edge = jnp.asarray(e / np.linalg.norm(e, axis=1, keepdims=True))

    def gap(self, c2):
        return float(np.pi**2 + c2)

    def mask(self, x):
        return 1 - jnp.sum(x**2)

    def inside(self, x):
        return jnp.sum(x**2) < 1

    def interior(self, key, n):
        return self.inner[jax.random.randint(key, (n,), 0, len(self.inner))]

    def boundary(self, key, n):
        return self.edge[jax.random.randint(key, (n,), 0, len(self.edge))]
