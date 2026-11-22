import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize


def sampler(dom, k, n, soft, Cn=None, power=0.0, pool=None, candidates=16, grid=0):
    def tuples(key, m):
        return dom.interior(key, m * k).reshape(m, k, dom.d)

    def sample(key):
        kx, kb, kj, ky, ks = jax.random.split(key, 5)
        if pool is not None:
            peak = pool[jax.random.randint(ks, (n // 2,), 0, len(pool))]
            batch = {"x": jnp.concatenate([tuples(kx, n - n // 2), peak]),
                     "w": jnp.concatenate([jnp.ones(n - n // 2), 1 / jnp.abs(jax.vmap(Cn)(peak))])}
        elif power:
            cand = tuples(kx, candidates * n)
            a = jnp.abs(jax.vmap(Cn)(cand)) ** power
            prob = 0.5 / len(cand) + 0.5 * a / a.sum()
            pick = jax.random.choice(ks, len(cand), (n,), p=prob)
            batch = {"x": cand[pick], "w": jnp.sqrt(1 / (len(cand) * prob[pick]))}
        else:
            batch = {"x": tuples(kx, n), "w": jnp.ones(n)}
        if grid:
            batch = {"g": dom.interior(kx, grid)}
        if soft:
            nb = max(n // 4, 1)
            b = dom.boundary(kb, nb)
            jb = jax.random.randint(kj, (nb,), 0, k)
            batch.update(b=b, jb=jb, xb=jnp.where((jnp.arange(k) == jb[:, None])[:, :, None], b[:, None, :], tuples(ky, nb)))
        return batch

    return sample


def moment_pool(key, dom, Cn, k, size, chains=4096, burn=1500, thin=10):
    logp = jax.vmap(lambda x: 2 * jnp.log(jnp.abs(Cn(x)) + 1e-300))
    inside = jax.vmap(lambda x: jnp.all(jax.vmap(dom.inside)(x)))

    @partial(jax.jit, static_argnums=3)
    def sweep(X, key, sigma, steps):
        def body(state, sub):
            X, L, acc = state
            ka, kb = jax.random.split(sub)
            Y = X + sigma * jax.random.normal(ka, X.shape)
            LY = jnp.where(inside(Y), logp(Y), -jnp.inf)
            take = jnp.log(jax.random.uniform(kb, L.shape)) < LY - L
            return (jnp.where(take[:, None, None], Y, X), jnp.where(take, LY, L), acc + take.mean()), None

        (X, _, acc), _ = jax.lax.scan(body, (X, logp(X), 0.0), jax.random.split(key, steps))
        return X, acc / steps

    X = dom.interior(key, chains * k).reshape(chains, k, dom.d)
    sigma = 0.3 / np.sqrt(k * dom.d)
    for i in range(burn // 50):
        X, acc = sweep(X, jax.random.fold_in(key, i), sigma, 50)
        sigma = sigma * (1.25 if acc > 0.3 else 0.8)
    out = []
    for i in range(size // chains):
        X, acc = sweep(X, jax.random.fold_in(key, 10**6 + i), sigma, thin)
        out.append(X)
    return jnp.concatenate(out), float(acc), float(sigma)


def terms_of(blocks):
    return lambda params, batch: jnp.stack([jnp.mean(b**2) for b in blocks(params, batch)])


def levenberg(params, blocks, batch, w, iters):
    flat, unravel = ravel_pytree(params)

    def res(z):
        return jnp.concatenate([jnp.sqrt(wi / b.size) * b.ravel() for wi, b in zip(w, blocks(unravel(z), batch))])

    @jax.jit
    def system(z):
        J = jax.lax.map(lambda e: jax.jvp(res, (z,), (e,))[1], jnp.eye(z.size), batch_size=256).T
        r = res(z)
        return J.T @ J, J.T @ r, r @ r

    @jax.jit
    def trial(z, H, g, mu):
        d = jnp.diag(H)
        step = jnp.linalg.solve(H + mu * jnp.diag(d + 1e-12 * d.max()), -g)
        r = res(z + step)
        return z + step, r @ r

    mu, hist = 1e-2, []
    H, g, loss = system(flat)
    for it in range(iters):
        new, trial_loss = trial(flat, H, g, mu)
        if trial_loss < loss:
            flat, mu = new, max(mu / 3, 1e-12)
            H, g, loss = system(flat)
        else:
            mu = mu * 4
            if mu > 1e8:
                break
        hist.append(float(loss))
    return unravel(flat), hist


def train(key, params, blocks, sample, cfg, w0, terms=None, final=None):
    terms = terms or terms_of(blocks)
    w = jnp.asarray(w0, float)
    lr = cfg.lr if cfg.const_lr else optax.cosine_decay_schedule(cfg.lr, max(cfg.adam, 1), 0.01)
    opt = optax.adam(lr)
    state = opt.init(params)
    fixed = (sample if cfg.fixed or final is None else final)(jax.random.fold_in(key, 2**31 - 1))

    def step(carry, it):
        params, state, w = carry
        batch = fixed if cfg.fixed else sample(jax.random.fold_in(key, it))
        loss, g = jax.value_and_grad(lambda p: w @ terms(p, batch))(params)
        upd, state = opt.update(g, state)
        return (optax.apply_updates(params, upd), state, w), loss

    @jax.jit
    def chunk(params, state, w, start):
        (params, state, _), losses = jax.lax.scan(step, (params, state, w), start + jnp.arange(100))
        return params, state, losses[-1]

    @jax.jit
    def balance(params, w, key):
        J = jax.jacrev(lambda p: terms(p, sample(key)))(params)
        sq = sum(jnp.sum(leaf.reshape(len(w), -1) ** 2, 1) for leaf in jax.tree_util.tree_leaves(J))
        norms = jnp.sqrt(sq)
        return 0.9 * w + 0.1 * norms.sum() / norms

    hist, times = [], {"compile": 0.0}
    t0 = time.perf_counter()
    for it in range(0, cfg.adam, 100):
        if cfg.balance and len(w) > 1:
            w = balance(params, w, jax.random.fold_in(key, it))
        params, state, loss = chunk(params, state, w, it)
        if it == 0:
            loss.block_until_ready()
            times["compile"] = time.perf_counter() - t0
            t0 = time.perf_counter()
        if it % max(cfg.adam // 20, 100) == 0:
            hist.append([it, float(loss)])
    jax.block_until_ready(params)
    times["adam"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cfg.lbfgs:
        flat, unravel = ravel_pytree(params)
        vg = jax.jit(jax.value_and_grad(lambda z: w @ terms(unravel(z), fixed)))

        def fun(z):
            v, g = vg(jnp.asarray(z))
            return float(v), np.asarray(g)

        res = minimize(fun, np.asarray(flat), jac=True, method="L-BFGS-B",
                       options={"maxiter": cfg.lbfgs, "maxfun": 2 * cfg.lbfgs, "maxcor": 50, "ftol": 0.0, "gtol": 1e-14})
        params = unravel(jnp.asarray(res.x))
        hist.append([cfg.adam + int(res.nit), float(res.fun)])
    times["lbfgs"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cfg.lm:
        params, h = levenberg(params, blocks, fixed, w, cfg.lm)
        hist.append([cfg.adam + cfg.lbfgs + len(h), h[-1] if h else float("nan")])
    times["lm"] = time.perf_counter() - t0
    return params, np.asarray(w), hist, times
