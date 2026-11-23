import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        (jax.random.normal(q, (m, n)) * jnp.sqrt(2 / (m + n)), jnp.zeros(n))
        for q, m, n in zip(keys, sizes[:-1], sizes[1:])
    ]


def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def value_and_second(f, x, e):
    (p, _), (_, dd) = jax.jvp(lambda y: jax.jvp(f, (y,), (e,)), (x,), (e,))
    return p, dd


def first(f, x, e):
    return jax.jvp(f, (x,), (e,))[1]


def elliptic(c2):
    def A(f, x, dirs):
        out = [value_and_second(f, x, e) for e in dirs]
        return out[0][0], c2 * out[0][0] - sum(dd for _, dd in out)

    return A


def heat(a):
    def A(f, x, dirs):
        p, dd = value_and_second(f, x, dirs[0])
        return p, first(f, x, dirs[1]) - a * dd

    return A


def copy_directions(k, d, i):
    return [jnp.zeros((k, d)).at[i, a].set(1.0) for a in range(d)]


def make_v(k, A, gamma, Cn, dom, cfg):
    amp = gamma**-k if cfg.scale else 1.0
    hard = cfg.bc == "hard"

    def init(key):
        return init_mlp(key, [k * dom.d] + [cfg.width] * cfg.depth + [1])

    def point(params, x):
        out = amp * mlp(params, x.ravel())[0]
        return out * jnp.prod(jax.vmap(dom.mask)(x)) if hard else out

    def residual(params, x):
        f = lambda y: point(params, y)
        for i in range(k):
            def f(y, g=f, dirs=copy_directions(k, dom.d, i)):
                return A(g, y, dirs)[1]
        return f(x) - Cn(x)

    def blocks(params, batch):
        out = [batch["w"] * jax.vmap(lambda x: residual(params, x))(batch["x"])]
        if not hard:
            out.append(jax.vmap(lambda x: point(params, x))(batch["xb"]) / amp)
        return out

    return init, jax.vmap(point, (None, 0)), blocks


def make_mo(k, A, gamma, Cn, dom, cfg):
    amp = gamma ** -(jnp.arange(k) + 1.0) if cfg.scale else jnp.ones(k)
    prev_amp = jnp.concatenate([jnp.ones(1), amp[:-1]])
    hard = cfg.bc == "hard"

    def init(key):
        if cfg.split:
            return [init_mlp(q, [k * dom.d] + [cfg.width] * cfg.depth + [1]) for q in jax.random.split(key, k)]
        return init_mlp(key, [k * dom.d] + [cfg.width] * cfg.depth + [k])

    def fields(params, x):
        out = amp * (jnp.concatenate([mlp(p, x.ravel()) for p in params]) if cfg.split else mlp(params, x.ravel()))
        if not hard:
            return out
        m = jax.vmap(dom.mask)(x)
        return out * (jnp.cumprod(m) if cfg.hardaux else jnp.ones(k).at[-1].set(jnp.prod(m)))

    def residual(params, x):
        v = fields(params, x)
        Av = jnp.stack([A(lambda y, j=j: fields(params, y)[j], x, copy_directions(k, dom.d, j))[1] for j in range(k)])
        prev = jax.lax.stop_gradient(v[:-1]) if cfg.sequential else v[:-1]
        return (Av - jnp.concatenate([Cn(x)[None], prev])) / prev_amp

    def boundary(params, x, j):
        return jnp.where(jnp.arange(k) >= j, fields(params, x) / amp, 0.0)

    def blocks(params, batch):
        out = list(jax.vmap(lambda x: residual(params, x))(batch["x"]).T)
        out[0] = batch["w"] * out[0]
        if not hard:
            out.append(jax.vmap(lambda x, j: boundary(params, x, j))(batch["xb"], batch["jb"]).ravel())
        return out

    return init, lambda params, X: jax.vmap(lambda x: fields(params, x)[-1])(X), blocks


def make_lr(k, A, gamma, Cn, dom, cfg):
    hard = cfg.bc == "hard"
    unit = 1 / gamma
    amp = unit if cfg.scale else 1.0
    dirs = list(jnp.eye(dom.d))
    c0 = 1.0 / cfg.rank

    def init(key):
        net = init_mlp(key, [dom.d] + [cfg.width] * cfg.depth + [cfg.rank])
        return {"net": net, "c": jnp.full(cfg.rank, np.log(np.expm1(c0)) if cfg.cpos else c0)}

    def coef(params):
        return jax.nn.softplus(params["c"]) if cfg.cpos else params["c"]

    def phi(params, t):
        out = amp * mlp(params["net"], t)
        return out * dom.mask(t) if hard else out

    def factors(g, params, X):
        return g(X.reshape(-1, dom.d)).reshape(X.shape[:2] + (cfg.rank,))

    def field(params, X):
        return jnp.prod(factors(jax.vmap(lambda t: phi(params, t)), params, X), 1) @ coef(params)

    def grid_residual(params, G):
        AP = jax.vmap(lambda t: A(lambda y: phi(params, y), t, dirs)[1])(G)
        letters = "abcdefgh"[:k]
        T = jnp.einsum(",".join(f"{l}r" for l in letters) + ",r->" + letters, *[AP] * k, coef(params))
        idx = jnp.stack(jnp.meshgrid(*[jnp.arange(len(G))] * k, indexing="ij"), -1).reshape(-1, k)
        return T.ravel() - jax.vmap(Cn)(G[idx])

    def blocks(params, batch):
        if "g" in batch:
            out = [grid_residual(params, batch["g"])]
        else:
            X = batch["x"]
            AP = factors(jax.vmap(lambda t: A(lambda y: phi(params, y), t, dirs)[1]), params, X)
            out = [batch["w"] * (jnp.prod(AP, 1) @ coef(params) - jax.vmap(Cn)(X))]
        if not hard:
            out.append(field(params, batch["xb"]) / unit**k)
        return out

    return init, field, blocks


MAKERS = {"v": make_v, "mo": make_mo, "lr": make_lr}
