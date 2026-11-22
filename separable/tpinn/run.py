import argparse
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tpinn import geom, ref
from tpinn.models import MAKERS, elliptic, heat as heat_operator
from tpinn.train import moment_pool, sampler, terms_of, train

NT = {1: 1000, 2: 200, 3: 64, 4: 32, 5: 16, 6: 10}
NCHEB = {1: 48, 2: 48, 3: 48, 4: 48, 5: 40, 6: 28}


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="sep", choices=["sep", "logn", "hole", "helm", "heat", "box", "ball"])
    ap.add_argument("--rhs", default="gauss", choices=["sep", "gauss", "exp"])
    ap.add_argument("--kappa", default=2 * np.pi, type=float)
    ap.add_argument("--corr", default=0.05, type=float)
    ap.add_argument("--arch", default="lr", choices=list(MAKERS))
    ap.add_argument("--k", default=2, type=int)
    ap.add_argument("--c2", default=0.0, type=float)
    ap.add_argument("--dim", default=2, type=int)
    ap.add_argument("--modes", default=8, type=int)
    ap.add_argument("--decay", default=2.0, type=float)
    ap.add_argument("--amp", default=1.0, type=float)
    ap.add_argument("--cutoff", default=0.3, type=float)
    ap.add_argument("--mesh", default=32, type=int)
    ap.add_argument("--width", default=64, type=int)
    ap.add_argument("--depth", default=4, type=int)
    ap.add_argument("--rank", default=16, type=int)
    ap.add_argument("--bc", default="hard", choices=["hard", "soft"])
    ap.add_argument("--scale", default=1, type=int)
    ap.add_argument("--norm", default=1, type=int)
    ap.add_argument("--hardaux", default=1, type=int)
    ap.add_argument("--split", default=0, type=int)
    ap.add_argument("--sequential", default=0, type=int)
    ap.add_argument("--weights", default="equal", choices=["equal", "legacy"])
    ap.add_argument("--wbc", default=1.0, type=float)
    ap.add_argument("--bctrace", default=0, type=int)
    ap.add_argument("--cpos", default=0, type=int)
    ap.add_argument("--grid", default=0, type=int)
    ap.add_argument("--maskrms", default=0, type=int)
    ap.add_argument("--balance", default=0, type=int)
    ap.add_argument("--n", default=4000, type=int)
    ap.add_argument("--fixed", default=0, type=int)
    ap.add_argument("--nl", default=0, type=int)
    ap.add_argument("--adam", default=20000, type=int)
    ap.add_argument("--lbfgs", default=5000, type=int)
    ap.add_argument("--lm", default=0, type=int)
    ap.add_argument("--importance", default=0.0, type=float)
    ap.add_argument("--peak", default=0, type=int)
    ap.add_argument("--tuples", default=0, type=int)
    ap.add_argument("--pool", default=409600, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--const-lr", dest="const_lr", default=0, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.environ.get("TPINN_OUT", "out"))
    return ap.parse_args(argv)


def cached(cfg, parts, compute):
    key = hashlib.md5(json.dumps(parts).encode()).hexdigest()
    path = Path(os.environ.get("TPINN_CACHE", Path(cfg.out) / "cache")) / f"ref_{key}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.npz")
        np.savez(tmp, **compute())
        os.replace(tmp, path)
    return dict(np.load(path))


def interval(cfg):
    k = cfg.k
    if cfg.problem == "sep":
        C = lambda xp: partial(ref.moment_sep, xp=xp)
    else:
        C = lambda xp: partial(ref.moment_logn, xp=xp, m=cfg.modes, p=cfg.decay, scale=cfg.amp)
    if cfg.problem == "logn" and (k > 6 or cfg.tuples):
        return interval_tuples(cfg)
    pts = (np.arange(NT[k]) + 0.5) / NT[k]
    X = np.stack(np.meshgrid(*[pts] * k, indexing="ij"), -1).reshape(-1, k)
    D = np.tile(np.linspace(0, 1, 203)[1:-1, None], (1, k))

    def truth():
        if cfg.problem == "sep":
            return {"full": ref.exact_sep(X), "diag": ref.exact_sep(D)}
        R = ref.Reference(C(np), k, cfg.c2, NCHEB[k])
        return {"full": R.grid(pts).ravel(), "diag": R.points(D)}

    parts = [cfg.problem, k, cfg.c2, cfg.modes, cfg.decay, cfg.amp, NT[k], NCHEB[k]]
    return geom.Interval(), lambda x: C(jnp)(x[:, 0]), lambda P: C(np)(P[..., 0]), X[..., None], D[..., None], lambda: cached(cfg, parts, truth)


def interval_tuples(cfg):
    k = cfg.k
    C = lambda xp: partial(ref.moment_logn, xp=xp, m=cfg.modes, p=cfg.decay, scale=cfg.amp)
    rng = np.random.default_rng(0)
    pts = rng.random(500)
    idx = np.concatenate([rng.integers(0, 500, (5000, k)), np.tile(np.arange(500)[:, None], (1, k))])

    def truth():
        val, quad = ref.lognormal_tuples(k, cfg.c2, cfg.modes, cfg.decay, cfg.amp, pts, idx)
        return {"full": val[:5000], "diag": val[5000:], "quad": quad}

    parts = ["tuples", k, cfg.c2, cfg.modes, cfg.decay, cfg.amp]
    return (geom.Interval(), lambda x: C(jnp)(x[:, 0]), lambda P: C(np)(P[..., 0]),
            pts[idx[:5000]][..., None], pts[idx[5000:]][..., None], lambda: cached(cfg, parts, truth))


def hole(cfg):
    from tpinn import fem

    k = cfg.k
    kw = dict(m=cfg.modes, p=cfg.decay, amp=cfg.amp)
    rng = np.random.default_rng(0)
    pts = fem.sample(rng, 500)
    idx = rng.integers(0, 500, (20000, k))
    diag = np.tile(np.arange(500)[:, None], (1, k))

    def truth():
        out = {}
        for name, n in (("coarse", cfg.mesh // 2), ("fine", cfg.mesh)):
            col = fem.Collocation(fem.holed_square(0.3, n, 4 * n), k, pts, cfg.c2, **kw)
            out[name] = col.tuples(idx)
        return {"full": out["fine"], "diag": col.tuples(diag), "coarse": out["coarse"], "quad": col.quadrature_error(idx)}

    parts = ["hole", k, cfg.c2, cfg.modes, cfg.decay, cfg.amp, cfg.mesh]
    return (geom.HoledSquare(width=cfg.cutoff), lambda x: fem.moment(x, jnp, **kw), lambda P: fem.moment(P, np, **kw),
            pts[idx], pts[diag], lambda: cached(cfg, parts, truth))


def helm(cfg):
    assert cfg.k == 2
    kw = dict(rhs=cfg.rhs, kappa=cfg.kappa, lam=cfg.corr)
    rng = np.random.default_rng(0)
    pts = rng.random((500, 2))
    idx = rng.integers(0, 500, (20000, 2))
    diag = np.tile(np.arange(500)[:, None], (1, 2))
    cfg.c2 = -cfg.kappa**2

    def truth():
        if cfg.rhs == "sep":
            return {"full": ref.exact_helmholtz(pts[idx], cfg.kappa), "diag": ref.exact_helmholtz(pts[diag], cfg.kappa)}
        S = {n: ref.square_kernel_moment(cfg.rhs, cfg.kappa, cfg.corr, cfg.c2, n, pts) for n in (4 * cfg.mesh, 8 * cfg.mesh)}
        fine = S[8 * cfg.mesh]
        return {"full": fine[idx[:, 0], idx[:, 1]], "diag": np.diag(fine), "coarse": S[4 * cfg.mesh][idx[:, 0], idx[:, 1]]}

    parts = ["helm", cfg.rhs, cfg.kappa, cfg.corr, cfg.mesh]
    return (geom.Square(), lambda x: ref.kernel(x, jnp, **kw), lambda P: ref.kernel(P, np, **kw),
            pts[idx], pts[diag], lambda: cached(cfg, parts, truth))


def box(cfg):
    assert cfg.k == 2
    rng = np.random.default_rng(0)
    pts = rng.random((500, cfg.dim))
    idx = rng.integers(0, 500, (20000, 2))
    diag = np.tile(np.arange(500)[:, None], (1, 2))

    def truth():
        if cfg.dim > 3 or cfg.rhs == "exp":
            S = {n: ref.box_highd(cfg.dim, cfg.corr, pts[idx], cfg.c2, n, rhs=cfg.rhs) for n in (64, 128)}
            return {"full": S[128], "diag": ref.box_highd(cfg.dim, cfg.corr, pts[diag], cfg.c2, rhs=cfg.rhs), "coarse": S[64]}
        S = {n: ref.box_kernel_moment(cfg.dim, cfg.c2, cfg.corr, n, pts) for n in (cfg.mesh, cfg.mesh + 16)}
        fine = S[cfg.mesh + 16]
        return {"full": fine[idx[:, 0], idx[:, 1]], "diag": np.diag(fine), "coarse": S[cfg.mesh][idx[:, 0], idx[:, 1]]}

    parts = ["box", cfg.dim, cfg.c2, cfg.corr, cfg.mesh] + ([cfg.rhs] if cfg.rhs != "gauss" else [])
    return (geom.Box(cfg.dim), lambda x: ref.iso_kernel(x, jnp, cfg.corr, cfg.rhs), lambda P: ref.iso_kernel(P, np, cfg.corr, cfg.rhs),
            pts[idx], pts[diag], lambda: cached(cfg, parts, truth))


def ball(cfg):
    k = cfg.k
    kw = dict(m=cfg.modes, p=cfg.decay, amp=cfg.amp)
    rng = np.random.default_rng(0)
    pts = geom.ball_points(rng, 500, 0.95)
    idx = rng.integers(0, 500, (20000, k))
    diag = np.tile(np.arange(500)[:, None], (1, k))

    def truth():
        return {"full": ref.ball_exact(pts[idx], np, **kw), "diag": ref.ball_exact(pts[diag], np, **kw)}

    parts = ["ball", k, cfg.c2, cfg.modes, cfg.decay, cfg.amp]
    return (geom.Ball(), lambda x: ref.ball_moment(x, jnp, c2=cfg.c2, **kw), lambda P: ref.ball_moment(P, np, c2=cfg.c2, **kw),
            pts[idx], pts[diag], lambda: cached(cfg, parts, truth))


def heat(cfg):
    assert cfg.k == 2
    kw = dict(rhs=cfg.rhs, sigma=10.0, lam=cfg.corr, rate=2.0)
    dom = geom.SpaceTime()
    rng = np.random.default_rng(0)
    pts = rng.random((500, 2))
    idx = rng.integers(0, 500, (20000, 2))
    diag = np.tile(np.arange(500)[:, None], (1, 2))

    def truth():
        S = {n: ref.heat_kernel_moment(cfg.rhs, 10.0, cfg.corr, 2.0, dom.a, n, pts) for n in (4 * cfg.mesh, 8 * cfg.mesh)}
        fine = S[8 * cfg.mesh]
        return {"full": fine[idx[:, 0], idx[:, 1]], "diag": np.diag(fine), "coarse": S[4 * cfg.mesh][idx[:, 0], idx[:, 1]]}

    parts = ["heat", cfg.rhs, cfg.corr, cfg.mesh]
    return (dom, lambda x: ref.heat_kernel(x, jnp, **kw), lambda P: ref.heat_kernel(P, np, **kw),
            pts[idx], pts[diag], lambda: cached(cfg, parts, truth))


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main(argv=None):
    cfg = parse(argv)
    k = cfg.k
    dom, C, Cnp, X, D, truth = {"hole": hole, "helm": helm, "heat": heat, "box": box, "ball": ball}.get(cfg.problem, interval)(cfg)
    soft = cfg.bc == "soft"
    key = jax.random.PRNGKey(cfg.seed)
    probe = np.asarray(sampler(dom, k, 200000, False)(jax.random.PRNGKey(123))["x"])
    S = float(np.sqrt(np.mean(Cnp(probe) ** 2))) if cfg.norm else 1.0
    Cn = lambda x: C(x) / S
    gamma = dom.gap(cfg.c2)
    if cfg.maskrms:
        base = dom.mask
        r = float(np.sqrt(np.mean(np.asarray(jax.vmap(base)(dom.interior(jax.random.PRNGKey(5), 100000))) ** 2)))
        dom.mask = lambda x: base(x) / r
    A = heat_operator(dom.a) if cfg.problem == "heat" else elliptic(cfg.c2)
    init, field, blocks = MAKERS[cfg.arch](k, A, gamma, Cn, dom, cfg)
    terms = terms_of(blocks)
    extra, pool = {}, None
    if cfg.peak:
        t0 = time.perf_counter()
        pool, acc, sigma = moment_pool(jax.random.fold_in(key, 7), dom, Cn, k, cfg.pool)
        extra = {"mcmc_time": time.perf_counter() - t0, "mcmc_accept": acc, "mcmc_sigma": sigma}
    sample = sampler(dom, k, cfg.n, soft, Cn, cfg.importance, pool, grid=cfg.grid)
    final = sampler(dom, k, cfg.nl or cfg.n, soft, Cn, cfg.importance, pool, grid=2 * cfg.grid)
    params = init(key)
    w0 = np.ones(len(terms(params, sample(key))))
    if cfg.arch == "mo" and cfg.weights == "legacy":
        w0[1:k] = 1000.0
    if soft:
        w0[-1] = cfg.wbc
    if cfg.sequential:
        stage = argparse.Namespace(**{**vars(cfg), "adam": cfg.adam // k, "lbfgs": cfg.lbfgs // k})
        hist, times = [], {"compile": 0.0, "adam": 0.0, "lbfgs": 0.0, "lm": 0.0}
        for j in range(k):
            params, w, h, t = train(jax.random.fold_in(key, 1 + j), params, blocks, sample, stage, np.eye(len(w0))[j], terms, final)
            hist += h
            times = {q: times[q] + t[q] for q in times}
    else:
        params, w, hist, times = train(jax.random.fold_in(key, 1), params, blocks, sample, cfg, w0, terms, final)

    f = jax.jit(field)
    pred = S * np.concatenate([np.asarray(f(params, jnp.asarray(X[i:i + 100000]))) for i in range(0, len(X), 100000)])
    pdiag = S * np.asarray(f(params, jnp.asarray(D)))
    t0 = time.perf_counter()
    T = truth()
    ntest = 4000 if cfg.arch == "v" and k * dom.d > 4 else 20000
    test = np.asarray(jax.jit(terms_of(blocks))(params, sampler(dom, k, ntest, soft)(jax.random.PRNGKey(10**6))))
    if cfg.peak and cfg.tuples:
        Z = np.asarray(pool[:500])
        tz, _ = ref.lognormal_tuples(k, cfg.c2, cfg.modes, cfg.decay, cfg.amp, Z[..., 0].ravel(), np.arange(Z.shape[0] * k).reshape(-1, k), chunk=2000)
        wz = 1 / Cnp(Z) ** 2
        extra["err_peak"] = float(np.sqrt(np.sum(wz * (S * np.asarray(f(params, jnp.asarray(Z))) - tz) ** 2) / np.sum(wz * tz**2)))
    if cfg.problem == "ball":
        rng = np.random.default_rng(1)
        W = geom.ball_points(rng, 500)[rng.integers(0, 500, (20000, k))]
        extra["err_whole"] = rel(S * np.asarray(f(params, jnp.asarray(W))), ref.ball_exact(W, np, m=cfg.modes, p=cfg.decay, amp=cfg.amp))
    norm = float(np.sqrt(np.mean(T["full"] ** 2)))
    j = np.arange(k)
    stage = gamma ** -(k - j) * (gamma**-j if cfg.scale else 1.0) if cfg.arch == "mo" else np.array([gamma**-k])
    out = {
        "cfg": vars(cfg), "S": S, "gamma": gamma, "weights": w.tolist(), "times": times, **extra,
        "ref_time": time.perf_counter() - t0, "err_full": rel(pred, T["full"]), "err_diag": rel(pdiag, T["diag"]),
        "ref_err": rel(T["coarse"], T["full"]) if "coarse" in T else None, "ref_quad": float(T["quad"]) if "quad" in T else None,
        "norm_L2": norm, "test_terms": test.tolist(),
        "bound_rel": float(S * np.sum(stage * np.sqrt(test[:len(stage)])) / norm),
        "nparams": int(sum(np.size(p) for p in jax.tree_util.tree_leaves(params))), "hist": hist,
    }
    name = "_".join(str(v) for v in [cfg.problem, cfg.arch, f"k{k}", cfg.bc, cfg.tag or "base", f"s{cfg.seed}"])
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out) / f"{name}.json").write_text(json.dumps(out))
    np.savez(Path(cfg.out) / f"{name}.npz", points=np.asarray(D), pred=pdiag, true=T["diag"])
    print(json.dumps({q: out[q] for q in ["err_full", "err_diag", "ref_err", "times", "test_terms", "bound_rel", "nparams"]}))


if __name__ == "__main__":
    main()
