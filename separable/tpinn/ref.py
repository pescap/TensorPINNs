import itertools
from functools import lru_cache
from math import comb

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


def moment_sep(X, xp=np, sigma=1.0):
    k = X.shape[-1]
    return xp.exp(0.5 * k**2 * sigma**2) * xp.prod(X, -1)


def exact_sep(X, sigma=1.0):
    k = X.shape[-1]
    return np.exp(0.5 * k**2 * sigma**2) * np.prod(X * (1 - X**2) / 6, -1)


def moment_logn(X, xp=np, m=8, p=2.0, scale=1.0):
    j = xp.arange(1, m + 1) * 1.0
    s = xp.sin(xp.pi * X[..., None] * j).sum(-2)
    return xp.exp(0.5 * scale * (j**-p * s**2).sum(-1))


def on_grid(C, xs):
    rest = np.stack(np.meshgrid(*xs[1:], indexing="ij"), -1).reshape(-1, len(xs) - 1) if len(xs) > 1 else np.zeros((1, 0))
    rows = [C(np.column_stack([np.full(len(rest), a), rest])) for a in xs[0]]
    return np.stack(rows).reshape([len(x) for x in xs])


def modes(T, Ms):
    for i, M in enumerate(Ms):
        T = np.moveaxis(np.tensordot(M, T, axes=(1, i)), 0, i)
    return T


def cheb(n):
    j = np.arange(n + 1)
    x = np.cos(np.pi * j / n)
    c = np.where((j == 0) | (j == n), 2.0, 1.0) * (-1.0) ** j
    D = np.outer(c, 1 / c) / (x[:, None] - x[None, :] + np.eye(n + 1))
    D -= np.diag(D.sum(1))
    return (x + 1) / 2, 2 * D


def bary(nodes, pts):
    w = (-1.0) ** np.arange(len(nodes))
    w[[0, -1]] *= 0.5
    d = pts[:, None] - nodes[None, :]
    hit = np.isclose(d, 0, atol=1e-14)
    d[hit] = 1.0
    E = w / d
    E /= E.sum(1, keepdims=True)
    E[hit.any(1)] = hit[hit.any(1)]
    return E


class Reference:
    def __init__(self, C, k, c2=0.0, n=48):
        self.k = k
        self.t, D = cheb(n)
        L = c2 * np.eye(n - 1) - (D @ D)[1:-1, 1:-1]
        self.S = modes(on_grid(C, [self.t[1:-1]] * k), [np.linalg.inv(L)] * k)

    def rows(self, pts):
        return bary(self.t, np.asarray(pts, float))[:, 1:-1]

    def grid(self, pts):
        return modes(self.S, [self.rows(pts)] * self.k)

    def points(self, P):
        E = [self.rows(P[:, i]) for i in range(self.k)]
        out = np.empty(len(P))
        for b in range(len(P)):
            T = self.S
            for i in reversed(range(self.k)):
                T = T @ E[i][b]
            out[b] = T
        return out


@lru_cache(None)
def fd(level, c2):
    n = 2**level - 1
    h = 2.0**-level
    L = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2 + c2 * np.eye(n)
    return h * np.arange(1, n + 1), np.linalg.inv(L)


def hat(x, pts):
    grid = np.r_[0.0, x, 1.0]
    eye = np.eye(len(x))
    return np.stack([np.interp(pts, grid, np.r_[0.0, e, 0.0]) for e in eye], 1)


def compositions(total, k):
    for bars in itertools.combinations(range(1, total), k - 1):
        yield tuple(b - a for a, b in zip((0,) + bars, bars + (total,)))


def fd_tensor(C, levels, c2, pts):
    xs, Ms = [], []
    for level in levels:
        x, Li = fd(level, c2)
        xs.append(x)
        Ms.append(hat(x, pts) @ Li)
    return modes(on_grid(C, xs), Ms), int(np.prod([len(x) for x in xs]))


def combination(C, k, c2, level, pts):
    out, dofs = 0.0, 0
    for q in range(k):
        for levels in compositions(level + k - 1 - q, k):
            T, n = fd_tensor(C, levels, c2, pts)
            out = out + (-1) ** q * comb(k - 1, q) * T
            dofs += n
    return out, dofs


def kernel(X, xp=np, rhs="gauss", kappa=2 * np.pi, lam=0.05):
    if rhs == "sep":
        return kappa**4 * xp.prod(xp.sin(kappa * X), (-2, -1))
    r = xp.abs(X[..., 0, 0] - X[..., 1, 0])
    return kappa**4 * xp.exp(-(r**2 if rhs == "gauss" else r) / (2 * lam))


def exact_helmholtz(X, kappa=2 * np.pi):
    return np.prod(np.sin(kappa * X), (-2, -1))


def square_second_moment(C, c2, n, pts):
    t, D = cheb(n)
    D2 = (D @ D)[1:-1, 1:-1]
    I = np.eye(n - 1)
    Li = np.linalg.inv(c2 * np.eye((n - 1) ** 2) - np.kron(D2, I) - np.kron(I, D2))
    g = np.stack(np.meshgrid(t[1:-1], t[1:-1], indexing="ij"), -1).reshape(-1, 2)
    G = C(np.stack([np.repeat(g, len(g), 0), np.tile(g, (len(g), 1))], 1)).reshape(len(g), len(g))
    bx, by = bary(t, pts[:, 0])[:, 1:-1], bary(t, pts[:, 1])[:, 1:-1]
    W = (bx[:, :, None] * by[:, None, :]).reshape(len(pts), -1) @ Li
    return W @ G @ W.T


def sine_kernel(K, nm, nq=4000):
    xq = (np.arange(nq) + 0.5) / nq
    Phi = np.sqrt(2) * np.sin(np.pi * xq[:, None] * np.arange(1, nm + 1))
    return Phi.T @ K(xq[:, None], xq[None, :]) @ Phi / nq**2


def square_kernel_moment(rhs, kappa, lam, c2, nm, pts):
    j = np.arange(1, nm + 1)
    Kh = sine_kernel(lambda x, y: kappa**4 * np.exp(-(np.abs(x - y) ** 2 if rhs == "gauss" else np.abs(x - y)) / (2 * lam)), nm)
    s = np.sqrt(2) * (1 - np.cos(j * np.pi)) / (j * np.pi)
    eig = np.pi**2 * (j[:, None] ** 2 + j[None, :] ** 2) + c2
    sx = np.sqrt(2) * np.sin(np.pi * pts[:, :1] * j)
    sy = np.sqrt(2) * np.sin(np.pi * pts[:, 1:] * j)
    U = sx * (sy @ (s / eig).T)
    return U @ Kh @ U.T


def iso_kernel(X, xp=np, lam=0.05, rhs="gauss"):
    r = X[..., 0, :] - X[..., 1, :]
    return xp.exp(-(r**2 if rhs == "gauss" else xp.abs(r)).sum(-1) / (2 * lam))


def kernel_1d(rhs, lam):
    return lambda x, y: np.exp(-((x - y) ** 2 if rhs == "gauss" else np.abs(x - y)) / (2 * lam))


def box_kernel_moment(d, c2, lam, n, pts):
    t, D = cheb(n)
    ev, P = np.linalg.eig((D @ D)[1:-1, 1:-1])
    ev, P = ev.real, P.real
    Pi = np.linalg.inv(P)
    K1 = np.exp(-(t[1:-1, None] - t[None, 1:-1]) ** 2 / (2 * lam))
    B = [bary(t, pts[:, i])[:, 1:-1] @ P for i in range(d)]
    U, lam_sum = B[0], ev
    for i in range(1, d):
        U = U[..., None] * B[i].reshape((len(pts),) + (1,) * i + (-1,))
        lam_sum = lam_sum[..., None] + ev
    U = U / (c2 - lam_sum)
    for _ in range(d):
        U = np.tensordot(U, Pi, axes=([1], [0]))
    V = U
    for _ in range(d):
        V = np.tensordot(V, K1, axes=([1], [0]))
    return U.reshape(len(pts), -1) @ V.reshape(len(pts), -1).T


def box_highd(d, lam, X, c2=0.0, N=128, hs=0.4, chunk=200, rhs="gauss"):
    G = sine_kernel(kernel_1d(rhs, lam), N)
    j = np.arange(1, N + 1)
    s = np.arange(-np.log(d * np.pi**2 * N**2) - 25, np.log(40 / (d * np.pi**2 + c2)), hs)
    t = np.exp(s)
    w = hs * t * np.exp(-t * c2)
    E = np.exp(-np.pi**2 * t[:, None] * j**2)
    out = []
    for c in range(0, len(X), chunk):
        x, y = X[c:c + chunk, 0], X[c:c + chunk, 1]
        acc = np.ones((len(x), len(t), len(t)))
        for i in range(d):
            Ax = np.sqrt(2) * np.sin(np.pi * x[:, i, None] * j)[:, None, :] * E
            Ay = np.sqrt(2) * np.sin(np.pi * y[:, i, None] * j)[:, None, :] * E
            acc *= np.einsum("pqn,nm,pkm->pqk", Ax, G, Ay, optimize=True)
        out.append(np.einsum("q,pqk,k->p", w, acc, w))
    return np.concatenate(out)


def heat_kernel(X, xp=np, rhs="gauss", sigma=10.0, lam=0.05, rate=2.0):
    r = xp.abs(X[..., 0, 0] - X[..., 1, 0])
    return sigma**2 * xp.exp(-(r**2 if rhs == "gauss" else r) / (2 * lam) - rate * (X[..., 0, 1] + X[..., 1, 1]))


def heat_kernel_moment(rhs, sigma, lam, rate, a, nm, pts):
    j = np.arange(1, nm + 1)
    Kh = sine_kernel(lambda x, y: sigma**2 * np.exp(-(np.abs(x - y) ** 2 if rhs == "gauss" else np.abs(x - y)) / (2 * lam)), nm)
    mu = a * (np.pi * j) ** 2
    t = pts[:, 1:]
    U = np.sqrt(2) * np.sin(np.pi * pts[:, :1] * j) * (np.exp(-rate * t) - np.exp(-mu * t)) / (mu - rate)
    return U @ Kh @ U.T


def lam(m, p, amp):
    return amp * (np.arange(1, m + 1) * 1.0) ** -p


def hermite(k, m, p, amp):
    b = k * np.sqrt(lam(m, p, amp))
    rules = [hermegauss(int(6 + 2.5 * v**2)) for v in b]
    xi = np.stack(np.meshgrid(*[r[0] for r in rules], indexing="ij"), -1).reshape(-1, m)
    w = np.prod(np.stack(np.meshgrid(*[r[1] for r in rules], indexing="ij"), -1).reshape(-1, m), 1)
    return xi, w / w.sum()


def lognormal_tuples(k, c2, m, p, amp, pts, idx, chunk=4000):
    t, D = cheb(64)
    M = bary(t, pts)[:, 1:-1] @ np.linalg.inv(c2 * np.eye(63) - (D @ D)[1:-1, 1:-1])
    j = np.arange(1, m + 1)
    B = np.sin(np.pi * t[1:-1, None] * j) * np.sqrt(lam(m, p, amp))
    E = np.sin(np.pi * pts[:, None] * j) * np.sqrt(lam(m, p, amp))
    xi, w = hermite(k, m, p, amp)
    out, check = np.zeros(len(idx)), np.zeros(len(idx))
    for q in range(0, len(w), chunk):
        V, F = M @ np.exp(B @ xi[q:q + chunk].T), np.exp(E @ xi[q:q + chunk].T)
        a, b = w[q:q + chunk] * 1.0, w[q:q + chunk] * 1.0
        for i in range(k):
            a, b = a * V[idx[:, i]], b * F[idx[:, i]]
        out, check = out + a.sum(1), check + b.sum(1)
    exact = moment_logn(pts[idx], np, m, p, amp)
    return out, float(np.linalg.norm(check - exact) / np.linalg.norm(exact))


BALL_FREQ = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 0, 0)], float)
BALL_PHASE = np.array([0.3, -0.2, 0.1])


def ball_parts(X, xp=np, m=4, p=2.0, amp=1.0):
    W = np.pi / 2 * BALL_FREQ[:m]
    arg = X[..., None, :] * W + BALL_PHASE * BALL_FREQ[:m]
    c, s = xp.cos(arg), xp.sin(arg)
    psi = c[..., 0] * c[..., 1] * c[..., 2]
    grad = -W * s * xp.stack([c[..., 1] * c[..., 2], c[..., 0] * c[..., 2], c[..., 0] * c[..., 1]], -1)
    a = np.sqrt(lam(m, p, amp))
    return a * psi, a[:, None] * grad, -a * (W**2).sum(-1) * psi


def ball_exact(X, xp=np, m=4, p=2.0, amp=1.0):
    h = ball_parts(X, xp, m, p, amp)[0]
    return xp.prod(1 - (X**2).sum(-1), -1) * xp.exp(0.5 * (h.sum(-2) ** 2).sum(-1))


def ball_moment(X, xp=np, m=4, p=2.0, amp=1.0, c2=0.0):
    k = X.shape[-2]
    if xp is np and X.ndim > 2 and X.shape[0] > 5000:
        return np.concatenate([ball_moment(X[i:i + 5000], np, m, p, amp, c2) for i in range(0, len(X), 5000)])
    g, wg = hermegauss(k + 1)
    xi = np.stack(np.meshgrid(*[g] * m, indexing="ij"), -1).reshape(-1, m)
    w = np.prod(np.stack(np.meshgrid(*[wg] * m, indexing="ij"), -1).reshape(-1, m), 1)
    w = w / w.sum()
    h, G, L = ball_parts(X, xp, m, p, amp)
    b = 1 - (X**2).sum(-1)
    H = h.sum(-2)
    Z = xi + H[..., None, :]
    out = 1.0
    for i in range(k):
        beta = 4 * (X[..., i, None, :] * G[..., i, :, :]).sum(-1) - b[..., i, None] * L[..., i, :]
        lin = (Z * beta[..., None, :]).sum(-1)
        quad = (xp.einsum("...mc,...qm->...qc", G[..., i, :, :], Z) ** 2).sum(-1)
        out = out * (6 + c2 * b[..., i, None] + lin - b[..., i, None] * quad)
    return xp.exp(0.5 * (H**2).sum(-1)) * (out * w).sum(-1)
