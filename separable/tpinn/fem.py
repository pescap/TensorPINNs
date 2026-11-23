import numpy as np
from scipy.sparse.linalg import eigsh, splu
from scipy.spatial import cKDTree
from skfem import Basis, BilinearForm, ElementTetP2, ElementTriP2, MeshTet, MeshTet2, MeshTri
from skfem.helpers import dot, grad

from tpinn.ref import ball_parts, hermite, lam


def holed_square(r0=0.3, ns=24, nt=96):
    th = 2 * np.pi * np.arange(nt) / nt
    d = np.stack([np.cos(th), np.sin(th)], 1)
    outer = d / np.abs(d).max(1, keepdims=True)
    s = np.linspace(0, 1, ns + 1)[:, None, None]
    p = ((1 - s) * r0 * d + s * outer).reshape(-1, 2)
    i, j = np.meshgrid(np.arange(ns), np.arange(nt), indexing="ij")
    a, b = i * nt + j, i * nt + (j + 1) % nt
    t = np.concatenate([np.stack([a, a + nt, b + nt], -1), np.stack([a, b + nt, b], -1)]).reshape(-1, 3)
    return MeshTri(p.T.copy(), t.T.copy())


def modes(x, m, xp=np):
    freq = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)][:m]
    return xp.stack([xp.cos(np.pi * a * x[..., 0] / 2 + a * 0.3) * xp.cos(np.pi * b * x[..., 1] / 2 - b * 0.2) for a, b in freq], -1)


def moment(X, xp=np, m=4, p=2.0, amp=1.0):
    s = modes(X, m, xp).sum(-2)
    return xp.exp(0.5 * (lam(m, p, amp) * s**2).sum(-1))


def lowest_eigenvalue(mesh):
    basis = Basis(mesh, ElementTriP2())
    K = BilinearForm(lambda u, v, _: dot(grad(u), grad(v))).assemble(basis)
    M = BilinearForm(lambda u, v, _: u * v).assemble(basis)
    I = basis.complement_dofs(basis.get_dofs())
    return float(eigsh(K[I][:, I].tocsc(), 1, M[I][:, I].tocsc(), sigma=0)[0][0])


def boundary(rng, n, r0=0.3):
    circle = rng.random(n) < 2 * np.pi * r0 / (8 + 2 * np.pi * r0)
    th = rng.uniform(0, 2 * np.pi, n)
    t, side = rng.uniform(-1, 1, n), rng.integers(0, 4, n)
    sq = np.where(side[:, None] < 2, np.stack([t, 2.0 * side - 1], 1), np.stack([2.0 * side - 5, t], 1))
    return np.where(circle[:, None], r0 * np.stack([np.cos(th), np.sin(th)], 1), sq)


class Collocation:
    def __init__(self, mesh, k, pts, c2=0.0, m=4, p=2.0, amp=1.0, chunk=512):
        self.k, self.args, self.pts = k, (m, p, amp), pts
        basis = Basis(mesh, ElementTriP2())
        self.xi, self.w = hermite(k, m, p, amp)
        K = BilinearForm(lambda u, v, _: dot(grad(u), grad(v)) + c2 * u * v).assemble(basis)
        M = BilinearForm(lambda u, v, _: u * v).assemble(basis)
        I = basis.complement_dofs(basis.get_dofs())
        lu = splu(K[I][:, I].tocsc())
        G = modes(basis.doflocs.T, m) * np.sqrt(lam(m, p, amp))
        P = basis.probes(pts.T).tocsr()[:, I]
        self.V = np.hstack([
            P @ lu.solve((M @ np.exp(G @ self.xi[q:q + chunk].T))[I]) for q in range(0, len(self.w), chunk)
        ])

    def tuples(self, idx, V=None):
        out = self.w * 1.0
        for j in range(self.k):
            out = out * (self.V if V is None else V)[idx[:, j]]
        return out.sum(1)

    def quadrature_error(self, idx):
        m, p, amp = self.args
        E = np.exp(modes(self.pts, m) * np.sqrt(lam(m, p, amp)) @ self.xi.T)
        exact = moment(self.pts[idx], np, m, p, amp)
        return float(np.linalg.norm(self.tuples(idx, E) - exact) / np.linalg.norm(exact))


def sample(rng, n, r0=0.3):
    out = np.empty((0, 2))
    while len(out) < n:
        q = rng.uniform(-1, 1, (2 * n, 2))
        out = np.vstack([out, q[(q**2).sum(1) > r0**2]])
    return out[:n]


def curved_ball(nrefs):
    m1 = MeshTet.init_ball(nrefs)
    m2 = MeshTet2.from_mesh(m1)
    f = m1.facets[:, m1.boundary_facets()]
    e = np.unique(np.sort(np.concatenate([f[[0, 1]], f[[1, 2]], f[[0, 2]]], 1), 0), axis=1)
    idx = cKDTree(m2.p.T).query(((m1.p[:, e[0]] + m1.p[:, e[1]]) / 2).T)[1]
    p = m2.p.copy()
    p[:, idx] /= np.linalg.norm(p[:, idx], axis=0)
    return m1, MeshTet2(p, m2.t)


def ball_solutions(nrefs, pts, xi, c2=0.0, m=4, p=2.0, amp=1.0, chunk=256, curved=False):
    m1, m2 = curved_ball(nrefs)
    basis = Basis(m2 if curved else m1, ElementTetP2())
    K = BilinearForm(lambda u, v, _: dot(grad(u), grad(v)) + c2 * u * v).assemble(basis)
    M = BilinearForm(lambda u, v, _: u * v).assemble(basis)
    I = basis.complement_dofs(basis.get_dofs())
    lu = splu(K[I][:, I].tocsc())
    P = Basis(m1, ElementTetP2()).probes(pts.T).tocsr()[:, I]
    x = basis.doflocs.T
    h, G, L = ball_parts(x, np, m, p, amp)
    b = 1 - (x**2).sum(1)
    out = []
    for q in range(0, len(xi), chunk):
        z = xi[q:q + chunk]
        dg = np.einsum("qm,nmc->qnc", z, G)
        f = np.exp(z @ h.T) * (6 + c2 * b + 4 * (dg * x).sum(-1) - b * (z @ L.T) - b * (dg**2).sum(-1))
        out.append(P @ lu.solve((M @ f.T)[I]))
    return np.hstack(out), int(basis.N)
