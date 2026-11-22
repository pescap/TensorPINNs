import numpy as np

from tpinn.fem import Collocation, holed_square, sample

RNG = np.random.default_rng(0)
PTS = sample(RNG, 60)


def run(ns, nt, k=2):
    idx = np.random.default_rng(1).integers(0, 60, (400, k))
    col = Collocation(holed_square(0.3, ns, nt), k, PTS)
    return col, idx, col.tuples(idx)


def test_quadrature_matches_closed_form():
    col, idx, _ = run(6, 24, 3)
    assert col.quadrature_error(idx) < 1e-8


def test_mesh_convergence():
    a, b, c = (run(ns, nt)[2] for ns, nt in ((6, 24), (12, 48), (24, 96)))
    assert np.linalg.norm(b - c) < 0.5 * np.linalg.norm(a - b)


def test_ball_fem_converges_to_manufactured():
    from tpinn.fem import ball_solutions
    from tpinn.geom import ball_points
    from tpinn.ref import ball_parts
    pts = ball_points(np.random.default_rng(7), 50, 0.9)
    xi = np.random.default_rng(8).standard_normal((3, 4))
    exact = (1 - (pts**2).sum(1))[:, None] * np.exp(ball_parts(pts)[0] @ xi.T)
    e = [np.linalg.norm(ball_solutions(n, pts, xi)[0] - exact) / np.linalg.norm(exact) for n in (2, 3)]
    assert 3 < e[0] / e[1] < 5
