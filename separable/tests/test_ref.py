import numpy as np

from tpinn.ref import Reference, combination, exact_sep, fd_tensor, moment_logn, moment_sep, on_grid

PTS = np.linspace(0.05, 0.95, 10)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_cheb_matches_exact_separable():
    for k in (1, 2, 3):
        ref = Reference(moment_sep, k, n=16)
        assert rel(ref.grid(PTS), on_grid(exact_sep, [PTS] * k)) < 1e-10


def test_points_match_grid():
    ref = Reference(moment_logn, 3, n=24)
    P = np.stack([PTS, PTS[::-1], PTS], 1)
    G = ref.grid(PTS)
    idx = np.arange(10)
    assert np.allclose(ref.points(P), G[idx, idx[::-1], idx], rtol=1e-10)


def test_cheb_converges_lognormal():
    a = Reference(moment_logn, 2, c2=16.0, n=40).grid(PTS)
    b = Reference(moment_logn, 2, c2=16.0, n=64).grid(PTS)
    assert rel(a, b) < 1e-8


def test_full_fd_second_order():
    ref = Reference(moment_logn, 2, n=64).grid(PTS)
    e = [rel(fd_tensor(moment_logn, (l, l), 0.0, PTS)[0], ref) for l in (5, 6, 7)]
    assert 3.5 < e[0] / e[1] < 4.5 and 3.5 < e[1] / e[2] < 4.5


def test_combination_converges():
    for k in (2, 3):
        ref = Reference(moment_logn, k, n=48).grid(PTS)
        e = [rel(combination(moment_logn, k, 0.0, l, PTS)[0], ref) for l in (5, 7)]
        assert e[1] < e[0] / 6


def test_box_matches_square_reference():
    from tpinn.ref import box_kernel_moment, iso_kernel, square_second_moment
    pts = np.random.default_rng(1).random((30, 2))
    a = box_kernel_moment(2, 3.0, 0.05, 24, pts)
    b = square_second_moment(lambda X: iso_kernel(X), 3.0, 24, pts)
    assert rel(a, b) < 1e-10


def test_box_reference_converges_3d():
    from tpinn.ref import box_kernel_moment
    pts = np.random.default_rng(2).random((40, 3))
    e = [rel(box_kernel_moment(3, 0.0, 0.05, n, pts), box_kernel_moment(3, 0.0, 0.05, 48, pts)) for n in (16, 32)]
    assert e[1] < e[0] / 4 and e[1] < 1e-3


def test_box_highd_matches_chebyshev():
    from tpinn.ref import box_highd, box_kernel_moment
    for d, c2 in ((2, 0.0), (3, 0.0), (2, 3.0)):
        pts = np.random.default_rng(3).random((20, d))
        S = box_kernel_moment(d, c2, 0.05, 40, pts)
        idx = np.random.default_rng(4).integers(0, 20, (50, 2))
        assert rel(box_highd(d, 0.05, pts[idx], c2), S[idx[:, 0], idx[:, 1]]) < 1e-4


def test_classical_box_converges():
    from tpinn.classical_box import combine, solve
    from tpinn.ref import box_highd
    P = np.random.default_rng(5).random((40, 2, 2))
    truth = box_highd(2, 0.05, P)
    dense = [rel(solve([l] * 4, 2, 0.05, 0.0, P)[0], truth) for l in (4, 5)]
    sg = [rel(combine([1] * 4, n, 2, 0.05, 0.0, P)[0], truth) for n in (8, 10)]
    assert 3 < dense[0] / dense[1] < 5 and sg[1] < sg[0] / 2


def test_ball_manufactured_moment():
    import jax
    import jax.numpy as jnp
    from tpinn.ref import ball_exact, ball_moment
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(6)
    X = rng.normal(size=(5, 2, 3))
    X *= 0.8 * rng.random((5, 2, 1)) / np.linalg.norm(X, axis=-1, keepdims=True)
    c2 = 2.0

    def A2(x):
        S = lambda a, b: ball_exact(jnp.stack([a, b]), jnp)
        lap = lambda f, z: jnp.trace(jax.hessian(f)(z))
        inner = lambda a: (lambda b: -lap(lambda bb: S(a, bb), b) + c2 * S(a, b))
        return -lap(lambda a: inner(a)(x[1]), x[0]) + c2 * inner(x[0])(x[1])

    got = np.array([A2(jnp.asarray(x)) for x in X])
    assert rel(got, ball_moment(X, np, c2=c2)) < 1e-10
