"""
Exact-reference validation of the low-rank hard-BC V-TPINN formulation (1-D, so an exact
tensor finite-difference solution is available).  D=[0,1], A=-u''+kappa^2 u, lognormal field.
2nd moment Sigma(x,y) solves A^(2)Sigma = C^2.  Symmetric low-rank ansatz Sigma=sum_r a_r(x)a_r(y),
a_r=x(1-x)*NN (hard BC), single loss ||A^(2)Sigma - C^2||^2, Adam + L-BFGS.
EXACT reference: tensor finite differences Sigma = Linv C^2 Linv^T.
Also writes a 3-panel validation figure (exact | TPINN | abs error).  PDF, publication style.
"""
import time, numpy as np, tensorflow as tf
tf.compat.v1.disable_eager_execution(); tf.compat.v1.disable_v2_behavior()
np.random.seed(0); tf.compat.v1.set_random_seed(0)

kappa, m, pdec, Ngrid = 4.0, 6, 2.0, 120
jj = np.arange(1, m+1, dtype=np.float64); lam = jj**(-pdec)
R, WIDTH, DEPTH, LR, EPOCHS, NCOL, LBFGS = 12, 64, 4, 1e-3, 8000, 4000, 8000

def Cg_np(a, b): return (lam*np.sin(np.pi*np.outer(a, jj))*np.sin(np.pi*np.outer(b, jj))).sum(1)

# ---- exact reference via tensor-FD ----
h = 1.0/(Ngrid+1); xg = np.linspace(h, 1-h, Ngrid)
L = (np.diag(2*np.ones(Ngrid)) + np.diag(-np.ones(Ngrid-1), 1) + np.diag(-np.ones(Ngrid-1), -1))/h**2 + kappa**2*np.eye(Ngrid)
Linv = np.linalg.inv(L)
XX, YY = np.meshgrid(xg, xg, indexing='ij')
C2grid = np.exp(0.5*(Cg_np(XX.ravel(), XX.ravel()) + Cg_np(YY.ravel(), YY.ravel())) + Cg_np(XX.ravel(), YY.ravel())).reshape(Ngrid, Ngrid)
Sig_exact = Linv @ C2grid @ Linv.T
SCALE = float(C2grid.max())
print(f"[ref] exact Sigma range [{Sig_exact.min():.3e},{Sig_exact.max():.3e}]  SCALE={SCALE:.3e}")

# ---- symmetric low-rank V-TPINN ----
def mlp(x, scope, reuse=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        hh = x
        for l in range(DEPTH): hh = tf.compat.v1.layers.dense(hh, WIDTH, tf.tanh, name=f"d{l}")
        raw = tf.compat.v1.layers.dense(hh, R, name="out")
    return x*(1-x)*raw                                        # hard BC a_r(0)=a_r(1)=0

xph = tf.compat.v1.placeholder(tf.float32, [None, 1]); yph = tf.compat.v1.placeholder(tf.float32, [None, 1])
A = mlp(xph, "A"); B = mlp(yph, "A", reuse=True)             # shared basis -> symmetric, SPD diagonal
def Aop(v, t):
    cols = []
    for r in range(R):
        vr = v[:, r:r+1]; g = tf.gradients(vr, t)[0]; vtt = tf.gradients(g, t)[0]
        cols.append(-vtt + kappa**2*vr)
    return tf.concat(cols, axis=1)
AA, AB = Aop(A, xph), Aop(B, yph)
Sigma = tf.reduce_sum(A*B, axis=1, keepdims=True)
op = tf.reduce_sum(AA*AB, axis=1, keepdims=True)
def s_tf(t): return tf.sin(np.pi*tf.constant(jj, tf.float32)*t)
def Cg_tf(a, b): return tf.reduce_sum(tf.constant(lam, tf.float32)*s_tf(a)*s_tf(b), axis=1, keepdims=True)
C2 = tf.exp(0.5*(Cg_tf(xph, xph) + Cg_tf(yph, yph)) + Cg_tf(xph, yph)) / SCALE
loss = tf.reduce_mean((op - C2)**2)
train = tf.compat.v1.train.AdamOptimizer(LR).minimize(loss)

_cfg = tf.compat.v1.ConfigProto(); _cfg.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=_cfg) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    t0 = time.perf_counter()
    for it in range(EPOCHS):
        X = np.random.rand(NCOL, 1).astype(np.float32); Y = np.random.rand(NCOL, 1).astype(np.float32)
        sess.run(train, {xph: X, yph: Y})
        if it % 2000 == 0: print(f"  it {it}: loss {sess.run(loss, {xph:X, yph:Y}):.3e} ({time.perf_counter()-t0:.0f}s)", flush=True)
    # L-BFGS refinement on a fixed collocation set
    Xf = np.random.rand(NCOL, 1).astype(np.float32); Yf = np.random.rand(NCOL, 1).astype(np.float32)
    fd = {xph: Xf, yph: Yf}
    if LBFGS > 0:
        from scipy.optimize import minimize
        vs = tf.compat.v1.trainable_variables(); grads = tf.gradients(loss, vs)
        shapes = [v.shape.as_list() for v in vs]; sizes = [int(np.prod(s)) for s in shapes]
        phs = [tf.compat.v1.placeholder(tf.float32, s) for s in shapes]
        assign = [v.assign(ph) for v, ph in zip(vs, phs)]
        def set_flat(z):
            i = 0; feed = {}
            for ph, sz, sh in zip(phs, sizes, shapes): feed[ph] = z[i:i+sz].reshape(sh); i += sz
            sess.run(assign, feed)
        x0 = np.concatenate([sess.run(v).ravel() for v in vs]).astype(np.float64)
        def fun(z):
            set_flat(z.astype(np.float32)); l, gs = sess.run([loss, grads], fd)
            return float(l), np.concatenate([np.asarray(g).ravel() for g in gs]).astype(np.float64)
        r = minimize(fun, x0, jac=True, method="L-BFGS-B",
                     options={"maxiter": LBFGS, "maxfun": LBFGS*2, "ftol": 1e-14, "gtol": 1e-12})
        set_flat(r.x.astype(np.float32)); print(f"  L-BFGS done: loss {sess.run(loss, fd):.3e}", flush=True)
    Xg = XX.ravel()[:, None].astype(np.float32); Yg = YY.ravel()[:, None].astype(np.float32)
    Sig_pred = (sess.run(Sigma, {xph: Xg, yph: Yg}).ravel() * SCALE).reshape(Ngrid, Ngrid)

rel = np.linalg.norm(Sig_pred - Sig_exact)/np.linalg.norm(Sig_exact)
corr = np.corrcoef(Sig_pred.ravel(), Sig_exact.ravel())[0, 1]
diagrel = np.linalg.norm(np.diag(Sig_pred)-np.diag(Sig_exact))/np.linalg.norm(np.diag(Sig_exact))
print(f"\nFORMCHECK sym low-rank V-TPINN vs exact tensor-FD: full-field rel.err {rel:.3e} | diag rel.err {diagrel:.3e} | corr {corr:.4f}")

# ---- publication figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
vmax = Sig_exact.max()
for a, val, ttl in [(ax[0], Sig_exact, r"exact tensor-FD  $\Sigma$"),
                    (ax[1], Sig_pred, r"low-rank V-TPINN  $\Sigma_\theta$")]:
    im = a.imshow(val, origin="lower", extent=[0,1,0,1], cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
    a.set_title(ttl); a.set_xlabel("$x$"); a.set_ylabel("$y$"); plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
err = np.abs(Sig_pred - Sig_exact)
im = ax[2].imshow(err, origin="lower", extent=[0,1,0,1], cmap="magma", aspect="equal")
ax[2].set_title(r"$|\Sigma_\theta-\Sigma|$" + f"  (rel. err {rel:.1e})")
ax[2].set_xlabel("$x$"); ax[2].set_ylabel("$y$"); plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig("results/lowrank_1d_exact.pdf", bbox_inches="tight")
print("[fig] results/lowrank_1d_exact.pdf")
