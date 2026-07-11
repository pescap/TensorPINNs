"""
Complex-geometry 2nd moment: LOW-RANK V-TPINN, hard BCs, SINGLE loss, CHEAP (2nd-order only).
Sigma(p1,p2) = sum_{r=1}^R a_r(p1) b_r(p2),  a_r=l(p) A(p)_r, b_r=l(p) B(p)_r  (hard BC on dD).
Then  A^(2)Sigma = sum_r (A a_r)(p1) (A b_r)(p2)  with A a_r = -Delta a_r + kappa^2 a_r  (2nd order
in a 2-D network -> no 4th-order nesting; fast to build and run).  One residual = one loss.
Coercive A = -Delta + kappa^2 on D = [-1,1]^2 minus disk r0.  Raw TF (compat.v1), Adam.
"""
import argparse, time, numpy as np, tensorflow as tf
tf.compat.v1.disable_eager_execution(); tf.compat.v1.disable_v2_behavior()

ap = argparse.ArgumentParser()
ap.add_argument("--R", default=8, type=int)
ap.add_argument("--width", default=64, type=int)
ap.add_argument("--depth", default=4, type=int)
ap.add_argument("--act", default="tanh")
ap.add_argument("--lr", default=1e-3, type=float)
ap.add_argument("--epochs", default=20000, type=int)
ap.add_argument("--N", default=8000, type=int)
ap.add_argument("--seed", default=1234, type=int)
ap.add_argument("--tag", default="base")
ap.add_argument("--bl", default=0.4, type=float)   # boundary-layer sampling fraction
ap.add_argument("--kappa", default=4.0, type=float)
ap.add_argument("--refsuf", default="")            # MC+FEM reference suffix (_hq, _k2)
ap.add_argument("--lbfgs", default=8000, type=int) # scipy L-BFGS iters after Adam (0=off)
ap.add_argument("--cw", default=0.30, type=float)  # hard-BC tanh cutoff width (~1/kappa)
ap.add_argument("--sym", default=1, type=int)      # 1: share basis b_r=a_r (SPD-correct 2nd moment)
args = ap.parse_args()
np.random.seed(args.seed); tf.compat.v1.set_random_seed(args.seed)

kappa, r0, m, p_decay = args.kappa, 0.3, 6, 2.0
jj = np.arange(1, m+1, dtype=np.float64); lam = jj**(-p_decay)
ACT = {"tanh": tf.tanh, "sin": tf.sin, "swish": tf.nn.swish}[args.act]

def _uniform_D(n):
    pts = np.empty((0, 2))
    while len(pts) < n:
        q = np.random.uniform(-1, 1, (2*n, 2)); q = q[q[:, 0]**2 + q[:, 1]**2 > r0**2]
        pts = np.vstack([pts, q])
    return pts[:n]

def sample_D(n, bl=0.4):               # bl-fraction concentrated in boundary layers (hole + outer)
    nb = int(n * bl); nu = n - nb
    u = _uniform_D(nu)
    # near the hole: annulus r in [r0, r0+w]
    nh = nb // 2; w = 0.25
    th = np.random.uniform(0, 2*np.pi, nh); rr = r0 + w*np.random.rand(nh)**1.5
    hole = np.stack([rr*np.cos(th), rr*np.sin(th)], 1)
    hole = hole[(np.abs(hole[:,0])<1) & (np.abs(hole[:,1])<1)]
    # near the outer square edges
    no = nb - len(hole); e = 0.15
    o = _uniform_D(no); side = np.random.randint(0, 4, len(o)); t = 1 - e*np.random.rand(len(o))**1.5
    for k,(ax,sg) in enumerate([(0,1),(0,-1),(1,1),(1,-1)]):
        msk = side == k; o[msk, ax] = sg*t[msk]
    o = o[o[:,0]**2 + o[:,1]**2 > r0**2]
    return np.vstack([u, hole, o])[:n].astype(np.float32)

def mlp(p, scope, reuse=False):        # 2-D point -> R, hard BC via cutoff l(p)
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        h = p
        for l in range(args.depth): h = tf.compat.v1.layers.dense(h, args.width, ACT, name=f"d{l}")
        raw = tf.compat.v1.layers.dense(h, args.R, name="out")
    x, y = p[:, 0:1], p[:, 1:2]
    # cutoff: outer square factor x tanh hole factor (saturates to ~1 away from the hole).
    # cw = tanh width; match it to the boundary-layer width ~1/kappa so d(ell) doesn't blow up the residual.
    ell = (1 - x**2) * (1 - y**2) * tf.tanh((x**2 + y**2 - r0**2) / args.cw)
    return ell * raw                    # (batch,R), each column vanishes on dD

p1 = tf.compat.v1.placeholder(tf.float32, [None, 2]); p2 = tf.compat.v1.placeholder(tf.float32, [None, 2])
# --sym: share the basis (b_r=a_r) -> Sigma symmetric & Sigma(p,p)=sum a_r(p)^2 >= 0 (correct SPD 2nd moment)
A = mlp(p1, "A"); B = mlp(p2, "A", reuse=True) if args.sym else mlp(p2, "B")   # (batch,R)

def Aop(v, p):                          # (-Delta + kappa^2) applied column-wise to v(p), 2-D
    cols = []
    for r in range(args.R):
        vr = v[:, r:r+1]
        gx = tf.gradients(vr, p)[0]
        vxx = tf.gradients(gx[:, 0:1], p)[0][:, 0:1]
        vyy = tf.gradients(gx[:, 1:2], p)[0][:, 1:2]
        cols.append(-(vxx + vyy) + kappa**2 * vr)
    return tf.concat(cols, axis=1)
AA, AB = Aop(A, p1), Aop(B, p2)         # (batch,R)

Sigma = tf.reduce_sum(A * B, axis=1, keepdims=True)
op = tf.reduce_sum(AA * AB, axis=1, keepdims=True)         # A^(2) Sigma

# RHS C^2 (lognormal field, depends on x-coords) and scale
def s_np(xc): return np.sin(np.pi * np.outer((xc + 1)/2, jj))
def Cg_np(a, b): return (lam * s_np(a) * s_np(b)).sum(1)
def logC2(x1, x2): return 0.5*(Cg_np(x1, x1) + Cg_np(x2, x2)) + Cg_np(x1, x2)
probe1, probe2 = sample_D(20000), sample_D(20000)
SCALE = float(np.exp(logC2(probe1[:, 0], probe2[:, 0]).max())); print(f"[scale] SCALE={SCALE:.3e}")
def s_tf(xc): return tf.sin(np.pi * tf.constant(jj, tf.float32) * (xc + 1.0)/2.0)
def Cg_tf(a, b): return tf.reduce_sum(tf.constant(lam, tf.float32) * s_tf(a) * s_tf(b), axis=1, keepdims=True)
C2 = tf.exp(0.5*(Cg_tf(p1[:,0:1], p1[:,0:1]) + Cg_tf(p2[:,0:1], p2[:,0:1])) + Cg_tf(p1[:,0:1], p2[:,0:1])) / SCALE

loss = tf.reduce_mean((op - C2)**2)     # SINGLE loss
opt = tf.compat.v1.train.AdamOptimizer(args.lr).minimize(loss)

_cfg = tf.compat.v1.ConfigProto()
_cfg.gpu_options.allow_growth = True     # shared node: grab only what we need, not all VRAM
with tf.compat.v1.Session(config=_cfg) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    P1, P2 = sample_D(args.N, args.bl), sample_D(args.N, args.bl)  # FIXED collocation (sample once)
    fd = {p1: P1, p2: P2}
    t0 = time.perf_counter()
    for it in range(args.epochs):
        sess.run(opt, fd)
        if it % 500 == 0:
            print(f"  it {it}: loss {sess.run(loss, fd):.3e}  ({(time.perf_counter()-t0):.0f}s)", flush=True)
    # ---- scipy L-BFGS refinement (breaks the Adam plateau) ----
    if args.lbfgs > 0:
        from scipy.optimize import minimize
        vs = tf.compat.v1.trainable_variables(); grads = tf.gradients(loss, vs)
        shapes = [v.shape.as_list() for v in vs]; sizes = [int(np.prod(s)) for s in shapes]
        phs = [tf.compat.v1.placeholder(tf.float32, s) for s in shapes]
        assign = [v.assign(ph) for v, ph in zip(vs, phs)]
        def set_flat(x):
            i = 0; feed = {}
            for ph, sz, sh in zip(phs, sizes, shapes): feed[ph] = x[i:i+sz].reshape(sh); i += sz
            sess.run(assign, feed)
        x0 = np.concatenate([sess.run(v).ravel() for v in vs]).astype(np.float64)
        def fun(x):
            set_flat(x.astype(np.float32)); l, gs = sess.run([loss, grads], fd)
            return float(l), np.concatenate([np.asarray(g).ravel() for g in gs]).astype(np.float64)
        r = minimize(fun, x0, jac=True, method="L-BFGS-B",
                     options={"maxiter": args.lbfgs, "maxfun": args.lbfgs*2, "ftol": 1e-14, "gtol": 1e-12})
        set_flat(r.x.astype(np.float32))
        print(f"  L-BFGS done: loss {sess.run(loss, fd):.3e}", flush=True)
    dt = time.perf_counter() - t0
    # compare diagonal to MC+FEM
    try:
        RS = args.refsuf
        coords = np.load(f"results/mcfem_coords{RS}.npy").astype(np.float32)
        mean = np.load(f"results/mcfem_mean{RS}.npy"); var = np.load(f"results/mcfem_var{RS}.npy")
        pred = sess.run(Sigma, {p1: coords, p2: coords}).ravel() * SCALE
        ref = mean**2 + var
        rel = np.linalg.norm(pred-ref)/np.linalg.norm(ref); corr = np.corrcoef(pred, ref)[0,1]
        print(f"\nRESULT[{args.tag}] low-rank V-TPINN kappa={kappa} R={args.R} w={args.width} d={args.depth} act={args.act} lr={args.lr} N={args.N} bl={args.bl} cw={args.cw} sym={args.sym} lbfgs={args.lbfgs} ref={RS or 'base'}: rel.err {rel:.3e} | corr {corr:.3f} | {dt:.0f}s", flush=True)
        np.save(f"results/lowrank_pred_{args.tag}.npy", pred); np.save(f"results/lowrank_ref_{args.tag}.npy", ref)
    except Exception as e:
        print("no MC+FEM baseline:", e)
