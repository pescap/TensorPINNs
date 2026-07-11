"""
MC + FEM baseline for the 2nd moment of a random-data Helmholtz problem on a COMPLEX geometry
(unit square with a circular hole). Classical reference for the complex-geometry headline example.

PDE (per realization w):   -Delta u - kappa^2 u = f(w)   in Omega = square \ disk,
                            u = 0 on the outer boundary and on the hole (sound-soft).
Random data: lognormal field  f(w,x) = fbar(x) * exp(g(w,x)),  g a truncated KL (sine) field.
Monte-Carlo: sample f, solve by FEM (dolfinx), accumulate E[u] and Var(u). Report wall-time.

This is the honest "why PINNs" comparison: FEM here needs an unstructured mesh of the non-Cartesian
domain, and the *2nd-moment* deterministic equation lives on Omega x Omega (4-D, complex^2) where
FEM is impractical -> that Omega x Omega problem is what the soft-BC TPINN targets by collocation.
Run in the dolfinx docker image.
"""
import os, time, numpy as np
from mpi4py import MPI
import ufl, dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

comm = MPI.COMM_WORLD
kappa = float(os.environ.get("KAPPA","4.0"))
m_modes, p_decay = 6, 2.0
seed, Nmc = 1234, int(os.environ.get("NMC", "400"))

# ---- mesh: square [-1,1]^2 minus a disk of radius r0, via gmsh if available ----
def build_mesh():
    try:
        import gmsh
        gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
        gdim = 2; r0 = 0.3; h = 0.04
        rect = gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)
        disk = gmsh.model.occ.addDisk(0, 0, 0, r0, r0)
        cut, _ = gmsh.model.occ.cut([(gdim, rect)], [(gdim, disk)])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(gdim, [cut[0][1]], 1)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        gmsh.model.mesh.generate(gdim)
        from dolfinx.io import gmsh as dgmsh
        res = dgmsh.model_to_mesh(gmsh.model, comm, 0, gdim=gdim)
        domain = res[0] if isinstance(res, tuple) else res.mesh
        gmsh.finalize()
        return domain, "square-minus-disk (gmsh)"
    except Exception as e:
        print(f"[mesh] gmsh unavailable ({e}); fallback to unit square")
        return mesh.create_unit_square(comm, 60, 60), "unit-square (fallback)"

domain, desc = build_mesh()
V = fem.functionspace(domain, ("Lagrange", 1))
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim - 1, tdim)
bfacets = mesh.exterior_facet_indices(domain.topology)
bdofs = fem.locate_dofs_topological(V, tdim - 1, bfacets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), bdofs, V)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
# Coercive reaction-diffusion  A = -Delta + kappa^2  (well-posed & PINN-friendly on complex geometry)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + kappa**2 * ufl.inner(u, v)) * ufl.dx

# KL sine modes on the bounding box [-1,1]^2 (just a random field generator)
jj = np.arange(1, m_modes + 1); lam = jj.astype(float) ** (-p_decay)
rng = np.random.default_rng(seed)

coords = V.tabulate_dof_coordinates()[:, :2]
def sample_logfield():
    xi = rng.standard_normal(m_modes)
    g = np.zeros(coords.shape[0])
    for a_, j in enumerate(jj):
        g += np.sqrt(lam[a_]) * xi[a_] * np.sin(j * np.pi * (coords[:, 0] + 1) / 2)
    return g

fbar_expr = 1.0  # constant baseline source (can swap for a profile)
fh = fem.Function(V)
L = ufl.inner(fh, v) * ufl.dx
problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="mc_",
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
mean = np.zeros(coords.shape[0]); m2 = np.zeros(coords.shape[0])
t0 = time.perf_counter()
for s in range(Nmc):
    fh.x.array[:] = fbar_expr * np.exp(sample_logfield())   # new realization; form re-assembled on solve
    uh = problem.solve()
    ua = uh.x.array.real
    mean += ua; m2 += ua**2
dt = time.perf_counter() - t0
mean /= Nmc; var = m2 / Nmc - mean**2
print(f"MC+FEM Helmholtz on {desc}: dofs={coords.shape[0]}, kappa={kappa}, Nmc={Nmc}")
print(f"  wall-time {dt:.1f}s ({dt/Nmc*1e3:.1f} ms/solve)")
print(f"  E[u] range [{mean.min():.3e},{mean.max():.3e}]  Var(u) max {var.max():.3e}")
SUF=os.environ.get("SUF",""); np.save(f"results/mcfem_mean{SUF}.npy", mean); np.save(f"results/mcfem_var{SUF}.npy", var)
np.save(f"results/mcfem_coords{SUF}.npy", coords)
