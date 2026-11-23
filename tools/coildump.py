"""Dump full coil sets (all field periods, stellarator copies) to npz for CoilPy finite-build export."""
import json, os, sys, numpy as np, warnings; warnings.simplefilter("ignore")
import jax.numpy as jnp
from phrport.coils import CoilSolver, CoilConfig, fr, sym_mod
src, out = sys.argv[1], sys.argv[2]
c = json.load(open(src))
cs = CoilSolver(CoilConfig()); cs.surface(c["boundary"]); P = cs.P
g, _, _ = fr.coils_curves(jnp.asarray(np.ravel(c["base_curve_dofs"])), P.cb, P.sym.C)
G = np.asarray(sym_mod.expand(g, P.sym))
I = np.asarray(c["base_currents_A"])[np.asarray(P.sym.coil)] * np.asarray(P.sym.sign)
base = np.asarray(P.sym.coil)
np.savez(out, xyz=G, current=I, base=base)
print(out, G.shape, "currents [kA]:", np.round(I / 1e3, 1)[:4])
