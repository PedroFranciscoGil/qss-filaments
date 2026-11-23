"""CoilPy finite-build VTK (rectangular winding pack, centroid frame) + thin-line VTK from a coildump npz."""
import sys, numpy as np
from coilpy.coils import Coil
npz, stem, w, h = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
d = np.load(npz); X = d["xyz"]; I = d["current"]
close = lambda a: np.concatenate([a, a[:1]])                 # closed curves (first point repeated)
coil = Coil(xx=[close(x[:, 0]) for x in X], yy=[close(x[:, 1]) for x in X], zz=[close(x[:, 2]) for x in X],
            II=list(I), names=[f"coil{int(b)+1}" for b in d["base"]], groups=[int(b) + 1 for b in d["base"]])
coil.toVTK(stem + "_finite_build.vtk", line=False, width=w, height=h)
coil.toVTK(stem + "_line", line=True)
print("wrote", stem)
