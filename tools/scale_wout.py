"""Rescale a VMEC/VMEX wout to a common minor radius and volume-averaged |B| (ARIES-CS: a = 1.704 m, <B> = 5.865 T).

Lengths scale by L = a_target / Aminor_p and fields by b = B_target / volavgB. Each Fourier table is scaled by its
dimension: R, Z ~ L; sqrt(g) (s, theta, phi) ~ L^3; |B| ~ b; covariant B_s, B_u, B_v ~ b L; contravariant
B^u, B^v ~ b / L; toroidal/poloidal flux ~ b L^2; currents ~ b L; pressure ~ b^2 (beta unchanged).
  scale_wout.py IN OUT [a_target B_target]
"""
import shutil, sys
import netCDF4
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
a_t = float(sys.argv[3]) if len(sys.argv) > 3 else 1.704
B_t = float(sys.argv[4]) if len(sys.argv) > 4 else 5.865
shutil.copyfile(src, dst)
with netCDF4.Dataset(dst, "r+") as ds:
    v = ds.variables
    L = a_t / float(v["Aminor_p"][()]); b = B_t / float(v["volavgB"][()])
    dims = {  # name: (power of L, power of b)
        "rmnc": (1, 0), "zmns": (1, 0), "rmns": (1, 0), "zmnc": (1, 0), "lmns": (0, 0), "lmnc": (0, 0),
        "raxis_cc": (1, 0), "zaxis_cs": (1, 0), "raxis_cs": (1, 0), "zaxis_cc": (1, 0),
        "Rmajor_p": (1, 0), "Aminor_p": (1, 0), "rmax_surf": (1, 0), "rmin_surf": (1, 0), "zmax_surf": (1, 0),
        "volume_p": (3, 0), "gmnc": (3, 0), "gmns": (3, 0), "vp": (3, 0),
        "bmnc": (0, 1), "bmns": (0, 1), "volavgB": (0, 1), "b0": (0, 1),
        "bsubsmns": (1, 1), "bsubsmnc": (1, 1), "bsubumnc": (1, 1), "bsubumns": (1, 1), "bsubvmnc": (1, 1), "bsubvmns": (1, 1),
        "bsupumnc": (-1, 1), "bsupumns": (-1, 1), "bsupvmnc": (-1, 1), "bsupvmns": (-1, 1),
        "phi": (2, 1), "phipf": (2, 1), "phips": (2, 1), "chi": (2, 1), "chipf": (2, 1),
        "jcuru": (-1, 1), "jcurv": (-1, 1), "ctor": (1, 1), "buco": (1, 1), "bvco": (1, 1),
        "presf": (0, 2), "pres": (0, 2), "am": (0, 2), "wb": (3, 2), "wp": (3, 2),
    }
    done = []
    for name, (pl, pb) in dims.items():
        if name in v:
            v[name][...] = np.asarray(v[name][...]) * L ** pl * b ** pb; done.append(name)
    print(f"{src} -> {dst}: L = {L:.4f}, b = {b:.4f}  (a {float(v['Aminor_p'][()]):.3f} m, <B> {float(v['volavgB'][()]):.3f} T)")
