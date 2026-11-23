"""Force balance of free-boundary equilibria: VMEX residuals and the radial force balance profile equif (fbal.f)."""
import sys, json, numpy as np
from netCDF4 import Dataset
out = {}
for f in sys.argv[1:]:
    w = Dataset(f); g = lambda k: np.array(w[k][:])
    eq = g("equif"); ns = len(eq); s = np.linspace(0, 1, ns); m = (s > 0.05) & (s < 0.95)
    p = g("presf"); jc = g("jcurv"); r = dict(ns=ns, fsqr=float(g("fsqr")), fsqz=float(g("fsqz")), fsql=float(g("fsql")),
             equif_max=float(np.abs(eq[m]).max()), equif_rms=float(np.sqrt((eq[m] ** 2).mean())),
             jnorm=float(4e-7 * np.pi * np.abs(np.r_[g("jcurv")[m], g("jcuru")[m]]).max() * float(g("Aminor_p")) / float(g("volavgB"))), ctor=float(g("ctor")), p0=float(p[0]), ier=int(g("ier_flag")), betatot=float(g("betatotal")))
    w.close(); out[f] = r; print(f, r)
json.dump(out, open("forcebal.json", "w"), indent=1)
