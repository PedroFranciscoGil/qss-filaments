"""Finite-beta version of a profile's design: kinetic profiles -> pressure, then a self-consistent
Redl bootstrap current (VMEX self_consistent_bootstrap, Picard), written as a new input deck.

  PHRPORT_PROBLEM=examples/qa_vacuum.json python -u tools/make_beta_design.py \\
      --out-deck inputs/input.my_beta_design --n0 2e19 --T0 1125
"""
import argparse
import dataclasses
import json
import warnings

import numpy as np

from phrport import equilibrium as E
from vmex.core import bootstrap as BS

QE = 1.602176634e-19


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-deck", required=True)
    ap.add_argument("--n0", type=float, default=2e19, help="flat n_e = n_i [1/m^3]")
    ap.add_argument("--T0", type=float, default=1125.0, help="T_e = T_i = T0 (1 - s) [eV]")
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--relax", type=float, default=1.0)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    p0 = 2.0 * QE * a.n0 * a.T0                       # p = e n (Te + Ti), Zeff = 1
    prof = BS.KineticProfiles(ne_coeffs=np.array([a.n0]), Te_coeffs=a.T0 * np.array([1.0, -1.0]),
                              Ti_coeffs=a.T0 * np.array([1.0, -1.0]), Zeff_coeffs=np.array([1.0]))
    inp = E.vmec_input(E.seed_boundary(), 4)
    inp = dataclasses.replace(inp, am=np.array([p0, -p0]), pres_scale=1.0, ncurr=1, curtor=0.0)
    with E.vmex_device_ctx():
        res = BS.self_consistent_bootstrap(inp, prof, int(E.QS_HELICITY_N), n_iter=a.n_iter, relax=a.relax,
                                           solve_kwargs=dict(E.VMEX_SOLVE_KWARGS))
        for h in res.history:
            print("  picard", {k: (float(v) if np.ndim(v) == 0 else v) for k, v in h.items()}, flush=True)
        m = E.evaluate_wout(E.vmec2000_wout(res.equilibrium))
    w = res.equilibrium.wout
    out = dict(converged=bool(res.converged), iterations=int(res.iterations), p0_Pa=p0,
               curtor_A=float(res.input.curtor), betatotal=float(w.betatotal),
               qs=float(m["qs_total"]), aspect=float(m["aspect"]), mean_iota=float(m["mean_iota"]),
               iota_axis=float(m["iota_axis"]), iota_edge=float(m["iota_edge"]),
               elongation=float(m["elongation"]), well=float(m["well"]), volume=float(w.volume_p))
    print("BETA DESIGN", json.dumps(out), flush=True)
    res.input.to_indata(a.out_deck)
    print("wrote", a.out_deck)


if __name__ == "__main__":
    main()
