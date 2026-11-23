"""Warm-start A/B setup for a problem profile (PHRPORT_PROBLEM): the two-stage arm and the QSS seed.

  PHRPORT_PROBLEM=examples/qa_vacuum.json python -u tools/warm_setup.py runs/warm_lpqa

Writes into OUT:
  two_stage_boundary.json   the design boundary (the profile's seed boundary)
  two_stage_coils.json/.npz the R1 final coil stage on it (CoilSolver.final_coil_stage), simsopt export
  two_stage.json            equilibrium metrics at the optimisation resolution (step) and at mpol 8 /
                            22x20 / ns101, PHR constraint values, coil metrics (flux, certification, elasticities)
  seed/al1_checkpoint_final.json
                            a PHR checkpoint at the design (step, lam = 0, rho = 1) that seeds the
                            flux-bound homotopy (phr --sweep --resume-from OUT/seed)
"""
import argparse
import json
import os
import time

import numpy as np

from phrport import equilibrium as E
from phrport import problem as PROB
from phrport.coils import CoilSolver, CoilConfig
from phrport.phr import PHRConfig, eq_constraint_labels, eq_constraint_values, export_coils


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def eq_summary(r):
    return dict(converged=bool(r.converged), qs=float(r.qs_total), aspect=float(r.aspect),
                mean_iota=float(r.mean_iota), elongation=float(r.elongation), well=float(r.well),
                message=r.message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=4, help="continuation step of the homotopy (4: modes <= 5)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "seed"), exist_ok=True)
    b = E.seed_boundary()
    cfg = PHRConfig()
    labels = eq_constraint_labels(cfg)
    with open(os.path.join(a.out, "two_stage_boundary.json"), "w") as fh:
        json.dump(b, fh, indent=1)
    t0 = time.time()
    r_opt = E.evaluate(b, a.step)
    r_m8 = E.evaluate_resolution(b, 8, 22, 20, "ns101")
    g = eq_constraint_values(r_opt, cfg)
    print(f"[{PROB.NAME}] design at step {a.step}: {eq_summary(r_opt)}", flush=True)
    print(f"  mpol 8 / ns101: QS {r_m8.qs_total:.4e}", flush=True)
    print(f"  constraints {dict(zip(labels, np.round(g, 8)))}  [{time.time() - t0:.0f}s]", flush=True)
    t0 = time.time()
    cs = CoilSolver(CoilConfig())
    r = cs.final_coil_stage(b)
    info = r.info or {}
    fs = info.get("final_stage") or {}
    print(f"  final coil stage: Jf={r.c0_local:.6e} ok={r.ok} certified={r.certified} "
          f"flux={(fs.get('flux_unfolded') or {}).get('bn_ratio_int')} start={fs.get('chosen_start')} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    p = export_coils(os.path.join(a.out, "two_stage_coils"), cs, r, b, source=f"warm_setup {PROB.NAME} two-stage")
    summary = dict(problem=PROB.NAME, profile=PROB.PATH, step=a.step, equilibrium=eq_summary(r_opt),
                   equilibrium_mpol8_ns101=eq_summary(r_m8), constraints=dict(zip(labels, g.tolist())),
                   targets=dict(aspect=cfg.aspect_target, aspect_two_sided=cfg.aspect_two_sided,
                                iota=cfg.iota_mean_target, iota_tol=cfg.iota_tol,
                                elongation=cfg.elongation_threshold, well=cfg.well_threshold),
                   coil_config=jsonable(CoilConfig().__dict__),
                   coils=dict(Jf=float(r.c0_local), ok=bool(r.ok), certified=bool(r.certified),
                              currents=jsonable(r.currents), cert=jsonable(info.get("cert")),
                              final_stage={k: jsonable(v) for k, v in fs.items()
                                           if k in ("chosen_start", "flux_50x50", "flux_unfolded", "elasticity",
                                                    "t_total")}),
                   export=str(p))
    with open(os.path.join(a.out, "two_stage.json"), "w") as fh:
        json.dump(jsonable(summary), fh, indent=1)
    names = E.free_dof_names(a.step)
    ck = dict(alm_mode="phr", step=a.step, al_iter=0, max_mode=a.step + 1,
              rho_eq=[cfg.rho_init] * len(labels), rho_coil=[cfg.rho_init], lam_eq=[0.0] * len(labels),
              lam_coil=[0.0], surface_dofs=[float(b[n]) for n in names], vmec_dofs=[float(b[n]) for n in names],
              vmec_input=str(E.INPUT_FILE), surface_dof_names=list(names), coil_targets_scaled=[0.0],
              flux_target_captured=False, final=True, completed=True, aborted=False,
              v_prev_eq=None, v_prev_coil=None, outers_done_in_step=0, note=f"warm_setup seed: {PROB.NAME} design")
    with open(os.path.join(a.out, "seed", "al1_checkpoint_final.json"), "w") as fh:
        json.dump(ck, fh, indent=1)
    print(f"  wrote {a.out}/two_stage.json, two_stage_coils.*, seed/al1_checkpoint_final.json", flush=True)


if __name__ == "__main__":
    main()
