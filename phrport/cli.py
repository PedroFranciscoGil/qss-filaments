"""Command-line entry points (installed as ``qss-run``, ``qss-coils`` and ``qss-check``).

qss-run   --profile P.json --run-dir DIR [--preset production|tiny] [phrport.phr options]
          The optimization (python -m phrport.phr with the profile selected). Every phrport.phr option is passed
          through; ``qss-run --help-phr`` lists them. --preset puts a flag set first: production = the settings of
          the published runs (alias paper), tiny = a few-minute laptop smoke run. A profile's "run" section adds options after the preset;
          the command line wins over both.
qss-coils --profile P.json (--run-dir RUN | --boundary B.json) --out STEM [--override JSON] [--quick]
          Coil stage on a fixed boundary with XCoil: cold starts, then the certified warm polish
          (CoilSolver.final_coil_stage), or one cold solve with --quick. Writes STEM.json (phrport-coils-v1).
qss-check  Environment report: versions, JAX devices / x64, VMEX, XCoil backend, input deck of a profile.

The profile is read when phrport is first imported, so these commands set PHRPORT_PROBLEM before importing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _set_profile(path):
    if path:
        p = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(p):
            raise SystemExit(f"profile {path!r} not found")
        if "phrport.problem" in sys.modules and os.environ.get("PHRPORT_PROBLEM") != p:
            raise SystemExit("phrport was imported before the profile was set; run qss-* in a fresh process")
        os.environ["PHRPORT_PROBLEM"] = p
    os.environ.setdefault("JAX_ENABLE_X64", "1")


# Flag presets for qss-run. "production" (the settings of the published runs; alias "paper"): exact equilibrium Jacobian, flux metric scale 500, adaptive flux bound 0.5 x c0
# per mode, anchored coil oracle, the exact speed-ups (skip rejected trials' coil solves, abort doomed solves, reuse the
# VMEX state, certify only accepted coils), carried trust radius and the full outer budget in every mode.
PRESETS = {
    "production": ["--eq-jacobian", "exact", "--max-nfev", "30", "--flux-metric-scale", "500", "--flux-target-frac", "0.5",
              "--coil-oracle", "anchored", "--oracle-k-max", "4", "--skip-coil-on-rejected", "--abort-doomed-coil-solves",
              "--no-certify-trials", "--reuse-vmex-state", "--carry-trust-radius", "--full-outer-budget",
              "--n-workers", "1", "--fd-ngroups", "1"],
    "tiny": ["--n-continuation-steps", "1", "--al-outer-iter", "1", "--max-nfev", "3", "--n-workers", "2",
             "--ls-verbose", "0"],
}
PRESET_ENV = {"production": {"PHRPORT_TRUST_RESET_STEP": "1", "PHRPORT_TRUST_FLOOR": "1e-3"}, "tiny": {}}
# deprecated preset names -> current name
PRESET_ALIASES = {"paper": "production"}


def _split_profile(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--profile", default=os.environ.get("PHRPORT_PROBLEM"))
    ap.add_argument("--preset", choices=sorted(PRESETS) + sorted(PRESET_ALIASES), default=None)
    ap.add_argument("--help-phr", action="store_true")
    return ap.parse_known_args(argv)


def run_main(argv=None):
    """qss-run: the PHR optimization of a profile (see the module docstring)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in ([], ["-h"], ["--help"]):
        print(__doc__.split("\n\n")[1].split("qss-coils")[0].rstrip())
        print("\nqss-run --help-phr lists the optimizer options (python -m phrport.phr --help).")
        return 0
    a, rest = _split_profile(argv)
    _set_profile(a.profile)
    from phrport import phr
    if a.help_phr:
        return phr.main(["--help"])
    # precedence (last wins): built-in defaults < profile sections < --preset < profile "run" < command line
    pre = []
    if a.preset in PRESET_ALIASES:
        print(f"[qss-run] note: --preset {a.preset} is deprecated, use --preset {PRESET_ALIASES[a.preset]}", flush=True)
        a.preset = PRESET_ALIASES[a.preset]
    if a.preset:
        for k, v in PRESET_ENV[a.preset].items():
            os.environ.setdefault(k, v)
        pre += PRESETS[a.preset]
        print(f"[qss-run] preset {a.preset}: {' '.join(PRESETS[a.preset])}", flush=True)
    if a.profile:
        from phrport import problem
        from phrport.options import validate_profile
        parser = phr.build_parser()
        for m in validate_profile(problem.DATA, run_flags=run_flags(parser)):
            print(f"[qss-run] WARNING profile {problem.PATH}: {m}", flush=True)
        prof_run = profile_run_args(problem.DATA.get("run") or {}, parser)
        if prof_run:
            print(f"[qss-run] profile run options: {' '.join(prof_run)}", flush=True)
        pre += prof_run
    return phr.main(pre + rest)


def run_flags(parser):
    """{name: action} accepted in a profile's "run" section: argparse dests and flag names without dashes."""
    out = {}
    for act in parser._actions:
        if act.dest == "help":
            continue
        out.setdefault(act.dest, act)          # a dest shared by --x / --no-x maps to the first (--x)
        for o in act.option_strings:
            out[o.lstrip("-").replace("-", "_")] = act
    return out


def profile_run_args(run, parser):
    """Command-line arguments equivalent to a profile's "run" section (see docs/profile.md)."""
    flags = run_flags(parser)
    args = []
    for k, v in run.items():
        if k.startswith("_"):
            continue
        act = flags.get(k)
        if act is None:
            continue                                  # reported by validate_profile
        opt = act.option_strings[0]
        twin = {type(o): o for o in parser._actions if o.dest == act.dest and o is not act}
        if isinstance(act, argparse._StoreTrueAction):
            if v:
                args.append(opt)
            elif argparse._StoreFalseAction in twin:          # "x": false undoes --x (e.g. from a preset)
                args.append(twin[argparse._StoreFalseAction].option_strings[0])
        elif isinstance(act, argparse._StoreFalseAction):
            # a "no_x" key: true means pass the flag; a dest key (e.g. enable_coils): false means pass the flag
            if (k == act.dest and not v) or (k != act.dest and v):
                args.append(opt)
            elif argparse._StoreTrueAction in twin:
                args.append(twin[argparse._StoreTrueAction].option_strings[0])
        elif v is not None:
            args += [opt, str(v)]
    return args


def run_parser():
    """The options qss-run adds in front of python -m phrport.phr's (see phrport.phr.build_parser)."""
    ap = argparse.ArgumentParser(prog="qss-run", add_help=False)
    ap.add_argument("--profile", default=None, help="problem profile JSON (sets PHRPORT_PROBLEM)")
    ap.add_argument("--preset", choices=sorted(PRESETS) + sorted(PRESET_ALIASES), default=None,
                    help="flag set put before the other options: production (the settings of the published runs; "
                         "'paper' is a deprecated alias) or tiny (a few-minute laptop smoke run); options given after "
                         "it win")
    ap.add_argument("--help-phr", action="store_true", help="list the optimizer options and exit")
    return ap


def coils_parser():
    ap = argparse.ArgumentParser(prog="qss-coils", description=coils_main.__doc__)
    ap.add_argument("--profile", default=os.environ.get("PHRPORT_PROBLEM"),
                    help="problem profile (coil limits, deck, finite beta)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="a qss-run output folder: its final boundary (al1_checkpoint_final.json) and "
                                       "coil settings (final_coils.json, when present)")
    src.add_argument("--boundary", help="JSON boundary {'rc(m,n)': value, 'zs(m,n)': value, ...} or a checkpoint "
                                        "with surface_dof_names / surface_dofs")
    ap.add_argument("--out", required=True, help="output stem: writes STEM.json (phrport-coils-v1)")
    ap.add_argument("--override", default=os.environ.get("COIL_OVERRIDE", "{}"),
                    help="JSON merged into the coil configuration, e.g. '{\"per_coil_length\": 3.85}'")
    ap.add_argument("--quick", action="store_true",
                    help="one cold solve (primary start + fallbacks, certified) instead of the full final stage")
    ap.add_argument("--step", type=int, default=None, help="continuation step of the equilibrium (finite beta)")
    return ap


def coils_main(argv=None):
    """qss-coils: certified XCoil coil stage on a fixed boundary."""
    a = coils_parser().parse_args(argv)
    _set_profile(a.profile)
    import time
    from phrport import equilibrium as E
    from phrport.coils import CoilSolver, CoilConfig
    from phrport.phr import export_coils

    b = E.seed_boundary()
    cc, step = {}, a.step
    if a.run_dir:
        ck = json.load(open(os.path.join(a.run_dir, "al1_checkpoint_final.json")))
        b.update(dict(zip(ck["surface_dof_names"], ck["surface_dofs"])))
        step = int(ck.get("step", 0)) if step is None else step
        fc_path = os.path.join(a.run_dir, "final_coils.json")
        if os.path.isfile(fc_path):
            fc = json.load(open(fc_path))
            cc = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (fc.get("coil_config") or {}).items()}
    else:
        d = json.load(open(a.boundary))
        if "surface_dof_names" in d:
            d = dict(zip(d["surface_dof_names"], d["surface_dofs"]))
        elif "boundary" in d:
            d = d["boundary"]
        b.update({k: float(v) for k, v in d.items()})
    cc.update(json.loads(a.override))
    cs = CoilSolver(CoilConfig(**cc))
    if a.run_dir and ck.get("ordering_axis") is not None:      # certify about the step's frozen ordering axis
        cs.set_ordering_axis(axis=ck["ordering_axis"])
    if E._PROB.get("phr", "finite_beta", False):
        from phrport.finite_beta import standalone_coupler
        cp = standalone_coupler(b, int(step or 0), log=lambda *x, **k: print(*x, flush=True))
        cs.set_plasma(cp.provider)
        print("finite beta: virtual-casing plasma field in the coil problem", flush=True)
    t0 = time.time()
    r = cs.solve(b, warm=None) if a.quick else cs.final_coil_stage(b)
    fs = (r.info or {}).get("final_stage") or {}
    summary = dict(c0=float(r.c0_local), ok=bool(r.ok), certified=bool(r.certified), linked=bool(r.linked),
                   v=(r.info or {}).get("v"), bn_ratio_int=(r.info or {}).get("bn_ratio_int"),
                   wall_s=time.time() - t0, mode="cold solve" if a.quick else "final_coil_stage",
                   **{k: fs.get(k) for k in ("chosen_start", "n_met", "n_polished", "Jf_polished", "v_target_met")})
    print(json.dumps(summary, default=str), flush=True)
    export_coils(a.out, cs, r, b, source="qss-coils (" + summary["mode"] + ")")
    print("wrote", a.out + ".json", flush=True)
    return 0 if r.ok else 2


def check_parser():
    ap = argparse.ArgumentParser(prog="qss-check", description=check_main.__doc__)
    ap.add_argument("--profile", default=os.environ.get("PHRPORT_PROBLEM"), help="problem profile to check")
    return ap


def check_main(argv=None):
    """qss-check: report the environment (versions, devices, VMEX, XCoil) and a profile's input deck."""
    a = check_parser().parse_args(argv)
    _set_profile(a.profile)
    ok = True
    import platform
    print(f"python      {platform.python_version()} ({sys.executable})")
    import numpy, scipy, jax
    jax.config.update("jax_enable_x64", True)
    print(f"numpy       {numpy.__version__}\nscipy       {scipy.__version__}\njax         {jax.__version__}")
    print(f"devices     {[str(d) for d in jax.devices()]}")
    print(f"x64         {bool(jax.config.jax_enable_x64)}")
    try:
        import vmex
        print(f"vmex        {getattr(vmex, '__version__', '?')} ({os.path.dirname(vmex.__file__)})")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"vmex        MISSING: {e}")
    for mod in ("virtual_casing_jax", "booz_xform_jax", "solvax"):
        try:
            m = __import__(mod)
            print(f"{mod:<11} {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"{mod:<11} MISSING: {e}")
    from phrport.backend import xcoil_available
    fok, msg = xcoil_available()
    ok &= fok
    print(f"xcoil       {'ok' if fok else 'UNAVAILABLE'}: {msg}")
    from phrport import problem as P
    print(f"profile     {P.PATH or '(none: PHR nfp2 QA defaults)'}")
    if P.PATH:
        from phrport.options import validate_profile
        from phrport.phr import build_parser
        msgs = validate_profile(P.DATA, run_flags=run_flags(build_parser()))
        for m in msgs:
            print(f"            WARNING {m}")
        print(f"profile keys {'ok' if not msgs else f'{len(msgs)} problem(s)'}")
        ok &= not msgs
    from phrport import equilibrium as E
    deck = str(E.INPUT_FILE)
    print(f"input deck  {deck} ({'found' if os.path.isfile(deck) else 'MISSING'})")
    ok &= os.path.isfile(deck)
    print("OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run_main())
