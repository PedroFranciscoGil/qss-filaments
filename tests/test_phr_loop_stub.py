"""The PHR outer loop end to end on cheap analytic stubs (tests/stubs.py): continuation steps, AL outers,
checkpoints, final metrics, resume from a checkpoint."""
import json
import os

import numpy as np

from phrport import phr as P
import stubs as S


def _runner(out, **cfgkw):
    eq = S.StubEquilibrium()
    pool = S.StubPool(eq)
    cfg = P.PHRConfig(**dict(dict(flux_metric_scale=100.0, n_continuation_steps=2, al_outer_iter=2, max_nfev=4),
                             **cfgkw))
    return P.PHRRunner(cfg, evaluate=eq.evaluate, evaluate_many=pool.evaluate_many, coil_solver=S.StubCoilSolver(),
                       free_dof_names=S.free_dof_names, seed_boundary=S.seed_boundary(), out_dir=str(out),
                       log_to_stdout=False)


def test_self_test():
    ok, _ = P.run_self_test(verbose=False)
    assert ok


def test_stub_run_and_resume(tmp_path):
    out = tmp_path / "run"
    _runner(out).run()
    files = os.listdir(out)
    assert "final_metrics.json" in files
    cks = sorted(f for f in files if f.startswith("al1_checkpoint_step"))
    assert cks, files
    fm = json.load(open(out / "final_metrics.json"))
    assert np.isfinite(fm["surface"]["qs_residual_sum_sq"]) and "g_eq" in fm["surface"]
    ck = json.load(open(out / cks[-1]))
    assert ck["eq_labels"] == P.eq_constraint_labels(P.PHRConfig())
    # resume the last checkpoint with one more continuation step
    out2 = tmp_path / "resumed"
    r2 = _runner(out2, n_continuation_steps=3)
    r2.resume_from = str(out / cks[-1])
    r2.run()
    assert "final_metrics.json" in os.listdir(out2)


def test_stub_run_with_extra_rows(tmp_path):
    """Optional rows switched on through the config (values come from the stub's attributes or NaN-safe)."""
    out = tmp_path / "rows"
    r = _runner(out, n_continuation_steps=1, al_outer_iter=1, aspect_two_sided=True, aspect_tol=0.05,
                iota_tol=0.01)
    r.run()
    fm = json.load(open(out / "final_metrics.json"))
    assert "aspect_lo" in json.dumps(fm)


def test_phase_timers(tmp_path):
    """--phase-timers (passive per-evaluation timing table) on the stub loop."""
    out = tmp_path / "pt"
    _runner(out, phase_timers=True, n_continuation_steps=1, al_outer_iter=1).run()
    lines = (out / "phase_timers.jsonl").read_text().splitlines()
    assert lines and json.loads(lines[0])["kind"] in ("f", "jac")
