"""Tutorial 1 end to end at its tiny settings (VMEX + XCoil, about 2 minutes on a laptop CPU)."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_tutorial1_tiny(tmp_path):
    run = tmp_path / "t1"
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", PHRPORT_PYCACHE=str(tmp_path / "pyc"))
    env.pop("PHRPORT_PROBLEM", None)
    cmd = [sys.executable, "-m", "phrport.cli", "--profile", os.path.join(ROOT, "examples", "qa_vacuum_tiny.json"),
           "--preset", "tiny", "--run-dir", str(run)]
    p = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    fm = json.load(open(run / "final_metrics.json"))
    s, c = fm["surface"], fm["coils"]
    assert abs(s["aspect_ratio"] - 6.0) < 0.05
    assert abs(s["mean_iota"] - 0.4193) < 0.02
    assert 0 < s["qs_residual_sum_sq"] < 1e-2
    assert set(s["g_eq"]) == {"aspect", "aspect_lo", "iota_mean_hi", "iota_mean_lo", "elong", "well"}
    assert 0 < c["squared_flux_Jf"] < 1e-2 and not c["linked"]
    fc = json.load(open(run / "final_coils.json"))
    assert fc["n_base_coils"] == 4 and len(fc["base_curve_dofs"]) == 4
    assert (run / "al1_checkpoint_final.json").exists() and (run / "wout_optimized.nc").exists()
