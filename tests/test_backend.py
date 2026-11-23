"""XCoil backend selection (phrport.backend) and certified coil solves."""
import subprocess
import sys

import numpy as np
import pytest

from phrport import backend as B


def _select(tree):
    code = ("from phrport.backend import select_xcoil as s; b = s(%r); print(b.kind, b.path); "
            "import xcoil; print(xcoil.__file__)" % str(tree))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_selection_and_errors(tmp_path):
    assert B.select_xcoil("").kind == "installed"
    assert B.select_xcoil("installed").kind == "installed"
    with pytest.raises(B.XcoilUnavailable):
        B.select_xcoil(str(tmp_path))                    # no xcoil package there
    # a local checkout (folder that contains xcoil/), in a fresh process
    (tmp_path / "co" / "xcoil").mkdir(parents=True)
    (tmp_path / "co" / "xcoil" / "__init__.py").write_text("")
    out = _select(tmp_path / "co").stdout.split()
    assert out[:2] == ["checkout", str(tmp_path / "co")] and out[2].startswith(str(tmp_path / "co"))

ok, why = B.xcoil_available()


@pytest.mark.xcoil
@pytest.mark.skipif(not ok, reason=f"XCoil backend unavailable: {why}")
def test_xcoil_cold_solve_certified():
    """Cold XCoil solve with the PHR coil limits on the PHR nfp2 QA seed boundary: certified, unlinked
    (about 10 s on a laptop CPU)."""
    from phrport import equilibrium as E
    from phrport.coils import CoilConfig, CoilSolver
    r = CoilSolver(CoilConfig()).solve(E.seed_boundary(), warm=None)
    assert r.ok and r.certified and not r.linked
    assert np.isfinite(r.c0_local) and 0 < r.c0_local < 0.05
    assert r.info["v"] <= CoilConfig().viol_tol
    assert 0 < r.info["bn_ratio_int"] < 0.2


@pytest.mark.xcoil
@pytest.mark.skipif(not ok, reason=f"XCoil backend unavailable: {why}")
def test_xcoil_cold_start_and_warm_polish_certification():
    """final_coil_stage on two starts with one polish: the polished coils meet the certified violation target
    v <= 1e-5 (about 30 s on a laptop CPU)."""
    from phrport import equilibrium as E
    from phrport.coils import CoilConfig, CoilSolver
    r = CoilSolver(CoilConfig()).final_coil_stage(E.seed_boundary(), labels=["L85", "L70"], max_polish=1)
    fs = r.info["final_stage"]
    assert fs["v_target_met"] and fs["n_polished"] == 1
    assert r.certified and not r.linked and r.info["v"] <= 1e-5


@pytest.mark.xcoil
@pytest.mark.slow
@pytest.mark.skipif(not ok, reason=f"XCoil backend unavailable: {why}")
def test_xcoil_tiny_coil_set():
    """Smallest coil set (2 coils per half period, order 2): the solver builds and returns a finite result."""
    from phrport import equilibrium as E
    from phrport.coils import CoilConfig, CoilSolver
    cfg = CoilConfig(n_coils=2, order=2, nq=32, phr_grid=(16, 16), dist_grid=(32, 16), max_starts=2)
    r = CoilSolver(cfg).solve(E.seed_boundary(), warm=None)
    assert np.isfinite(r.c0_local) and "cert" in r.info
