"""Seed decks (qss-seed) and seed coils (coils.seed_coils)."""
import json
import os

import numpy as np
import pytest

from phrport import seed as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deck_boundary(path):
    from vmex import VmecInput
    inp = VmecInput.from_file(path)
    nt = int(inp.ntor)
    out = {}
    for arr, t in ((np.asarray(inp.rbc), "rc"), (np.asarray(inp.zbs), "zs")):
        for i, j in zip(*np.nonzero(np.abs(arr) > 0)):
            out[f"{t}({j},{i - nt})"] = float(arr[i, j])
    return out, int(inp.nfp)


EXPECTED = {
    "qa": {"rc(0,0)": 1.0, "rc(1,0)": 1 / 6, "zs(1,0)": 1 / 6, "rc(1,1)": 1e-3, "zs(1,1)": 1e-3, "rc(2,1)": 1e-3,
           "zs(2,1)": 1e-3},
    "qi": {"rc(0,0)": 1.0, "rc(1,0)": 0.1, "zs(1,0)": 0.1, "rc(1,1)": -2e-3, "zs(1,1)": 2e-3, "rc(0,1)": 2e-3,
           "zs(0,1)": -2e-3},
}


@pytest.mark.parametrize("brk,deck,nfp,aspect", [("qa", "input.tokamak_nfp2", 2, 6.0),
                                                 ("qi", "input.tokamak_nfp5", 5, 10.0)])
def test_tokamak_seed_decks(tmp_path, brk, deck, nfp, aspect):
    """qss-seed tokamak gives the documented boundary, and the shipped example decks are its output."""
    out = S.write_deck(S.tokamak_boundary(aspect=aspect, brk=brk), str(tmp_path / "d"), nfp=nfp)
    got, nfp_got = _deck_boundary(out)
    assert nfp_got == nfp and set(got) == set(EXPECTED[brk])
    assert all(abs(got[k] - EXPECTED[brk][k]) < 1e-12 for k in got)
    ref, nfp_ref = _deck_boundary(os.path.join(ROOT, "inputs", deck))
    assert nfp_ref == nfp and ref == pytest.approx(got)


def test_tokamak_shapes_and_round_trip(tmp_path):
    b = S.tokamak_boundary(aspect=6, kappa=1.3, delta=0.3, brk="none")
    assert all(k.endswith(",0)") for k in b)                 # axisymmetric without a break
    assert b["zs(1,0)"] > b["rc(1,0)"]                         # elongated
    bj = tmp_path / "b.json"
    json.dump(S.rotating_ellipse_boundary(aspect=8, amp=0.1), open(bj, "w"))
    deck = S.write_deck(S.boundary_from_json(str(bj)), str(tmp_path / "re"))
    got, _ = _deck_boundary(deck)
    assert got["rc(1,1)"] == pytest.approx(-0.1 / 8) and got["zs(1,1)"] == pytest.approx(0.1 / 8)
    with pytest.raises(ValueError):                            # a checkpoint needs from-checkpoint
        json.dump({"surface_dofs": [1.0], "step": 0}, open(tmp_path / "ck.json", "w"))
        S.boundary_from_json(str(tmp_path / "ck.json"))


def test_qss_seed_cli(tmp_path):
    assert S.main(["tokamak", "--aspect", "6", "--break", "qa", "--out", str(tmp_path / "t")]) == 0
    assert "written by qss-seed" in open(tmp_path / "t").read()


ok_fc = __import__("phrport.backend", fromlist=["x"]).xcoil_available()[0]


@pytest.mark.xcoil
@pytest.mark.skipif(not ok_fc, reason="XCoil unavailable")
def test_seed_coils_from_export_and_points(tmp_path):
    """A run's coil export (JSON, NPZ, other order) and a points JSON all seed the first cold solve."""
    from phrport import coils as C
    from phrport import equilibrium as E
    from phrport.phr import export_coils
    b = E.seed_boundary()
    cs = C.CoilSolver(C.CoilConfig())
    r = cs.solve(b)
    path = export_coils(str(tmp_path / "c"), cs, r, b)
    f = json.load(open(path))
    cb = C.fr.make_basis("fourier", 64, int(f["order"]))
    P = np.asarray(C.fr.coils_curves(C.jnp.asarray(np.ravel(f["base_curve_dofs"])), cb, 4)[0])
    pts = tmp_path / "pts.json"
    json.dump({"points": P.tolist(), "currents_A": f["base_currents_A"]}, open(pts, "w"))
    for seed, order in ((path, 5), (path.replace(".json", ".npz"), 5), (str(pts), 6)):
        C._SEED_USED[0] = False
        r2 = C.CoilSolver(C.CoilConfig(seed_coils=seed, order=order)).solve(b)
        assert r2.info["start"] == "seed" and r2.c0_local == pytest.approx(r.c0_local, rel=2e-2)
    C._SEED_USED[0] = False
    with pytest.raises(Exception):
        C.CoilSolver(C.CoilConfig(seed_coils=path, n_coils=5)).seed_start()
