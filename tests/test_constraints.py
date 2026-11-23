"""Every equilibrium constraint row: profile/config key -> label, value (g <= 0 feasible) and gradient key."""
from dataclasses import dataclass, field

import numpy as np
import pytest

from phrport import phr as P


@dataclass
class Eq:
    aspect: float = 6.0
    mean_iota: float = 0.42
    iota_axis: float = 0.40
    iota_edge: float = 0.45
    elongation: float = 3.0
    well: float = 0.02
    mirror: float = 0.20
    maxj: float = -0.1
    bmaxsp: float = -0.01
    volume: float = 2.0
    aminor: float = 0.17
    merc: float = 0.05
    qs_total: float = 1e-4
    qs_residuals: np.ndarray = field(default_factory=lambda: np.zeros(3))


ALL_ON = dict(aspect_two_sided=True, aspect_target=6.0, aspect_tol=0.1, iota_mean_target=0.42, iota_tol=0.01,
              elongation_threshold=4.0, vacuum_well=True, well_threshold=0.01, mirror_max=0.25, mirror_min=0.1,
              maxj=True, bmaxsp=True, iota_axis_target=0.4, iota_edge_target=0.45,
              iota_end_tol=0.01, volume_target=2.0, aminor_target=0.17, mercier=True, mercier_margin=0.0,
              mercier_scale=0.1, qs_max=2e-4)

EXPECTED = ["aspect", "aspect_lo", "iota_mean_hi", "iota_mean_lo", "elong", "well", "mirror", "mirror_lo", "maxj",
            "bmaxsp", "iota_axis_hi", "iota_axis_lo", "iota_edge_hi", "iota_edge_lo",
            "volume_hi", "volume_lo", "aminor_hi", "aminor_lo", "merc", "qs_cap"]


def test_default_rows_are_phr():
    cfg = P.PHRConfig()
    assert P.eq_constraint_labels(cfg) == ["aspect", "iota_mean_hi", "iota_mean_lo", "elong"]


def test_all_rows_labels_and_values():
    cfg = P.PHRConfig(**ALL_ON)
    labels = P.eq_constraint_labels(cfg)
    assert labels == EXPECTED
    g = P.eq_constraint_values(Eq(), cfg)
    assert g.shape == (len(labels),)
    assert np.all(np.isfinite(g))
    v = dict(zip(labels, g))
    assert v["aspect"] == pytest.approx(6.0 - 6.1) and v["aspect_lo"] == pytest.approx(5.9 - 6.0)
    assert v["well"] == pytest.approx(0.01 - 0.02)
    assert v["mirror"] == pytest.approx(0.20 - 0.25) and v["mirror_lo"] == pytest.approx(0.1 - 0.20)
    assert v["merc"] == pytest.approx((0.0 - 0.05) / 0.1)
    assert v["qs_cap"] == pytest.approx(1e-4 / 2e-4 - 1.0)
    assert v["volume_hi"] == pytest.approx(1.0 - (1.0 + cfg.volume_tol))
    assert all(x <= 0 for x in g), "the test point is feasible for every row"


def test_one_sided_aspect_and_violations():
    cfg = P.PHRConfig(aspect_target=5.0, elongation_threshold=2.0)
    g = dict(zip(P.eq_constraint_labels(cfg), P.eq_constraint_values(Eq(), cfg)))
    assert g["aspect"] == pytest.approx(1.0) and g["elong"] == pytest.approx(1.0)


def test_every_row_has_a_gradient_key():
    cfg = P.PHRConfig(**ALL_ON)
    jac = {k: np.ones(5) for k in ("aspect", "mean_iota", "elongation", "well", "mirror", "maxj", "bmaxsp", 
                                   "iota_axis", "iota_edge", "volume", "aminor", "merc", "qs_total")}
    dg = P.exact_constraint_grads(jac, cfg)
    missing = [k for k in P.eq_constraint_labels(cfg) if k not in dg]
    assert not missing, missing
    assert np.allclose(dg["merc"], -1.0 / cfg.mercier_scale)
    assert np.allclose(dg["mirror_lo"], -1.0) and np.allclose(dg["aspect_lo"], -1.0)


def test_maxj_schedule_holds_row_inactive():
    cfg = P.PHRConfig(maxj=True, maxj_from_step=2, aspect_target=10.0)
    lab = P.eq_constraint_labels(cfg)
    P._MAXJ_STEP[0] = 0
    try:
        assert dict(zip(lab, P.eq_constraint_values(Eq(maxj=5.0), cfg)))["maxj"] == -1.0
        P._MAXJ_STEP[0] = 2
        assert dict(zip(lab, P.eq_constraint_values(Eq(maxj=5.0), cfg)))["maxj"] == 5.0
    finally:
        P._MAXJ_STEP[0] = None


def test_resume_multiplier_alignment():
    labels = ["aspect", "iota_mean_hi", "iota_mean_lo", "elong", "merc"]
    lam, rho = P.align_eq_multipliers(dict(lam_eq=[1, 2, 3, 4], rho_eq=[5, 5, 5, 5]), labels, 1.0)
    assert list(lam) == [1, 2, 3, 4, 0] and list(rho) == [5, 5, 5, 5, 1.0]
    lam, _ = P.align_eq_multipliers(dict(lam_eq=[1, 2], rho_eq=[1, 1], eq_labels=["elong", "aspect"]), labels, 1.0)
    assert list(lam) == [2, 0, 0, 1, 0]
    with pytest.raises(RuntimeError):
        P.align_eq_multipliers(dict(lam_eq=[1, 2], rho_eq=[1, 1]), labels, 1.0)


def test_coil_config_has_every_coil_limit():
    from phrport.coils import CoilConfig
    f = CoilConfig.__dataclass_fields__
    for k in ("per_coil_length", "length_mode", "total_length", "max_curvature", "lp_curvature_max", "lp_curvature_p",
              "lp_curvature_threshold", "max_msc", "min_cc", "min_cs", "slot_rows", "slot_gap_deg", "link_guard",
              "total_current", "free_currents", "max_force", "certify", "ok_viol"):
        assert k in f, k


def test_flux_bound_options():
    f = P.PHRConfig.__dataclass_fields__
    for k in ("flux_target_frac", "flux_phr_target", "flux_target_floor", "final_mode_flux_target", "flux_target_bn",
              "flux_metric_scale", "cs_threshold"):
        assert k in f, k


def test_checkpoint_with_unknown_rows_loads_with_warning():
    labels = ["aspect", "iota_mean_hi", "iota_mean_lo", "elong"]
    logs = []
    lam, _ = P.align_eq_multipliers(dict(lam_eq=[1, 2, 3, 4, 5], rho_eq=[1] * 5,
                                         eq_labels=["aspect", "iota_mean_hi", "iota_mean_lo", "elong", "extra_row"]),
                                    labels, 1.0, logs.append)
    assert list(lam) == [1, 2, 3, 4]
    assert any("not in this problem" in m for m in logs)
