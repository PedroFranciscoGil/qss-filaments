"""Roadmap A4 / G1: collect front points and existing finals into one branch-aware QS / flux front.

Inputs (read-only):
  --sweeps DIR ...   sweep run directories (PHRRunner.sweep / python -m phrport.phr --sweep): every
                     front_point_*.json (front_point_seed.json included);
  --finals DIR ...   ordinary run directories: final_metrics.json + intermediate_state.json['boundary'].

For every point (duplicates with bitwise-identical boundaries merged):
  * QS is RE-EVALUATED at mpol = ntor = 8, 22 x 20, ladder ns101 (phrport.equilibrium.evaluate_resolution
    (b, 8, 22, 20, "ns101"); roadmap G3 "report QS at mpol 8"), together with the constraint values there;
  * flux = <|B.n|>/<|B|> as a ratio of area integrals (coils.py info['bn_ratio_int'], the
    definition), from the R1 final coil stage: the converged-grid value when present, else 50x50.  Finals
    run without the R1 stage have no bn_ratio_int: --recompute-coils runs CoilSolver.final_coil_stage on
    their boundary (about 1 min each), otherwise they are listed but left out of the front.
Then:
  * feasible  = max PHR-resolution constraint value g <= --feas-tol (g at mpol 8 is reported);
  * front     = the non-dominated feasible points in (QS_m8, flux), both minimised;
  * clusters  = single-linkage groups at --cluster-dist in the boundary L2 distance, made invariant
                under the half-period shift (phi -> phi + pi/nfp flips every odd-n coefficient):
                d(a, b) = min(|a - b|, |a - flip(b)|) (the rigor addendum's canonicalisation; clusters
                there were >= 0.05 apart with members within 0.038); clusters are labelled A, B, ... by
                increasing best QS, and a sweep's points form its branch.

Outputs (--out-dir, default runs/analysis/front): front_collect.json (all points, front, clusters,
pairwise cluster distances, per-branch counts: points on the pooled front / feasible points = the A4
"at least half non-dominated" check) and front_L.png (log-log flux vs QS_m8, the "L-shape" figure: sweep
branches drawn as labelled lines with k on every point, run finals labelled, the front as a staircase).
The mpol-8 solves are cached in --cache (default <out-dir>/front_m8_cache.json) by boundary hash.

Run (VMEX venv; ~10 s per mpol-8 solve):
  nice -n 10 python tools/front_collect.py \\
      --sweeps runs/sweep_A --finals runs/final_A \\
      --out-dir runs/analysis/front
--no-reeval uses the PHR-resolution QS instead (no VMEX; for quick looks and tests).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = os.environ.get("PHRPORT_PYCACHE") or sys.pycache_prefix or str(ROOT / ".pycache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

M8 = dict(mpol=8, ntheta=22, nzeta=20, ns_ladder="ns101")
TARGETS = dict(aspect=4.5, iota=0.43, elong=4.0)
_NAME = re.compile(r"^(rc|zs)\((-?\d+),(-?\d+)\)$")


# ------------------------------------------------------------------------------------------- loading

def _branch_name(d):
    """Branch label of a run / sweep dir; a standalone run's in-mode ramp (<run>/ramp) is labelled <run>."""
    p = os.path.normpath(d)
    return os.path.basename(os.path.dirname(p)) if os.path.basename(p) == "ramp" else os.path.basename(p)

def _load(p):
    with open(p) as f:
        return json.load(f)


def boundary_key(b):
    """Hash of a boundary (sorted names, float64 bits)."""
    h = hashlib.sha1()
    for k in sorted(b):
        h.update(k.encode())
        h.update(np.float64(b[k]).tobytes())
    return h.hexdigest()[:16]


def _flux_from_coils(c):
    """bn_ratio_int from a coils dict (front point 'coils' / final_metrics 'coils'): converged grid first."""
    if not c:
        return None, None
    for key, src in (("bn_ratio_int_converged", "converged_grid"), ("bn_ratio_int", "grid_50x50")):
        v = c.get(key)
        if v is not None and np.isfinite(v):
            return float(v), src
    fm = c.get("flux_metrics") or {}
    for src in ("converged_grid", "grid_50x50"):
        v = (fm.get(src) or {}).get("bn_ratio_int")
        if v is not None and np.isfinite(v):
            return float(v), src
    return None, None


def load_sweep(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "front_point_*.json"))):
        if p.endswith("_coils.json"):
            continue
        r = _load(p)
        if r.get("kind") != "front_point":
            continue
        flux, fsrc = _flux_from_coils(r.get("coils"))
        name = _branch_name(d)
        out.append(dict(label=f"{name}:{'seed' if r.get('is_seed') else 'k=%d' % r['k']}", kind="sweep",
                        source=p, branch=name, k=r.get("k"), is_seed=bool(r.get("is_seed")),
                        boundary=r["boundary"], qs_phr=r.get("qs_phr"), g_phr=r.get("g_eq"),
                        Jf=r.get("Jf"), flux=flux, flux_source=fsrc, eps=r.get("eps"),
                        target_met=r.get("target_met"), stationarity=(r.get("kkt") or {}).get("stationarity"),
                        lambda_c=r.get("lambda_c")))
    return out


def load_final(d):
    fm = _load(os.path.join(d, "final_metrics.json"))
    st = _load(os.path.join(d, "intermediate_state.json"))
    s, c = fm.get("surface", {}), fm.get("coils", {})
    flux, fsrc = _flux_from_coils(c)
    fs = c.get("final_stage") or {}
    kk = (fm.get("optimization") or {}).get("kkt_final") or {}
    name = _branch_name(d)
    return [dict(label=name, kind="final", source=d, branch=name, k=None, is_seed=False, boundary=st["boundary"],
                 qs_phr=s.get("qs_residual_sum_sq"), g_phr=s.get("g_eq"),
                 Jf=(fs.get("Jf_polished") if fs.get("Jf_polished") is not None else c.get("squared_flux_Jf")),
                 flux=flux, flux_source=fsrc, eps=None, target_met=None, stationarity=kk.get("stationarity"),
                 lambda_c=((fm.get("optimization") or {}).get("lam_coil_final") or [None])[0])]


def merge_duplicates(points):
    """Bitwise-identical boundaries -> one point (the first keeps its fields; labels are joined)."""
    seen, out = {}, []
    for p in points:
        key = boundary_key(p["boundary"])
        p["bkey"] = key
        if key in seen:
            q = seen[key]
            q["label"] += " = " + p["label"]
            for f in ("flux", "flux_source", "Jf", "stationarity"):
                if q.get(f) is None and p.get(f) is not None:
                    q[f] = p[f]
            continue
        seen[key] = p
        out.append(p)
    return out


# ---------------------------------------------------------------------------------------- evaluation
def violation(g):
    """max(0, g_i) over the PHR inequality constraints (dict label -> g)."""
    if not g:
        return None
    vals = [float(v) for v in g.values() if v is not None]
    return max([0.0] + vals) if vals and all(np.isfinite(vals)) else None


def m8_metrics(eq):
    if not bool(eq.converged):
        return dict(converged=False, qs=None, g=None, message=getattr(eq, "message", ""))
    g = dict(aspect=float(eq.aspect) - TARGETS["aspect"], iota_mean_hi=float(eq.mean_iota) - TARGETS["iota"],
             iota_mean_lo=TARGETS["iota"] - float(eq.mean_iota), elong=float(eq.elongation) - TARGETS["elong"])
    return dict(converged=True, qs=float(eq.qs_total), g=g, aspect=float(eq.aspect),
                mean_iota=float(eq.mean_iota), elongation=float(eq.elongation))


def reevaluate(points, evaluate_fn=None, cache_path=None, log=print):
    """QS (and constraints) at mpol 8 / 22x20 / ns101 for every point; evaluate_fn(boundary) -> EqResult
    (default phrport.equilibrium.evaluate_resolution(b, 8, 22, 20, 'ns101')); cached by boundary key."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = _load(cache_path)
    if evaluate_fn is None:
        from phrport import equilibrium as eqmod

        def evaluate_fn(b):
            return eqmod.evaluate_resolution(b, M8["mpol"], M8["ntheta"], M8["nzeta"], M8["ns_ladder"])
    for i, p in enumerate(points):
        key = p["bkey"]
        if key not in cache:
            log(f"  [{i + 1}/{len(points)}] mpol-8 solve: {p['label']}")
            cache[key] = m8_metrics(evaluate_fn(p["boundary"]))
            if cache_path:
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=1)
        m = cache[key]
        p["m8"] = m
        p["qs_m8"] = m.get("qs")
        p["viol_m8"] = violation(m.get("g"))
    return points


def recompute_coils(points, log=print):
    """R1 final coil stage for points without a flux value (CoilSolver.final_coil_stage)."""
    todo = [p for p in points if p.get("flux") is None]
    if not todo:
        return points
    from phrport.coils import CoilSolver, CoilConfig
    cs = CoilSolver(CoilConfig())
    for p in todo:
        log(f"  final coil stage: {p['label']}")
        r = cs.final_coil_stage(p["boundary"])
        info = getattr(r, "info", None) or {}
        fs = info.get("final_stage") or {}
        fu = fs.get("flux_unfolded") or {}
        if fu.get("bn_ratio_int") is not None:
            p["flux"], p["flux_source"] = float(fu["bn_ratio_int"]), "converged_grid (recomputed)"
        elif info.get("bn_ratio_int") is not None:
            p["flux"], p["flux_source"] = float(info["bn_ratio_int"]), "grid_50x50 (recomputed)"
        p["Jf"] = float(r.c0_local)
    return points


# ---------------------------------------------------------------------------------- front / clusters
def non_dominated(points, xkey="qs", ykey="flux"):
    """Indices of the points not dominated in (x, y) (both minimised; a point is dominated when another is
    <= in both and < in one).  Points with a missing / non-finite value are skipped."""
    ok = [i for i, p in enumerate(points)
          if p.get(xkey) is not None and p.get(ykey) is not None
          and np.isfinite(p[xkey]) and np.isfinite(p[ykey])]
    out = []
    for i in ok:
        xi, yi = points[i][xkey], points[i][ykey]
        dom = any((points[j][xkey] <= xi and points[j][ykey] <= yi
                   and (points[j][xkey] < xi or points[j][ykey] < yi)) for j in ok if j != i)
        if not dom:
            out.append(i)
    return sorted(out, key=lambda i: points[i][xkey])


def flip_half_period(b):
    """phi -> phi + pi/nfp: every odd-n coefficient changes sign."""
    out = {}
    for k, v in b.items():
        m = _NAME.match(k)
        out[k] = -float(v) if (m and int(m.group(3)) % 2) else float(v)
    return out


def boundary_distance(a, b):
    """min(|a - b|, |a - flip(b)|) over the union of coefficient names (missing = 0)."""
    keys = sorted(set(a) | set(b))
    va = np.array([float(a.get(k, 0.0)) for k in keys])
    vb = np.array([float(b.get(k, 0.0)) for k in keys])
    fb = flip_half_period(b)
    vf = np.array([float(fb.get(k, 0.0)) for k in keys])
    return float(min(np.linalg.norm(va - vb), np.linalg.norm(va - vf)))


def cluster(points, dist):
    """Single-linkage clusters at threshold dist -> (labels list, distance matrix)."""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = boundary_distance(points[i]["boundary"], points[j]["boundary"])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= dist:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    def best(g):
        qs = [points[i].get("qs") for i in g if points[i].get("qs") is not None]
        return min(qs) if qs else math.inf
    order = sorted(groups.values(), key=lambda g: (best(g), min(g)))
    labels = [None] * n
    for c, g in enumerate(order):
        name = chr(ord("A") + c) if c < 26 else f"C{c}"
        for i in g:
            labels[i] = name
    return labels, D


# ------------------------------------------------------------------------------------------- figure
def figure(points, front_idx, path, qs_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    clusters = sorted({p["cluster"] for p in points})
    cmap = plt.get_cmap("tab10")
    col = {c: cmap(i % 10) for i, c in enumerate(clusters)}
    for c in clusters:
        for kind, mk in (("sweep", "o"), ("final", "s")):
            ps = [p for p in points if p["cluster"] == c and p["kind"] == kind and p.get("qs") and p.get("flux")]
            if not ps:
                continue
            ax.scatter([p["qs"] for p in ps], [p["flux"] for p in ps], marker=mk, s=42, color=col[c],
                       edgecolor=["k" if p["feasible"] else "r" for p in ps], linewidth=1.0,
                       label=f"cluster {c} ({kind})", zorder=3)
    branches = sorted({p["branch"] for p in points if p["kind"] == "sweep"})
    for ib, br in enumerate(branches):
        ps = sorted([p for p in points if p["branch"] == br and p["kind"] == "sweep" and p.get("qs") and p.get("flux")],
                    key=lambda p: (p["k"] if p["k"] is not None else -1))
        if not ps:
            continue
        ls = ("-", "-.", ":", (0, (5, 2)))[ib % 4]
        if len(ps) > 1:
            ax.plot([p["qs"] for p in ps], [p["flux"] for p in ps], linestyle=ls, color="0.45", lw=0.9, zorder=1,
                    label=f"branch {br} (sweep, k = {ps[0]['k']}..{ps[-1]['k']})")
        # branch label at its last (lowest-target) point; k on every sweep point
        ax.annotate(br, (ps[-1]["qs"], ps[-1]["flux"]), fontsize=7, fontweight="bold", xytext=(4, -9),
                    textcoords="offset points", color="0.2")
        for p in ps:
            ax.annotate("seed" if p.get("is_seed") else f"k{p['k']}", (p["qs"], p["flux"]), fontsize=5,
                        xytext=(-3, 4), textcoords="offset points", ha="right", color="0.35")
    fr = [points[i] for i in front_idx]
    if fr:
        xs, ys = [fr[0]["qs"]], [fr[0]["flux"]]
        for p in fr[1:]:
            xs += [p["qs"], p["qs"]]
            ys += [ys[-1], p["flux"]]
        ax.plot(xs, ys, "k--", lw=1.0, zorder=2, label="non-dominated front")
    for p in points:
        if p.get("qs") and p.get("flux") and p["kind"] == "final":
            ax.annotate(p["label"].split(" = ")[0], (p["qs"], p["flux"]), fontsize=6, xytext=(3, 2),
                        textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(qs_label)
    ax.set_ylabel("<|B.n|>/<|B|>  (ratio of area integrals, final coil stage)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title("QS / flux front (red edge: infeasible; circles: sweep points, squares: run finals)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- main
def collect(sweeps=(), finals=(), *, out_dir, reeval=True, evaluate_fn=None, cache_path=None,
            recompute=False, feas_tol=1e-3, cluster_dist=0.04, make_figure=True, log=print):
    pts = []
    for d in sweeps:
        got = load_sweep(d)
        log(f"  sweep {d}: {len(got)} point(s)")
        pts += got
    for d in finals:
        pts += load_final(d)
    pts = merge_duplicates(pts)
    os.makedirs(out_dir, exist_ok=True)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    if recompute:
        recompute_coils(pts, log=log)
    if reeval:
        reevaluate(pts, evaluate_fn=evaluate_fn, cache_path=cache_path, log=log)
        qs_key, qs_label = "qs_m8", "QS residual at mpol 8 / 22x20 / ns101"
    else:
        for p in pts:
            p["qs_m8"], p["viol_m8"] = None, None
        qs_key, qs_label = "qs_phr", "QS residual at the PHR resolution (no mpol-8 re-evaluation)"
    for p in pts:
        p["qs"] = p.get(qs_key)
        p["viol_phr"] = violation(p.get("g_phr"))
        p["feasible"] = bool(p["viol_phr"] is not None and p["viol_phr"] <= feas_tol
                             and (not reeval or p.get("qs_m8") is not None))
        p["missing_flux"] = p.get("flux") is None
    labels, D = cluster(pts, cluster_dist)
    for p, c in zip(pts, labels):
        p["cluster"] = c
    feas = [p for p in pts if p["feasible"]]
    front_local = non_dominated(feas, "qs", "flux")
    front = [feas[i] for i in front_local]
    front_idx = [pts.index(p) for p in front]
    for p in pts:
        p["on_front"] = p in front
    cl_names = sorted(set(labels))
    cl_dist = {}
    for a in cl_names:
        for b in cl_names:
            if a < b:
                ia = [i for i, c in enumerate(labels) if c == a]
                ib = [i for i, c in enumerate(labels) if c == b]
                cl_dist[f"{a}-{b}"] = float(min(D[i, j] for i in ia for j in ib))
    branches = {}
    for br in sorted({p["branch"] for p in pts}):
        bp = [p for p in pts if p["branch"] == br]
        bq = [p["qs"] for p in bp if p.get("qs") is not None]
        nf = sum(1 for p in bp if p["feasible"])
        branches[br] = dict(kind=bp[0]["kind"], n_points=len(bp), n_feasible=nf,
                            n_on_front=sum(1 for p in bp if p["on_front"]),
                            frac_non_dominated=(sum(1 for p in bp if p["on_front"]) / nf if nf else None),
                            clusters=sorted({p["cluster"] for p in bp}),
                            qs_range=([min(bq), max(bq)] if bq else None))
    os.makedirs(out_dir, exist_ok=True)
    res = dict(qs_metric=qs_label, flux_metric="bn_ratio_int (ratio of area integrals of |B.n| and |B|)",
               feas_tol=feas_tol, cluster_dist=cluster_dist, sweeps=list(sweeps), finals=list(finals),
               n_points=len(pts), n_feasible=len(feas), n_missing_flux=sum(p["missing_flux"] for p in pts),
               front=[dict(label=p["label"], qs=p["qs"], flux=p["flux"], Jf=p["Jf"], cluster=p["cluster"],
                           kind=p["kind"]) for p in front],
               clusters={c: dict(members=[p["label"] for p in pts if p["cluster"] == c],
                                 n_on_front=sum(1 for p in front if p["cluster"] == c),
                                 best_qs=min([p["qs"] for p in pts if p["cluster"] == c and p["qs"] is not None]
                                             or [None], key=lambda v: math.inf if v is None else v))
                         for c in cl_names},
               cluster_min_distance=cl_dist, branches=branches,
               points=[{k: v for k, v in p.items()} for p in pts])
    with open(os.path.join(out_dir, "front_collect.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    if make_figure:
        figure(pts, front_idx, os.path.join(out_dir, "front_L.png"), qs_label)
    log(f"  {len(pts)} points ({len(feas)} feasible, {res['n_missing_flux']} without a flux value), "
        f"{len(front)} on the front, clusters {cl_names}")
    for p in front:
        log(f"    front: {p['label']:40s} QS={p['qs']:.3e} flux={p['flux']:.4e} cluster {p['cluster']}")
    for br, b in branches.items():
        log(f"    branch {br}: {b['n_on_front']}/{b['n_feasible']} feasible point(s) on the front "
            f"({b['n_points']} total, clusters {b['clusters']})")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweeps", nargs="*", default=[])
    ap.add_argument("--finals", nargs="*", default=[])
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "analysis" / "front"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--no-reeval", dest="reeval", action="store_false", default=True)
    ap.add_argument("--recompute-coils", action="store_true", default=False)
    ap.add_argument("--feas-tol", type=float, default=1e-3)
    ap.add_argument("--cluster-dist", type=float, default=0.04)
    ap.add_argument("--no-figure", dest="figure", action="store_false", default=True)
    a = ap.parse_args(argv)
    if not a.sweeps and not a.finals:
        ap.error("give --sweeps and / or --finals")
    cache = a.cache or os.path.join(a.out_dir, "front_m8_cache.json")
    os.makedirs(a.out_dir, exist_ok=True)
    collect(a.sweeps, a.finals, out_dir=a.out_dir, reeval=a.reeval, cache_path=cache,
            recompute=a.recompute_coils, feas_tol=a.feas_tol, cluster_dist=a.cluster_dist,
            make_figure=a.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
