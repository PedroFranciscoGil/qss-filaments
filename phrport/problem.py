"""Problem profile: the design a run optimizes (JSON, selected with ``PHRPORT_PROBLEM`` or ``qss-run --profile``).

``PHRPORT_PROBLEM=<profile.json>`` is read once at import (VMEX pool workers inherit it). It overrides the
defaults of :mod:`phrport.equilibrium` (input deck, boundary resolution, nfp, physics, QS helicity / QI
options, seed ellipse, VMEC resolution and radial ladder), of :class:`phrport.phr.PHRConfig` (physics
targets and constraint rows) and of :class:`phrport.coils.CoilConfig` (coil count, order, engineering
limits, ordering rows, flux grid). Without the variable every default is PHR's nfp2 QA problem.

Profile layout (all keys optional; see docs/constraints.md for every key):
  {"name": "LP_QA",
   "equilibrium": {"input_file", "physics", "surf_mpol", "surf_ntor", "nfp", "helicity_m", "helicity_n",
                   "seed_ellipse", "vmec_mpol", "ns_array", "ftol_array", "niter_array", "qi_options", ...},
   "phr": {PHRConfig field: value, ...},
   "coils": {CoilConfig field: value, ...}}

Paths: ``equilibrium.input_file`` may be absolute, relative to the profile's own folder, relative to the
repository root, or start with ``__ROOT__/`` (the repository root). Relative names are resolved here,
so the rest of the code always sees an absolute path.
"""
import json
import os
from pathlib import Path

PATH = os.environ.get("PHRPORT_PROBLEM") or None
DATA = json.loads(Path(PATH).read_text()) if PATH else {}
NAME = DATA.get("name", "nfp2_QA")
ACTIVE = bool(DATA)
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value, base=None):
    """Absolute path for a profile path value (see the module docstring); unchanged if nothing matches."""
    if not value:
        return value
    s = str(value)
    if s.startswith("__ROOT__"):
        return str(REPO_ROOT / s[len("__ROOT__"):].lstrip("/\\"))
    p = Path(s).expanduser()
    if p.is_absolute():
        return str(p)
    for root in ([Path(base)] if base else []) + [REPO_ROOT, REPO_ROOT / "inputs"]:
        q = root / p
        if q.exists():
            return str(q.resolve())
    return str(p)


if PATH and DATA.get("equilibrium", {}).get("input_file"):
    DATA["equilibrium"]["input_file"] = resolve_path(DATA["equilibrium"]["input_file"],
                                                     Path(PATH).resolve().parent)


def get(section, key, default):
    """Profile value section.key, else ``default``."""
    return DATA.get(section, {}).get(key, default)
