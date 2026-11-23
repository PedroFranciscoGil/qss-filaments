"""PHR envelope QSS port (XCoil coils + VMEX equilibria). Sandbox package; originals are read-only.

No user-specific absolute paths: everything derives from the package location or from env vars.

  PACKAGE_ROOT    the repository root holding this package (code location; always from __file__)
  SANDBOX_ROOT    data root (runs_phrport/, analysis_phrport/, phrport_ref/): $PHRPORT_ROOT, else
                  PACKAGE_ROOT
  PYCACHE_PREFIX  sys.pycache_prefix, set here before anything else (XCoil) is imported: $PHRPORT_PYCACHE, else an already-set prefix
                  (PYTHONPYCACHEPREFIX / -X pycache_prefix), else PACKAGE_ROOT/.pycache
  SCRATCH_ROOT    scratch output of tests / verification scripts: $PHRPORT_SCRATCH, else
                  <tempfile.gettempdir()>/phrport_scratch

The VMEC input deck is phrport.equilibrium.INPUT_FILE (the profile's input_file, else $PHRPORT_INPUT, else the
repository's inputs/input.nfp2_QA).
"""
import os as _os
import sys as _sys
from pathlib import Path as _Path


def _env_path(name):
    v = _os.environ.get(name)
    return _Path(v).expanduser() if v else None


PACKAGE_ROOT = _Path(__file__).resolve().parents[1]
SANDBOX_ROOT = (_env_path("PHRPORT_ROOT") or PACKAGE_ROOT).resolve()
PYCACHE_PREFIX = (_os.environ.get("PHRPORT_PYCACHE") or _sys.pycache_prefix
                  or str(PACKAGE_ROOT / ".pycache"))
_sys.pycache_prefix = PYCACHE_PREFIX

import tempfile as _tempfile  # noqa: E402  (after the pycache prefix)

SCRATCH_ROOT = _env_path("PHRPORT_SCRATCH") or _Path(_tempfile.gettempdir()) / "phrport_scratch"


def scratch_dir(*parts, create=True):
    """SCRATCH_ROOT/<parts...> as a str (created unless create=False)."""
    p = SCRATCH_ROOT.joinpath(*parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return str(p)
