"""Coil-backend selection (XCoil).

The coil subproblem is solved by XCoil (a JAX augmented-Lagrangian / Levenberg-Marquardt coil optimizer with exact
certification; github.com/PedroFranciscoGil/XCoil). It is a regular dependency (pyproject.toml pins a commit).
Another tree can be selected before phrport.coils is imported:

  PHRPORT_XCOIL unset                  the installed ``xcoil`` package (default)
  PHRPORT_XCOIL=/path/to/checkout      a local checkout: the folder that CONTAINS the ``xcoil`` package (the root of
                                       an XCoil clone), put first on sys.path

There is no simsopt coil backend: phrport never imports simsopt. XCoil runs on any JAX device; on a machine without a
GPU it runs on the CPU (JAX_PLATFORMS=cpu forces it), precision "mixed" or "fp64" (CoilConfig.precision).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

ENV_VARS = ("PHRPORT_XCOIL",)
INSTALL_HINT = ('pip install "xcoil @ git+https://github.com/PedroFranciscoGil/XCoil@<commit>" (the commit pinned in '
                'pyproject.toml), or set PHRPORT_XCOIL to a local checkout')


class XcoilUnavailable(ImportError):
    """XCoil (the coil backend) cannot be imported."""



@dataclass
class XcoilBackend:
    kind: str          # "installed" | "checkout"
    path: str | None   # folder put first on sys.path (None for "installed")

    def describe(self):
        return {"installed": "installed xcoil package", "checkout": f"local checkout {self.path}"}[self.kind]


def _setting(env):
    if env is not None:
        return env.strip()
    for k in ENV_VARS:
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return ""


def select_xcoil(env=None) -> XcoilBackend:
    """Resolve PHRPORT_XCOIL and put the selected tree first on sys.path. Idempotent."""
    v = _setting(env)
    if v in ("", "installed"):
        b = XcoilBackend("installed", None)
    else:
        p = os.path.abspath(os.path.expanduser(v)).rstrip(os.sep)
        if os.path.basename(p) == "xcoil" and os.path.isfile(os.path.join(p, "__init__.py")):
            p = os.path.dirname(p)                      # the package folder itself was given
        if os.path.isfile(os.path.join(p, "xcoil", "__init__.py")):
            b = XcoilBackend("checkout", p)
        else:
            raise XcoilUnavailable(f"PHRPORT_XCOIL={v!r}: no xcoil package found there; give the folder "
                                   f"that contains xcoil/ (the root of an XCoil clone), or unset it to use the "
                                   f"installed package")
    if b.path is not None:
        if b.path in sys.path:
            sys.path.remove(b.path)
        sys.path.insert(0, b.path)
        pkg = "xcoil"
        loaded = sys.modules.get(pkg)
        if loaded is not None and not os.path.abspath(getattr(loaded, "__file__", "") or "").startswith(b.path):
            raise XcoilUnavailable(f"{pkg} was already imported from {loaded.__file__}, not from {b.path}; set "
                                   f"PHRPORT_XCOIL before importing phrport.coils")
    return b



def xcoil_available() -> tuple[bool, str]:
    """(importable, message) for the selected backend, without raising."""
    try:
        b = select_xcoil()
        import xcoil  # noqa: F401
        from xcoil.fixed import starts  # noqa: F401
        from xcoil.verify import certify  # noqa: F401
        return True, b.describe()
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}. {INSTALL_HINT}"

