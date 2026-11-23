"""pytest setup: x64 JAX on the CPU, the repository root and the stub components (tests/stubs.py) on sys.path.

Markers: ``slow`` (VMEX / XCoil solves, minutes on a laptop CPU; skipped unless QSS_RUN_SLOW=1 or -m slow)
and ``xcoil`` (needs the XCoil backend; skipped when it cannot be imported).
"""
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.pop("PHRPORT_PROBLEM", None)          # unit tests use the built-in (PHR nfp2 QA) defaults

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402


def pytest_collection_modifyitems(config, items):
    run_slow = os.environ.get("QSS_RUN_SLOW", "0") not in ("", "0") or "slow" in (config.getoption("-m") or "")
    if run_slow:
        return
    skip = pytest.mark.skip(reason="slow test: set QSS_RUN_SLOW=1 (or pytest -m slow) to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
