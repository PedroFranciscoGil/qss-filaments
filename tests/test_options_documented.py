"""The option dictionary is exhaustive and current, profiles validate, and the profile "run" section maps to flags."""
import glob
import json
import os
import re
import subprocess
import sys

from phrport import cli, options, phr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_option_is_documented():
    assert options.undocumented() == []
    inv = {r["key"] for r in options.inventory()}
    assert not (set(options.DOC) - inv), sorted(set(options.DOC) - inv)     # no stale entries


def test_reference_md_is_current():
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_reference.py"), "--check"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_example_profiles_are_valid():
    flags = cli.run_flags(phr.build_parser())
    for path in glob.glob(os.path.join(ROOT, "examples", "*.json")):
        assert options.validate_profile(json.load(open(path)), run_flags=flags) == [], path


def test_validator_reports_unknown_keys():
    bad = {"phr": {"aspect_targt": 5}, "coils": {"max_forces": 1}, "equilibrium": {"qi_options": {"nalph": 3}},
           "run": {"bogus": 1}, "extra": {}}
    msgs = options.validate_profile(bad, run_flags=cli.run_flags(phr.build_parser()))
    assert len(msgs) == 5


def test_profile_run_section_to_flags():
    parser = phr.build_parser()
    args = cli.profile_run_args({"n_continuation_steps": 5, "no_coils": True, "carry_trust_radius": True,
                                 "full_outer_budget": False, "eq_jacobian": "exact", "_note": "x"}, parser)
    assert args == ["--n-continuation-steps", "5", "--no-coils", "--carry-trust-radius", "--no-full-outer-budget",
                    "--eq-jacobian", "exact"]
    a = parser.parse_args(["--run-dir", "x"] + args + ["--n-continuation-steps", "7"])   # command line wins
    assert a.n_continuation_steps == 7 and a.enable_coils is False and a.eq_jacobian == "exact"


def test_doc_links_resolve():
    """Every relative markdown link in README / docs points at an existing file (and heading, when given)."""
    def anchors(path):
        out = set()
        for line in open(path):
            if line.startswith("#"):
                t = line.lstrip("#").strip().lower()
                out.add("".join(ch for ch in t.replace(" ", "-") if ch.isalnum() or ch in "-_"))
        return out
    files = [os.path.join(ROOT, "README.md")] + glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True)
    bad = []
    for f in files:
        for target in re.findall(r"\]\(([^)\s]+)\)", open(f).read()):
            if target.startswith(("http", "#local_", "mailto")):
                continue
            path, _, anchor = target.partition("#")
            full = os.path.normpath(os.path.join(os.path.dirname(f), path)) if path else f
            if not os.path.exists(full):
                bad.append(f"{os.path.relpath(f, ROOT)} -> {target}")
            elif anchor and full.endswith(".md") and anchor not in anchors(full):
                bad.append(f"{os.path.relpath(f, ROOT)} -> {target} (no heading)")
    assert not bad, bad


def test_profile_run_section_negations():
    parser = phr.build_parser()
    assert cli.profile_run_args({"carry_trust_radius": False}, parser) == ["--no-carry-trust-radius"]
    assert cli.profile_run_args({"no_carry_trust_radius": True}, parser) == ["--no-carry-trust-radius"]
    assert cli.profile_run_args({"enable_coils": False}, parser) == ["--no-coils"]
    a = parser.parse_args(["--run-dir", "x", "--carry-trust-radius"]
                          + cli.profile_run_args({"carry_trust_radius": False}, parser))
    assert a.carry_trust_radius is False                 # the profile undoes the preset's flag


def test_presets_and_undo_forms():
    """production is the preset name ('paper' an alias) and every on/off option it switches on can be undone."""
    assert "production" in cli.PRESETS and cli.PRESET_ALIASES == {"paper": "production"}
    parser = phr.build_parser()
    flags = cli.PRESETS["production"]
    on = [f for f in flags if f.startswith("--") and isinstance(
        next(a for a in parser._actions if f in a.option_strings), (__import__("argparse")._StoreTrueAction,
                                                                   __import__("argparse")._StoreFalseAction))]
    a = parser.parse_args(["--run-dir", "x"] + flags)
    undo = {"--skip-coil-on-rejected": "--no-skip-coil-on-rejected", "--abort-doomed-coil-solves":
            "--no-abort-doomed-coil-solves", "--no-certify-trials": "--certify-trials", "--reuse-vmex-state":
            "--no-reuse-vmex-state", "--carry-trust-radius": "--no-carry-trust-radius",
            "--full-outer-budget": "--no-full-outer-budget"}
    assert set(on) == set(undo)
    b = parser.parse_args(["--run-dir", "x"] + flags + list(undo.values()))
    for f in on:
        dest = next(x for x in parser._actions if f in x.option_strings).dest
        assert getattr(a, dest) != getattr(b, dest), f
    run = {"skip_coil_on_rejected": False, "full_outer_budget": False, "certify_trials": True}
    c = parser.parse_args(["--run-dir", "x"] + flags + cli.profile_run_args(run, parser))
    assert not c.skip_coil_on_rejected and not c.full_outer_budget and c.certify_trials
