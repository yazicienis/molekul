#!/usr/bin/env python3
"""
scripts/status.py — MOLEKUL project status summary.

Reads outputs/logs/*.json and the test suite, then prints:
  - Implemented phases and their validation status
  - Latest benchmark / optimisation results
  - Available output artifacts

Usage
-----
  python scripts/status.py
  python scripts/status.py --json        # machine-readable output
  python scripts/status.py --run-tests   # also runs pytest

No scientific computation is performed here; all data come from previously
written logs and the pytest result cache.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO / "outputs" / "logs"
OUTPUTS_DIR = REPO / "outputs"
CHECKPOINTS_DIR = REPO / "outputs" / "checkpoints"

SEP  = "─" * 68
SEP2 = "=" * 68


# ---------------------------------------------------------------------------
# Phase registry — ground truth of what the project is supposed to contain
# ---------------------------------------------------------------------------

PHASES: List[Dict[str, Any]] = [
    {
        "id":          "phase1",
        "name":        "XYZ I/O & Molecule",
        "modules":     ["io_xyz", "molecule", "atoms", "constants"],
        "log_key":     "phase1_molecules",
        "script":      "scripts/run_example.py",
        "test_files":  ["tests/test_io_xyz.py", "tests/test_molecule.py",
                        "tests/test_atoms.py"],
        "description": "XYZ file reader/writer, Molecule dataclass, nuclear repulsion",
    },
    {
        "id":          "phase2",
        "name":        "One-Electron Integrals",
        "modules":     ["integrals", "basis", "basis_sto3g"],
        "log_key":     "phase2_integrals",
        "script":      "scripts/validate_phase2.py",
        "test_files":  ["tests/test_integrals.py"],
        "description": "Overlap S, kinetic T, nuclear V, H_core — McMurchie-Davidson",
    },
    {
        "id":          "phase3",
        "name":        "Two-Electron ERIs",
        "modules":     ["eri"],
        "log_key":     None,   # covered by test suite only
        "script":      "scripts/compare_pyscf.py",
        "test_files":  ["tests/test_eri.py"],
        "description": "4-centre ERIs (ab|cd), McMurchie-Davidson, 8-fold symmetry",
    },
    {
        "id":          "phase4",
        "name":        "RHF SCF",
        "modules":     ["rhf"],
        "log_key":     "phase4_rhf",
        "script":      "scripts/validate_rhf.py",
        "test_files":  ["tests/test_rhf.py"],
        "description": "Restricted Hartree-Fock, DIIS, STO-3G, H2/HeH+/H2O",
    },
    {
        "id":          "phase5",
        "name":        "Geometry Optimisation",
        "modules":     ["grad", "optimizer"],
        "log_key":     "phase5_optimizer",
        "script":      "scripts/validate_optimizer.py",
        "test_files":  ["tests/test_optimizer.py"],
        "description": "Numerical gradient, BFGS, XYZ trajectory, JSON history",
    },
    {
        "id":          "phase6",
        "name":        "CUBE Export",
        "modules":     ["cube", "geom"],
        "log_key":     "phase6_cube",
        "script":      "scripts/export_cube_h2o.py",
        "test_files":  ["tests/test_cube.py"],
        "description": "Electron density & MO amplitude on 3D grid, Gaussian CUBE format",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_log(key: str) -> Optional[Dict[str, Any]]:
    path = LOGS_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _module_exists(name: str) -> bool:
    return (REPO / "src" / "molekul" / f"{name}.py").exists()


def _test_file_exists(rel: str) -> bool:
    return (REPO / rel).exists()


def _artifact_exists(rel: str) -> bool:
    return (REPO / rel).exists()


def _git_info() -> Dict[str, str]:
    info: Dict[str, str] = {}
    try:
        info["sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip())
    except Exception:
        pass
    return info


def _run_pytest(test_files: List[str]) -> Dict[str, Any]:
    """Run pytest on given files and return pass/fail counts."""
    existing = [f for f in test_files if _test_file_exists(f)]
    if not existing:
        return {"passed": 0, "failed": 0, "error": "no test files found"}
    cmd = [sys.executable, "-m", "pytest"] + existing + [
        "-q", "--tb=no", "--no-header"
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=REPO, timeout=300
        )
        last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        # Parse "X passed" or "X failed" from pytest summary line
        passed = failed = 0
        for word, prev in zip(last_line.split(), [""] + last_line.split()):
            if word == "passed":
                try:
                    passed = int(prev)
                except ValueError:
                    pass
            if word == "failed":
                try:
                    failed = int(prev)
                except ValueError:
                    pass
        return {"passed": passed, "failed": failed, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"passed": 0, "failed": 0, "error": "timeout"}
    except Exception as e:
        return {"passed": 0, "failed": 0, "error": str(e)}


def _status_icon(status: str) -> str:
    return {"PASS": "✓", "FAIL": "✗", "PARTIAL": "~", "MISSING": "?"}.get(status, "?")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_phase(phase: Dict[str, Any], run_tests: bool) -> List[str]:
    lines = []
    pid    = phase["id"]
    pname  = phase["name"]
    log    = _load_log(phase["log_key"]) if phase["log_key"] else None
    mods   = [m for m in phase["modules"] if _module_exists(m)]
    missing_mods = [m for m in phase["modules"] if not _module_exists(m)]

    # Implementation status
    impl_ok = len(missing_mods) == 0
    impl_str = f"{len(mods)}/{len(phase['modules'])} modules"

    # Log status
    if log:
        log_status  = log.get("status", "?")
        log_time    = log.get("timestamp", "unknown")[:10]
        log_sha     = log.get("git_sha", "n/a")
        log_elapsed = log.get("elapsed_s", "?")
        n_passed    = log.get("n_passed", 0)
        n_failed    = log.get("n_failed", 0)
    else:
        log_status = "MISSING"
        log_time   = "—"
        log_sha    = "—"
        log_elapsed = "—"
        n_passed = n_failed = 0

    icon = _status_icon(log_status)
    lines.append(f"  {icon} {pid.upper()}  {pname}")
    lines.append(f"      {phase['description']}")
    lines.append(f"      Modules  : {impl_str}" +
                 (f"  (missing: {', '.join(missing_mods)})" if missing_mods else ""))
    sha_display = f"  sha={log_sha}" if log_sha and log_sha not in ("n/a", "None", None) else ""
    lines.append(f"      Log      : {log_status}  ({n_passed} passed, {n_failed} failed)"
                 + (f"  dated {log_time}{sha_display}" if log_status != "MISSING" else
                    "  — run validation script to generate"))

    # Key metrics from log
    if log and log.get("metrics"):
        m = log["metrics"]
        # Phase-specific metric display
        if pid == "phase1":
            n_mol = m.get("n_molecules", "?")
            lines.append(f"      Molecules: {n_mol} processed")
        elif pid == "phase2":
            s12  = m.get("h2_S12",    "?")
            err  = m.get("h2_S12_err","?")
            lines.append(f"      H2 S12   : {s12:.8f}  err={err:.2e}" if isinstance(s12, float) else f"      S12: {s12}")
        elif pid == "phase4":
            h2e = m.get("h2_energy_ha", "?")
            h2d = m.get("h2_pyscf_diff_ha", "?")
            h2o = m.get("h2o_energy_ha", "?")
            if isinstance(h2e, float):
                lines.append(f"      H2 E     : {h2e:.10f} Ha  Δ(PySCF)={h2d:.2e}")
            if isinstance(h2o, float):
                lines.append(f"      H2O E    : {h2o:.10f} Ha")
        elif pid == "phase5":
            R    = m.get("h2_opt_R_bohr",         "?")
            E    = m.get("h2_opt_energy_ha",       "?")
            g    = m.get("h2_opt_grad_max_ha_bohr","?")
            dR   = m.get("delta_R_vs_pyscf_bohr",  "?")
            steps= m.get("h2_opt_n_steps",         "?")
            if isinstance(R, float):
                lines.append(f"      H2 R_eq  : {R:.6f} bohr  ΔR(PySCF)={dR:+.4f} bohr")
                lines.append(f"      H2 E_eq  : {E:.10f} Ha  in {steps} BFGS steps")
                lines.append(f"      |g|_max  : {g:.2e} Ha/bohr at convergence")
        elif pid == "phase6":
            n_e  = m.get("n_electrons_integrated", "?")
            rmax = m.get("rho_max_e_bohr3",        "?")
            hn   = m.get("homo_norm_integrated",   "?")
            he   = m.get("homo_energy_ev",         "?")
            npts = m.get("grid_n_points",          "?")
            if isinstance(n_e, float):
                lines.append(f"      ∫ρ dr    : {n_e:.4f} e  (target 10)")
                lines.append(f"      ρ_max    : {rmax:.2f} e/bohr³   grid pts: {npts:,}")
            if isinstance(hn, float):
                lines.append(f"      HOMO norm: {hn:.4f}  ε_HOMO={he:.4f} eV")

    # Test results (only if --run-tests)
    if run_tests:
        tres = _run_pytest(phase["test_files"])
        t_icon = "✓" if tres.get("failed", 1) == 0 and tres.get("passed", 0) > 0 else "✗"
        lines.append(f"      Tests    : {t_icon} {tres.get('passed',0)} passed, "
                     f"{tres.get('failed',0)} failed"
                     + (f"  [{tres['error']}]" if "error" in tres else ""))

    return lines


def _list_artifacts() -> List[str]:
    """Return a sorted list of artifact paths relative to REPO (no duplicates)."""
    seen = set()
    artifacts = []
    # Walk OUTPUTS_DIR recursively once, grouping by parent
    for p in sorted(OUTPUTS_DIR.rglob("*")):
        if p.is_file() and p.name not in (".gitkeep",):
            rel = str(p.relative_to(REPO))
            if rel not in seen:
                seen.add(rel)
                artifacts.append(rel)
    return artifacts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_tests: bool = False, as_json: bool = False) -> None:
    git = _git_info()

    if as_json:
        status_doc: Dict[str, Any] = {
            "git": git,
            "phases": [],
            "artifacts": _list_artifacts(),
        }
        for phase in PHASES:
            log = _load_log(phase["log_key"]) if phase["log_key"] else None
            status_doc["phases"].append({
                "id":          phase["id"],
                "name":        phase["name"],
                "log_status":  log.get("status", "MISSING") if log else "MISSING",
                "log_timestamp": log.get("timestamp") if log else None,
                "metrics":     log.get("metrics", {}) if log else {},
                "modules_ok":  all(_module_exists(m) for m in phase["modules"]),
            })
        print(json.dumps(status_doc, indent=2))
        return

    # Human-readable output
    print(f"\n{SEP2}")
    print(f"  MOLEKUL — Project Status")
    sha_str   = git.get("sha") or "not a git repo"
    branch    = git.get("branch") or ""
    dirty_str = "  (uncommitted changes)" if git.get("dirty") else ""
    loc_str   = f" on {branch}" if branch else ""
    print(f"  git  : {sha_str}{loc_str}{dirty_str}")
    print(SEP2)

    # Count logs present
    n_logs = sum(1 for p in PHASES if p["log_key"] and _load_log(p["log_key"]))
    n_phases = len(PHASES)
    print(f"\n  {n_logs}/{n_phases} phases have validation logs\n")

    print(SEP)
    print("  PHASES")
    print(SEP)
    for phase in PHASES:
        for line in _fmt_phase(phase, run_tests=run_tests):
            print(line)
        print()

    print(SEP)
    print("  OUTPUT ARTIFACTS")
    print(SEP)
    artifacts = _list_artifacts()
    if artifacts:
        # Group by subdirectory
        by_dir: Dict[str, List[str]] = {}
        for a in artifacts:
            d = str(Path(a).parent)
            by_dir.setdefault(d, []).append(Path(a).name)
        for d, files in sorted(by_dir.items()):
            print(f"\n  {d}/")
            for f in sorted(files):
                print(f"    {f}")
    else:
        print("  (none yet — run validation scripts to generate)")

    print(f"\n{SEP}")
    print("  HOW TO RE-GENERATE LOGS")
    print(SEP)
    print("""
  python scripts/run_example.py          # phase1_molecules.txt / .json
  python scripts/validate_phase2.py      # phase2_integrals.txt / .json
  python -m pytest tests/test_eri.py -v  # phase3 (test suite only)
  python scripts/validate_rhf.py         # phase4_rhf.txt / .json
  python scripts/validate_optimizer.py   # phase5_optimizer.txt / .json
  python scripts/export_cube_h2o.py      # phase6_cube.txt / .json
  python scripts/status.py --run-tests   # full status + live test run
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOLEKUL project status")
    parser.add_argument("--json",       action="store_true",
                        help="Output machine-readable JSON instead of human-readable text")
    parser.add_argument("--run-tests",  action="store_true",
                        help="Run pytest for each phase and include results")
    args = parser.parse_args()
    main(run_tests=args.run_tests, as_json=args.json)
