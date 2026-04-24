"""
Experiment logging utilities for MOLEKUL.

Provides structured dual-format logging (human-readable .txt + machine-readable
.json) to outputs/logs/, plus checkpoint helpers for outputs/checkpoints/.

Schema for every JSON log
-------------------------
{
  "phase":         "phase4",          # phase identifier string
  "name":          "rhf",             # short experiment name
  "timestamp":     "2026-04-01T...",  # ISO-8601 wall-clock time
  "git_sha":       "abcdef1" | null,  # git HEAD SHA if available
  "status":        "PASS" | "FAIL" | "PARTIAL",
  "n_passed":      int,               # checks that passed
  "n_failed":      int,               # checks that failed
  "elapsed_s":     float,             # total wall-clock time in seconds
  "metrics":       { ... },           # key scalar results
  "artifacts":     [ ... ],           # paths of files written
  "details":       { ... }            # optional free-form section
}
"""

from __future__ import annotations

import datetime
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_outputs_dir() -> Path:
    d = _repo_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logs_dir() -> Path:
    d = get_outputs_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_checkpoints_dir() -> Path:
    d = get_outputs_dir() / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_repo_root(),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with run_example.py)
# ---------------------------------------------------------------------------

def save_json_log(data: Dict[str, Any], filename: str) -> Path:
    if not filename.endswith(".json"):
        filename += ".json"
    out = get_outputs_dir() / filename
    out.write_text(json.dumps(data, indent=2))
    return out


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self):
        self._start = time.perf_counter()


# ---------------------------------------------------------------------------
# ExperimentLogger — dual txt + json output
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Accumulates experiment metrics and writes dual-format reports.

    Usage
    -----
    >>> log = ExperimentLogger("phase4", "rhf")
    >>> log.metric("h2_energy_ha", -1.1167143190)
    >>> log.check("H2 energy within 1e-6 of PySCF", True)
    >>> log.artifact("outputs/phase4_rhf.txt")
    >>> log.save()   # writes outputs/logs/phase4_rhf.txt + .json

    The .txt file is written with the provided header and lines.
    The .json file uses the standard MOLEKUL log schema.
    """

    def __init__(self, phase: str, name: str):
        self.phase = phase
        self.name = name
        self._timer = Timer()
        self._metrics: Dict[str, Any] = {}
        self._details: Dict[str, Any] = {}
        self._artifacts: List[str] = []
        self._passed: List[str] = []
        self._failed: List[str] = []
        self._txt_lines: List[str] = []

    # --- Accumulation methods -----------------------------------------------

    def metric(self, key: str, value: Any) -> None:
        """Record a scalar metric (will appear in JSON metrics section)."""
        self._metrics[key] = value

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        """Record a pass/fail check."""
        if ok:
            self._passed.append(label)
        else:
            self._failed.append(label)
        return ok

    def detail(self, key: str, value: Any) -> None:
        """Record a free-form detail (will appear in JSON details section)."""
        self._details[key] = value

    def artifact(self, path: str) -> None:
        """Record a path to an output artifact."""
        self._artifacts.append(str(path))

    def line(self, text: str = "") -> None:
        """Append a line to the human-readable .txt report."""
        self._txt_lines.append(text)

    # --- Save ---------------------------------------------------------------

    def save(
        self,
        txt_lines: Optional[List[str]] = None,
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> tuple[Path, Path]:
        """
        Write outputs/logs/{phase}_{name}.txt and .json.

        Parameters
        ----------
        txt_lines  : additional lines to append to the .txt report
                     (combined with lines added via self.line())
        extra_json : additional top-level keys for the JSON document

        Returns
        -------
        (txt_path, json_path)
        """
        elapsed = self._timer.elapsed()
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        sha = _git_sha()
        n_passed = len(self._passed)
        n_failed = len(self._failed)
        n_total = n_passed + n_failed
        if n_failed == 0 and n_total > 0:
            status = "PASS"
        elif n_passed == 0 and n_total > 0:
            status = "FAIL"
        elif n_total == 0:
            status = "PASS"  # no explicit checks — assume script ran OK
        else:
            status = "PARTIAL"

        logs_dir = get_logs_dir()
        base = f"{self.phase}_{self.name}"

        # --- JSON -----------------------------------------------------------
        doc: Dict[str, Any] = {
            "phase":     self.phase,
            "name":      self.name,
            "timestamp": timestamp,
            "git_sha":   sha,
            "status":    status,
            "n_passed":  n_passed,
            "n_failed":  n_failed,
            "elapsed_s": round(elapsed, 3),
            "metrics":   self._metrics,
            "artifacts": self._artifacts,
            "details":   self._details,
        }
        if self._failed:
            doc["failed_checks"] = self._failed
        if extra_json:
            doc.update(extra_json)

        json_path = logs_dir / f"{base}.json"
        json_path.write_text(json.dumps(doc, indent=2))

        # --- TXT ------------------------------------------------------------
        all_lines = [
            f"MOLEKUL Experiment Report",
            f"Phase    : {self.phase}",
            f"Name     : {self.name}",
            f"Time     : {timestamp}",
            f"Git SHA  : {sha or 'n/a'}",
            f"Status   : {status}  ({n_passed}/{n_total} checks passed)",
            f"Elapsed  : {elapsed:.2f} s",
            "─" * 60,
        ]
        all_lines += self._txt_lines
        if txt_lines:
            all_lines += txt_lines
        all_lines += [
            "─" * 60,
            "Metrics:",
        ]
        for k, v in self._metrics.items():
            all_lines.append(f"  {k} = {v}")
        if self._artifacts:
            all_lines += ["", "Artifacts:"]
            for a in self._artifacts:
                all_lines.append(f"  {a}")
        if self._failed:
            all_lines += ["", "Failed checks:"]
            for f in self._failed:
                all_lines.append(f"  FAIL  {f}")

        txt_path = logs_dir / f"{base}.txt"
        txt_path.write_text("\n".join(all_lines) + "\n")

        return txt_path, json_path


# ---------------------------------------------------------------------------
# Stdout capture helper
# ---------------------------------------------------------------------------

class StdoutCapture:
    """
    Context manager that tees stdout to both the terminal and a string buffer.

    Example
    -------
    >>> with StdoutCapture() as cap:
    ...     print("hello")
    hello
    >>> cap.text
    'hello\\n'
    """

    def __init__(self, tee: bool = True):
        self._tee = tee
        self._buf = io.StringIO()
        self._old = None

    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = self
        return self

    def write(self, s: str):
        self._buf.write(s)
        if self._tee:
            self._old.write(s)

    def flush(self):
        self._buf.flush()
        if self._tee and self._old:
            self._old.flush()

    def __exit__(self, *args):
        sys.stdout = self._old

    @property
    def text(self) -> str:
        return self._buf.getvalue()
