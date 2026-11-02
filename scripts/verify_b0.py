"""Validate B0's release files, source hashes, executed notebook and English text.

Run with --run-tests for the full repository pytest suite and B0 Ruff checks.
Personal reference notes live outside this repository and are not inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd
from PIL import Image

from scripts.make_b0_artifacts import audit

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/b0"
CODE = [
    "emotionshift/models/b0.py",
    "tests/test_b0.py",
    "scripts/train_b0.py",
    "scripts/make_b0_artifacts.py",
    "scripts/explain_b0_inertia.py",
    "scripts/analyze_b0_paired.py",
    "scripts/analyze_b0_transitions.py",
    "scripts/verify_b0.py",
]
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".ipynb"}


def release_files():
    files = [
        ROOT / "README.md",
        ROOT / "configs/b0_baselines.yaml",
        *(ROOT / p for p in CODE),
        ROOT / "docs/B0_BASELINES.md",
        ROOT / "docs/B0_PR_DESCRIPTION.md",
        ROOT / "notebooks/b0_baselines.ipynb",
    ]
    files += [
        p
        for folder in [REPORT, ROOT / "deliverables/01_inertia"]
        for p in folder.rglob("*")
        if p.is_file()
    ]
    files += list((ROOT / "predictions").glob("b0_*.csv"))
    return sorted(set(files))


def text_hash_matches(path, expected):
    """Allow only Git LF/CRLF conversion; substantive text must still match.

    Raw downloaded data uses exact bytes instead (checked separately).
    """
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {
        hashlib.sha256(value).hexdigest() for value in (raw, lf, lf.replace(b"\n", b"\r\n"))
    }


def check_hashes(manifest, field):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for relative, expected in data[field].items():
        if not text_hash_matches(ROOT / relative, expected):
            raise ValueError(f"Source hash mismatch in {manifest}: {relative}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    count = audit()  # Recognition, derived shifts and source row identity.
    run = json.loads((REPORT / "run_manifest.json").read_text(encoding="utf-8"))
    for name, expected in run["source_sha256"].items():
        if not text_hash_matches(ROOT / name, expected):
            raise ValueError(f"Training source changed since run: {name}")
    for name, expected in run["raw_sha256"].items():
        path = ROOT / "data/raw" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Raw input changed since run: {name}")
    paired = check_hashes(REPORT / "paired/manifest.json", "prediction_sha256")
    transitions = check_hashes(REPORT / "transitions/manifest.json", "source_sha256")
    for manifest, source in [
        (paired, "scripts/analyze_b0_paired.py"),
        (transitions, "scripts/analyze_b0_transitions.py"),
    ]:
        if not text_hash_matches(ROOT / source, manifest["analysis_sha256"]):
            raise ValueError(f"Analysis source changed: {source}")
    if not text_hash_matches(ROOT / "emotionshift/eval/metrics.py", paired["helper_sha256"]):
        raise ValueError("Shared paired metric implementation changed")
    differences = pd.read_csv(REPORT / "paired/differences.csv")
    if len(differences) != paired["point_differences_recomputed"]:
        raise ValueError("Incomplete paired comparison table")
    notebook = nbformat.read(ROOT / "notebooks/b0_baselines.ipynb", as_version=4)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    if not all(c.execution_count is not None and len(c.source.splitlines()) <= 15 for c in cells):
        raise ValueError("Notebook has unexecuted or oversized code cells")
    if any(o.output_type == "error" for c in cells for o in c.outputs):
        raise ValueError("Notebook contains execution errors")
    files = release_files()
    chinese = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]")
    hits = []
    for path in files:
        if path.suffix in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
            if path.suffix in {".json", ".ipynb"}:
                text = json.dumps(json.loads(text), ensure_ascii=False)
            if chinese.search(text):
                hits.append(str(path.relative_to(ROOT)))
    if hits:
        raise ValueError(f"Chinese text in release files: {hits}")
    if any(path.stat().st_size >= 50_000_000 for path in files):
        raise ValueError("A release artifact exceeds the team's 50 MB limit")
    dimensions = {}
    for path in REPORT.rglob("*.png"):
        with Image.open(path) as image:
            dimensions[str(path.relative_to(ROOT))] = list(image.size)
            if image.width < 1600:
                raise ValueError(f"Figure too small: {path}")
    validation = {
        "prediction_files_checked": count,
        "executed_notebook_code_cells": len(cells),
        "paired_contrasts_checked": len(differences),
        "english_release_no_chinese": True,
        "training_inputs_and_code_hashes_verified": True,
        "git_text_hash_policy": "LF/CRLF-equivalent text accepted; raw downloads require exact byte hashes",
        "paired_and_transition_provenance_verified": True,
        "all_release_files_below_50_mb": True,
        "figure_dimensions": dimensions,
    }
    if args.run_tests:
        for name, command in [
            ("pytest", [sys.executable, "-X", "utf8", "-m", "pytest", "-q"]),
            ("ruff", [sys.executable, "-m", "ruff", "check", *CODE]),
        ]:
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=300,
            )
            print(result.stdout, end="", flush=True)
            if result.returncode:
                print(result.stderr, end="", flush=True)
                raise RuntimeError(f"{name} failed")
            validation[name] = result.stdout.strip().splitlines()[-1]
    (REPORT / "release_check.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print("B0 release verification passed.", flush=True)


if __name__ == "__main__":
    main()
