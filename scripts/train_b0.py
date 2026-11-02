"""Run B0 on all three native label spaces; CPU only, no gold at inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from emotionshift.data.labels import emotions
from emotionshift.data.loaders import DATA_ROOT, load
from emotionshift.eval import metrics as M
from emotionshift.models.b0 import HistoryBaselines
from emotionshift.models.tfidf import TfidfBaseline, TfidfConfig

ROOT = Path(__file__).resolve().parents[1]
PRED_COLS = [
    "dataset",
    "split",
    "dialogue_id",
    "turn",
    "speaker",
    "text",
    "emotion",
    "pred",
    "stratum",
    "shift_type",
    "model",
    "seed",
]
MODELS = ["majority", "donor", "inertia_pred", "markov_pred", "inertia_oracle", "markov_oracle"]
DETERMINISTIC = {"majority", "inertia_oracle", "markov_oracle"}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dialogue_intervals(frame, labels, n_boot=1000, seed=0):
    """Same dialogue bootstrap as shared helper, aggregated to confusion matrices.

    All scores use the shared scores_from_confusion; avoids recomputing sklearn
    scores six times for each of 1,000 samples. Unit test checks exact equivalence.
    """
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    dialogue = pd.factorize(frame.dialogue_id, sort=False)[0]
    pair = frame.emotion.map(index).to_numpy() * k + frame.pred.map(index).to_numpy()
    matrices = np.bincount(dialogue * k * k + pair, minlength=(dialogue.max() + 1) * k * k).reshape(
        -1, k, k
    )
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(matrices), len(matrices))
        samples.append(M.scores_from_confusion(matrices[sampled].sum(axis=0), labels))
    return {
        metric: list(np.percentile([s[metric] for s in samples], [2.5, 97.5]))
        for metric in M.METRICS
    }


def forecast_result(frame, probabilities, labels):
    eligible = frame.stratum.isin(["stable", "shift"])
    y = frame.loc[eligible, "stratum"].eq("shift").astype(int)
    p = np.asarray(probabilities)[eligible.to_numpy()]
    return {
        "n": len(y),
        "base_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "auprc": float(average_precision_score(y, p)) if y.sum() else None,
        "brier": float(np.mean((y.to_numpy() - p) ** 2)),
        "scope": "same-speaker non-first labelled turns; no current text input",
    }


def run(dataset, cfg, base):
    start = time.perf_counter()
    labels, headline = emotions(dataset), M.HEADLINE[dataset]
    df = load(dataset)
    # System rows have no gold target; they are never trained on or scored.
    df = df[df.emotion.notna()].copy()
    train, dev = df[df.split == "train"], df[df.split == "dev"]
    out = base / "reports" / "b0" / dataset
    out.mkdir(parents=True, exist_ok=True)
    (base / "predictions").mkdir(parents=True, exist_ok=True)
    baseline = HistoryBaselines(labels).fit(train)
    grid = []
    for c in cfg["donor_C"]:
        model = TfidfBaseline(TfidfConfig(features="word", C=c, class_weight="balanced", seed=0))
        model.fit(train.text, train.emotion)
        grid.append({"C": c, **M.scores(dev.emotion, model.predict(dev.text), labels)})
    grid = pd.DataFrame(grid).sort_values(headline, ascending=False, kind="stable")
    grid.to_csv(out / "donor_dev_grid.csv", index=False)
    selected = float(grid.iloc[0].C)
    # Persist selection BEFORE test prediction.
    write_json(
        out / "selection.json",
        {
            "selected_C": selected,
            "headline": headline,
            "selected_on": "dev only",
            "features": "word 1-2grams",
            "class_weight": "balanced",
        },
    )
    pd.DataFrame(baseline.transition, index=labels, columns=labels).to_csv(out / "transition.csv")
    pd.DataFrame(baseline.transition_counts, index=labels, columns=labels).to_csv(
        out / "transition_counts.csv"
    )
    records, forecasts, ci = [], [], {}
    for seed in cfg["seeds"]:
        donor = TfidfBaseline(
            TfidfConfig(features="word", C=selected, class_weight="balanced", seed=seed)
        )
        fitted = time.perf_counter()
        donor.fit(train.text, train.emotion)
        fit_seconds = time.perf_counter() - fitted
        for split in cfg["test_splits"][dataset]:
            test = df[df.split == split].copy()
            recognized = pd.Series(donor.predict(test.text), index=test.index)
            predicted = baseline.predict(test, recognized)
            oracle = baseline.oracle(test)
            predictions = {
                "donor": recognized,
                **{m: predicted[m] for m in MODELS if m in predicted},
                **{m: oracle[m] for m in ["inertia_oracle", "markov_oracle"]},
            }
            for name in MODELS:
                # No fake across-seed variation or redundant files for deterministic rules.
                if name in DETERMINISTIC and seed != cfg["seeds"][0]:
                    continue
                result = test.copy()
                result["pred"] = predictions[name]
                result["model"], result["seed"] = "b0_" + name, seed
                result["donor_pred"] = recognized
                result["history_pred"] = predicted.history_pred
                extra = [c for c in result if c not in PRED_COLS]
                result[PRED_COLS + extra].to_csv(
                    base / "predictions" / f"b0_{name}_{dataset}_{split}_seed{seed}.csv",
                    index=False,
                )
                record = {
                    "dataset": dataset,
                    "split": split,
                    "model": name,
                    "seed": seed,
                    "oracle": "oracle" in name,
                    **M.scores(result.emotion, result.pred, labels),
                    **M.derived_shift_detection(result),
                    "donor_fit_seconds": fit_seconds,
                }
                # The derived-shift helper's n counts eligible turns; preserve total n.
                record["n"] = len(result)
                records.append(record)
                if seed == cfg["seeds"][0]:
                    suffix = f"{name}_{split}"
                    M.stratified(result, labels=labels).to_csv(
                        out / f"stratified_{suffix}.csv", index=False
                    )
                    M.per_class_f1(result.emotion, result.pred, labels).to_csv(
                        out / f"per_class_{suffix}.csv", index=False
                    )
                    M.confusion(result.emotion, result.pred, labels).to_csv(
                        out / f"confusion_{suffix}.csv"
                    )
                    ci[suffix] = dialogue_intervals(result, labels, cfg["bootstrap"])
            for kind, previous in [
                ("pred", predicted.history_pred),
                ("oracle", oracle.history_pred),
            ]:
                probabilities = (predicted if kind == "pred" else oracle).forecast_shift_prob
                forecasts.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "seed": seed,
                        "model": "markov_" + kind,
                        **forecast_result(test, probabilities, labels),
                    }
                )
                forecast = test[[*PRED_COLS[:7], "stratum"]].copy()
                forecast["history_label"] = previous
                forecast["shift_prob"] = probabilities
                forecast["forecast_emotion"] = predictions["markov_" + kind]
                if kind == "pred" or seed == cfg["seeds"][0]:
                    forecast.to_csv(out / f"forecast_{kind}_{split}_seed{seed}.csv", index=False)
        print(f"{dataset}: finished seed {seed}", flush=True)
    records = pd.DataFrame(records)
    records.to_csv(out / "per_seed.csv", index=False)
    pd.DataFrame(forecasts).to_csv(out / "forecast.csv", index=False)
    summary = []
    for (split, name), group in records.groupby(["split", "model"], sort=False):
        row = {
            "dataset": dataset,
            "split": split,
            "model": name,
            "oracle": "oracle" in name,
            "headline_metric": headline,
            "n": int(group.iloc[0].n),
            "seeds": group.seed.tolist(),
        }
        for metric in [*M.METRICS, "shift_f1"]:
            row[metric] = {
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=0)),
            }
            if metric in M.METRICS:
                row[metric]["ci95_seed0"] = ci[f"{name}_{split}"][metric]
        summary.append(row)
    write_json(
        out / "summary.json",
        {
            "dataset": dataset,
            "labels": labels,
            "majority": baseline.majority,
            "native_label_space": True,
            "bootstrap": cfg["bootstrap"],
            "donor_C": selected,
            "transition_smoothing": "one train-marginal pseudo-count",
            "seconds": time.perf_counter() - start,
            "results": summary,
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/b0_baselines.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = args.datasets or cfg["datasets"]
    results = [row for dataset in datasets for row in run(dataset, cfg, args.output_root)]
    out = args.output_root / "reports/b0"
    rows = [
        {
            "dataset": r["dataset"],
            "split": r["split"],
            "model": r["model"],
            "oracle": r["oracle"],
            "n": r["n"],
            "headline_metric": r["headline_metric"],
            "headline": r[r["headline_metric"]]["mean"],
            "headline_std": r[r["headline_metric"]]["std"],
            "ci_low": r[r["headline_metric"]]["ci95_seed0"][0],
            "ci_high": r[r["headline_metric"]]["ci95_seed0"][1],
            "macro_f1": r["macro_f1"]["mean"],
            "shift_f1": r["shift_f1"]["mean"],
        }
        for r in results
    ]
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    hashes = {
        p.relative_to(DATA_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for d in datasets
        for p in sorted((DATA_ROOT / d).glob("*"))
        if p.is_file() and p.suffix in {".json", ".csv"}
    }
    write_json(
        out / "run_manifest.json",
        {
            "config": cfg,
            "datasets": datasets,
            "python": platform.python_version(),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "raw_sha256": hashes,
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), ROOT / "emotionshift/models/b0.py", args.config]
            },
        },
    )
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
