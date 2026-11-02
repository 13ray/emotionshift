"""Paired dialogue-bootstrap differences from saved B0 seed-0 predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from emotionshift.data.labels import emotions
from emotionshift.eval.metrics import HEADLINE, paired_bootstrap, scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/b0/paired"
KEYS = ["dataset", "split", "dialogue_id", "turn", "speaker"]
# Direction is always candidate - reference. Oracle is diagnostic only.
COMPARISONS = [
    ("majority", "inertia_pred", False),
    ("inertia_pred", "donor", False),
    ("inertia_pred", "markov_pred", False),
    ("inertia_pred", "inertia_oracle", True),
]


def aligned_predictions(dataset, split, hashes):
    merged = None
    for name in dict.fromkeys(m for a, b, _ in COMPARISONS for m in [a, b]):
        path = ROOT / "predictions" / f"b0_{name}_{dataset}_{split}_seed0.csv"
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        source = pd.read_csv(path)
        if source.duplicated(KEYS).any():
            raise ValueError(f"Duplicate prediction key: {path}")
        part = source[[*KEYS, "emotion", "stratum", "pred"]].rename(columns={"pred": name})
        if merged is None:
            merged = part
        else:
            joined = merged.merge(
                part,
                on=KEYS,
                how="outer",
                validate="one_to_one",
                suffixes=("", "_check"),
                indicator=True,
            )
            if not joined["_merge"].eq("both").all():
                raise ValueError(f"Prediction populations differ: {path}")
            for col in ["emotion", "stratum"]:
                if not joined[col].eq(joined[col + "_check"]).all():
                    raise ValueError(f"Evaluation metadata differ: {path}")
            merged = joined.drop(columns=["emotion_check", "stratum_check", "_merge"])
    return merged


def main():
    cfg = yaml.safe_load((ROOT / "configs/b0_baselines.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for dataset in cfg["datasets"]:
        metric, labels = HEADLINE[dataset], emotions(dataset)
        for split in cfg["test_splits"][dataset]:
            frame = aligned_predictions(dataset, split, hashes)
            if frame.emotion.isna().any() or not frame.emotion.isin(labels).all():
                raise ValueError("Only labelled targets in the native label space may be scored")
            for reference, candidate, oracle in COMPARISONS:
                for group in ["all", "stable", "shift"]:
                    mask = None if group == "all" else frame.stratum.eq(group)
                    selected = frame if mask is None else frame[mask]
                    if selected.empty:
                        continue
                    result = paired_bootstrap(
                        frame,
                        reference,
                        candidate,
                        metric=metric,
                        mask=mask,
                        labels=labels,
                        n_boot=cfg["bootstrap"],
                        seed=0,
                    )
                    before = scores(selected.emotion, selected[reference], labels)[metric]
                    after = scores(selected.emotion, selected[candidate], labels)[metric]
                    if not np.isclose(result["diff"], after - before):
                        raise AssertionError(
                            "Paired difference disagrees with shared score function"
                        )
                    direction = (
                        "above_zero"
                        if result["ci_low"] > 0
                        else "below_zero"
                        if result["ci_high"] < 0
                        else "includes_zero"
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "group": group,
                            "reference": reference,
                            "candidate": candidate,
                            "oracle": oracle,
                            "metric": metric,
                            "checkpoint_seed": 0,
                            "bootstrap_seed": 0,
                            "bootstrap_resamples": cfg["bootstrap"],
                            "dialogues_resampled": frame.dialogue_id.nunique(),
                            "reference_score": before,
                            "candidate_score": after,
                            **result,
                            "diff_pp": result["diff"] * 100,
                            "ci_low_pp": result["ci_low"] * 100,
                            "ci_high_pp": result["ci_high"] * 100,
                            "interval_direction": direction,
                        }
                    )
            print(f"Paired bootstrap finished: {dataset}/{split}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "differences.csv", index=False)
    headline = table[
        table.group.eq("all") & ~(table.dataset.eq("dailydialog") & table.split.eq("test"))
    ]
    headline.to_csv(OUT / "headline_differences.csv", index=False)
    text = [
        "# B0 paired dialogue-bootstrap differences",
        "",
        "Difference = candidate - reference, in **percentage points**, not relative percent.",
        "Each replicate resamples the same dialogues for both methods (1,000 replicates; seed 0).",
        "Use the shared `paired_bootstrap` helper without changing its resampling or metric definitions.",
        "Native dataset labels and dataset-specific headline metrics are retained.",
        "These are seed-0 checkpoint comparisons, not differences between five-seed means.",
        "",
        "## Overall headline splits",
        "",
        "| Dataset / split | Candidate - reference | Delta (pp) | 95% CI (pp) | Interval |",
        "|---|---|---:|---|---|",
    ]
    for r in headline.itertuples(index=False):
        name = f"{r.candidate} - {r.reference}" + (" [oracle diagnostic]" if r.oracle else "")
        text.append(
            f"| {r.dataset} / {r.split} | {name} | {r.diff_pp:+.2f} | "
            f"[{r.ci_low_pp:+.2f}, {r.ci_high_pp:+.2f}] | {r.interval_direction} |"
        )
    text += [
        "",
        "## Interpretation and limits",
        "",
        "- An interval entirely above zero supports a positive difference for this fixed comparison and checkpoint under dialogue resampling; below zero supports a negative difference.",
        "- An interval including zero does not establish a clear direction; it does not prove equivalence.",
        "- Read differences directly. Overlap of two separately estimated method intervals is not a paired comparison.",
        "- Stable/shift estimates resample the original split's whole dialogue population, then score only the selected turns. Dialogues without those turns remain in the resampling population.",
        "- Majority -> inertia asks whether lagged recognition helps over a constant floor. Inertia -> donor measures the gain from using current-text recognition instead of copying its previous judgement.",
        "- Inertia -> Markov compares two history-only rules using the same donor. Inertia -> oracle is a gold-history diagnostic, never a deployable model improvement.",
        "- These additional post-hoc analyses are exploratory. Intervals are pointwise and not adjusted for multiple comparisons; no family-wide significance or causal effect is claimed.",
        "- This is inference about held-out dialogue sampling for fixed trained checkpoints, not uncertainty over training seeds or domains. Scores and differences across datasets use different headline metrics and must not be ranked as equivalent tasks.",
        "",
        "`differences.csv` includes all/stable/shift groups and the secondary DailyDialog official test. Prediction files are unchanged; there is no retraining or test-based model selection.",
        "",
        "Reproduce: `uv run python -X utf8 scripts/analyze_b0_paired.py`.",
    ]
    (OUT / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "bootstrap_resamples": cfg["bootstrap"],
                "bootstrap_seed": 0,
                "checkpoint_seed": 0,
                "comparisons": COMPARISONS,
                "prediction_sha256": hashes,
                "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "helper_sha256": hashlib.sha256(
                    (ROOT / "emotionshift/eval/metrics.py").read_bytes()
                ).hexdigest(),
                "row_alignment_and_gold_checked": True,
                "point_differences_recomputed": len(table),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure(headline)


def figure(table):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    names = {
        "majority": "Majority",
        "inertia_pred": "Inertia",
        "donor": "Current text",
        "markov_pred": "Markov",
        "inertia_oracle": "Oracle inertia",
    }
    metrics = {
        "meld": "Weighted F1",
        "dailydialog": "Non-neutral micro-F1",
        "emowoz": "Non-neutral weighted F1",
    }
    for ax, dataset in zip(axes, ["meld", "dailydialog", "emowoz"], strict=True):
        rows = table[table.dataset.eq(dataset)]
        for y, r in enumerate(rows.itertuples(index=False)):
            color = "#777777" if r.oracle else "#35648b"
            ax.plot([r.ci_low_pp, r.ci_high_pp], [y, y], color=color, linewidth=2)
            ax.scatter(r.diff_pp, y, marker="s" if r.oracle else "o", color=color)
        ax.set_yticks(
            range(len(rows)),
            [f"{names[r.candidate]} - {names[r.reference]}" for r in rows.itertuples(index=False)],
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.invert_yaxis()
        ax.set_xlabel("Candidate - reference (percentage points)")
        ax.set_title(dataset + "\n" + metrics[dataset])
    fig.suptitle("Paired dialogue bootstrap: seed-0 differences with pointwise 95% CIs")
    fig.savefig(OUT / "headline_differences.png", dpi=160, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    main()
