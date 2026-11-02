"""Explain B0's fixed training transition matrices and saved predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emotionshift.data.labels import emotions
from emotionshift.data.loaders import load
from emotionshift.models.b0 import HistoryBaselines

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/b0/transitions"
KEYS = ["dataset", "split", "dialogue_id", "turn", "speaker"]


def heatmap(dataset, matrix, support):
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    labels = matrix.columns.tolist()
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), [f"{lab} (n={int(support[lab])})" for lab in labels])
    ax.set_xlabel("Current same-speaker emotion")
    ax.set_ylabel("Previous same-speaker emotion; n = observed train transitions")
    ax.set_title(
        f"{dataset}: P(current | previous), training only\n"
        "One train-marginal pseudo-count; each row sums to 1"
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, label="Conditional probability")
    fig.savefig(OUT / f"{dataset}_transition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, distributions, effects, hashes = [], [], [], {}
    for dataset in ["meld", "dailydialog", "emowoz"]:
        labels = emotions(dataset)
        # Reconstruct exclusively from train to audit the already fitted priors.
        train = load(dataset, splits=["train"])
        fitted = HistoryBaselines(labels).fit(train)
        matrix_path = ROOT / f"reports/b0/{dataset}/transition.csv"
        count_path = ROOT / f"reports/b0/{dataset}/transition_counts.csv"
        matrix = pd.read_csv(matrix_path, index_col=0).loc[labels, labels]
        counts = pd.read_csv(count_path, index_col=0).loc[labels, labels]
        np.testing.assert_allclose(matrix.to_numpy(), fitted.transition)
        np.testing.assert_array_equal(counts.to_numpy(), fitted.transition_counts)
        np.testing.assert_allclose(matrix.sum(axis=1), 1)
        support = counts.sum(axis=1)
        matrix.to_csv(OUT / f"{dataset}_probabilities.csv")
        counts.to_csv(OUT / f"{dataset}_counts.csv")
        heatmap(dataset, matrix, support)
        for label in labels:
            summary.append(
                {
                    "dataset": dataset,
                    "previous_emotion": label,
                    "train_transition_support": int(support[label]),
                    "self_probability": matrix.loc[label, label],
                    "to_neutral_probability": matrix.loc[label, "neutral"],
                    "argmax_next_emotion": matrix.loc[label].idxmax(),
                    "argmax_probability": matrix.loc[label].max(),
                    "argmax_matches_inertia": matrix.loc[label].idxmax() == label,
                    "empirical_self_probability": (
                        counts.loc[label, label] / support[label] if support[label] else None
                    ),
                }
            )
        splits = ["test_dedup", "test"] if dataset == "dailydialog" else ["test"]
        for split in splits:
            joined = None
            for name in ["majority", "donor", "inertia_pred", "markov_pred"]:
                path = ROOT / f"predictions/b0_{name}_{dataset}_{split}_seed0.csv"
                source = pd.read_csv(path)
                if source.duplicated(KEYS).any() or not source.pred.isin(labels).all():
                    raise ValueError(f"Invalid prediction export: {path}")
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                if joined is None:
                    joined = source[[*KEYS, "emotion", "stratum", "pred"]].rename(
                        columns={"pred": name}
                    )
                else:
                    incoming = source[[*KEYS, "pred", "emotion"]].rename(columns={"pred": name})
                    joined = joined.merge(
                        incoming,
                        on=KEYS,
                        how="outer",
                        validate="one_to_one",
                        suffixes=("", "_check"),
                        indicator=True,
                    )
                    if (
                        not joined._merge.eq("both").all()
                        or not joined.emotion.eq(joined.emotion_check).all()
                    ):
                        raise ValueError("Saved prediction populations or gold labels differ")
                    joined = joined.drop(columns=["emotion_check", "_merge"])
                for label in labels:
                    distributions.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "model": name,
                            "emotion": label,
                            "gold_n": int(source.emotion.eq(label).sum()),
                            "predicted_n": int(source.pred.eq(label).sum()),
                            "predicted_fraction": float(source.pred.eq(label).mean()),
                        }
                    )
            for group in ["all", "stable", "shift"]:
                part = joined if group == "all" else joined[joined.stratum.eq(group)]
                inertia_ok = part.inertia_pred.eq(part.emotion)
                markov_ok = part.markov_pred.eq(part.emotion)
                changes = part.inertia_pred.ne(part.markov_pred)
                effects.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "group": group,
                        "n": len(part),
                        "changed_predictions": int(changes.sum()),
                        "changed_to_neutral": int((changes & part.markov_pred.eq("neutral")).sum()),
                        "markov_correct_inertia_wrong": int((markov_ok & ~inertia_ok).sum()),
                        "inertia_correct_markov_wrong": int((inertia_ok & ~markov_ok).sum()),
                        "both_correct": int((inertia_ok & markov_ok).sum()),
                        "both_wrong": int((~inertia_ok & ~markov_ok).sum()),
                    }
                )
        for path in [matrix_path, count_path]:
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = pd.DataFrame(summary)
    distribution = pd.DataFrame(distributions)
    effect = pd.DataFrame(effects)
    summary.to_csv(OUT / "transition_summary.csv", index=False)
    distribution.to_csv(OUT / "prediction_distribution.csv", index=False)
    effect.to_csv(OUT / "markov_vs_inertia_cases.csv", index=False)
    notes = [
        "# What does the Markov prior learn?",
        "",
        "Rows condition on the previous labelled utterance by the **same speaker**, not the previous conversation turn. Columns are that speaker's next/current emotion.",
        "The diagonal is emotional persistence. The neutral column measures return to neutral. Each row sums to one, after the fixed train-marginal pseudo-count.",
        "Heatmap annotations are rounded to two decimals: 0.00 need not be exactly zero. Row support is the observed transition count before smoothing.",
        "The source counts and probabilities are verified against a fresh **train-only** reconstruction. No test transition is fitted, and no predictions or hyperparameters change.",
        "",
        "## Learned dynamics and decision rules",
        "",
        "| Dataset | Previous emotion | Train support | Stay probability | To neutral | Argmax next label |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in summary.itertuples(index=False):
        notes.append(
            f"| {r.dataset} | {r.previous_emotion} | {r.train_transition_support} | "
            f"{r.self_probability:.3f} | {r.to_neutral_probability:.3f} | {r.argmax_next_emotion} |"
        )
    notes += ["", "## Why a learned transition prior need not beat inertia", ""]
    for dataset in ["meld", "dailydialog", "emowoz"]:
        rows = summary[summary.dataset.eq(dataset)]
        collapsed = rows[rows.argmax_next_emotion.eq("neutral")].previous_emotion.tolist()
        split = "test_dedup" if dataset == "dailydialog" else "test"
        predicted = distribution[
            distribution.dataset.eq(dataset)
            & distribution.split.eq(split)
            & distribution.model.eq("markov_pred")
        ]
        never = predicted[predicted.predicted_n.eq(0)].emotion.tolist()
        neutral_share = predicted.loc[predicted.emotion.eq("neutral"), "predicted_fraction"].iloc[0]
        notes.append(
            f"- **{dataset}:** {len(collapsed)} of 7 history labels map to neutral under argmax "
            f"({', '.join(collapsed)}). On {split}, Markov predicts neutral for "
            f"{neutral_share:.1%} of targets and never predicts: {', '.join(never) or 'none'}."
        )
    notes += [
        "",
        "A row can have substantial self-transition probability without persistence being the most likely outcome. For example, if P(joy|joy) is below P(neutral|joy), argmax predicts neutral rather than copying joy.",
        "The transition estimator learns a population average; it cannot observe what the current utterance says. Even a correct previous emotion cannot identify an unexpected turning point.",
        "The conditioning emotion comes from a fallible donor recognizer, adding history error. Oracle history removes that input error but not the model's lack of current-utterance evidence.",
        "Argmax prioritizes the most likely single label, not the dataset's non-neutral F1. Collapsing many histories to neutral can therefore hurt emotional-class recall, especially on EmoWOZ, whose headline excludes neutral.",
        "",
        "## Per-turn changes are diagnostics, not an F1 decomposition",
        "",
        "`markov_vs_inertia_cases.csv` counts changes and cases where only one method is correct. Both methods use exactly aligned seed-0 targets. These accuracy-style counts do not algebraically decompose class-weighted or micro-F1.",
        "For example, on MELD test Markov corrects 282 inertia errors but loses 136 previously correct targets, increasing accuracy by 146/2610 (5.59 percentage points). Yet its weighted-F1 is 4.56 points lower. More correct majority-class targets need not compensate for lost emotional-class precision/recall in F1.",
        "`prediction_distribution.csv` includes gold/predicted counts for every native label, including zero predictions. Only labelled targets are counted. First turns have the same train-majority fallback and do not explain differences between the rules.",
        "Read the [paired F1 comparison](../paired/README.md) for uncertainty on score differences: Markov is below inertia on MELD and EmoWOZ in these fixed-checkpoint comparisons; DailyDialog's overall difference interval crosses zero.",
        "",
        "## Limits",
        "",
        "Small training row supports imply less reliable transition estimates; the heatmaps display support beside every previous label. This analysis does not estimate uncertainty on individual transition probabilities.",
        "Within-EmoWOZ native labels are retained. Differences in probability patterns across datasets are descriptive and do not establish causal domain effects or directly comparable task difficulty.",
        "",
        "Reproduce: `uv run python -X utf8 scripts/analyze_b0_transitions.py`. Each PNG is at least 1600 pixels wide and has matching probability/count CSVs.",
    ]
    (OUT / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "training_reconstruction_verified": True,
                "source_sha256": hashes,
                "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Verified all three train priors and wrote", OUT)


if __name__ == "__main__":
    main()
