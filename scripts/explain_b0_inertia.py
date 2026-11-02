"""Explain lagged donor inertia with a controlled toy and saved MELD examples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from emotionshift.data.labels import EMOTIONS
from emotionshift.data.shifts import add_shift_labels
from emotionshift.models.b0 import HistoryBaselines

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/b0/inertia_examples"
KEYS = ["dataset", "split", "dialogue_id", "turn", "speaker"]
DISPLAY = [
    "turn",
    "speaker",
    "text",
    "emotion",
    "donor_pred",
    "history_pred",
    "inertia_pred",
    "inertia_oracle",
    "stratum",
]


def illustrate():
    """Manual toy recognizer outputs, not actual SVM predictions or benchmark data."""
    train = pd.DataFrame(
        {
            "dataset": ["illustration"] * 5,
            "split": ["train"] * 5,
            "dialogue_id": [0] * 5,
            "turn": range(5),
            "speaker": ["A"] * 5,
            "emotion": ["neutral", "neutral", "neutral", "joy", "anger"],
        }
    )
    frame = pd.DataFrame(
        {
            "dataset": ["illustration"] * 6,
            "split": ["test"] * 6,
            "dialogue_id": [0] * 6,
            "turn": range(6),
            "speaker": ["A", "B", "A", "B", "A", "A"],
            "text": [
                "Great, we got the tickets!",
                "Okay.",
                "I am still so happy!",
                "Wonderful, I can join you!",
                "Wait, you cancelled them? I am furious!",
                "I am still angry about it.",
            ],
            "emotion": ["joy", "neutral", "joy", "joy", "anger", "anger"],
        }
    )
    frame = add_shift_labels(frame)
    # Intentionally wrong first A prediction exposes one-step error carry-over.
    donor = pd.Series(["anger", "neutral", "joy", "joy", "anger", "anger"], index=frame.index)
    model = HistoryBaselines(EMOTIONS).fit(train)
    predicted = model.predict(frame[KEYS], donor)
    oracle = model.oracle(frame)
    frame["donor_pred"] = donor
    frame["history_pred"] = predicted.history_pred
    frame["inertia_pred"] = predicted.inertia_pred
    frame["inertia_oracle"] = oracle.inertia_oracle
    frame["explanation"] = [
        "First A turn: train-majority neutral; current donor anger is saved for A's next turn.",
        "First B turn: independent B history; A's prediction is not copied.",
        "Stable A: previous donor anger was wrong and is copied now, although current donor joy is correct.",
        "B shifts: previous B donor neutral is copied; intervening A predictions do not enter B history.",
        "A shifts: previous donor joy was correct, but is now stale; even gold-history oracle misses the shift.",
        "Stable A: copy the previous donor anger correctly. The earlier copied output joy is NOT fed back.",
    ]
    return frame


def real_cases():
    frame = pd.read_csv(ROOT / "predictions/b0_inertia_pred_meld_test_seed0.csv")
    oracle = pd.read_csv(ROOT / "predictions/b0_inertia_oracle_meld_test_seed0.csv")
    frame = frame.merge(
        oracle[[*KEYS, "pred"]].rename(columns={"pred": "inertia_oracle"}),
        on=KEYS,
        validate="one_to_one",
    )
    frame = frame.rename(columns={"pred": "inertia_pred"}).sort_values(KEYS[:-1])
    scope = ["dataset", "split", "dialogue_id", "speaker"]
    frame["previous_turn"] = frame.groupby(scope).turn.shift()
    history = frame.groupby(scope).donor_pred.shift()
    pd.testing.assert_series_equal(frame.history_pred, history, check_names=False)
    frame["target_row"] = False
    selected = []
    conditions = {
        "stable_error_propagation": (
            frame.stratum.eq("stable")
            & frame.inertia_pred.ne(frame.emotion)
            & frame.inertia_oracle.eq(frame.emotion)
            & frame.donor_pred.eq(frame.emotion)
        ),
        "shift_even_correct_history_fails": (
            frame.stratum.eq("shift")
            & frame.history_pred.eq(frame.prev_emotion)
            & frame.inertia_pred.ne(frame.emotion)
            & frame.donor_pred.eq(frame.emotion)
        ),
    }
    for name, mask in conditions.items():
        candidates = frame[mask].copy()
        # Prefer readable sentences over fragments such as "I'm" or "Oh.".
        word_count = candidates.text.str.split().str.len()
        readable = candidates[word_count.between(4, 25)]
        if not readable.empty:
            candidates = readable.copy()
        if candidates.empty:
            raise ValueError(f"No explanatory MELD example for {name}")
        # Compact examples for explanation; not random or representative sampling.
        candidates["length"] = candidates.text.str.len()
        target = candidates.sort_values(["length", "dialogue_id", "turn"], kind="stable").iloc[0]
        context = frame[
            frame.dialogue_id.eq(target.dialogue_id)
            & frame.turn.between(target.previous_turn, target.turn)
        ].copy()
        context["target_row"] = context.turn.eq(target.turn)
        context["case"] = name
        context["selection_candidates"] = len(candidates)
        selected.append(context)
    return pd.concat(selected, ignore_index=True)


def markdown_table(frame, columns):
    def cell(value):
        return "—" if pd.isna(value) else str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *[
                "| " + " | ".join(cell(v) for v in row) + " |"
                for row in frame[columns].itertuples(index=False, name=None)
            ],
        ]
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    toy, real = illustrate(), real_cases()
    toy.to_csv(OUT / "controlled_example.csv", index=False)
    real.to_csv(OUT / "meld_seed0_cases.csv", index=False)
    text = [
        "# What does inertia copy?",
        "",
        "**Copy the same speaker's previous SVM prediction, not their previous gold emotion or previous inertia output.**",
        "",
        "Emit inertia from stored history first, then update that speaker's history with the current SVM prediction.",
        "Other speakers do not overwrite this history. Dialogues and splits reset independently; first turns use the training majority, neutral in this example.",
        "",
        "## 1. Controlled illustration (not a benchmark result)",
        "",
        "Text, gold labels and donor predictions are manually specified to isolate the mechanism. Turns are zero-based.",
        "donor_pred is the current SVM prediction; history_pred is its previous prediction stored before the current prediction.",
        "emotion is gold, used for evaluation only and never as an input to predicted-history inference.",
        "",
        markdown_table(toy, DISPLAY),
        "",
        "- **A at turn 2:** gold remains joy, but the donor misclassified turn 0 as anger. Inertia copies that error even though the current donor correctly predicts joy.",
        "- **A at turn 4:** previous joy was correct, but A now shifts to anger. Even the gold-history oracle copies stale joy and fails.",
        "- **A at turn 5:** copying the donor's anger from turn 4 restores correctness; inertia's erroneous joy output is not fed back.",
        "- **Independent B history:** at turn 3, B copies B's neutral from turn 1, not A's joy from turn 2.",
        "",
        "**Two failures: an earlier recognition error carries over, or correct history becomes stale after a genuine emotion shift.**",
        "This is one-step lagged recognition, not an indefinite chain of recursively copied inertia outputs.",
        "",
        "## 2. Real saved-prediction MELD cases (seed 0)",
        "",
        "Cases read existing prediction exports. No models are retrained and no experimental results change.",
        "Within each condition, prefer targets of 4 to 25 words, then select the shortest target. These post-hoc examples are neither representative sampling nor tuning.",
        "Each case includes the target speaker's previous turn and any intervening speakers. target_row=True identifies the explained turn.",
    ]
    for name, group in real.groupby("case", sort=False):
        text += [
            "",
            f"### {name}",
            "",
            markdown_table(group, ["dialogue_id", *DISPLAY, "target_row"]),
        ]
    text += [
        "",
        "## Suggested explanation",
        "",
        "Predicted-history inertia uses the recognizer's previous judgement as a prior. It can inherit recognition errors; even perfectly recognized history lags behind a genuine emotion shift. The oracle helps distinguish these mechanisms but is not a deployable result.",
        "",
        "Reproduce: `uv run python -X utf8 scripts/explain_b0_inertia.py`. Models, labels, test scores and hyperparameters are unchanged.",
    ]
    (OUT / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("Wrote controlled illustration and saved-prediction MELD cases to", OUT)


if __name__ == "__main__":
    main()
