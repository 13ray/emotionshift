import numpy as np
import pandas as pd
import pytest

from emotionshift.models.b0 import HistoryBaselines


def table():
    return pd.DataFrame(
        {
            "dataset": ["toy"] * 6,
            "split": ["train"] * 6,
            "dialogue_id": [0, 0, 0, 0, 1, 1],
            "turn": [0, 1, 2, 3, 0, 1],
            "speaker": ["A", "B", "A", "B", "A", "A"],
            "emotion": ["neutral", "joy", "joy", "neutral", "neutral", "neutral"],
        }
    )


def test_train_transition_estimation_and_unseen_history():
    model = HistoryBaselines(["neutral", "joy", "fear"]).fit(table())
    assert model.majority == "neutral"
    np.testing.assert_array_equal(model.transition_counts, [[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    np.testing.assert_allclose(model.transition.sum(axis=1), 1)
    np.testing.assert_allclose(model.transition[2], model.marginal)


def test_causal_history_speaker_dialogue_and_split_boundaries():
    frame = table()
    frame.loc[5, "split"] = "test"
    donor = pd.Series(["joy", "neutral", "neutral", "joy", "joy", "neutral"])
    model = HistoryBaselines(["neutral", "joy"]).fit(table())
    output = model.predict(frame.drop(columns="emotion"), donor)
    assert output.inertia_pred.tolist() == [
        "neutral",
        "neutral",
        "joy",
        "neutral",
        "neutral",
        "neutral",
    ]
    changed = donor.copy()
    changed.loc[2] = "joy"
    # Current/future evidence cannot affect a prediction made before that evidence.
    pd.testing.assert_frame_equal(output.iloc[:3], model.predict(frame, changed).iloc[:3])
    # Reordering input retains alignment and chronological causality.
    shuffled = frame.sample(frac=1, random_state=2)
    pd.testing.assert_frame_equal(model.predict(shuffled, donor).sort_index(), output)


def test_gold_independence_and_oracle_separation():
    model = HistoryBaselines(["neutral", "joy"]).fit(table())
    frame = table()
    donor = pd.Series(["joy"] * len(frame))
    predicted = model.predict(frame, donor)
    frame["emotion"] = "joy"
    frame["prev_emotion"] = "joy"
    frame["stratum"] = "shift"
    pd.testing.assert_frame_equal(predicted, model.predict(frame, donor))
    assert "inertia_oracle" in model.oracle(frame)


def test_unlabelled_context_does_not_update_history():
    frame = table().iloc[:3].copy()
    frame["speaker"] = "A"
    model = HistoryBaselines(["neutral", "joy"]).fit(table())
    result = model.predict(frame, pd.Series(["joy", None, "neutral"]))
    assert result.inertia_pred.tolist() == ["neutral", "joy", "joy"]
    with pytest.raises(ValueError, match="outside label space"):
        model.predict(frame, pd.Series(["bad"] * 3))


def test_bootstrap_matches_shared_dialogue_resampling():
    from emotionshift.eval.metrics import bootstrap_ci
    from scripts.train_b0 import dialogue_intervals

    frame = table()
    frame["pred"] = ["neutral", "joy", "neutral", "joy", "neutral", "joy"]
    labels = ["neutral", "joy"]
    actual = dialogue_intervals(frame, labels, n_boot=30)
    for metric in actual:
        np.testing.assert_allclose(
            actual[metric], bootstrap_ci(frame, metric, n_boot=30, labels=labels)
        )
