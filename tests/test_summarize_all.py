import numpy as np

from emotionshift.data.labels import EMOTIONS
from emotionshift.eval.metrics import scores
from scripts.summarize_all import (
    batch_scores,
    bootstrap_seed_mean,
    dialogue_counts,
    paired_test,
)

K = len(EMOTIONS)


def _toy(n=600, dialogues=60, seeds=3, accuracy=0.6, seed=0):
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, K, n)
    preds = np.where(rng.random((seeds, n)) < accuracy, gold, rng.integers(0, K, (seeds, n)))
    dialogue = np.sort(rng.integers(0, dialogues, n))
    dialogue = np.unique(dialogue, return_inverse=True)[1]
    return gold, preds, dialogue, int(dialogue.max()) + 1


def test_batch_scores_match_sklearn():
    gold, preds, dialogue, d = _toy()
    counts = dialogue_counts(gold, preds, dialogue, np.ones(len(gold), bool), d)
    batch = batch_scores(counts.sum(0).reshape(len(preds), K, K))
    for s, pred in enumerate(preds):
        expected = scores([EMOTIONS[i] for i in gold], [EMOTIONS[i] for i in pred])
        for metric, values in batch.items():
            assert np.isclose(values[s], expected[metric])


def test_dialogue_counts_respect_mask():
    gold, preds, dialogue, d = _toy()
    keep = gold % 2 == 0
    counts = dialogue_counts(gold, preds, dialogue, keep, d)
    assert counts.shape == (d, len(preds), K * K)
    assert np.all(counts.sum(axis=(0, 2)) == keep.sum())


def test_identical_systems_are_not_different():
    gold, preds, dialogue, d = _toy()
    counts = dialogue_counts(gold, preds, dialogue, np.ones(len(gold), bool), d)
    result = paired_test(counts, counts, "weighted_f1", 200, 200, np.random.default_rng(0))
    assert result["diff"] == 0
    assert result["p"] == 1


def test_better_system_is_detected():
    gold, weak, dialogue, d = _toy(accuracy=0.4)
    strong = np.where(np.random.default_rng(1).random(weak.shape) < 0.5, gold, weak)
    keep = np.ones(len(gold), bool)
    a = dialogue_counts(gold, weak, dialogue, keep, d)
    b = dialogue_counts(gold, strong, dialogue, keep, d)
    result = paired_test(a, b, "macro_f1", 500, 500, np.random.default_rng(0))
    assert result["diff"] > 0 and result["ci_low"] > 0 and result["p"] < 0.01


def test_seed_mean_ci_brackets_estimate():
    gold, preds, dialogue, d = _toy()
    counts = dialogue_counts(gold, preds, dialogue, np.ones(len(gold), bool), d)
    estimate = batch_scores(counts.sum(0).reshape(len(preds), K, K))["weighted_f1"].mean()
    low, high = bootstrap_seed_mean(counts, "weighted_f1", 500, np.random.default_rng(0))
    assert low < estimate < high
