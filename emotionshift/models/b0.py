"""B0 floors: train majority, lagged recognizer inertia and Markov prior.

Predicted-history methods never read evaluation emotion/stratum columns.
The history is the previous *donor recognizer* prediction, not the recursively
copied B0 output. This follows notebooks/02_quick_baselines.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HistoryBaselines:
    def __init__(self, labels):
        self.labels = list(labels)

    def fit(self, train):
        """Use labelled train rows only; estimate within-speaker transitions."""
        ordered = train.sort_values(["dataset", "split", "dialogue_id", "turn"])
        ordered = ordered[ordered.emotion.notna()].copy()
        if ordered.empty or not ordered.emotion.isin(self.labels).all():
            raise ValueError("Training labels must be nonempty and in the label space")
        counts = ordered.emotion.value_counts().reindex(self.labels, fill_value=0)
        self.majority = counts.idxmax()  # ties follow the dataset's fixed label order
        self.marginal = counts.to_numpy(dtype=float) / counts.sum()
        keys = ["dataset", "split", "dialogue_id", "speaker"]
        previous = ordered.groupby(keys, sort=False).emotion.shift()
        transitions = (
            pd.crosstab(previous, ordered.emotion)
            .reindex(index=self.labels, columns=self.labels, fill_value=0)
            .to_numpy(dtype=float)
        )
        self.transition_counts = transitions
        # One fixed marginal pseudo-count; unseen histories fall back to train marginal.
        smoothed = transitions + self.marginal[None, :]
        self.transition = smoothed / smoothed.sum(axis=1, keepdims=True)
        return self

    def predict(self, frame, donor_predictions):
        """Return predictions aligned with frame.index, including unlabelled context.

        donor_predictions is a Series with that same index; null means no
        recognizer observation (e.g. a system turn). State is local to each call.
        Initial output is train majority; no current text prediction is used.
        """
        donor = pd.Series(donor_predictions, index=frame.index)
        if not donor.dropna().isin(self.labels).all():
            raise ValueError("Donor predictions outside label space")
        history, rows = {}, {}
        order = frame.sort_values(["dataset", "split", "dialogue_id", "turn"])
        ids = {label: i for i, label in enumerate(self.labels)}
        for ix, row in order.iterrows():
            key = (row.dataset, row.split, row.dialogue_id, row.speaker)
            previous = history.get(key)
            distribution = self.marginal if previous is None else self.transition[ids[previous]]
            rows[ix] = {
                "majority": self.majority,
                "inertia_pred": previous if previous is not None else self.majority,
                "markov_pred": self.labels[int(distribution.argmax())],
                "history_pred": previous,
                "forecast_shift_prob": (
                    np.nan if previous is None else 1 - distribution[ids[previous]]
                ),
            }
            # Update only AFTER emitting the current prediction.
            if pd.notna(donor.loc[ix]):
                history[key] = donor.loc[ix]
        return pd.DataFrame.from_dict(rows, orient="index").reindex(frame.index)

    def oracle(self, frame):
        """Explicit gold-history references; never headline/deployable results."""
        result = self.predict(frame, frame.emotion)
        return result.rename(
            columns={
                "inertia_pred": "inertia_oracle",
                "markov_pred": "markov_oracle",
            }
        )
