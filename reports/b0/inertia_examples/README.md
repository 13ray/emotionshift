# What does inertia copy?

**Copy the same speaker's previous SVM prediction, not their previous gold emotion or previous inertia output.**

Emit inertia from stored history first, then update that speaker's history with the current SVM prediction.
Other speakers do not overwrite this history. Dialogues and splits reset independently; first turns use the training majority, neutral in this example.

## 1. Controlled illustration (not a benchmark result)

Text, gold labels and donor predictions are manually specified to isolate the mechanism. Turns are zero-based.
donor_pred is the current SVM prediction; history_pred is its previous prediction stored before the current prediction.
emotion is gold, used for evaluation only and never as an input to predicted-history inference.

| turn | speaker | text | emotion | donor_pred | history_pred | inertia_pred | inertia_oracle | stratum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | A | Great, we got the tickets! | joy | anger | — | neutral | neutral | first |
| 1 | B | Okay. | neutral | neutral | — | neutral | neutral | first |
| 2 | A | I am still so happy! | joy | joy | anger | anger | joy | stable |
| 3 | B | Wonderful, I can join you! | joy | joy | neutral | neutral | neutral | shift |
| 4 | A | Wait, you cancelled them? I am furious! | anger | anger | joy | joy | joy | shift |
| 5 | A | I am still angry about it. | anger | anger | anger | anger | anger | stable |

- **A at turn 2:** gold remains joy, but the donor misclassified turn 0 as anger. Inertia copies that error even though the current donor correctly predicts joy.
- **A at turn 4:** previous joy was correct, but A now shifts to anger. Even the gold-history oracle copies stale joy and fails.
- **A at turn 5:** copying the donor's anger from turn 4 restores correctness; inertia's erroneous joy output is not fed back.
- **Independent B history:** at turn 3, B copies B's neutral from turn 1, not A's joy from turn 2.

**Two failures: an earlier recognition error carries over, or correct history becomes stale after a genuine emotion shift.**
This is one-step lagged recognition, not an indefinite chain of recursively copied inertia outputs.

## 2. Real saved-prediction MELD cases (seed 0)

Cases read existing prediction exports. No models are retrained and no experimental results change.
Within each condition, prefer targets of 4 to 25 words, then select the shortest target. These post-hoc examples are neither representative sampling nor tuning.
Each case includes the target speaker's previous turn and any intervening speakers. target_row=True identifies the explained turn.

### stable_error_propagation

| dialogue_id | turn | speaker | text | emotion | donor_pred | history_pred | inertia_pred | inertia_oracle | stratum | target_row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 140 | 12 | Joey | Y'know I mean I-I-I'm an actor, I don't have the discipline that takes, y'know? | sadness | neutral | anger | anger | anger | shift | False |
| 140 | 13 | Joey | I can't do it. | sadness | sadness | neutral | neutral | sadness | stable | True |

### shift_even_correct_history_fails

| dialogue_id | turn | speaker | text | emotion | donor_pred | history_pred | inertia_pred | inertia_oracle | stratum | target_row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 58 | 12 | Chandler | Yeah! Totally! Totally, and you? | joy | joy | neutral | neutral | sadness | shift | False |
| 58 | 13 | Monica | Great! It's so amazing! I mean, last time Dr. Roger came over, I was | joy | joy | neutral | neutral | neutral | shift | False |
| 58 | 14 | Chandler | I bet he can. | neutral | neutral | joy | joy | joy | shift | True |

## Suggested explanation

Predicted-history inertia uses the recognizer's previous judgement as a prior. It can inherit recognition errors; even perfectly recognized history lags behind a genuine emotion shift. The oracle helps distinguish these mechanisms but is not a deployable result.

Reproduce: `uv run python -X utf8 scripts/explain_b0_inertia.py`. Models, labels, test scores and hyperparameters are unchanged.
