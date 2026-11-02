# What does the Markov prior learn?

Rows condition on the previous labelled utterance by the **same speaker**, not the previous conversation turn. Columns are that speaker's next/current emotion.
The diagonal is emotional persistence. The neutral column measures return to neutral. Each row sums to one, after the fixed train-marginal pseudo-count.
Heatmap annotations are rounded to two decimals: 0.00 need not be exactly zero. Row support is the observed transition count before smoothing.
The source counts and probabilities are verified against a fresh **train-only** reconstruction. No test transition is fitted, and no predictions or hyperparameters change.

## Learned dynamics and decision rules

| Dataset | Previous emotion | Train support | Stay probability | To neutral | Argmax next label |
|---|---|---:|---:|---:|---|
| meld | neutral | 3409 | 0.588 | 0.588 | neutral |
| meld | joy | 1235 | 0.333 | 0.407 | neutral |
| meld | surprise | 852 | 0.251 | 0.353 | neutral |
| meld | anger | 820 | 0.408 | 0.269 | anger |
| meld | sadness | 482 | 0.280 | 0.347 | neutral |
| meld | disgust | 185 | 0.215 | 0.314 | neutral |
| meld | fear | 191 | 0.172 | 0.325 | neutral |
| dailydialog | neutral | 55807 | 0.878 | 0.878 | neutral |
| dailydialog | joy | 6291 | 0.604 | 0.371 | joy |
| dailydialog | surprise | 1264 | 0.111 | 0.716 | neutral |
| dailydialog | anger | 586 | 0.523 | 0.409 | anger |
| dailydialog | sadness | 675 | 0.354 | 0.544 | neutral |
| dailydialog | disgust | 208 | 0.407 | 0.511 | neutral |
| dailydialog | fear | 103 | 0.404 | 0.469 | neutral |
| emowoz | neutral | 45355 | 0.702 | 0.702 | neutral |
| emowoz | fearful | 355 | 0.062 | 0.623 | neutral |
| emowoz | dissatisfied | 3651 | 0.471 | 0.433 | dissatisfied |
| emowoz | apologetic | 646 | 0.014 | 0.636 | neutral |
| emowoz | abusive | 62 | 0.175 | 0.424 | neutral |
| emowoz | excited | 752 | 0.050 | 0.709 | neutral |
| emowoz | satisfied | 6420 | 0.442 | 0.520 | neutral |

## Why a learned transition prior need not beat inertia

- **meld:** 6 of 7 history labels map to neutral under argmax (neutral, joy, surprise, sadness, disgust, fear). On test, Markov predicts neutral for 94.0% of targets and never predicts: joy, surprise, sadness, disgust, fear.
- **dailydialog:** 5 of 7 history labels map to neutral under argmax (neutral, surprise, sadness, disgust, fear). On test_dedup, Markov predicts neutral for 92.2% of targets and never predicts: surprise, sadness, disgust, fear.
- **emowoz:** 6 of 7 history labels map to neutral under argmax (neutral, fearful, apologetic, abusive, excited, satisfied). On test, Markov predicts neutral for 94.1% of targets and never predicts: fearful, apologetic, abusive, excited, satisfied.

A row can have substantial self-transition probability without persistence being the most likely outcome. For example, if P(joy|joy) is below P(neutral|joy), argmax predicts neutral rather than copying joy.
The transition estimator learns a population average; it cannot observe what the current utterance says. Even a correct previous emotion cannot identify an unexpected turning point.
The conditioning emotion comes from a fallible donor recognizer, adding history error. Oracle history removes that input error but not the model's lack of current-utterance evidence.
Argmax prioritizes the most likely single label, not the dataset's non-neutral F1. Collapsing many histories to neutral can therefore hurt emotional-class recall, especially on EmoWOZ, whose headline excludes neutral.

## Per-turn changes are diagnostics, not an F1 decomposition

`markov_vs_inertia_cases.csv` counts changes and cases where only one method is correct. Both methods use exactly aligned seed-0 targets. These accuracy-style counts do not algebraically decompose class-weighted or micro-F1.
For example, on MELD test Markov corrects 282 inertia errors but loses 136 previously correct targets, increasing accuracy by 146/2610 (5.59 percentage points). Yet its weighted-F1 is 4.56 points lower. More correct majority-class targets need not compensate for lost emotional-class precision/recall in F1.
`prediction_distribution.csv` includes gold/predicted counts for every native label, including zero predictions. Only labelled targets are counted. First turns have the same train-majority fallback and do not explain differences between the rules.
Read the [paired F1 comparison](../paired/README.md) for uncertainty on score differences: Markov is below inertia on MELD and EmoWOZ in these fixed-checkpoint comparisons; DailyDialog's overall difference interval crosses zero.

## Limits

Small training row supports imply less reliable transition estimates; the heatmaps display support beside every previous label. This analysis does not estimate uncertainty on individual transition probabilities.
Within-EmoWOZ native labels are retained. Differences in probability patterns across datasets are descriptive and do not establish causal domain effects or directly comparable task difficulty.

Reproduce: `uv run python -X utf8 scripts/analyze_b0_transitions.py`. Each PNG is at least 1600 pixels wide and has matching probability/count CSVs.
