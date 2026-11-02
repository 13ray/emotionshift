# B0 results: majority, predicted-history inertia and Markov

All values below are percentages except AUROC/AUPRC. Native label spaces;
different dataset headline metrics are not directly comparable.
Recognition means use five donor seeds; deterministic rules use seed 0.
95% intervals resample dialogues 1,000 times for seed 0, not across seeds.

| Dataset / split | Method | Headline F1 | 95% CI (seed 0) | Macro-F1 | Derived shift F1 |
|---|---|---:|---|---:|---:|
| meld / test | majority | 31.27 ± 0.00 | [28.62, 34.31] | 9.28 | 0.00 |
| meld / test | donor | 56.78 ± 0.00 | [54.59, 58.94] | 38.39 | 68.53 |
| meld / test | inertia_pred | 38.30 ± 0.00 | [36.02, 40.71] | 19.49 | 57.94 |
| meld / test | markov_pred | 33.74 ± 0.00 | [31.16, 36.54] | 12.04 | 19.20 |
| meld / test | inertia_oracle (oracle) | 44.86 ± 0.00 | [42.37, 47.45] | 28.49 | 61.34 |
| meld / test | markov_oracle (oracle) | 36.84 ± 0.00 | [34.26, 39.48] | 15.05 | 23.45 |
| dailydialog / test_dedup | majority | 0.00 ± 0.00 | [0.00, 0.00] | 12.86 | 0.00 |
| dailydialog / test_dedup | donor | 53.03 ± 0.00 | [49.94, 55.95] | 46.16 | 50.57 |
| dailydialog / test_dedup | inertia_pred | 21.68 ± 0.00 | [18.62, 24.94] | 29.04 | 32.14 |
| dailydialog / test_dedup | markov_pred | 21.21 ± 0.00 | [17.80, 24.62] | 19.60 | 25.45 |
| dailydialog / test_dedup | inertia_oracle (oracle) | 39.02 ± 0.00 | [34.99, 43.11] | 45.15 | 41.29 |
| dailydialog / test_dedup | markov_oracle (oracle) | 36.84 ± 0.00 | [32.21, 41.34] | 26.98 | 30.78 |
| dailydialog / test | majority | 0.00 ± 0.00 | [0.00, 0.00] | 12.84 | 0.00 |
| dailydialog / test | donor | 53.45 ± 0.00 | [50.54, 56.14] | 48.09 | 50.45 |
| dailydialog / test | inertia_pred | 22.74 ± 0.00 | [19.71, 25.47] | 30.56 | 31.38 |
| dailydialog / test | markov_pred | 22.34 ± 0.00 | [19.16, 25.41] | 20.62 | 25.03 |
| dailydialog / test | inertia_oracle (oracle) | 39.69 ± 0.00 | [35.77, 43.44] | 45.28 | 41.11 |
| dailydialog / test | markov_oracle (oracle) | 37.61 ± 0.00 | [33.30, 41.87] | 27.29 | 30.91 |
| emowoz / test | majority | 0.00 ± 0.00 | [0.00, 0.00] | 11.73 | 0.00 |
| emowoz / test | donor | 75.39 ± 0.00 | [73.50, 77.28] | 52.54 | 73.07 |
| emowoz / test | inertia_pred | 29.23 ± 0.00 | [27.29, 31.35] | 22.37 | 40.47 |
| emowoz / test | markov_pred | 9.33 ± 0.00 | [7.26, 11.43] | 17.47 | 15.99 |
| emowoz / test | inertia_oracle (oracle) | 32.60 ± 0.00 | [30.32, 35.00] | 26.16 | 46.16 |
| emowoz / test | markov_oracle (oracle) | 11.25 ± 0.00 | [8.70, 14.03] | 18.73 | 19.96 |

Headlines: MELD weighted-F1; DailyDialog micro-F1 excluding neutral;
EmoWOZ weighted-F1 excluding neutral. All-neutral output therefore scores
zero on the DailyDialog/EmoWOZ headline despite often high accuracy.

## What the results show

On MELD, majority reaches 31.27 weighted-F1; lagged predicted-history
inertia reaches 38.30, compared with 44.86 for oracle inertia and 56.78
for the supporting current-text recognizer. Oracle inertia reproduces
the proposal's 0.449 pilot; it must not be presented as deployable.
Pure persistence cannot detect a new emotion from the current utterance.
The native-label EmoWOZ run is in-domain and does not establish transfer
performance. Its score must not be compared directly with Ekman-label F1.
Five donor fits converged to identical test predictions/scores in this run;
zero standard deviation does not mean zero uncertainty (see dialogue CIs).

## Markov forecasting (seed 0)

| Dataset / split | History | Eligible turns | Shift rate | AUROC | AUPRC | Brier |
|---|---|---:|---:|---:|---:|---:|
| meld / test | markov_pred | 1864 | 0.538 | 0.558 | 0.570 | 0.255 |
| meld / test | markov_oracle | 1864 | 0.538 | 0.663 | 0.673 | 0.227 |
| dailydialog / test_dedup | markov_pred | 5237 | 0.180 | 0.588 | 0.250 | 0.148 |
| dailydialog / test_dedup | markov_oracle | 5237 | 0.180 | 0.658 | 0.359 | 0.128 |
| dailydialog / test | markov_pred | 5740 | 0.180 | 0.581 | 0.242 | 0.151 |
| dailydialog / test | markov_oracle | 5740 | 0.180 | 0.657 | 0.356 | 0.128 |
| emowoz / test | markov_pred | 7534 | 0.352 | 0.569 | 0.408 | 0.228 |
| emowoz / test | markov_oracle | 7534 | 0.352 | 0.608 | 0.463 | 0.211 |

Forecast probabilities use no current-utterance text; gold-history
oracle rows are references only. AUPRC should be compared with that
split's shift base rate, and Brier is a probability-error metric (lower
is better). These are point estimates, without forecast confidence intervals.

## Runtime and verification

- meld: 4.0 seconds including dev selection, fitting, recognition CIs and exports.
- dailydialog: 53.7 seconds including dev selection, fitting, recognition CIs and exports.
- emowoz: 36.1 seconds including dev selection, fitting, recognition CIs and exports.

CPU run, Python 3.12. The artifact audit recomputed all 72 prediction
files' recognition metrics and verified source identity and CSV schema.
Tests: 70 passed, 5 skipped in this environment (optional neural dependencies absent).
The source manifest hashes raw files, config and training/rule code.
See docs/B0_BASELINES.md for definitions and notebooks/b0_baselines.ipynb
for the executed walkthrough. Training/evaluation commands are not run
again inside the presentation notebook.

## Paired method differences

See [paired/README.md](paired/README.md) for candidate-minus-reference
percentage-point differences and pointwise 95% paired dialogue-bootstrap
intervals. These fixed seed-0 comparisons directly assess the difference,
rather than overlap between separate model intervals. They are exploratory
and do not include multiple-comparison correction.

## Training transition interpretation

See [transitions/README.md](transitions/README.md) for verified train-only
heatmaps, per-history supports, argmax rules, and seed-0 prediction
distributions. It explains neutral collapse and why transition argmax
need not improve non-neutral F1. No current-text evidence enters the prior.
