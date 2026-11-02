# B0 paired dialogue-bootstrap differences

Difference = candidate - reference, in **percentage points**, not relative percent.
Each replicate resamples the same dialogues for both methods (1,000 replicates; seed 0).
Use the shared `paired_bootstrap` helper without changing its resampling or metric definitions.
Native dataset labels and dataset-specific headline metrics are retained.
These are seed-0 checkpoint comparisons, not differences between five-seed means.

## Overall headline splits

| Dataset / split | Candidate - reference | Delta (pp) | 95% CI (pp) | Interval |
|---|---|---:|---|---|
| meld / test | inertia_pred - majority | +7.04 | [+5.05, +8.92] | above_zero |
| meld / test | donor - inertia_pred | +18.48 | [+16.16, +20.73] | above_zero |
| meld / test | markov_pred - inertia_pred | -4.56 | [-6.25, -2.74] | below_zero |
| meld / test | inertia_oracle - inertia_pred [oracle diagnostic] | +6.55 | [+4.52, +8.78] | above_zero |
| dailydialog / test_dedup | inertia_pred - majority | +21.68 | [+18.62, +24.94] | above_zero |
| dailydialog / test_dedup | donor - inertia_pred | +31.35 | [+28.14, +34.49] | above_zero |
| dailydialog / test_dedup | markov_pred - inertia_pred | -0.47 | [-1.92, +0.84] | includes_zero |
| dailydialog / test_dedup | inertia_oracle - inertia_pred [oracle diagnostic] | +17.34 | [+14.01, +20.64] | above_zero |
| emowoz / test | inertia_pred - majority | +29.24 | [+27.29, +31.35] | above_zero |
| emowoz / test | donor - inertia_pred | +46.16 | [+43.51, +48.75] | above_zero |
| emowoz / test | markov_pred - inertia_pred | -19.90 | [-21.69, -18.12] | below_zero |
| emowoz / test | inertia_oracle - inertia_pred [oracle diagnostic] | +3.36 | [+1.66, +5.07] | above_zero |

## Interpretation and limits

- An interval entirely above zero supports a positive difference for this fixed comparison and checkpoint under dialogue resampling; below zero supports a negative difference.
- An interval including zero does not establish a clear direction; it does not prove equivalence.
- Read differences directly. Overlap of two separately estimated method intervals is not a paired comparison.
- Stable/shift estimates resample the original split's whole dialogue population, then score only the selected turns. Dialogues without those turns remain in the resampling population.
- Majority -> inertia asks whether lagged recognition helps over a constant floor. Inertia -> donor measures the gain from using current-text recognition instead of copying its previous judgement.
- Inertia -> Markov compares two history-only rules using the same donor. Inertia -> oracle is a gold-history diagnostic, never a deployable model improvement.
- These additional post-hoc analyses are exploratory. Intervals are pointwise and not adjusted for multiple comparisons; no family-wide significance or causal effect is claimed.
- This is inference about held-out dialogue sampling for fixed trained checkpoints, not uncertainty over training seeds or domains. Scores and differences across datasets use different headline metrics and must not be ranked as equivalent tasks.

`differences.csv` includes all/stable/shift groups and the secondary DailyDialog official test. Prediction files are unchanged; there is no retraining or test-based model selection.

Reproduce: `uv run python -X utf8 scripts/analyze_b0_paired.py`.
