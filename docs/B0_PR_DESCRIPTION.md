# Complete B0 majority, inertia and Markov baselines

Add the workstream-1 B0 rules and reproducible results on MELD, DailyDialog
(de-duplicated headline plus official test) and native-label EmoWOZ. Predicted
history comes from a train-only utterance SVM selected on dev, following the
existing quick-baseline notebook. Gold-history variants are explicitly oracle
references. First turns use train majority; speaker history resets across
dialogues and splits. System turns are excluded from targets.

Includes five-seed supporting-recognizer experiments, train transitions,
stratified/per-class metrics, derived shift detection, 1,000-resample dialogue
recognition CIs, Markov forecasting scores, 72 audited prediction exports,
an executed notebook, 2400px figures with source CSVs, and narration/reference
drafts. Deterministic majority/oracle recognition rules are exported once.
Native EmoWOZ is an in-domain extension, not the optional cross-domain valence
experiment. No existing shared data or metric definitions are modified.

Also includes an explicit inertia walkthrough and 48 exploratory paired
seed-0 contrasts over overall/stable/shift groups, using shared dialogue-level
resampling. Difference intervals are pointwise, not adjusted for multiple
comparisons; oracle contrasts are diagnostics only. Prediction hashes and
point-score checks make the analysis reproducible without retraining.

Train-only transition heatmaps with row supports and matching source CSVs
explain self-persistence, return to neutral and argmax label collapse. Saved
prediction distributions and per-turn correctness diagnostics link the rules
to their observed limitations; native labels and existing scores are unchanged.

Validation: 70 passed, 5 skipped (optional neural dependencies unavailable);
new-file Ruff checks pass. Audit verifies original twelve-column ordering,
source row identity, unique keys, native labels and recomputed recognition scores.
Oracle MELD inertia reproduces 0.449 weighted-F1; oracle Markov reproduces
AUROC 0.663 / AUPRC 0.673. Forecast uncertainty intervals are not estimated.

Run: `uv run python -X utf8 scripts/train_b0.py`, then
`uv run python -X utf8 scripts/make_b0_artifacts.py`.
See `docs/B0_BASELINES.md` for definitions and `reports/b0/results.md` for results.

Final preflight: `uv run python -X utf8 scripts/verify_b0.py --run-tests`.
Submission documentation and notebook explanations are in English; personal
reference notes remain outside the repository. Slides and own-voice narration
are separate presentation deliverables.
