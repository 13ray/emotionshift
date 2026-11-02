# B0: Majority, emotional inertia and Markov transition prior (workstream 1)

This implements the B0 row of `docs/proposal.md`, Section 5.1. It is separate
from B1 (TF-IDF/SVM) and from EmoWOZ dataset integration. No shared loader,
label space, shift definition, or evaluation helper is changed.

## Run

From the repository root with the current Python 3.12 uv environment:

```powershell
uv sync --locked
uv run python -X utf8 scripts/train_b0.py
uv run python -X utf8 scripts/make_b0_artifacts.py
uv run python -X utf8 scripts/verify_b0.py --run-tests
```

CPU only; no additional dependencies or neural checkpoints. Measured full-run
times are recorded per dataset in `reports/b0/<dataset>/summary.json`.
UTF-8 mode also avoids Windows GBK errors in the existing MELD-FR tests.
Raw data must already be downloaded into `data/raw/`.

## Rules fixed before test evaluation

| Name | Rule | Reads test gold? |
|---|---|---|
| majority | Always output the most frequent **training** class; ties follow the fixed dataset label order | No |
| donor | Supporting utterance-only TF-IDF/SVM recognizer | No |
| inertia_pred | Copy the donor's prediction at the previous labelled turn of the **same speaker** | No |
| markov_pred | Argmax of the training transition distribution, conditioned on that previous donor prediction | No |
| inertia_oracle | Copy the previous same-speaker **gold** emotion | Yes; reference only |
| markov_oracle | Training transition distribution conditioned on previous same-speaker gold emotion | Yes; reference only |

History resets at every dataset/split/dialogue/speaker boundary. First turns
use the training majority; the Markov distribution there is the training
marginal. Unlabelled system turns are never donor observations or scored targets.
The oracle variants use the same first-turn convention, so the MELD oracle
inertia weighted-F1 reproduces the proposal's 0.449 reference.

The donor is needed to define a useful *predicted-history* baseline. We follow
the existing `notebooks/02_quick_baselines.ipynb`: copy its previous prediction,
not the previous output of the copying baseline itself. Pure recursive inertia
initialized at the majority would remain the same class forever and duplicate
the majority baseline. Reports explicitly name this dependency; it is not a
history-free floor or a replacement for the B1 author's full dev grid.

The donor uses the existing `TfidfBaseline` implementation: word 1–2-grams,
balanced LinearSVC, C in [0.1, 0.3, 1.0], selected on the dataset's dev headline
metric with seed 0. Selection is written before any test predictions, then
train-only fitting is repeated for seeds 0, 1, 2, 3, 4. Test data never chooses
hyperparameters. No dev refit or test label feedback is used.

Markov counts use only training within-speaker labelled transitions. Add a
fixed single pseudo-count distributed by the train label marginal; an unseen
history row falls back to that marginal. No smoothing search uses test data.
Forecast shift probability is `1 - P(previous_label | previous_label)` and is
scored only on non-first labelled turns, before the current utterance is read.
AUROC, AUPRC, base rate and Brier score are reported, with predicted-history
and gold-history/oracle results separated. Forecast CIs are not estimated.

## Datasets and scoring

- MELD: official test; weighted-F1 headline.
- DailyDialog: `test_dedup` headline, official `test` secondary;
  micro-F1 excluding neutral headline.
- EmoWOZ: **native** seven-label space via `emotions("emowoz")`; labelled user
  targets only; weighted-F1 excluding neutral headline. This is an in-domain
  extension, not the proposal's optional cross-domain valence experiment.
  No lossy Ekman mapping or comparison of equivalent label names is claimed.

All recognition metrics, confusion matrices, per-class and first/stable/shift
breakdowns use the shared evaluation functions with the appropriate label list.
Derived T2 shift detection uses each baseline's own previous output, not donor
history; a model can change donor-conditioned emotion output yet still have
poor derived shift detection.

Seed variation describes donor fitting only. Deterministic majority and oracle
rules are evaluated/exported once at seed 0; pretending five identical runs
were independent would add no uncertainty information. Each recognition metric
has a 95% percentile interval from 1,000 dialogue resamples of seed 0. The
confusion-aggregation implementation uses shared `scores_from_confusion` and a
unit test verifies equivalence to shared `bootstrap_ci`. Intervals are not
across-seed confidence intervals. F1 includes the fixed full native label list,
including classes absent in a stratum; all-neutral predictions can therefore
have nonzero weighted-F1 and zero non-neutral headline F1.

## Files to hand in

### Markov transition interpretation

`uv run python -X utf8 scripts/analyze_b0_transitions.py` verifies the saved
counts and smoothed matrices against a fresh train-only reconstruction. Each
dataset has a >=1600px annotated heatmap with matching probability/count CSVs.
`reports/b0/transitions/` also contains self-transition/neutral-return rates,
row supports, argmax next-label rules, seed-0 gold/prediction distributions and
Markov-vs-inertia correctness counts. These counts diagnose decision changes;
they are not a decomposition of F1. The explanation distinguishes rare-history
uncertainty, donor history errors, absent current-text evidence and argmax's
neutral preference. Native EmoWOZ labels are retained. No cellwise transition
probability CIs or causal domain effects are claimed. The notebook includes
all three heatmaps and tables; no retraining is needed.

### Paired differences

Run `uv run python -X utf8 scripts/analyze_b0_paired.py` to compare saved
seed-0 predictions using shared `paired_bootstrap`. Four contrasts: inertia
minus majority, donor minus inertia, Markov minus inertia, and oracle inertia
minus predicted inertia (diagnostic only). Report all/stable/shift and both
DailyDialog test views. Each of 1,000 replicates uses the same sampled dialogues
for both methods. `reports/b0/paired/` contains percentage-point differences,
pointwise 95% intervals, source tables, a figure and prediction hashes. These
exploratory post-hoc comparisons are not adjusted for multiple testing or
training-seed uncertainty. Intervals crossing zero do not prove equivalence.
No prediction or selected model changes. The executed notebook includes the
headline table and figure.

### Inertia walkthrough

`uv run python -X utf8 scripts/explain_b0_inertia.py` generates an explicitly
artificial two-speaker example and two real saved-prediction MELD cases in
`reports/b0/inertia_examples/`. The English README distinguishes copying a
wrong previous recognizer label from failing to anticipate an actual shift.
It shows why this is one-step error carry-over, not recursive inertia-output
feedback. The executed notebook includes both cases. Real cases are selected
post hoc for explanation only; they do not estimate error prevalence or tune
the model. No retraining or scoring rules change.

| Artifact | Location |
|---|---|
| Model rules | `emotionshift/models/b0.py` |
| Reproducible experiment/config | `scripts/train_b0.py`, `configs/b0_baselines.yaml` |
| Summary comparison and manifest | `reports/b0/comparison.csv`, `run_manifest.json` |
| Dev selection, train transitions, per-seed metrics, summaries | `reports/b0/<dataset>/` |
| Stratified, class and confusion tables (seed 0) | same directory |
| Forecast probabilities and forecast metrics | same directory |
| Recognition predictions | `predictions/b0_<name>_<dataset>_<split>_seed<k>.csv` |
| Executed walkthrough | `notebooks/b0_baselines.ipynb` |
| Figures at least 1600px wide + source CSVs | `reports/b0/figures/` |
| Video script, references, contribution draft | `deliverables/01_inertia/` |
| Release audit (schema, scores, hashes, English text, notebook, figures, tests) | `scripts/verify_b0.py`, `reports/b0/release_check.json` |

Prediction CSVs retain the original twelve columns **first**, then provenance,
previous emotion and donor/history diagnostics. Oracle filenames and summary
flags prevent mixing them into deployable headline comparisons. No system row
appears in training or evaluated prediction CSVs.

For the video, use notebook sections **Baseline rules**, **Main comparison**,
and **Shift vs stable**. Each code cell is at most 15 lines; full training is
run once by the script, not during the presentation. See the generated results
notes for measured findings, not assumed improvements. Slides must use the
shared team template, and narration/signatures/forms remain personal tasks.

## Submission boundary

Commit only the B0 source/config/tests, prediction exports, reports, executed
notebook, English documentation and English presentation drafts. Personal
reference notes are stored outside this repository and are not part of the
submission. Raw data, environments and model caches remain ignored.

`verify_b0.py --run-tests` checks the full repository test suite and B0 source
style, audits recognition and derived-shift metrics from every prediction CSV,
verifies recorded training/analysis input hashes, checks all notebook code
cells were executed without errors, and rejects Chinese text in B0 release
files (including decoded notebook content). It also checks figure widths and
the team's 50 MB per-file limit. This audit does not create slides, record
narration or sign personal forms.

Git text hash checks tolerate LF/CRLF-only conversion, so Windows/Linux checkouts
can verify the same text. Downloaded raw inputs still require exact byte hashes.
