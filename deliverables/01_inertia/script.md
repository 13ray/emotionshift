# B0: proposed narration (use your own voice)

## Slide 1 — What is the minimum we need to beat? (~35 seconds)

We use three simple B0 rules. Majority always predicts the most common training
emotion. Inertia predicts the current emotion by copying the same speaker's
previous text-model prediction. Markov uses training transition counts to
predict from that previous prediction. The first turn starts from the training
majority. These rules do not use test gold labels. We report gold-history
versions separately as oracle references. MELD and DailyDialog share their
labels; EmoWOZ keeps its own native labels and only user turns are scored.

Show: notebook "Baseline rules" cell (8 lines); mention that donor means the
supporting utterance-only SVM. The first slide should define oracle explicitly.

Optional replacement for the first slide's explanation: use notebook
"Inertia: one-step error carry-over" and the real saved-prediction examples.
At A's turn 2, an earlier SVM error is copied even though the current SVM is
correct. At turn 4, A's emotion changes and even a correct previous gold label
is stale. At turn 5, the updated SVM history restores the correct prediction;
the baseline does not recursively copy its own erroneous output. The toy SVM
outputs are manually specified for teaching, not a measured result. Use only
one of these explanations to stay within the original time budget.

## Slide 2 — History alone is not enough (~35 seconds)

On MELD, majority reaches 31.27 percent weighted F1. Predicted-history inertia
reaches 38.30, while the oracle version reaches 44.86. The supporting text
recognizer reaches 56.78. This shows both that history is useful relative to
the constant floor and that current text carries information simple inertia
misses. On de-duplicated DailyDialog and native EmoWOZ the gap is also large,
using each dataset's own headline metric. The metrics and label spaces differ,
so we do not rank the datasets by those numbers.

Show: reports/b0/figures/headline_comparison.png and its source CSV.

## Slide 3 — Forecasting and limits (~20 seconds)

Before reading the current utterance, the MELD Markov prior obtains AUROC 0.558
using predicted history, versus 0.663 using oracle history. So the oracle pilot
is a reference, not a deployable result. These baselines establish the floors
for judging whether the team's richer models improve on simple persistence
and transition patterns. Stable-versus-shift results are included separately.

Optional visual: shift_vs_stable.png; forecast table in the notebook.
Alternatively, show the training transition heatmap: rows condition on the
same speaker's previous emotion, and columns show the next emotion. On MELD,
joy remains joy with probability about 0.33 but returns to neutral with about
0.41, so Markov argmax predicts neutral. This helps explain why a learned
transition rule need not beat simple inertia. Use this as a replacement for
another explanation, not additional narration beyond the time budget.
Total target: about 90 seconds; adjust after recording in your own voice.

Slides must use the shared team template. No narration has been recorded and
no signed forms are supplied by this code contribution.
