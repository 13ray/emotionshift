# Add native EmoWOZ experiments for workstreams 2 and 3

Extend the existing TF-IDF/SVM, BiGRU and BiLSTM baselines to the official
EmoWOZ partitions and native seven-class label space. Score labelled user
turns only; system rows remain unlabelled context. No lossy label mapping is used.

## Changes

- Make model probability/target/checkpoint label order explicit, preserving the
  original defaults and old checkpoint compatibility for MELD/DailyDialog.
- Add a separate extension config/runner. SVM selects among sixteen fixed dev
  settings. RNN structures/hyperparameters follow the original B2 experiment;
  checkpoints and primary architecture are chosen on dev only.
- Run seeds 0–4, export fifteen shared-format test prediction CSVs, and report
  native non-neutral weighted/macro-F1, full metrics, strata and confusion tables.
- Include 1,000-resample dialogue CIs for seed 0, CPU inference efficiency,
  source/data/runtime manifests, training histories and resumable completed seeds.
- Supply English documentation, figures with source CSVs and an executed
  notebook with a native-label checkpoint demo.

## Validation

- Original model and shared pipeline regression tests plus new native-label,
  absent-class probability and checkpoint round-trip tests.
- Artifact audit checks row identity, metadata, label order, dev selection,
  training provenance and recomputed scores for all fifteen prediction CSVs.
- Existing MELD/DailyDialog experiment reports and predictions are unchanged.
- Raw data, local environments and personal reference notes are excluded.

See `docs/EMOWOZ_B1_B2.md` and `reports/emowoz_baselines/results.md` for protocol,
commands, results and source attribution. Original B1/B2 implementations are
reused; this contribution adds native-label support and the EmoWOZ experiments.
These are in-domain scores, not cross-dataset transfer results. Slides,
own-voice narration and personal forms remain separate deliverables.
