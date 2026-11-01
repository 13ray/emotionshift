# ShiftAware results (test, mean ± sample std over seeds)

Checkpoints and variants are selected on dev only. Shift-turn F1 is the seed mean.

| dataset | split | model | seeds | weighted F1 | macro F1 | micro F1 (no neutral) | weighted F1 on shift turns | shift-head AUROC |
|---|---|---|---|---|---|---|---|---|
| meld | test | B1 TF-IDF | 5 | 56.78 ± 0.00 | 38.39 ± 0.00 | 40.91 ± 0.00 | 49.15 | n/a |
| meld | test | shiftaware | 5 | 62.68 ± 0.52 | 41.71 ± 1.72 | 50.43 ± 0.49 | 53.94 | 73.78 ± 1.33 |
| meld | test | shiftaware_cw-inverse | 1 | 60.81 (1 seed) | 45.33 (1 seed) | 47.97 (1 seed) | 54.04 | 74.90 (1 seed) |
| meld | test | shiftaware_cw-sqrt | 5 | 62.52 ± 0.57 | 46.33 ± 0.80 | 49.64 ± 0.81 | 54.95 | 74.75 ± 0.54 |
| dailydialog | test_dedup | B1 TF-IDF | 5 | 83.88 ± 0.00 | 46.16 ± 0.00 | 53.03 ± 0.00 | 70.06 | n/a |
| dailydialog | test | B1 TF-IDF | 5 | 83.74 ± 0.00 | 48.09 ± 0.00 | 53.45 ± 0.00 | 70.60 | n/a |
| dailydialog | test_dedup | shiftaware | 5 | 85.42 ± 0.36 | 45.09 ± 5.34 | 57.39 ± 0.71 | 67.40 | 84.17 ± 0.24 |
| dailydialog | test | shiftaware | 5 | 85.33 ± 0.32 | 46.84 ± 5.47 | 57.60 ± 0.62 | 67.38 | 83.69 ± 0.19 |
| dailydialog | test_dedup | shiftaware_cw-sqrt | 5 | 84.42 ± 0.25 | 51.24 ± 1.97 | 57.17 ± 0.96 | 71.34 | 84.05 ± 0.40 |
| dailydialog | test | shiftaware_cw-sqrt | 5 | 84.16 ± 0.29 | 52.12 ± 1.63 | 57.12 ± 0.90 | 71.38 | 83.44 ± 0.47 |
| dailydialog | test_dedup | shiftaware_tf-d768-0layer | 1 | 84.28 (1 seed) | 35.57 (1 seed) | 53.58 (1 seed) | 68.38 | 74.23 (1 seed) |
| dailydialog | test | shiftaware_tf-d768-0layer | 1 | 84.03 (1 seed) | 35.47 (1 seed) | 53.23 (1 seed) | 68.12 | 73.77 (1 seed) |
| dailydialog | test_dedup | shiftaware_tf-d768-0layer-hm | 3 | 84.86 ± 0.13 | 46.97 ± 0.48 | 56.30 ± 0.38 | 71.12 | 74.35 ± 0.26 |
| dailydialog | test | shiftaware_tf-d768-0layer-hm | 3 | 84.81 ± 0.12 | 48.79 ± 1.21 | 56.53 ± 0.16 | 71.03 | 74.08 ± 0.19 |
| dailydialog | test_dedup | shiftaware_tf-d768-2layer | 1 | 86.00 (1 seed) | 47.77 (1 seed) | 58.43 (1 seed) | 66.69 | 84.73 (1 seed) |
| dailydialog | test | shiftaware_tf-d768-2layer | 1 | 85.75 (1 seed) | 49.39 (1 seed) | 58.36 (1 seed) | 66.40 | 84.24 (1 seed) |
| dailydialog | test_dedup | shiftaware_tf-d768-2layer-hm | 3 | 85.28 ± 0.31 | 51.56 ± 1.74 | 57.84 ± 1.21 | 68.63 | 83.70 ± 0.22 |
| dailydialog | test | shiftaware_tf-d768-2layer-hm | 3 | 85.31 ± 0.30 | 53.30 ± 1.68 | 58.35 ± 1.06 | 68.84 | 83.29 ± 0.17 |
| dailydialog | test_dedup | shiftaware_tf-d768-4layer | 1 | 84.97 (1 seed) | 35.70 (1 seed) | 56.45 (1 seed) | 66.84 | 83.74 (1 seed) |
| dailydialog | test | shiftaware_tf-d768-4layer | 1 | 84.61 (1 seed) | 35.44 (1 seed) | 55.91 (1 seed) | 66.10 | 83.36 (1 seed) |
