# ShiftAware selection for the comparison table

Chosen on dev only, among variants with five seeds. Seeds are 0, 1, 2, 3 and 42.
Recompute dev from `reports/<variant>/<dataset>/seed<k>/summary.json` and test from
`predictions/`.

| dataset | variant | dev headline (mean) | other 5-seed variant (dev) | best dev seed (demo) | test headline (mean ± sample SD) |
|---|---|---|---|---|---|
| MELD (weighted F1) | `shiftaware_cw-sqrt` | 60.78 | `shiftaware` 59.94 | 42 | 62.52 ± 0.57 |
| DailyDialog test_dedup (micro F1, no neutral) | `shiftaware` | 60.99 | `shiftaware_cw-sqrt` 58.49 | 3 | 57.39 ± 0.71 |

The transformer variants and all EmoWOZ runs have at most three seeds, so they stay
exploratory and are not in the comparison table. Most of these runs were trained
from a dirty working tree (`git.dirty` in each `summary.json`).
