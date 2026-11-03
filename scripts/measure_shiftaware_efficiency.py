"""E9 efficiency for ShiftAware, measured without the trained checkpoints.

    uv run scripts/measure_shiftaware_efficiency.py

The checkpoints are not in Git, so this builds the same architecture
(configs/shiftaware.yaml on top of RoBERTa-base) from the RoBERTa-base config,
with untrained weights everywhere. Parameter count, file size and CPU latency do
not depend on the weight values, so they hold for the trained model too; the
predictions of this untrained model are thrown away. Only the tokenizer and
config are needed locally, not the pretrained weights.

Latency is measured on MELD test dialogues the way a live demo would run without
caching: to label turn t, re-run the dialogue prefix 1..t in one call. We also
report the whole-dialogue cost divided by its turns (offline labelling).

Writes reports/shiftaware/efficiency.json (read by scripts/summarize_all.py).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModel, AutoTokenizer

from emotionshift.data.loaders import load
from emotionshift.eval.efficiency import file_size_mb, machine
from emotionshift.models.shiftaware.config import ShiftAwareConfig
from emotionshift.models.shiftaware.encoder import (
    ShiftAwarePredictor,
    TextShiftAware,
    TransformerUtteranceEncoder,
)
from emotionshift.models.shiftaware.model import ShiftAware

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "FacebookAI/roberta-base"


def main(n_dialogues: int = 60, warmup: int = 5, seed: int = 0):
    torch.manual_seed(seed)
    config = ShiftAwareConfig(**yaml.safe_load((ROOT / "configs/shiftaware.yaml").read_text()))
    backbone = AutoModel.from_config(AutoConfig.from_pretrained(MODEL_NAME, local_files_only=True))
    encoder = TransformerUtteranceEncoder(backbone, backbone.config.hidden_size, frozen=False)
    model = TextShiftAware(encoder, ShiftAware(config)).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    predictor = ShiftAwarePredictor(model, tokenizer)

    parameters = {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "dynamics": sum(p.numel() for p in model.dynamics.parameters()),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save(model.state_dict(), path)
        size = file_size_mb(path)

    test = load("meld", ["test"])
    dialogues = [g for _, g in test.groupby("dialogue_id", sort=False)]
    rng = np.random.default_rng(seed)
    picked = [dialogues[i] for i in rng.choice(len(dialogues), n_dialogues, replace=False)]

    def run(d):
        predictor.predict(d["text"].tolist(), speakers=d["speaker"].astype(str).tolist())

    for d in picked[:warmup]:
        run(d)
    per_turn, amortized = [], []
    for d in picked:
        t = int(rng.integers(1, len(d) + 1))  # label a random turn from its prefix
        start = time.perf_counter()
        run(d.iloc[:t])
        per_turn.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        run(d)
        amortized.append((time.perf_counter() - start) * 1000 / len(d))

    result = {
        "model": "shiftaware (architecture of configs/shiftaware.yaml)",
        "note": "untrained weights; latency and size do not depend on weight values",
        "efficiency": {
            "parameters": parameters["encoder"] + parameters["dynamics"],
            "parameters_by_part": parameters,
            "model_file_mb": round(size, 2),
            "latency_one_utterance": {
                "what": "label one new turn by re-running its dialogue prefix (live demo)",
                "n": len(per_turn),
                "median_ms": float(np.median(per_turn)),
                "p95_ms": float(np.percentile(per_turn, 95)),
            },
            "latency_amortized": {
                "what": "whole dialogue in one call, divided by its turns (offline)",
                "n": len(amortized),
                "median_ms": float(np.median(amortized)),
                "p95_ms": float(np.percentile(amortized, 95)),
            },
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "machine": machine(),
            "note": "untrained weights, same architecture; one new turn incl. re-encoding its prefix",
        },
    }
    out = ROOT / "reports/shiftaware/efficiency.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
