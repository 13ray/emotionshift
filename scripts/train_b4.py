"""Train B4 (RoBERTa with speaker names and the past k turns) and B4+ on a GPU.

    python scripts/train_b4.py --out /content/b4_out     # everything (Colab T4: several hours)
    python scripts/train_b4.py --stages sweep main       # MELD only
    python scripts/train_b4.py --quick                   # DistilRoBERTa, tiny subset (smoke test)

Stages, in this order (``--stages`` picks a subset):

    sweep        MELD, k in {1, 3, 5, 10, all} x 3 seeds                (E3)
    main         MELD at the dev-best k, the remaining seeds            (E1)
    b4plus       MELD at the dev-best k, class-weighted loss, 5 seeds   (R3)
    dailydialog  DailyDialog at the same k, 5 seeds                     (E1)

Interrupted? Run the same command again: finished runs are reused (one state file each
under ``<out>/reports/b4/<dataset>/runs/``), so a Colab session that dies loses at most
the run in progress.

This script only trains and predicts. It writes small files:

    <out>/reports/b4/<dataset>/runs/<variant>_k<k>_seed<s>.json    dev history, time, efficiency
    <out>/reports/b4/<dataset>/preds/<variant>_k<k>_seed<s>.csv    split, dialogue_id, turn, pred
                                                                   (+ E4 control predictions)
    <out>/reports/b4/selection.json                                k chosen on dev

``scripts/analyse_b4.py`` (CPU, seconds) turns them into the shared prediction CSVs,
tables and figures.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionshift.data.loaders import load  # noqa: E402
from emotionshift.eval import efficiency as E  # noqa: E402
from emotionshift.eval import metrics as M  # noqa: E402
from emotionshift.models.roberta import RobertaConfig  # noqa: E402
from emotionshift.models.roberta_context import RobertaContext, k_name  # noqa: E402

STAGES = ["sweep", "main", "b4plus", "dailydialog"]
CONTROLS = ["none", "shuffled", "speakers"]          # E4, on the headline test split only
HEADLINE_SPLIT = {"meld": "test", "dailydialog": "test_dedup"}
OUT = ROOT


def run_name(variant: str, k, seed: int) -> str:
    return f"{variant}_k{k_name(k)}_seed{seed}"


def paths(dataset: str, variant: str, k, seed: int) -> tuple[Path, Path]:
    base = OUT / "reports" / "b4" / dataset
    name = run_name(variant, k, seed)
    return base / "runs" / f"{name}.json", base / "preds" / f"{name}.csv"


def measure_efficiency(m: RobertaContext, inputs: list[str], n: int) -> dict:
    """E9 on the model in memory: size, GPU throughput, CPU latency of one utterance with context."""
    tmp = Path(tempfile.mkdtemp(prefix="b4_size_"))
    m.save(tmp)
    size = round(sum(f.stat().st_size for f in tmp.iterdir() if f.is_file()) / 1e6, 1)
    shutil.rmtree(tmp, ignore_errors=True)
    thr = {**E.throughput(m.predict, inputs), "device": m.device}
    device = m.device
    m.model.to("cpu")
    m.device = "cpu"
    lat = E.latency_ms(m.predict, inputs, n=n)
    m.model.to(device)
    m.device = device
    return {"parameters": m.num_parameters(), "model_dir_mb": size, "latency_one_utterance_cpu": lat,
            "batch_throughput": thr, "machine": E.machine()}


def run(dataset: str, variant: str, k, seed: int, cfg: dict, data: dict, quick: bool, fresh: bool) -> dict:
    state, pred_file = paths(dataset, variant, k, seed)
    if state.exists() and pred_file.exists() and not fresh:
        done = json.loads(state.read_text())
        print(f"[{dataset}] {run_name(variant, k, seed)}: already finished (dev {done['dev_score']:.4f})", flush=True)
        return done
    state.parent.mkdir(parents=True, exist_ok=True)
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    df = data[dataset]
    mcfg = {**cfg["model"], "epochs": cfg["epochs"][dataset], "select_on": cfg["select_on"][dataset],
            "class_weight": cfg["b4plus"]["class_weight"] if variant == "b4plus" else None}
    if quick:
        mcfg.update(model_name="distilroberta-base", epochs=1)
    print(f"[{dataset}] {run_name(variant, k, seed)}: training {mcfg['model_name']}, {mcfg['epochs']} epochs", flush=True)
    t0 = time.perf_counter()
    m = RobertaContext(RobertaConfig(**mcfg, seed=seed), k=k)
    m.fit_frames(df[df.split == "train"], df[df.split == "dev"])
    seconds = time.perf_counter() - t0
    dev_score = max(h[mcfg["select_on"]] for h in m.history)

    out, test_scores = [], {}
    for split in cfg["predict_splits"][dataset]:
        t = df[df.split == split]
        p = t[["split", "dialogue_id", "turn"]].copy()
        p["pred"] = m.predict_frame(t)
        test_scores[split] = M.scores(t.emotion, p.pred)[M.HEADLINE[dataset]]
        if split == HEADLINE_SPLIT[dataset]:
            for mode in CONTROLS:
                p[f"pred_{mode}"] = m.predict_frame(t, mode=mode, seed=seed)
        out.append(p)
    pd.concat(out).to_csv(pred_file, index=False)

    done = {"dataset": dataset, "variant": variant, "k": k_name(k), "seed": seed, "config": mcfg,
            "dev_score": dev_score, "history": m.history, "train_seconds": seconds,
            "headline": M.HEADLINE[dataset], "scores": test_scores}
    if seed == 0 and variant == "b4":   # once per (dataset, k): the cost of a longer window
        t = df[df.split == HEADLINE_SPLIT[dataset]]
        done["efficiency"] = measure_efficiency(m, m.inputs(t), n=50 if quick else 300)
    state.write_text(json.dumps(done, indent=1, default=float))   # written last: half-done runs are redone
    print(f"[{dataset}] {run_name(variant, k, seed)}: dev {dev_score:.4f}, "
          f"{HEADLINE_SPLIT[dataset]} {test_scores[HEADLINE_SPLIT[dataset]]:.4f} ({seconds / 60:.1f} min)", flush=True)
    del m
    return done


def choose_k(cfg: dict, dataset: str) -> dict:
    """Window with the best mean dev score over the sweep seeds (ties: the smaller window)."""
    rows = []
    for k in cfg["windows"]:
        scores = [json.loads(paths(dataset, "b4", k, s)[0].read_text())["dev_score"] for s in cfg["sweep_seeds"]]
        rows.append({"k": k_name(k), "dev_mean": float(np.mean(scores)), "dev_per_seed": scores})
    best = max(rows, key=lambda r: r["dev_mean"])   # max keeps the first (smallest) window on ties
    sel = {"dataset": dataset, "selected_on": f"dev {cfg['select_on'][dataset]}, mean over seeds {cfg['sweep_seeds']}",
           "best_k": best["k"], "windows": rows}
    (OUT / "reports" / "b4" / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "b4_context.yaml"))
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", help="output root (default: the repo)")
    ap.add_argument("--fresh", action="store_true", help="retrain instead of reusing finished runs")
    a = ap.parse_args()
    global OUT
    OUT = Path(a.out) if a.out else (ROOT / "outputs_quick" if a.quick else ROOT)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.quick:
        cfg.update(seeds=[0, 1], sweep_seeds=[0], windows=[1, "all"])
    (OUT / "reports" / "b4").mkdir(parents=True, exist_ok=True)

    sweep_ds = cfg["sweep_dataset"]
    need = {sweep_ds} | ({"dailydialog"} if "dailydialog" in a.stages else set())
    data = {d: load(d) for d in need}
    if a.quick:
        data = {d: df[df.dialogue_id < 12] for d, df in data.items()}
    go = lambda *args: run(*args, cfg, data, a.quick, a.fresh)  # noqa: E731

    t0 = time.perf_counter()
    if "sweep" in a.stages:
        for seed in cfg["sweep_seeds"]:
            for k in cfg["windows"]:
                go(sweep_ds, "b4", k, seed)
    sel_file = OUT / "reports" / "b4" / "selection.json"
    if all(paths(sweep_ds, "b4", k, s)[0].exists() for k in cfg["windows"] for s in cfg["sweep_seeds"]):
        sel = choose_k(cfg, sweep_ds)
    elif sel_file.exists():
        sel = json.loads(sel_file.read_text())
    else:
        sys.exit("The window sweep is not finished; run the sweep stage first.")
    best_k = sel["best_k"]
    print(f"window chosen on {sweep_ds} dev: k = {best_k}", flush=True)

    if "main" in a.stages:
        for seed in cfg["seeds"]:
            go(sweep_ds, "b4", best_k, seed)
    if "b4plus" in a.stages:
        for d in cfg["b4plus"]["datasets"]:
            for seed in cfg["seeds"]:
                go(d, "b4plus", best_k, seed)
    if "dailydialog" in a.stages:
        for seed in cfg["seeds"]:
            go("dailydialog", "b4", best_k, seed)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
