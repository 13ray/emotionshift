"""Turn the B4 training output into shared prediction CSVs, tables and figures (CPU, seconds).

    python scripts/analyse_b4.py

Reads what ``scripts/train_b4.py`` wrote under ``reports/b4/`` (run states, slim
predictions, the window chosen on dev) and writes

    predictions/b4_<dataset>_<split>_seed<k>.csv       B4 at the dev-best window (shared format)
    predictions/b4plus_meld_test_seed<k>.csv           B4+ (class-weighted)
    reports/b4/<dataset>/summary.json                  config, dev, test metrics, efficiency (E9)
    reports/b4/<dataset>/stratified_<split>.csv, per_class_<split>.csv, confusion_<split>.csv
    reports/b4/meld/window_sweep.csv                   E3: score against the window k, by stratum
    reports/b4/meld/context_gain.csv                   E2 / H1: B4 - B3 by stratum and shift type
    reports/b4/meld/controls.csv                       E4 / H2: no / shuffled context, shuffled speakers
    reports/b4/meld/b4plus_vs_b3.json                  R3: paired macro-F1 difference with CI
    reports/b4/figures/window_sweep_meld.png, context_gain_meld.png
    reports/b4/results.md

Anything not trained yet is skipped, so this also runs on partial results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionshift.data.labels import EMOTIONS  # noqa: E402
from emotionshift.data.loaders import load  # noqa: E402
from emotionshift.eval import metrics as M  # noqa: E402

B4 = ROOT / "reports" / "b4"
PRED = ROOT / "predictions"
PRED_COLS = ["dataset", "split", "dialogue_id", "turn", "speaker", "text", "emotion", "pred",
             "stratum", "shift_type", "model", "seed"]
KEYS = ["weighted_f1", "macro_f1", "micro_f1_no_neutral", "accuracy"]
HEADLINE_SPLIT = {"meld": "test", "dailydialog": "test_dedup"}
GROUPS = ["all", "stratum:first", "stratum:stable", "stratum:shift"]
N_BOOT = 1000


def state(dataset, variant, k, seed):
    f = B4 / dataset / "runs" / f"{variant}_k{k}_seed{seed}.json"
    return json.loads(f.read_text()) if f.exists() else None


def slim(dataset, variant, k, seed):
    f = B4 / dataset / "preds" / f"{variant}_k{k}_seed{seed}.csv"
    return pd.read_csv(f) if f.exists() else None


def full(gold: pd.DataFrame, s: pd.DataFrame, split: str, col: str = "pred") -> pd.DataFrame:
    """Gold rows of ``split`` with the predictions of a slim file, in gold order."""
    t = gold[gold.split == split]
    p = s[s.split == split][["dialogue_id", "turn", col]].rename(columns={col: "pred"})
    out = t.merge(p, on=["dialogue_id", "turn"], how="left", validate="one_to_one")
    assert len(out) == len(t) and out.pred.notna().all(), f"{split}: predictions do not cover the split"
    return out


def group_scores(t: pd.DataFrame) -> pd.DataFrame:
    return M.stratified(t).set_index("group")


def seed_table(frames: list[pd.DataFrame], metric: str) -> pd.DataFrame:
    """Mean and sample SD over seeds of ``metric`` for every stratum / shift type."""
    per = pd.concat([group_scores(t)[metric].rename(i) for i, t in enumerate(frames)], axis=1)
    return pd.DataFrame({"n": group_scores(frames[0])["n"], "mean": per.mean(axis=1),
                         "std": per.std(axis=1, ddof=1) if per.shape[1] > 1 else 0.0})


def _f1_scores(cm: np.ndarray) -> dict[str, np.ndarray]:
    """The shared metrics from confusion matrices stacked on the leading axes (..., k, k)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * tp / (support + predicted))
        emo = np.array([e != "neutral" for e in EMOTIONS])
        micro = np.nan_to_num(2 * tp[..., emo].sum(-1) / (predicted[..., emo].sum(-1) + support[..., emo].sum(-1)))
        weighted = np.nan_to_num((f1 * support).sum(-1) / support.sum(-1))
    return {"weighted_f1": weighted, "macro_f1": f1.mean(-1), "micro_f1_no_neutral": micro}


def _dialogue_counts(frames: list[pd.DataFrame], keep: np.ndarray) -> np.ndarray:
    """Confusion counts per dialogue and seed, (dialogues, seeds, k, k), on the rows in ``keep``."""
    index = {lab: i for i, lab in enumerate(EMOTIONS)}
    k = len(EMOTIONS)
    base = frames[0]
    gold = base.emotion.map(index).to_numpy()
    dia = pd.factorize(base.dialogue_id)[0]
    n_d = dia.max() + 1
    rows = np.flatnonzero(keep)
    out = np.zeros((n_d, len(frames), k * k))
    for s, t in enumerate(frames):
        cell = dia[rows] * k * k + gold[rows] * k + t.pred.map(index).to_numpy()[rows]
        out[:, s] = np.bincount(cell, minlength=n_d * k * k).reshape(n_d, k * k)
    return out.reshape(n_d, len(frames), k, k)


def _resample(n_dialogues: int, seed: int) -> np.ndarray:
    """How often each dialogue is drawn in each of N_BOOT resamples with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_dialogues, np.full(n_dialogues, 1 / n_dialogues), size=N_BOOT).astype(float)


def seed_mean_ci(frames: list[pd.DataFrame], metric: str, seed: int = 0) -> tuple[float, float]:
    """95% CI of the seed-mean ``metric``, resampling whole dialogues."""
    counts = _dialogue_counts(frames, np.ones(len(frames[0]), bool))
    boots = _f1_scores(np.einsum("bd,dskl->bskl", _resample(len(counts), seed), counts))[metric].mean(-1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def paired_seed_mean(a: list[pd.DataFrame], b: list[pd.DataFrame], metric: str, mask=None, seed: int = 0) -> dict:
    """Seed-mean metric(b) - metric(a) with a 95% CI from resampling whole dialogues."""
    n = min(len(a), len(b))
    keep = np.ones(len(a[0]), bool) if mask is None else np.asarray(mask, bool)
    ca, cb = _dialogue_counts(a[:n], keep), _dialogue_counts(b[:n], keep)

    def mean_metric(cm):
        return _f1_scores(cm)[metric].mean(-1)

    w = _resample(len(ca), seed)
    boots = mean_metric(np.einsum("bd,dskl->bskl", w, cb)) - mean_metric(np.einsum("bd,dskl->bskl", w, ca))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(mean_metric(cb.sum(0)) - mean_metric(ca.sum(0))), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(keep.sum()), "seeds": n}


def b3_runs(dataset, split):
    files = sorted(PRED.glob(f"b3_{dataset}_{split}_seed*.csv"))
    return [pd.read_csv(f) for f in files]


def write_shared(frames, prefix, model, dataset, split):
    for seed, t in frames.items():
        t = t.copy()
        t["model"], t["seed"] = model, seed
        t[PRED_COLS].to_csv(PRED / f"{prefix}_{dataset}_{split}_seed{seed}.csv", index=False)


def summarise(dataset, variant, k, seeds, gold, cfg, label) -> dict | None:
    """summary.json-style block for one model, plus the per-split tables."""
    states = {s: state(dataset, variant, k, s) for s in seeds}
    states = {s: st for s, st in states.items() if st}
    if not states:
        return None
    out_dir = B4 / dataset
    prefix = "b4" if variant == "b4" else "b4plus"
    res = {"dataset": dataset, "model": label, "k": k, "selected_on_dev": cfg["select_on"][dataset],
           "config": next(iter(states.values()))["config"], "seeds": list(states),
           "dev": {f"seed{s}": {"best": st["dev_score"], "per_epoch": st["history"]} for s, st in states.items()},
           "dev_scores": {}, "test": {}}
    for split in cfg["predict_splits"][dataset]:
        frames = {s: full(gold, slim(dataset, variant, k, s), split) for s in states}
        runs = list(frames.values())
        per = [M.scores(t.emotion, t.pred) for t in runs]
        r = {key: {"mean": float(np.mean([p[key] for p in per])),
                   "std": float(np.std([p[key] for p in per], ddof=1)) if len(per) > 1 else 0.0,
                   "per_seed": [round(p[key], 4) for p in per]} for key in KEYS}
        for key in KEYS[:3]:
            r[key]["ci95"] = seed_mean_ci(runs, key)
        r["n"] = len(runs[0])
        shift = [M.derived_shift_detection(t) for t in runs]
        r["derived_shift_detection"] = {**shift[0], "shift_f1_mean": float(np.mean([s["shift_f1"] for s in shift])),
                                        "shift_f1_std": float(np.std([s["shift_f1"] for s in shift]))}
        res["test" if split != "dev" else "dev_scores"][split] = r
        if split == "dev":
            continue
        write_shared(frames, prefix, f"{prefix}_roberta_context", dataset, split)
        tag = split if variant == "b4" else f"{split}_b4plus"
        strat = pd.concat([group_scores(t) for t in runs]).groupby(level=0, sort=False).mean().round(4)
        strat.reset_index().to_csv(out_dir / f"stratified_{tag}.csv", index=False)
        M.per_class_f1(runs[0].emotion, runs[0].pred).round(4).to_csv(out_dir / f"per_class_{tag}.csv", index=False)
        M.confusion(runs[0].emotion, runs[0].pred).to_csv(out_dir / f"confusion_{tag}.csv")
    secs = [st["train_seconds"] for st in states.values()]
    eff = next((st["efficiency"] for st in states.values() if "efficiency" in st), None)
    if eff is None and variant == "b4plus":           # same network as B4: reuse its measurements
        b4 = state(dataset, "b4", k, 0)
        eff = b4.get("efficiency") if b4 else None
    res["efficiency"] = {**(eff or {}), "train_seconds": {"mean": float(np.mean(secs)),
                                                          "per_seed": [round(s, 1) for s in secs]}}
    return res


def window_sweep(cfg, gold) -> pd.DataFrame | None:
    """E3: MELD test and dev weighted-F1 against the window, over the sweep seeds (k = 0 is B3)."""
    seeds, rows = cfg["sweep_seeds"], []
    b3 = b3_runs("meld", "test")[:len(seeds)]
    if b3:
        tab = seed_table(b3, "weighted_f1")
        rows += [{"k": "0", "split": "test", "group": g, "seeds": len(b3), **tab.loc[g].to_dict()} for g in GROUPS]
    for k in map(str, cfg["windows"]):
        slims = [s for s in (slim("meld", "b4", k, sd) for sd in seeds) if s is not None]
        if not slims:
            continue
        for split in ("dev", "test"):
            tab = seed_table([full(gold, s, split) for s in slims], "weighted_f1")
            rows += [{"k": k, "split": split, "group": g, "seeds": len(slims), **tab.loc[g].to_dict()} for g in GROUPS]
    if not rows:
        return None
    df = pd.DataFrame(rows).round(4)
    df.to_csv(B4 / "meld" / "window_sweep.csv", index=False)

    test = df[df.split == "test"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    names = {"all": "all turns", "stratum:stable": "stable turns", "stratum:shift": "shift turns"}
    for g, colour in zip(names, ["#16202E", "#256ABF", "#C0392B"], strict=True):
        d = test[test.group == g]
        ax.errorbar(range(len(d)), 100 * d["mean"], yerr=100 * d["std"], marker="o", capsize=3, color=colour,
                    label=names[g])
    ks = test[test.group == "all"].k.tolist()
    ax.set_xticks(range(len(ks)), [f"{k}\n(B3)" if k == "0" else k for k in ks])
    ax.set_xlabel("past turns in the input (k)")
    ax.set_ylabel("weighted-F1 on MELD test (%)")
    ax.set_title(f"Context window sweep, mean ± SD over {len(seeds)} seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    (B4 / "figures").mkdir(exist_ok=True)
    fig.savefig(B4 / "figures" / "window_sweep_meld.png", dpi=160)
    plt.close(fig)
    return df


def context_gain(b4: list[pd.DataFrame], dataset: str, split: str) -> pd.DataFrame | None:
    """E2 / H1: B4 - B3 weighted-F1 per stratum and shift type, paired by seed, dialogue-bootstrap CI."""
    b3 = b3_runs(dataset, split)
    if not b3 or not b4:
        return None
    base = b3[0]
    assert (base[["dialogue_id", "turn"]].to_numpy() == b4[0][["dialogue_id", "turn"]].to_numpy()).all()
    groups = [("all", None)] + [(f"stratum:{s}", base.stratum == s) for s in ["first", "stable", "shift"]] + \
             [(f"shift_type:{s}", base.shift_type == s) for s in sorted(base.shift_type.dropna().unique())]
    n = min(len(b3), len(b4))                      # pair the same number of seeds on both sides
    b3, b4 = b3[:n], b4[:n]
    b3_tab, b4_tab = seed_table(b3, "weighted_f1"), seed_table(b4, "weighted_f1")
    rows = []
    for name, mask in groups:
        d = paired_seed_mean(b3, b4, "weighted_f1", mask)
        rows.append({"group": name, "n": d["n"], "b3": b3_tab.loc[name, "mean"], "b4": b4_tab.loc[name, "mean"],
                     "gain": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"]})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(B4 / dataset / "context_gain.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    d = df.iloc[::-1]
    err = [100 * (d.gain - d.ci_low), 100 * (d.ci_high - d.gain)]
    colours = ["#C0392B" if "shift" in g else "#256ABF" for g in d.group]
    ax.barh(d.group.str.replace("stratum:", "").str.replace("shift_type:", "shift: "), 100 * d.gain, xerr=err,
            color=colours, capsize=3)
    ax.axvline(0, color="#16202E", lw=1)
    ax.set_xlabel(f"B4 - B3 weighted-F1 on {dataset} {split} (points, 95% CI)")
    fig.tight_layout()
    fig.savefig(B4 / "figures" / f"context_gain_{dataset}.png", dpi=160)
    plt.close(fig)
    return df


def controls(dataset, k, seeds, gold) -> pd.DataFrame | None:
    """E4 / H2: the trained B4 models with their test input changed."""
    split = HEADLINE_SPLIT[dataset]
    slims = [s for s in (slim(dataset, "b4", k, sd) for sd in seeds) if s is not None]
    if not slims:
        return None
    cols = {"real context (B4)": "pred", "no context at test": "pred_none",
            "shuffled context": "pred_shuffled", "shuffled speakers": "pred_speakers"}
    metric = M.HEADLINE[dataset]
    rows = []
    b3 = b3_runs(dataset, split)
    if b3:
        tab = seed_table(b3, metric)
        rows += [{"input": "B3 (trained without context)", "group": g, **tab.loc[g].to_dict()} for g in GROUPS]
    for name, col in cols.items():
        tab = seed_table([full(gold, s, split, col) for s in slims], metric)
        rows += [{"input": name, "group": g, **tab.loc[g].to_dict()} for g in GROUPS]
    df = pd.DataFrame(rows).round(4)
    df.insert(1, "metric", metric)
    df.to_csv(B4 / dataset / "controls.csv", index=False)
    return df


def pct(x):
    return f"{100 * x:.2f}"


def results_md(summaries, sel, sweep, gain, ctrl, r3) -> str:
    def cell(r, key):
        lo, hi = r[key]["ci95"]
        return f"{pct(r[key]['mean'])} ± {pct(r[key]['std'])} [{100 * lo:.1f}, {100 * hi:.1f}]"

    lines = ["# B4 RoBERTa with context (workstream 5) results", "",
             f"RoBERTa-base with speaker names and the past k turns. Window chosen on MELD dev: **k = {sel['best_k']}** "
             f"({sel['selected_on']}). Mean ± SD over seeds, [95% dialogue-bootstrap CI of the seed mean], in %.", "",
             "| Model | Dataset | Test split | Seeds | n | Weighted-F1 | Macro-F1 | Micro-F1 excl. neutral | Derived shift F1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        for split, r in s["test"].items():
            d = r["derived_shift_detection"]
            lines.append(f"| {s['model']} | {s['dataset']} | {split} | {len(s['seeds'])} | {r['n']} | "
                         f"{cell(r, 'weighted_f1')} | {cell(r, 'macro_f1')} | {cell(r, 'micro_f1_no_neutral')} | "
                         f"{pct(d['shift_f1_mean'])} ± {pct(d['shift_f1_std'])} |")
    if r3:
        lines += ["", "## R3: B4+ against B3 on MELD macro-F1", "",
                  f"B4+ - B3 = **{100 * r3['diff']:+.2f}** points, 95% CI [{100 * r3['ci_low']:+.2f}, "
                  f"{100 * r3['ci_high']:+.2f}] ({r3['seeds']} seed pairs, dialogue bootstrap)."]
    if sweep is not None:
        lines += ["", "## E3: context window sweep (MELD, weighted-F1, %)", "",
                  "| k | Seeds | Dev, all | Test, all | Test, first | Test, stable | Test, shift |", "|---|---|---|---|---|---|---|"]
        for k in sweep.k.unique():
            d = sweep[sweep.k == k].set_index(["split", "group"])

            def v(split, g, d=d):
                return f"{pct(d.loc[(split, g), 'mean'])} ± {pct(d.loc[(split, g), 'std'])}" if (split, g) in d.index else "-"
            lines.append(f"| {k}{' (B3)' if k == '0' else ''} | {int(d.seeds.iloc[0])} | {v('dev', 'all')} | "
                         f"{v('test', 'all')} | {v('test', 'stratum:first')} | {v('test', 'stratum:stable')} | "
                         f"{v('test', 'stratum:shift')} |")
        lines += ["", "![window sweep](figures/window_sweep_meld.png)"]
    for dataset, g in gain.items():
        lines += ["", f"## E2 / H1: where the context gain comes from ({dataset}, weighted-F1, %)", "",
                  "| Group | n | B3 | B4 | B4 - B3 | 95% CI |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.group} | {r.n} | {pct(r.b3)} | {pct(r.b4)} | {100 * r.gain:+.2f} | "
                  f"[{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}] |" for r in g.itertuples()]
        lines += ["", f"![context gain](figures/context_gain_{dataset}.png)"]
    for dataset, c in ctrl.items():
        lines += ["", f"## E4: input controls ({dataset} {HEADLINE_SPLIT[dataset]}, {c.metric.iloc[0]}, %)", "",
                  "| Input | All | First | Stable | Shift |", "|---|---|---|---|---|"]
        for name in c.input.unique():
            d = c[c.input == name].set_index("group")
            lines.append(f"| {name} | " + " | ".join(f"{pct(d.loc[g, 'mean'])} ± {pct(d.loc[g, 'std'])}" for g in GROUPS) + " |")
    eff = [s for s in summaries if "parameters" in s["efficiency"]]
    if eff:
        lines += ["", "## Efficiency (E9)", "",
                  "| Model | Dataset | Parameters | Model files | Train time per seed | CPU latency per utterance (median / p95) | Batch throughput |",
                  "|---|---|---|---|---|---|---|"]
        for s in eff:
            e = s["efficiency"]
            lat, thr = e["latency_one_utterance_cpu"], e["batch_throughput"]
            lines.append(f"| {s['model']} | {s['dataset']} | {e['parameters'] / 1e6:.1f} M | {e['model_dir_mb']:.0f} MB | "
                         f"{e['train_seconds']['mean'] / 60:.1f} min | {lat['median_ms']:.1f} / {lat['p95_ms']:.1f} ms | "
                         f"{thr['utterances_per_second']:,.0f} utt/s ({thr['device']}) |")
    return "\n".join(lines) + "\n"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "b4_context.yaml").read_text())
    sel_file = B4 / "selection.json"
    if not sel_file.exists():
        sys.exit("reports/b4/selection.json is missing: the window sweep has not finished.")
    sel = json.loads(sel_file.read_text())
    k, seeds = sel["best_k"], cfg["seeds"]
    (B4 / "figures").mkdir(parents=True, exist_ok=True)
    data = {d: load(d) for d in ["meld", "dailydialog"] if (B4 / d / "runs").exists()}

    summaries, gain, ctrl, r3 = [], {}, {}, None
    for dataset, gold in data.items():
        s = summarise(dataset, "b4", k, seeds, gold, cfg, f"B4 RoBERTa + context (k = {k})")
        if not s:
            continue
        (B4 / dataset / "summary.json").write_text(json.dumps(s, indent=1, default=float))
        summaries.append(s)
        split = HEADLINE_SPLIT[dataset]
        b4 = [full(gold, slim(dataset, "b4", k, sd), split) for sd in s["seeds"]]
        if (g := context_gain(b4, dataset, split)) is not None:
            gain[dataset] = g
        if (c := controls(dataset, k, s["seeds"], gold)) is not None:
            ctrl[dataset] = c
        if dataset in cfg["b4plus"]["datasets"]:
            p = summarise(dataset, "b4plus", k, seeds, gold, cfg, f"B4+ class-weighted (k = {k})")
            if p:
                (B4 / dataset / "summary_b4plus.json").write_text(json.dumps(p, indent=1, default=float))
                summaries.append(p)
                plus = [full(gold, slim(dataset, "b4plus", k, sd), split) for sd in p["seeds"]]
                b3 = b3_runs(dataset, split)
                if b3 and dataset == "meld":
                    r3 = {"metric": "macro_f1", "a": "b3", "b": "b4plus", **paired_seed_mean(b3, plus, "macro_f1")}
                    (B4 / dataset / "b4plus_vs_b3.json").write_text(json.dumps(r3, indent=1))
    sweep = window_sweep(cfg, data["meld"]) if "meld" in data else None
    md = results_md(summaries, sel, sweep, gain, ctrl, r3)
    (B4 / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
