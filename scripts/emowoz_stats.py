"""Export reproducible source tables and summary; no model or GPU required."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotionshift.data.loaders import DATA_ROOT, load_emowoz_shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("reports/emowoz"))
    args = parser.parse_args()
    df = load_emowoz_shared(root=args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, group in df.groupby("split"):
        eligible = group[group.stratum != "first"]
        summary[split] = {
            "dialogues": int(group.dialogue_id.nunique()), "utterances": len(group),
            "first": int((group.stratum == "first").sum()),
            "stable": int((group.stratum == "stable").sum()),
            "shift": int((group.stratum == "shift").sum()),
            "shift_rate_excluding_first": float((eligible.stratum == "shift").mean()),
        }
    pd_tables = {
        "native_labels": ["split", "source", "emotion_original_id", "emotion_original"],
        "shared_labels": ["split", "emotion"],
        "strata": ["split", "stratum"], "shift_types": ["split", "shift_type"],
    }
    for name, columns in pd_tables.items():
        df.groupby(columns).size().rename("n").reset_index().to_csv(args.output / f"{name}.csv", index=False)
    summary["provenance"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((args.root / "emowoz").glob("*.json")) if p.name != "download_manifest.json"
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
