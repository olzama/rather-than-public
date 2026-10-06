#!/usr/bin/env python3
r"""
Convert the human pairwise preferences of HelpSteer2 and HelpSteer3
(nvidia/HelpSteer2 preference/preference.jsonl.gz, nvidia/HelpSteer3
preference/train.jsonl.gz on Hugging Face, CC-BY-4.0) into chosen/rejected
pairs for counting/preference_pairs.py. Ties are dropped. HelpSteer2: a
positive preference_strength means response 2 is better. HelpSteer3: a
negative overall_preference means response 1 is better; only English
"general" and "stem" items are kept (code and multilingual left out).

Usage:
    python3 helpsteer_pairs.py hs2_pref.jsonl.gz hs3_pref.jsonl.gz --out-dir .
Writes hs2_pairs.parquet and hs3_pairs.parquet (needs pandas and pyarrow).
"""
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hs2")
    ap.add_argument("hs3")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    out = Path(args.out_dir)
    rows = []
    for r in map(json.loads, gzip.open(args.hs2, "rt")):
        s = int(r["preference_strength"])
        if s == 0:
            continue
        c, j = (r["response_2"], r["response_1"]) if s > 0 else (r["response_1"], r["response_2"])
        rows.append({"chosen": c, "rejected": j, "source": "helpsteer2", "strength": abs(s)})
    pd.DataFrame(rows).to_parquet(out / "hs2_pairs.parquet")
    print(f"HelpSteer2: {len(rows)} pairs")
    rows = []
    for r in map(json.loads, gzip.open(args.hs3, "rt")):
        if r["domain"] not in ("general", "stem") or str(r.get("language", "english")).lower() != "english":
            continue
        s = int(r["overall_preference"])
        if s == 0:
            continue
        c, j = (r["response1"], r["response2"]) if s < 0 else (r["response2"], r["response1"])
        rows.append({"chosen": c, "rejected": j, "source": "helpsteer3-" + r["domain"], "strength": abs(s)})
    pd.DataFrame(rows).to_parquet(out / "hs3_pairs.parquet")
    print(f"HelpSteer3: {len(rows)} pairs")


if __name__ == "__main__":
    main()
