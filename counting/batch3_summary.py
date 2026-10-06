#!/usr/bin/env python3
r"""
Annotation batch 3 (data/annotation_b3) for one annotator: annoying rate per
source with Wilson 95% intervals; LLM-written vs. arXiv 2026 items (Fisher's
exact test, two-sided); LLM items by stage (generated / revised) and by the
corpus of the source paper; the same annotator's rates in the main pool
(labeled with the earlier instructions); and hidden-repeat consistency.
With --other NAME: agreement with a second annotator on the items both have
labeled, and per-source rates on those items for each annotator and for
Either (annoying if either labeled it annoying, legitimate if both labeled
it legitimate).
Rates exclude unsure and garbled items.

Usage:
    python3 batch3_summary.py ../data/annotation_b3 ../data/annotation --annotator A2 [--json out.json]
"""
import argparse
import collections
import json
import re
from math import sqrt
from pathlib import Path

from scipy.stats import fisher_exact

MAIN_ARXIV = ("arxiv2026", "arxiv_v1", "arxiv_latest")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def load_labels(d, name):
    rows = [json.loads(l) for l in open(Path(d) / "human_labels.jsonl")]
    return {r["item_id"]: r["label"] for r in rows if r["annotator"].strip().lower() == name.lower()}


def rate(labels, ids):
    ls = [labels[i] for i in ids if i in labels]
    k = sum(l == "annoying" for l in ls)
    n = sum(l in ("annoying", "legitimate") for l in ls)
    return {"annoying": k, "n": n, "unsure": sum(l == "unsure" for l in ls),
            "garbled": sum(l == "garbled" for l in ls), "labeled": len(ls),
            "rate": k / n if n else float("nan"), "ci": wilson(k, n)}


def fisher(a, b):
    return fisher_exact([[a["annoying"], a["n"] - a["annoying"]], [b["annoying"], b["n"] - b["annoying"]]])[1]


def shared(la, lb, groups, name_a, name_b):
    ids = [i for i in la if i in lb and not i.startswith("rep__")]
    binary = [i for i in ids if la[i] in ("annoying", "legitimate") and lb[i] in ("annoying", "legitimate")]
    a = [la[i] == "annoying" for i in binary]
    b = [lb[i] == "annoying" for i in binary]
    n = len(binary)
    po = sum(x == y for x, y in zip(a, b)) / n if n else float("nan")
    pa, pb = (sum(a) / n, sum(b) / n) if n else (0, 0)
    pe = pa * pb + (1 - pa) * (1 - pb)
    either = {}
    for i in ids:
        if "annoying" in (la[i], lb[i]):
            either[i] = "annoying"
        elif la[i] == lb[i] == "legitimate":
            either[i] = "legitimate"
    idset = set(ids)
    rates = {g: {name_a: rate(la, [i for i in m if i in idset]), name_b: rate(lb, [i for i in m if i in idset]),
                 "Either": rate(either, [i for i in m if i in idset])}
             for g, m in sorted(groups.items())}
    return {"n": len(ids), "same": sum(la[i] == lb[i] for i in ids), "n_binary": n, "agree_binary": po,
            "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"),
            "both_annoying": sum(x and y for x, y in zip(a, b)), "only_a": sum(x and not y for x, y in zip(a, b)),
            "only_b": sum(y and not x for x, y in zip(a, b)), "rates": rates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("b3_dir")
    ap.add_argument("main_dir")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--other", help="second annotator: agreement and Either rates on shared items")
    ap.add_argument("--json")
    args = ap.parse_args()
    prov = {r["item_id"]: r for r in map(json.loads, open(Path(args.b3_dir) / "items_provenance.jsonl"))}
    lab = load_labels(args.b3_dir, args.annotator)
    groups = collections.defaultdict(list)
    for i, p in prov.items():
        groups[p["corpus"]].append(i)
        if p["corpus"].startswith("llm_"):
            model, stage, src = p["doc_id"].split("__")
            groups["llm_all"].append(i)
            if model != "gpt-4o":
                groups["llm_recent"].append(i)
                groups[f"stage_{stage}"].append(i)
                groups["source_" + ("arxiv2026" if re.fullmatch(r"\d{4}\.\d{4,5}", src) else "acl2019")].append(i)
    out = {"annotator": args.annotator, "batch3": {g: rate(lab, ids) for g, ids in sorted(groups.items())}}
    b = out["batch3"]
    out["tests"] = {f"{g} vs arxiv2026": fisher(b[g], b["arxiv2026"])
                    for g in ("llm_gpt-4o", "llm_gpt-5.6-sol", "llm_gpt-6-sol", "llm_recent", "acl2019")}
    out["tests"]["stage_revised vs stage_generated"] = fisher(b["stage_revised"], b["stage_generated"])
    out["tests"]["source_arxiv2026 vs source_acl2019"] = fisher(b["source_arxiv2026"], b["source_acl2019"])

    mprov = {r["item_id"]: r["corpus"] for r in map(json.loads, open(Path(args.main_dir) / "items_provenance.jsonl"))}
    mlab = load_labels(args.main_dir, args.annotator)
    main_pool = {"arxiv2026": rate(mlab, [i for i, c in mprov.items() if c in MAIN_ARXIV]),
                 "acl2019": rate(mlab, [i for i, c in mprov.items() if c == "acl2019"])}
    out["main_pool"] = main_pool
    out["tests"]["arxiv2026: batch 3 vs main pool"] = fisher(b["arxiv2026"], main_pool["arxiv2026"])
    for s in ("arxiv2026", "acl2019"):
        m = main_pool[s]
        out["tests"][f"{s} unsure share: batch 3 vs main pool"] = fisher_exact(
            [[b[s]["unsure"], b[s]["labeled"] - b[s]["unsure"]], [m["unsure"], m["labeled"] - m["unsure"]]])[1]

    reps = [(i[5:], l) for i, l in lab.items() if i.startswith("rep__")]
    out["repeats"] = {"n": len(reps), "same": sum(lab.get(i) == l for i, l in reps),
                      "annoying_either": sum(lab.get(i) == "annoying" or l == "annoying" for i, l in reps),
                      "annoying_both": sum(lab.get(i) == "annoying" and l == "annoying" for i, l in reps)}

    print(f"Batch 3, {args.annotator}")
    print(f"{'group':<22}{'annoying':>9}{'n':>5}{'rate':>7}  {'95% CI':<13}{'unsure':>7}")
    for g, r in b.items():
        print(f"{g:<22}{r['annoying']:>9}{r['n']:>5}{r['rate']:>7.1%}  {r['ci'][0]:.1%}-{r['ci'][1]:.1%}{r['unsure']:>7}")
    for s, r in main_pool.items():
        print(f"{'main pool ' + s:<22}{r['annoying']:>9}{r['n']:>5}{r['rate']:>7.1%}  {r['ci'][0]:.1%}-{r['ci'][1]:.1%}{r['unsure']:>7}")
    for t, p in out["tests"].items():
        print(f"  {t}: p = {p:.3g}")
    r = out["repeats"]
    print(f"hidden repeats: {r['same']}/{r['n']} same label; annoying in either {r['annoying_either']}, both {r['annoying_both']}")
    if args.other:
        out["shared"] = shared(lab, load_labels(args.b3_dir, args.other), groups, args.annotator, args.other)
        sh = out["shared"]
        print(f"\nshared with {args.other}: {sh['n']} items; same label {sh['same']}; "
              f"both annoying/legitimate {sh['n_binary']}: agreement {sh['agree_binary']:.1%}, kappa {sh['kappa']:.2f}; "
              f"annoying both {sh['both_annoying']}, only {args.annotator} {sh['only_a']}, only {args.other} {sh['only_b']}")
        for g, r in sh["rates"].items():
            print(f"  {g:<20}" + "  ".join(f"{who} {v['annoying']}/{v['n']} ({v['rate']:.1%})" for who, v in r.items()))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
