#!/usr/bin/env python3
r"""
Build the item sets for coding whether Y (the rejected alternative) is a
straw man, blind to the annoying/legitimate label.

  dev   --dev-unsure arXiv 2026 items labeled unsure plus --dev-acl ACL 2019
        items labeled legitimate: for refining the definition; not in test.
  test  every annoying arXiv 2026 item plus --test-legit random legitimate
        arXiv 2026 items, shuffled.

Writes <out_dir>/dev.jsonl and test.jsonl (item_id, sentence, context,
Y's position in the sentence; no labels) and test_key.jsonl (item_id,
label, stratum), which the coding page never sees.

Usage:
    python3 strawman_sets.py ../data/annotation ../data/annotation/strawman \
        [--annotator A1] [--dev-unsure 20] [--dev-acl 10] [--test-legit 80] [--seed 1]
"""
import argparse
import json
import random
import re
from pathlib import Path

ARXIV2026 = {"arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"}


def locate(sentence, phrase):
    """Character span of phrase in sentence, tolerating whitespace differences."""
    pat = r"\s+".join(re.escape(w) for w in phrase.split())
    m = re.search(pat, sentence)
    return (m.start(), m.end()) if m else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--dev-unsure", type=int, default=20)
    ap.add_argument("--dev-acl", type=int, default=10)
    ap.add_argument("--test-legit", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    d, out = Path(args.annotation_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prov = {json.loads(l)["item_id"]: json.loads(l) for l in open(d / "items_provenance.jsonl")}
    pub = {json.loads(l)["item_id"]: json.loads(l) for l in open(d / "items_public.jsonl")}
    xy = {json.loads(l)["item_id"]: json.loads(l) for l in open(d / "xy_alternatives.jsonl")}
    labels = {}
    for l in open(d / "human_labels.jsonl"):
        r = json.loads(l)
        if r["annotator"] == args.annotator:
            labels[r["item_id"]] = r["label"]

    def usable(i):
        s, e = locate(pub[i]["sentence"], xy[i]["y"]) if i in xy else (None, None)
        return s is not None and "repeat_of" not in prov[i]

    rng = random.Random(args.seed)
    by = lambda corp, lab: sorted(i for i in pub if (prov[i]["corpus"] in corp) and labels.get(i) == lab and usable(i))
    dev = rng.sample(by(ARXIV2026, "unsure"), args.dev_unsure) + rng.sample(by({"acl2019"}, "legitimate"), args.dev_acl)
    test = by(ARXIV2026, "annoying") + rng.sample(by(ARXIV2026, "legitimate"), args.test_legit)
    rng.shuffle(dev)
    rng.shuffle(test)

    def record(i):
        it = pub[i]
        ys, ye = locate(it["sentence"], xy[i]["y"])
        return {"item_id": i, "sentence": it["sentence"], "y_start": ys, "y_end": ye,
                "rt_start": it["local_start"], "rt_end": it["local_end"],
                "context_before": it.get("context_before", ""), "context_after": it.get("context_after", "")}

    for name, ids in (("dev", dev), ("test", test)):
        with open(out / f"{name}.jsonl", "w") as f:
            for i in ids:
                f.write(json.dumps(record(i)) + "\n")
    with open(out / "test_key.jsonl", "w") as f:
        for i in test:
            f.write(json.dumps({"item_id": i, "label": labels[i], "stratum": prov[i]["corpus"]}) + "\n")
    print(f"dev {len(dev)}, test {len(test)} "
          f"({sum(labels[i] == 'annoying' for i in test)} annoying) -> {out}")


if __name__ == "__main__":
    main()
