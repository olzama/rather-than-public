#!/usr/bin/env python3
r"""
Score the minimal pairs from build_reward_pairs.py with open reward models:
does a reward model prefer the sentence with its "rather than Y" clause?

Each text is scored as the assistant turn of a two-turn conversation whose
user turn gives the preceding sentence of the paper and asks for the next
sentence. Scores (the reward head's output) are appended to --out, one line
per item and model; items already scored for a model are skipped, so the
script resumes.

Whitespace is collapsed in all texts. Needs torch and transformers >= 4.51
(Qwen3 models). --device auto (default) uses a CUDA GPU if present, then
Apple silicon (mps), then the CPU; the 8B models need about 16 GB of GPU
memory in bfloat16.

Usage:
    python3 reward_scores.py ../data/reward_pairs/pairs.jsonl ../data/reward_pairs/scores.jsonl \
        --models Skywork/Skywork-Reward-V2-Qwen3-0.6B Skywork/Skywork-Reward-V2-Qwen3-8B \
        [--device auto] [--dtype bfloat16] [--limit N] [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

PROMPT = ("Here is a passage from an NLP research paper:\n\n{context}\n\n"
          "Write the next sentence of the paper.")


def flat(text):
    """Collapse whitespace, so PDF line breaks (ACL 2019) and LaTeX-source text (arXiv) look alike."""
    return " ".join(text.split())


def conversation(context, text):
    return [{"role": "user", "content": PROMPT.format(context=flat(context) or "(start of section)")},
            {"role": "assistant", "content": flat(text)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="print two formatted conversations and exit")
    args = ap.parse_args()
    pairs = [json.loads(l) for l in open(args.pairs)][:args.limit]
    if args.dry_run:
        for p in pairs[:2]:
            print(json.dumps(conversation(p["context_before"], p["original"]), indent=1))
            print(json.dumps(conversation(p["context_before"], p["deleted"]), indent=1))
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.device == "auto":
        args.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        if args.device == "cpu" and args.dtype != "float32":
            args.dtype = "float32"
        print(f"device: {args.device}, dtype: {args.dtype}", file=sys.stderr)

    done = set()
    if Path(args.out).exists():
        done = {(r["model"], r["item_id"]) for r in map(json.loads, open(args.out))}
    dtype = getattr(torch, args.dtype)
    with open(args.out, "a") as out:
        for name in args.models:
            todo = [p for p in pairs if (name, p["item_id"]) not in done]
            print(f"{name}: {len(todo)} pairs to score", file=sys.stderr)
            if not todo:
                continue
            tok = AutoTokenizer.from_pretrained(name)
            if args.device == "cuda" and torch.cuda.device_count() > 1:
                # shard across GPUs (fills GPU 0 first, so small models stay on one)
                model = AutoModelForSequenceClassification.from_pretrained(
                    name, dtype=dtype, num_labels=1, device_map="auto")
            else:
                model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype, num_labels=1)
                model.to(args.device)
            model.eval()

            def score(conv):
                text = tok.apply_chat_template(conv, tokenize=False)
                if tok.bos_token and text.startswith(tok.bos_token):
                    text = text[len(tok.bos_token):]  # the tokenizer adds it again
                enc = tok(text, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    return float(model(**enc).logits[0][0].float())

            for k, p in enumerate(todo):
                r = {"model": name, "item_id": p["item_id"],
                     "original": score(conversation(p["context_before"], p["original"])),
                     "deleted": score(conversation(p["context_before"], p["deleted"]))}
                out.write(json.dumps(r) + "\n")
                out.flush()
                if k % 50 == 0:
                    print(f"  {k}/{len(todo)}", file=sys.stderr)
            del model
            if args.device == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
