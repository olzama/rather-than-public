#!/usr/bin/env python3
r"""
Summarize code-release repos with Claude Code in headless mode (`claude -p`),
one call per repo: the rendered bundle from
../data_collection/build_code_summary_bundles.py --render-dir goes in on
stdin, summary_prompt.md is the system prompt, and a Markdown summary comes
out. No paper text is involved at any point.

Designed to run on a Claude Code subscription without surprises:
  - Resumable: repos whose summary file already exists are skipped, so it can
    be stopped (Ctrl-C, --limit, a usage limit) and restarted any time.
  - Minimal per-call overhead: the default Claude Code system prompt is
    replaced, and tools, MCP servers, skills and session persistence are off
    (~500 tokens of overhead per call, measured). --bare is not used because it
    ignores OAuth, i.e. subscription logins.
  - Stops at the first usage/rate-limit error instead of burning retries;
    rerun after the limit resets.
  - Logs token usage per call to <out_dir>/usage.jsonl, so actual consumption
    can be compared against /usage before scaling up.

Usage:
    python3 summarize_code_bundles.py <selection.txt> <bundles_dir> <out_dir> \
        [--pilot N | --limit N | --only owner/repo ...] [--workers 1] \
        [--model sonnet] [--effort LEVEL] [--dry-run]

<selection.txt>: one owner/repo per line (e.g. code_summary_selection.txt).
--pilot N: N repos spread evenly over the bundle-size range (deterministic),
    for measuring usage and checking quality before a full run.
--limit N: summarize at most N (not-yet-done) repos this run, in selection order.
Output: <out_dir>/<owner>__<repo>.md, plus usage.jsonl and errors.jsonl.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT_PATH = Path(__file__).with_name("summary_prompt.md")
REQUIRED_SECTIONS = ["## Overview", "## Task and data", "## Methods and algorithms",
                     "## Baselines, variants and ablations", "## Experimental pipeline",
                     "## Evaluation", "## Outputs and results", "## Dependencies and resources",
                     "## Implementation status and gaps", "## Keywords"]
LIMIT_MARKERS = ("usage limit", "rate limit", "limit reached", "rate_limit", "429",
                 "overloaded", "credit balance", "quota")


def claude_cmd(args, system_prompt):
    cmd = ["claude", "-p", "--model", args.model, "--output-format", "json",
           "--tools", "", "--system-prompt", system_prompt, "--strict-mcp-config",
           "--disable-slash-commands", "--no-session-persistence"]
    if args.effort:
        cmd += ["--effort", args.effort]
    return cmd


def pick_pilot(repos, bundles_dir, n):
    sized = sorted(repos, key=lambda r: bundle_path(bundles_dir, r).stat().st_size)
    if n >= len(sized):
        return sized
    step = (len(sized) - 1) / (n - 1) if n > 1 else 0
    return [sized[round(i * step)] for i in range(n)]


def bundle_path(bundles_dir, repo):
    return Path(bundles_dir) / (repo.replace("/", "__") + ".txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("selection")
    p.add_argument("bundles_dir")
    p.add_argument("out_dir")
    p.add_argument("--pilot", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--model", default="sonnet")
    p.add_argument("--effort", default=None, help="passed to claude --effort (e.g. low)")
    p.add_argument("--timeout", type=int, default=900, help="seconds per call")
    p.add_argument("--dry-run", action="store_true", help="list what would run, call nothing")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = PROMPT_PATH.read_text()
    repos = [l.strip() for l in open(args.selection) if l.strip()]
    missing = [r for r in repos if not bundle_path(args.bundles_dir, r).is_file()]
    if missing:
        sys.exit(f"{len(missing)} selected repos have no bundle, e.g. {missing[:3]}")
    if args.only:
        repos = [r for r in repos if r in set(args.only)]
    if args.pilot:
        repos = pick_pilot(repos, args.bundles_dir, args.pilot)
    todo = [r for r in repos if not (out_dir / (r.replace("/", "__") + ".md")).exists()]
    n_done = len(repos) - len(todo)
    if args.limit is not None:
        todo = todo[: args.limit]
    total_chars = sum(bundle_path(args.bundles_dir, r).stat().st_size for r in todo)
    print(f"{n_done} already done; {len(todo)} to run, "
          f"~{total_chars / 3.7e6:.1f}M input tokens (estimate)", file=sys.stderr)
    if args.dry_run:
        for r in todo:
            print(r, bundle_path(args.bundles_dir, r).stat().st_size)
        return

    stop = threading.Event()
    lock = threading.Lock()
    totals = {"done": 0, "in": 0, "out": 0}

    def log(name, record):
        with lock, open(out_dir / name, "a") as f:
            f.write(json.dumps(record) + "\n")

    def run(repo):
        if stop.is_set():
            return
        bundle = bundle_path(args.bundles_dir, repo)
        started = time.time()
        try:
            with open(bundle) as stdin:
                proc = subprocess.run(claude_cmd(args, system_prompt), stdin=stdin,
                                      capture_output=True, text=True, timeout=args.timeout,
                                      cwd=out_dir)
        except subprocess.TimeoutExpired:
            log("errors.jsonl", {"repo": repo, "error": "timeout", "ts": time.time()})
            return
        try:
            res = json.loads(proc.stdout)
        except ValueError:
            res = {"is_error": True, "result": (proc.stdout + proc.stderr)[-2000:]}
        text = res.get("result") or ""
        if res.get("is_error") or proc.returncode != 0 or not text.strip():
            msg = (text or proc.stderr or "")[-2000:]
            log("errors.jsonl", {"repo": repo, "returncode": proc.returncode,
                                 "subtype": res.get("subtype"), "error": msg, "ts": time.time()})
            if any(m in msg.lower() for m in LIMIT_MARKERS):
                stop.set()
                print(f"!! usage/rate limit hit on {repo}; stopping. Rerun later to resume.\n   {msg[:300]}",
                      file=sys.stderr)
            else:
                print(f"!! {repo}: error, see errors.jsonl", file=sys.stderr)
            return
        text = text.strip()
        missing_sections = [s for s in REQUIRED_SECTIONS if s not in text]
        (out_dir / (repo.replace("/", "__") + ".md")).write_text(text + "\n")
        u = res.get("usage", {})
        n_in = (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                + u.get("cache_read_input_tokens", 0))
        n_out = u.get("output_tokens", 0)
        log("usage.jsonl", {"repo": repo, "model": list(res.get("modelUsage", {}) or [args.model]),
                            "bundle_chars": bundle.stat().st_size, "input_tokens": n_in,
                            "output_tokens": n_out,
                            "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
                            "summary_words": len(text.split()), "missing_sections": missing_sections,
                            "duration_s": round(time.time() - started, 1),
                            "api_equivalent_usd": res.get("total_cost_usd"), "ts": time.time()})
        with lock:
            totals["done"] += 1
            totals["in"] += n_in
            totals["out"] += n_out
            print(f"[{totals['done']}/{len(todo)}] {repo}: {n_in} in / {n_out} out tokens, "
                  f"{time.time() - started:.0f}s{'  (missing: ' + ', '.join(missing_sections) + ')' if missing_sections else ''}"
                  f"  | cumulative {totals['in'] / 1e6:.2f}M in / {totals['out'] / 1e3:.0f}k out",
                  file=sys.stderr)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run, todo))
    except KeyboardInterrupt:
        stop.set()
        print("interrupted; finished summaries are kept, rerun to resume", file=sys.stderr)
    print(f"DONE this run: {totals['done']} summaries, {totals['in'] / 1e6:.2f}M input / "
          f"{totals['out'] / 1e3:.0f}k output tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
