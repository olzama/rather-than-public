# Reward-model minimal pairs

`pairs.jsonl`: 424 annotation-pool items (batch 1), each sentence as written
and with its "rather than Y" clause deleted (`data_collection/build_reward_pairs.py`).
Controls: `pairs_control.jsonl` / `replacements.jsonl` (clause replaced by a non-contrastive
clause of the same length, `data_collection/build_length_controls.py`) and `sentiment.jsonl`
(LLM sentiment ratings of original, control and deleted, `data_collection/sentiment_llm.py`).
Results: `scores.jsonl`, `scores_control.jsonl`, `control_analysis.txt`, `sentiment_control.txt`, `reward_analysis.txt` (vs. annotation labels),
`reward_annotation_length.txt` (annoying vs. legitimate, length-adjusted; `counting/reward_annotation_length.py`).

## Running the scoring (see also ../../REWARD_MODEL_RUN.txt)

```
python3 -m venv ~/rm-env && source ~/rm-env/bin/activate
pip install torch "transformers>=4.51" accelerate numpy scipy
cd rather-than-public/counting
# check the prompt format, then a 5-item test with the smallest model
python3 reward_scores.py ../data/reward_pairs/pairs.jsonl ../data/reward_pairs/scores.jsonl --models x --dry-run
python3 reward_scores.py ../data/reward_pairs/pairs.jsonl /tmp/test_scores.jsonl \
    --models Skywork/Skywork-Reward-V2-Qwen3-0.6B --limit 5
# full run (resumable; models download from Hugging Face on first use, ~1-16 GB each)
python3 reward_scores.py ../data/reward_pairs/pairs.jsonl ../data/reward_pairs/scores.jsonl \
    --models Skywork/Skywork-Reward-V2-Qwen3-0.6B Skywork/Skywork-Reward-V2-Qwen3-1.7B \
             Skywork/Skywork-Reward-V2-Qwen3-8B Skywork/Skywork-Reward-V2-Llama-3.1-8B
python3 reward_analysis.py ../data/reward_pairs/pairs.jsonl ../data/reward_pairs/scores.jsonl ../data/annotation
# controls: score the length-matched texts with the same models, then analyze
python3 reward_scores.py ../data/reward_pairs/pairs_control.jsonl ../data/reward_pairs/scores_control.jsonl \
    --models Skywork/Skywork-Reward-V2-Qwen3-0.6B Skywork/Skywork-Reward-V2-Qwen3-1.7B \
             Skywork/Skywork-Reward-V2-Qwen3-8B Skywork/Skywork-Reward-V2-Llama-3.1-8B
python3 reward_control_analysis.py ../data/reward_pairs/scores.jsonl \
    ../data/reward_pairs/pairs_control.jsonl ../data/reward_pairs/scores_control.jsonl
python3 sentiment_control.py ../data/reward_pairs/pairs_control.jsonl ../data/reward_pairs/scores.jsonl \
    ../data/reward_pairs/scores_control.jsonl ../data/reward_pairs/sentiment.jsonl
```
If a run stops with CUDA out of memory, rerun it with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(finished items are skipped). Without `bin/activate` (e.g. conda), call the environment's python directly.

848 forward passes per model: a few minutes per model on a GPU. The device is
chosen automatically (CUDA GPU, then Apple silicon, then CPU). Commit `scores.jsonl` and `scores_control.jsonl` afterwards (`git add -f`, data/ is ignored).
