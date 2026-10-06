from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

DEFAULT_SOURCES = {
    "acl2019": "data/acl2019/sections.jsonl.gz",
    "arxiv2026": "data/arxiv2026/sections.jsonl.gz",
}


@dataclass
class ModelCfg:
    name: str
    device: str = "auto"
    dtype: str = "auto"
    is_chat_model: bool = True


@dataclass
class DecodingCfg:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 9000
    max_completion_tokens: Optional[int] = None
    do_sample: bool = True
    num_return_sequences: int = 1
    num_beams: int = 1


@dataclass
class QuantizationCfg:
    mode: str = "none"

    # 4-bit
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False
    llm_int8_threshold: float = 6.0
    offload_folder: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None


@dataclass
class ChatCfg:
    system_prompt: str = "You are a helpful assistant."


DEFAULT_ACL_SECTIONS = [
    "Abstract",
    "1 Introduction",
    "2 Related Work",
    "3 Method",
    "4 Experimental Setup",
    "5 Results",
    "6 Analysis",
    "7 Conclusion",
    "Limitations",
]


@dataclass
class PaperCfg:
    target_pages: int = 8
    words_per_page: int = 800
    sections: List[str] = field(default_factory=lambda: list(DEFAULT_ACL_SECTIONS))

    @property
    def target_words(self) -> int:
        return self.target_pages * self.words_per_page


DEFAULT_REVIEW_SECTIONS = [
    "Summary",
    "Strengths",
    "Weaknesses",
    "Suggestions for Improvement",
    "Questions for the Authors",
    "Overall Assessment",
]


@dataclass
class EvaluationCfg:
    sections: List[str] = field(default_factory=lambda: list(DEFAULT_REVIEW_SECTIONS))


@dataclass
class GenerationConfig:
    """Decoding, chat, and output-shape settings for the paper-generation stage."""

    decoding: DecodingCfg
    chat: ChatCfg
    paper: PaperCfg


@dataclass
class EvaluationStageConfig:
    """Decoding, chat, and output-shape settings for the review stage."""

    model_name: str
    decoding: DecodingCfg
    chat: ChatCfg
    evaluation: EvaluationCfg


@dataclass
class RevisionConfig:
    """Decoding and chat settings for the revision stage.

    Reuses the generation stage's `PaperCfg` (target length, section list) so the
    revised paper is held to the same output shape as the original.
    """
    model_name: str
    decoding: DecodingCfg
    chat: ChatCfg


@dataclass
class PipelineConfig:
    """Full three-stage config. `model`/`quantization` are shared across stages so a
    local model is loaded once and reused for generation, evaluation, and revision."""

    model: ModelCfg
    quantization: QuantizationCfg
    generation: GenerationConfig
    evaluation: EvaluationStageConfig
    revision: RevisionConfig
    input_jsonl: str = "data/acl2019/sections.jsonl"
    # language_filter.py output; ACL 2019 doc_ids listed there are never generated.
    # Set to null in the config to disable (not recommended).
    excluded_documents: str | None = "data/excluded_documents.tsv"
    # Sections files papers are drawn from (--num-papers draws at random, split evenly).
    sources: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SOURCES))
    # Registry of every paper drawn for generation (batch, corpus, doc_id); each new
    # batch is appended to it, and draws never repeat a paper listed there.
    used_papers: str = "paper_pipeline/used_papers.tsv"


def _stage_decoding(raw_stage: dict) -> DecodingCfg:
    return DecodingCfg(**raw_stage.get("decoding", {}))


def _stage_chat(raw_stage: dict) -> ChatCfg:
    return ChatCfg(**raw_stage.get("chat", {}))


def load_config(path: str = "config.yaml") -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model = ModelCfg(**raw["model"])
    quantization = QuantizationCfg(**raw.get("quantization", {}))

    gen_raw = raw.get("generation", {})
    generation = GenerationConfig(
        decoding=_stage_decoding(gen_raw),
        chat=_stage_chat(gen_raw),
        paper=PaperCfg(**gen_raw.get("paper", {})),
    )

    if generation.decoding.num_return_sequences != 1:
        raise ValueError("Paper generation requires generation.decoding.num_return_sequences: 1")

    eval_raw = raw.get("evaluation", {})
    evaluation_cfg = EvaluationCfg(**{"sections": eval_raw["sections"]} if "sections" in eval_raw else {})
    if not isinstance(evaluation_cfg.sections, list) or not evaluation_cfg.sections or any(
        not isinstance(section, str) or not section.strip()
        for section in evaluation_cfg.sections
    ):
        raise ValueError("evaluation.sections must contain non-empty headings")
    eval_decoding = _stage_decoding(eval_raw)
    if eval_decoding.num_return_sequences != 1:
        raise ValueError("Paper evaluation requires evaluation.decoding.num_return_sequences: 1")
    evaluation = EvaluationStageConfig(
        model_name=eval_raw.get("model_name", model.name),
        decoding=eval_decoding,
        chat=_stage_chat(eval_raw),
        evaluation=evaluation_cfg,
    )

    rev_raw = raw.get("revision", {})
    revision = RevisionConfig(
        model_name=rev_raw.get("model_name", model.name),
        decoding=_stage_decoding(rev_raw),
        chat=_stage_chat(rev_raw),
    )

    return PipelineConfig(
        model=model,
        quantization=quantization,
        generation=generation,
        evaluation=evaluation,
        revision=revision,
        input_jsonl=raw.get("input_jsonl", "data/acl2019/sections.jsonl"),
        excluded_documents=raw.get("excluded_documents", "data/excluded_documents.tsv"),
        sources=dict(raw.get("sources") or DEFAULT_SOURCES),
        used_papers=raw.get("used_papers", "paper_pipeline/used_papers.tsv"),
    )
