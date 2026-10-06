import logging
from typing import Tuple

from .config_loader import ChatCfg, DecodingCfg, EvaluationCfg
from .model_backend import ChatModelBackend
from .prompts import build_evaluation_prompt

logger = logging.getLogger(__name__)


class PaperEvaluator:
    """Reviews a plain-text paper, producing strengths/weaknesses/suggestions."""

    def __init__(
        self,
        backend: ChatModelBackend,
        decoding_cfg: DecodingCfg,
        chat_cfg: ChatCfg,
        evaluation_cfg: EvaluationCfg,
    ):
        self.backend = backend
        self.decoding_cfg = decoding_cfg
        self.chat_cfg = chat_cfg
        self.evaluation_cfg = evaluation_cfg

    def evaluate_paper(self, paper_text: str) -> Tuple[str, float]:
        """Return (review_text, estimated_cost_usd) for one non-empty paper."""
        if not isinstance(paper_text, str) or not paper_text.strip():
            raise ValueError("Paper text must be a non-empty string")
        prompt = build_evaluation_prompt(paper_text, self.evaluation_cfg)
        review, cost = self.backend.generate(prompt, self.decoding_cfg, self.chat_cfg)
        if not review.strip():
            raise ValueError("The model returned an empty review")
        return review, cost
