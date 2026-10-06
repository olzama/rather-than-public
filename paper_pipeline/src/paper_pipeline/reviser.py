import logging
from typing import Tuple

from .config_loader import ChatCfg, DecodingCfg, PaperCfg
from .model_backend import ChatModelBackend
from .prompts import build_revision_prompt

logger = logging.getLogger(__name__)


class PaperReviser:
    """Rewrites a paper to act on a review's methodology/development suggestions."""

    def __init__(
        self,
        backend: ChatModelBackend,
        decoding_cfg: DecodingCfg,
        chat_cfg: ChatCfg,
        paper_cfg: PaperCfg,
    ):
        self.backend = backend
        self.decoding_cfg = decoding_cfg
        self.chat_cfg = chat_cfg
        self.paper_cfg = paper_cfg

    def revise_paper(self, paper_text: str, review_text: str) -> Tuple[str, float]:
        """Return (revised_paper_text, estimated_cost_usd)."""
        if not isinstance(paper_text, str) or not paper_text.strip():
            raise ValueError("Paper text must be a non-empty string")
        if not isinstance(review_text, str) or not review_text.strip():
            raise ValueError("Review text must be a non-empty string")
        prompt = build_revision_prompt(paper_text, review_text, self.paper_cfg)
        revised, cost = self.backend.generate(prompt, self.decoding_cfg, self.chat_cfg)
        if not revised.strip():
            raise ValueError("The model returned an empty revision")
        return revised, cost
