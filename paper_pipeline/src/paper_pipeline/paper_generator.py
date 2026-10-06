import logging
from typing import Any, Dict, Tuple

from .config_loader import ChatCfg, DecodingCfg, PaperCfg
from .model_backend import ChatModelBackend
from .prompts import build_paper_prompt

logger = logging.getLogger(__name__)


class PaperGenerator:
    """Generates ACL-style academic papers as Markdown, one title/abstract seed at a time."""

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

    def generate_paper(self, seed: Dict[str, Any]) -> Tuple[str, float]:
        """Generate one paper as plain text. Returns (paper_text, estimated_cost_usd)."""
        prompt = build_paper_prompt(seed, self.paper_cfg)
        text, cost = self.backend.generate(prompt, self.decoding_cfg, self.chat_cfg)
        if not text.strip():
            raise ValueError("The model returned an empty paper")
        return text, cost
