from typing import Any, Dict, Tuple

from .config_loader import ModelCfg, PipelineConfig
from .evaluator import PaperEvaluator
from .paper_generator import PaperGenerator
from .model_backend import ChatModelBackend
from .reviser import PaperReviser


class PaperPipeline:
    """Wires one shared model backend to the three pipeline stages.

    Generation and revision share one `ChatModelBackend` built from the config's
    shared `model` and `quantization` settings. Review can use a separate model
    name so it can run with different provider behavior or cost profile.
    """

    def __init__(self, config: PipelineConfig):
        backend = ChatModelBackend(config.model, config.quantization)
        review_model_cfg = ModelCfg(
            name=config.evaluation.model_name,
            device=config.model.device,
            dtype=config.model.dtype,
            is_chat_model=config.model.is_chat_model,
        )
        review_backend = (
            backend
            if review_model_cfg.name == config.model.name
            else ChatModelBackend(review_model_cfg, config.quantization)
        )

        self.generator = PaperGenerator(
            backend, config.generation.decoding, config.generation.chat, config.generation.paper
        )
        self.evaluator = PaperEvaluator(
            review_backend, config.evaluation.decoding, config.evaluation.chat, config.evaluation.evaluation
        )
        # Revision reuses the generation stage's PaperCfg so the revised paper is
        # held to the same section list and target length as the original.
        self.reviser = PaperReviser(
            backend, config.revision.decoding, config.revision.chat, config.generation.paper
        )

    def run(self, seed: Dict[str, Any]) -> Tuple[str, str, str, float]:
        """Generate, review, and revise one title/abstract seed.

        Returns (paper, review, revised_paper, total_estimated_cost_usd).
        """
        paper_text, generation_cost = self.generator.generate_paper(seed)
        review_text, revised_text, review_revision_cost = self.review_and_revise(
            paper_text
        )
        return (
            paper_text,
            review_text,
            revised_text,
            generation_cost + review_revision_cost,
        )

    def review_and_revise(self, paper_text: str) -> Tuple[str, str, float]:
        """Review and revise an existing paper. Returns (review, revised_paper, cost)."""
        review_text, review_cost = self.evaluator.evaluate_paper(paper_text)
        revised_text, revision_cost = self.reviser.revise_paper(paper_text, review_text)
        return review_text, revised_text, review_cost + revision_cost
