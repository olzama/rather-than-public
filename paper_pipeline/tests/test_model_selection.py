import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from paper_pipeline.pipeline import PaperPipeline


class ModelSelectionTests(unittest.TestCase):
    def test_review_uses_separate_model(self):
        config = SimpleNamespace(
            model=SimpleNamespace(name="gpt-5.5", device="auto", dtype="auto", is_chat_model=True),
            quantization=SimpleNamespace(),
            generation=SimpleNamespace(decoding=Mock(), chat=Mock(), paper=Mock()),
            evaluation=SimpleNamespace(
                model_name="gpt-4o",
                decoding=Mock(),
                chat=Mock(),
                evaluation=Mock(),
            ),
            revision=SimpleNamespace(decoding=Mock(), chat=Mock()),
        )
        generation_backend = Mock(name="generation_backend")
        review_backend = Mock(name="review_backend")

        with patch("paper_pipeline.pipeline.ChatModelBackend", side_effect=[generation_backend, review_backend]) as backend_cls:
            pipeline = PaperPipeline(config)

        self.assertEqual([call.args[0].name for call in backend_cls.call_args_list], ["gpt-5.5", "gpt-4o"])
        self.assertIs(pipeline.generator.backend, generation_backend)
        self.assertIs(pipeline.evaluator.backend, review_backend)
        self.assertIs(pipeline.reviser.backend, generation_backend)


if __name__ == "__main__":
    unittest.main()