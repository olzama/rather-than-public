import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from paper_pipeline.main import extract_antithesis, main, track_antithesis


class AntithesisTests(unittest.TestCase):
    def test_stage_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp) / 'generated'
            revised = Path(tmp) / 'revised'
            for directory in (generated, revised):
                directory.mkdir()
                (directory / 'paper.md').write_text('Paper')
            for stage, paper, expected in (
                ('all', None, [generated, revised]),
                ('all', Path('existing.md'), [revised]),
                ('generate', None, [generated]),
                ('revise', None, [revised]),
                ('review', None, []),
            ):
                with self.subTest(stage=stage, paper=paper), patch(
                    'paper_pipeline.main.extract_antithesis'
                ) as extract:
                    track_antithesis(SimpleNamespace(stage=stage, paper=paper,
                                                     generated_dir=generated, revised_dir=revised))
                    self.assertEqual([c.args[0] for c in extract.call_args_list], expected)

    def test_partial_failure_still_extracts_saved_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.jsonl'
            source.write_text(json.dumps({'doc_id': 'paper', 'title': 'Title', 'sections': [
                {'category': 'abstract', 'text': 'Abstract'}]}))
            pipeline = Mock()
            pipeline.generator.generate_paper.return_value = ('Draft', 0)
            pipeline.evaluator.evaluate_paper.side_effect = RuntimeError('Review failed')
            with patch('sys.argv', ['pipeline', '--input-jsonl', str(source),
                                    '--generated-dir', str(root/'generated'),
                                    '--revised-dir', str(root/'revised')]), patch(
                'paper_pipeline.main.PaperPipeline', return_value=pipeline
            ), patch('paper_pipeline.main.extract_antithesis') as extract:
                with self.assertRaises(SystemExit):
                    main()
                extract.assert_called_once_with(str(root/'generated'))

    def test_actual_markdown_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'2019.test-1.1.md').write_text('This is not a failure but a success.')
            extract_antithesis(root)
            output = root/'antithesis/antithesis_instances.jsonl'
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(records)
            self.assertEqual(records[0]['doc_id'], '2019.test-1.1')
