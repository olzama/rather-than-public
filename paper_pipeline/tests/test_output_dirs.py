import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from paper_pipeline.main import configure_output_dirs, main, model_logging


class OutputDirectoryTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(generated_dir=None,
                               evaluated_dir=None, revised_dir=None)

    def test_names_and_overrides(self):
        args = self.args()
        args.evaluated_dir = '/tmp/custom_reviews'
        configure_output_dirs(args, 'Qwen/Qwen2.5-14B-Instruct')
        self.assertEqual(args.generated_dir, Path('Qwen_Qwen2.5-14B-Instruct/generated_papers'))
        self.assertEqual(args.evaluated_dir, '/tmp/custom_reviews')
        self.assertEqual(args.revised_dir.parent, args.generated_dir.parent)

    def test_full_workflow_separates_models(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                for name in ('model-a', 'model-b'):
                    pipeline = Mock()
                    pipeline.generator.generate_paper.return_value = ('Paper', 0)
                    pipeline.evaluator.evaluate_paper.return_value = ('Review', 0)
                    pipeline.reviser.revise_paper.return_value = ('Revision', 0)
                    Path('input.jsonl').write_text('{"doc_id":"2019.test-1.1","title":"Test","sections":[{"category":"abstract","text":"Abstract"}]}\n')
                    with patch('sys.argv', ['pipeline', '--batch', 't1']), patch(
                        'paper_pipeline.main.load_config', return_value=SimpleNamespace(model=SimpleNamespace(name=name), input_jsonl="input.jsonl", excluded_documents=None,
                                                                         sources={"acl2019": "input.jsonl"}, used_papers="used.tsv")
                    ), patch('paper_pipeline.main.PaperPipeline', return_value=pipeline), patch(
                        'paper_pipeline.main.extract_antithesis'
                    ) as extract:
                        main()
                    for folder in ('generated_papers', 'evaluated_papers', 'revised_papers'):
                        self.assertTrue(Path(name, folder, '2019.test-1.1.md').is_file())
                    self.assertIn('Paper summary:', Path(name, 'pipeline.log').read_text())
                    self.assertIn('acl2019\t2019.test-1.1', Path(name, 'sample_ids.tsv').read_text())
                    self.assertEqual([call.args[0] for call in extract.call_args_list],
                                     [Path(name, 'generated_papers'), Path(name, 'revised_papers')])
                # both models used batch t1: one draw, recorded once
                self.assertEqual([l for l in Path('used.tsv').read_text().splitlines() if not l.startswith('#')],
                                 ['t1\tacl2019\t2019.test-1.1'])
                self.assertFalse(Path('generated_topics').exists())
            finally:
                os.chdir(previous)

    def test_single_stage_creates_complete_model_folder(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with patch('sys.argv', ['pipeline', '--stage', 'generate']), patch(
                    'paper_pipeline.main.load_config',
                    return_value=SimpleNamespace(model=SimpleNamespace(name='gpt-4o'))
                ), patch('paper_pipeline.main.run_pipeline'):
                    main()
                model_dir = Path('gpt-4o')
                for folder in ('generated_papers', 'evaluated_papers', 'revised_papers'):
                    self.assertTrue((model_dir / folder).is_dir())
                    self.assertFalse(Path(folder).exists())
                self.assertTrue((model_dir / 'pipeline.log').is_file())
            finally:
                os.chdir(previous)

    def test_extract_does_not_require_config(self):
        with patch('sys.argv', ['pipeline', '--stage', 'extract', '--papers-dir', 'papers']), patch(
            'paper_pipeline.main.extract_antithesis'
        ), patch('paper_pipeline.main.load_config') as config:
            main()
        config.assert_not_called()


class ModelLoggingTests(unittest.TestCase):
    def test_logs_append_and_stay_with_their_model(self):
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'model-a', Path(tmp) / 'model-b'
            for folder, message in ((first, 'first run'), (second, 'second model'),
                                    (first, 'another run')):
                with model_logging(folder):
                    logging.getLogger('paper_pipeline.seeds').info(message)
            first_log = (first / 'pipeline.log').read_text(encoding='utf-8')
            second_log = (second / 'pipeline.log').read_text(encoding='utf-8')
            self.assertIn('first run', first_log)
            self.assertIn('another run', first_log)
            self.assertNotIn('second model', first_log)
            self.assertIn('second model', second_log)
            self.assertNotIn('another run', second_log)
        self.assertEqual(root_logger.handlers, original_handlers)
        self.assertEqual(root_logger.level, original_level)

    def test_failure_is_logged_and_handler_closed(self):
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'test failure'):
                with model_logging(folder):
                    raise ValueError('test failure')
            contents = (folder / 'pipeline.log').read_text(encoding='utf-8')
            self.assertIn('Traceback', contents)
            self.assertIn('test failure', contents)
        self.assertEqual(root_logger.handlers, original_handlers)
