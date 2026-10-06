import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from paper_pipeline.main import main, parse_args


class BatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.papers = self.root / 'papers'
        self.reviews = self.root / 'reviews'
        self.revised = self.root / 'revised'
        self.papers.mkdir()
        self.reviews.mkdir()
        for stem in ('b', 'a'):
            (self.papers / f'{stem}.txt').write_text(f'Paper {stem}')
            (self.reviews / f'{stem}.md').write_text(f'Review {stem}')
        (self.papers / 'nested').mkdir()
        (self.papers / 'nested/ignored.txt').write_text('Ignored')
        self.pipeline = Mock()
        self.pipeline.evaluator.evaluate_paper.return_value = ('New review', 0.1)
        self.pipeline.reviser.revise_paper.return_value = ('Revised paper', 0.2)

    def run_cli(self, *args):
        with patch('sys.argv', ['pipeline', *args, '--evaluated-dir', str(self.reviews),
                                '--revised-dir', str(self.revised)]), patch(
            'paper_pipeline.main.PaperPipeline', return_value=self.pipeline
        ) as model:
            main()
        return model

    def test_batch_review_sorted_nonrecursive(self):
        self.run_cli('--stage', 'review', '--papers-dir', str(self.papers))
        self.assertEqual([c.args[0] for c in self.pipeline.evaluator.evaluate_paper.call_args_list],
                         ['Paper a', 'Paper b'])
        self.pipeline.reviser.revise_paper.assert_not_called()
        self.assertEqual((self.reviews / 'a.md').read_text(), 'New review\n')

    def test_batch_revision_matching(self):
        self.run_cli('--stage', 'revise', '--papers-dir', str(self.papers),
                     '--reviews-dir', str(self.reviews))
        self.assertEqual([c.args for c in self.pipeline.reviser.revise_paper.call_args_list],
                         [('Paper a', 'Review a'), ('Paper b', 'Review b')])
        self.assertTrue((self.revised / 'b.md').exists())
        self.pipeline.evaluator.evaluate_paper.assert_not_called()

    def test_missing_review_stops_before_model(self):
        (self.reviews / 'b.md').unlink()
        with patch('paper_pipeline.main.PaperPipeline') as model:
            with self.assertRaisesRegex(ValueError, 'Cannot start batch'):
                self.run_cli('--stage', 'revise', '--papers-dir', str(self.papers),
                             '--reviews-dir', str(self.reviews))
            model.assert_not_called()
        self.pipeline.reviser.revise_paper.assert_not_called()

    def test_skip_all_avoids_model(self):
        model = self.run_cli('--stage', 'review', '--papers-dir', str(self.papers), '--skip-existing')
        model.assert_not_called()

    def test_single_revision(self):
        self.run_cli('--stage', 'revise', '--paper', str(self.papers / 'a.txt'),
                     '--review', str(self.reviews / 'a.md'))
        self.pipeline.reviser.revise_paper.assert_called_once_with('Paper a', 'Review a')

    def test_failure_continues_and_exits_nonzero(self):
        self.pipeline.evaluator.evaluate_paper.side_effect = [RuntimeError('Failure'), ('Good', 0.1)]
        with self.assertRaises(SystemExit) as error:
            self.run_cli('--stage', 'review', '--papers-dir', str(self.papers))
        self.assertEqual(error.exception.code, 1)
        self.assertEqual((self.reviews / 'b.md').read_text(), 'Good\n')

    def test_empty_directory(self):
        (self.root / 'papers/nested/empty').mkdir()
        with self.assertRaisesRegex(ValueError, 'No .txt or .md papers'):
            self.run_cli('--stage', 'review', '--papers-dir', str(self.root / 'papers/nested/empty'))

    def test_conflicting_inputs(self):
        with patch('sys.argv', ['pipeline', '--stage', 'review', '--paper', 'a.txt', '--papers-dir', 'papers']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == '__main__':
    unittest.main()
