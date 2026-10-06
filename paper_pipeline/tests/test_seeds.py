import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from paper_pipeline.config_loader import load_config
from paper_pipeline.main import DEFAULT_CONFIG, generate_papers, parse_args
from paper_pipeline.pipeline import PaperPipeline
import gzip

from paper_pipeline.seeds import filename_stem, iter_seeds, load_excluded, load_ids, load_source_ids, select_seeds


def record(doc_id='2019.test-1.1', **updates):
    value = {'doc_id': doc_id, 'title': 'Seed title', 'n_references': 987654,
             'sections': [{'category': 'introduction', 'text': 'SECRET BODY'},
                          {'category': 'abstract', 'text': 'Seed abstract'},
                          {'category': 'results', 'text': 'SECRET RESULTS'}]}
    value.update(updates)
    return value


class SeedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'sections.jsonl'
        self.args = SimpleNamespace(num_papers=None, generated_dir=self.root/'generated',
                                    evaluated_dir=self.root/'reviews', revised_dir=self.root/'revised')

    def write_records(self, records):
        self.source.write_text('\n'.join(json.dumps(r) for r in records) + '\n')

    def test_multiple_and_abstract_by_category(self):
        self.write_records([record(), record('second')])
        seeds = list(iter_seeds(self.source))
        self.assertEqual([s['doc_id'] for s in seeds], ['2019.test-1.1', 'second'])
        self.assertEqual(seeds[0], {'doc_id': '2019.test-1.1', 'title': 'Seed title',
                                    'abstract': 'Seed abstract'})

    def test_invalid_and_duplicates_reported(self):
        self.write_records([record('missing-title', title=' '), record('no-abstract', sections=[]),
                            record('', title='Present'), None, record(), record(), record('last')])
        with self.source.open('a') as out:
            out.write('{broken json\n')
        with self.assertLogs('paper_pipeline.seeds', level='WARNING') as logs:
            seeds = list(iter_seeds(self.source))
        self.assertEqual([s['doc_id'] for s in seeds], ['2019.test-1.1', 'last'])
        self.assertEqual(len(logs.output), 6)
        self.assertIn('duplicate doc_id', '\n'.join(logs.output))

    def test_limit_counts_valid_unique_records(self):
        self.write_records([record('bad', sections=[]), record('a'), record('a'), record('b'), record('c')])
        with self.assertLogs('paper_pipeline.seeds'):
            self.assertEqual([s['doc_id'] for s in iter_seeds(self.source, 2)], ['a', 'b'])

    def test_filename_safety_and_mapping(self):
        ids = ['2019.ccnlg-1.1', '../escape', '%2E%2E/escape', 'a/b', 'a%2Fb', '.', '..', 'CON', 'name.']
        stems = [filename_stem(value) for value in ids]
        self.assertEqual(stems[0], ids[0])
        self.assertEqual(len(set(stems)), len(ids))
        for stem in stems:
            self.assertNotIn('/', stem)
            self.assertNotIn('\\', stem)
            self.assertNotIn(stem, ('.', '..'))

    def test_full_pipeline_one_request_per_stage_no_source_leak(self):
        self.write_records([record(), record('second')])
        config = load_config(DEFAULT_CONFIG)
        backend = Mock()
        backend.generate.side_effect = [('# Paper one', 1), ('Review one', 2), ('Revision one', 3),
                                        ('# Paper two', 1), ('Review two', 2), ('Revision two', 3)]
        with patch('paper_pipeline.pipeline.ChatModelBackend', return_value=backend):
            pipeline = PaperPipeline(config)
        self.assertEqual(generate_papers(self.args, pipeline, self.source, True), 12)
        self.assertEqual(backend.generate.call_count, 6)
        for index in (0, 3):
            prompt, decoding, chat = backend.generate.call_args_list[index].args
            self.assertIn('Seed title', prompt)
            self.assertIn('Seed abstract', prompt)
            for forbidden in ('SECRET', '987654', '2019.test-1.1', 'doc_id'):
                self.assertNotIn(forbidden, prompt)
            self.assertIs(decoding, config.generation.decoding)
            self.assertIs(chat, config.generation.chat)
        for directory in (self.args.generated_dir, self.args.evaluated_dir, self.args.revised_dir):
            self.assertEqual(sorted(p.name for p in directory.iterdir()), ['2019.test-1.1.md', 'second.md'])
        self.assertEqual((self.args.generated_dir/'2019.test-1.1.md').read_text(), '# Paper one\n')
        self.assertIn('# Paper one', backend.generate.call_args_list[1].args[0])
        self.assertIn('Review one', backend.generate.call_args_list[2].args[0])

    def test_each_stage_failure_continues_and_preserves_outputs(self):
        for failure in ('generator', 'evaluator', 'reviser'):
            with self.subTest(failure=failure):
                self.write_records([record('first'), record('second')])
                pipeline = Mock()
                methods = [pipeline.generator.generate_paper, pipeline.evaluator.evaluate_paper,
                           pipeline.reviser.revise_paper]
                for method in methods:
                    method.return_value = ('Success', 0)
                methods[['generator', 'evaluator', 'reviser'].index(failure)].side_effect = [RuntimeError('failed'), ('Success', 0)]
                with self.assertLogs('paper_pipeline.main'), self.assertRaises(SystemExit):
                    generate_papers(self.args, pipeline, self.source, True)
                self.assertTrue((self.args.revised_dir/'second.md').exists())
                self.assertEqual(pipeline.generator.generate_paper.call_count, 2)
                if failure != 'generator':
                    self.assertTrue((self.args.generated_dir/'first.md').exists())

    def test_generation_limit_and_duplicates(self):
        self.write_records([record('a'), record('a'), record('b'), record('c')])
        self.args.num_papers = 2
        pipeline = Mock()
        pipeline.generator.generate_paper.return_value = ('Paper', 0)
        with self.assertLogs('paper_pipeline.seeds'):
            generate_papers(self.args, pipeline, self.source)
        self.assertEqual(pipeline.generator.generate_paper.call_count, 2)
        pipeline.evaluator.evaluate_paper.assert_not_called()

    def test_prompt_allowlist_even_with_extra_fields(self):
        from paper_pipeline.prompts import build_paper_prompt
        from paper_pipeline.config_loader import PaperCfg
        source = record()
        source.update(abstract='Allowed abstract', area='SECRET AREA', keywords=['SECRET KEYWORD'])
        prompt = build_paper_prompt(source, PaperCfg())
        self.assertIn('Allowed abstract', prompt)
        self.assertNotIn('SECRET', prompt)
        self.assertNotIn(source['doc_id'], prompt)

    def test_prompts_contain_no_antithesis(self):
        import re
        from paper_pipeline.prompts import build_evaluation_prompt, build_paper_prompt, build_revision_prompt
        from paper_pipeline.config_loader import EvaluationCfg, PaperCfg
        prompts = [build_paper_prompt({'title': 'T', 'abstract': 'A'}, PaperCfg()),
                   build_evaluation_prompt('P', EvaluationCfg()),
                   build_revision_prompt('P', 'R', PaperCfg())]
        pattern = re.compile(r"rather than|instead of|as opposed to|not (only|just|simply|merely)\b|"
                             r"\bnot [^.,;]{1,40}\bbut\b|, (not|never)\b", re.I)
        for prompt in prompts:
            self.assertIsNone(pattern.search(prompt), prompt)
        self.assertIn('6400 words', prompts[0])

    def test_reader_yields_before_reading_later_lines(self):
        source = Mock()
        source.__enter__ = Mock(return_value=iter([json.dumps(record()), '{invalid']))
        source.__exit__ = Mock(return_value=False)
        with patch('builtins.open', return_value=source), patch('paper_pipeline.seeds.logger.warning') as warning:
            seeds = iter_seeds(self.source)
            self.assertEqual(next(seeds)['title'], 'Seed title')
            warning.assert_not_called()
            seeds.close()

    def test_config_and_cli(self):
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config.input_jsonl, 'data/acl2019/sections.jsonl')
        self.assertEqual(config.generation.decoding.top_k, 50)
        self.assertEqual(config.generation.decoding.max_completion_tokens, 18000)
        with patch('sys.argv', ['pipeline', '--num-papers', '2', '--input-jsonl', 'custom.jsonl']):
            args = parse_args()
        self.assertEqual(args.num_papers, 2)
        self.assertEqual(args.input_jsonl, Path('custom.jsonl'))
        for options in (['--num-papers', '0'], ['--num-topics', '1'], ['--topics', 'old.json']):
            with patch('sys.argv', ['pipeline', *options]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_excluded_ids_are_skipped(self):
        self.write_records([record('a'), record('b'), record('c')])
        tsv = self.root / 'excluded.tsv'
        tsv.write_text('corpus\tdoc_id\treason\nacl2019\tb\tnon_english\narxiv2026\tc\tno_text\n')
        exclude = load_excluded(tsv)
        self.assertEqual(exclude, frozenset({'b'}))
        self.assertEqual([s['doc_id'] for s in iter_seeds(self.source, exclude=exclude)], ['a', 'c'])
        self.assertEqual([s['doc_id'] for s in iter_seeds(self.source, 1, exclude=exclude)], ['a'])

    def test_only_ids_and_excluded_request(self):
        self.write_records([record('a'), record('b'), record('c')])
        ids = self.root / 'ids.txt'
        ids.write_text('# sample\nc\na\n')
        only = load_ids(ids)
        self.assertEqual([s['doc_id'] for s in iter_seeds(self.source, only=only)], ['a', 'c'])
        with self.assertRaises(ValueError):
            list(iter_seeds(self.source, only=['a', 'b'], exclude=frozenset({'b'})))

    def test_config_has_exclusion_and_cli_options(self):
        self.assertEqual(load_config(DEFAULT_CONFIG).excluded_documents, 'data/excluded_documents.tsv')
        with patch('sys.argv', ['pipeline', '--ids-file', 'ids.txt', '--excluded', 'x.tsv']):
            args = parse_args()
        self.assertEqual((args.ids_file, args.excluded), (Path('ids.txt'), Path('x.tsv')))
        with patch('sys.argv', ['pipeline', '--stage', 'review', '--papers-dir', 'd', '--ids-file', 'i']), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()

    def two_sources(self):
        acl, arx = self.root / 'acl.jsonl', self.root / 'arx.jsonl.gz'
        acl.write_text('\n'.join(json.dumps(record(f'A{i}')) for i in range(10)) + '\n')
        with gzip.open(arx, 'wt') as out:
            out.write('\n'.join(json.dumps(record(f'X{i}')) for i in range(10)) + '\n')
        tsv = self.root / 'excluded.tsv'
        tsv.write_text('corpus\tdoc_id\nacl2019\tA0\narxiv2026\tX0\narxiv2026\tA1\n')
        return {'acl2019': acl, 'arxiv2026': arx}, tsv

    def test_select_random_split_reproducible_and_excluded(self):
        sources, tsv = self.two_sources()
        picked = select_seeds(sources, tsv, num=7, seed=3)
        corpora = [c for c, _ in picked]
        self.assertEqual((corpora.count('acl2019'), corpora.count('arxiv2026')), (4, 3))
        ids = {s['doc_id'] for _, s in picked}
        self.assertFalse(ids & {'A0', 'X0'})
        self.assertEqual(picked, select_seeds(sources, tsv, num=7, seed=3))
        self.assertNotEqual([s['doc_id'] for _, s in picked],
                            [s['doc_id'] for _, s in select_seeds(sources, tsv, num=7, seed=4)])
        self.assertEqual(len(select_seeds(sources, tsv)), 18)
        used = [(c, s['doc_id']) for c, s in picked]
        rest = select_seeds(sources, tsv, used=used)
        self.assertEqual(len(rest), 18 - 7)
        self.assertFalse({(c, s['doc_id']) for c, s in rest} & set(used))
        with self.assertRaises(ValueError):
            select_seeds(sources, tsv, num=20)

    def test_draw_batch_registry(self):
        from paper_pipeline.seeds import draw_batch, read_registry
        sources, tsv = self.two_sources()
        reg = self.root / 'used.tsv'
        first, new = draw_batch(sources, tsv, reg, 'b1', num=6, seed=1)
        self.assertTrue(new)
        self.assertEqual(len(read_registry(reg)), 6)
        again, new = draw_batch(sources, tsv, reg, 'b1', num=6, seed=99)
        self.assertFalse(new)
        self.assertEqual({s['doc_id'] for _, s in again}, {s['doc_id'] for _, s in first})
        with self.assertRaises(ValueError):
            draw_batch(sources, tsv, reg, 'b1', num=4)
        second, new = draw_batch(sources, tsv, reg, 'b2', num=8, seed=1)
        self.assertTrue(new)
        self.assertFalse({s['doc_id'] for _, s in second} & {s['doc_id'] for _, s in first})
        self.assertEqual(len(read_registry(reg)), 14)
        with self.assertRaises(ValueError):
            draw_batch(sources, tsv, reg, 'bad name', num=1)

    def test_select_only_listed_pairs(self):
        sources, tsv = self.two_sources()
        ids = self.root / 'ids.tsv'
        ids.write_text('# drawn earlier\narxiv2026\tX3\nA2\n')
        pairs = load_source_ids(ids)
        self.assertEqual(pairs, [('arxiv2026', 'X3'), ('acl2019', 'A2')])
        self.assertEqual([s['doc_id'] for _, s in select_seeds(sources, tsv, only=pairs)], ['X3', 'A2'])
        with self.assertRaises(ValueError):
            select_seeds(sources, tsv, only=[('acl2019', 'A0')])


if __name__ == '__main__':
    unittest.main()
