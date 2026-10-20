#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2014--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from os import makedirs, remove
from os.path import exists, islink, join, realpath
from collections import Counter
from glob import glob
from shutil import copy, rmtree
from contextlib import redirect_stdout
from io import StringIO

from unittest import TestCase, main
import pandas as pd
from numpy import isnan, nan
from tempfile import mkdtemp
from tax_credit.framework_functions import (
    generate_novel_sequence_sets,
    generate_simulated_datasets,
    clean_database,
    evaluate_classification,
    find_last_common_ancestor,
    novel_taxa_classification_evaluation,
    extract_per_level_accuracy,
    seq_count,
    simulated_reads_filepath,
)
from tax_credit.paths import (
    QUERY_FASTA,
    QUERY_TAXA_TSV,
    QUERY_TAX_ASSIGNMENTS_TXT,
    REF_SEQS_FASTA,
    REF_TAXA_TSV,
)
from tax_credit.simulation_names import (
    cross_validated_root,
    cross_validated_trad_root,
    novel_taxa_simulations_root,
    ref_dbs_root,
)
from tax_credit.taxa_manipulator import (import_to_list,
                                         import_taxonomy_to_dict,
                                         extract_fasta_ids,
                                         extract_taxa_names)


def assert_level_lists_equal(observed, expected):
    """Compare per-level metric lists, treating NaN as equal to NaN."""
    assert len(observed) == len(expected), (observed, expected)
    for level, (got, want) in enumerate(zip(observed, expected)):
        if isinstance(want, float) and isnan(want):
            assert isnan(got), f'level {level}: expected NaN, got {got}'
        else:
            assert got == want, f'level {level}: expected {want}, got {got}'


class EvalFrameworkTests(TestCase):

    def test_generate_simulated_datasets(self):
        generate_simulated_datasets(
            self.ref_data, self.tmpdir, 2, read_length=100,
            levelrange=range(6, 5, -1))
        # cross-validated q0 should be ref1 and vice versa
        q0 = import_to_list(
            join(self.cvdir, 'B1-REF-iter0', QUERY_TAXA_TSV), field=1)
        q1 = import_to_list(
            join(self.cvdir, 'B1-REF-iter1', QUERY_TAXA_TSV), field=1)
        ref0 = import_to_list(
            join(self.cvdir, 'B1-REF-iter0', REF_TAXA_TSV), field=1)
        ref1 = import_to_list(
            join(self.cvdir, 'B1-REF-iter1', REF_TAXA_TSV), field=1)
        for q_taxon in q0:
            for r_taxon in ref1:
                if r_taxon.startswith(q_taxon):
                    break
            else:
                self.fail(q_taxon + ' not in reference taxa')
        for q_taxon in q1:
            for r_taxon in ref0:
                if r_taxon.startswith(q_taxon):
                    break
            else:
                self.fail(q_taxon + ' not in reference taxa')
        # ref0 + ref1 should equal ref
        ref = import_to_list(join(self.tmpdir, self.query_fp), field=1)
        self.assertEqual(Counter(ref), Counter(ref0 + ref1))
        # confirm that query seq IDs are not in ref for cross-val
        for i in [0, 1]:
            # test that cross-validated queries have pair in ref,
            # but keys do not match
            query_ids = import_to_list(
                join(self.cvdir, 'B1-REF-iter{0}'.format(i), QUERY_TAXA_TSV),
                field=0)
            ref_ids = import_to_list(
                join(self.cvdir, 'B1-REF-iter{0}/ref_taxa.tsv'.format(i)),
                field=0)
            # seq ID keys do not match
            self.assertEqual(set(query_ids).intersection(set(ref_ids)), set())
            # test that novel-taxa queries have no match in ref
            query_taxa = import_taxonomy_to_dict(
                join(self.ntdir, 'B1-REF-L6-iter{0}'.format(i), QUERY_TAXA_TSV))
            ref_taxa = import_taxonomy_to_dict(
                join(self.ntdir, 'B1-REF-L6-iter{0}/ref_taxa.tsv'.format(i)))
            for key, value in query_taxa.items():
                self.assertNotIn(key, ref_taxa)
                self.assertNotIn(value, ref_taxa.values())
        # default behavior should also generate traditional CV folds
        self.assertTrue(exists(join(
            cross_validated_trad_root(self.tmpdir), 'B1-REF-iter0')))

    def test_generate_simulated_datasets_trim_primers_optional(self):
        tmp = mkdtemp()
        try:
            copy(join(self.tmpdir, 'ref1.txt'), join(tmp, 'ref1.txt'))
            name = 'B1-REF-L6-iter0'
            qdir = join(tmp, name)
            makedirs(qdir)
            copy(self.query_fp, join(qdir, QUERY_TAXA_TSV))
            databases = {
                'B1-REF': [
                    join(tmp, 'ref1.txt'),
                    join(qdir, QUERY_TAXA_TSV),
                    'ref1',
                    'GTGCCAGCMGCCGCGGTAA',
                    'GGACTACHVGGGTWTCTAAT',
                    '515f',
                    '806r',
                ],
            }
            ref_data = pd.DataFrame.from_dict(databases, orient='index')
            ref_data.columns = [
                'Reference file path',
                'Reference tax path',
                'Reference id',
                'Fwd primer',
                'Rev primer',
                'Fwd primer id',
                'Rev primer id',
            ]
            generate_simulated_datasets(
                ref_data, tmp, 2, read_length=100,
                levelrange=range(6, 5, -1), trim_primers=False)
            db_dir = join(ref_dbs_root(tmp), 'ref1')
            paths = glob(join(db_dir, '*_noprimer_trunc.fasta'))
            self.assertEqual(len(paths), 1)
            self.assertGreater(seq_count(paths[0]), 0)
        finally:
            rmtree(tmp)

    def test_generate_simulated_datasets_min_read_length_optional(self):
        tmp = mkdtemp()
        try:
            copy(join(self.tmpdir, 'ref1.txt'), join(tmp, 'ref1.txt'))
            name = 'B1-REF-L6-iter0'
            qdir = join(tmp, name)
            makedirs(qdir)
            copy(self.query_fp, join(qdir, QUERY_TAXA_TSV))
            databases = {
                'B1-REF': [
                    join(tmp, 'ref1.txt'),
                    join(qdir, QUERY_TAXA_TSV),
                    'ref1',
                    'GTGCCAGCMGCCGCGGTAA',
                    'GGACTACHVGGGTWTCTAAT',
                    '515f',
                    '806r',
                ],
            }
            ref_data = pd.DataFrame.from_dict(databases, orient='index')
            ref_data.columns = [
                'Reference file path',
                'Reference tax path',
                'Reference id',
                'Fwd primer',
                'Rev primer',
                'Fwd primer id',
                'Rev primer id',
            ]
            generate_simulated_datasets(
                ref_data, tmp, 2, read_length=100,
                levelrange=range(6, 5, -1),
                trim_primers=False, min_read_length=None)
            db_dir = join(ref_dbs_root(tmp), 'ref1')
            paths = glob(join(db_dir, '*_noprimer_trunc_nominlen.fasta'))
            self.assertEqual(len(paths), 1)
            self.assertGreater(seq_count(paths[0]), 0)
        finally:
            rmtree(tmp)

    def test_generate_simulated_datasets_truncation_optional(self):
        tmp = mkdtemp()
        try:
            copy(join(self.tmpdir, 'ref1.txt'), join(tmp, 'ref1.txt'))
            name = 'B1-REF-L6-iter0'
            qdir = join(tmp, name)
            makedirs(qdir)
            copy(self.query_fp, join(qdir, QUERY_TAXA_TSV))
            databases = {
                'B1-REF': [
                    join(tmp, 'ref1.txt'),
                    join(qdir, QUERY_TAXA_TSV),
                    'ref1',
                    'GTGCCAGCMGCCGCGGTAA',
                    'GGACTACHVGGGTWTCTAAT',
                    '515f',
                    '806r',
                ],
            }
            ref_data = pd.DataFrame.from_dict(databases, orient='index')
            ref_data.columns = [
                'Reference file path',
                'Reference tax path',
                'Reference id',
                'Fwd primer',
                'Rev primer',
                'Fwd primer id',
                'Rev primer id',
            ]
            generate_simulated_datasets(
                ref_data, tmp, 2, levelrange=range(6, 5, -1),
                trim_primers=False, truncate=False)
            db_dir = join(ref_dbs_root(tmp), 'ref1')
            paths = glob(join(db_dir, '*_noprimer_full.fasta'))
            self.assertEqual(len(paths), 1)
            self.assertGreater(seq_count(paths[0]), 0)
        finally:
            rmtree(tmp)

    def test_generate_simulated_datasets_read_length_required_when_truncating(
            self):
        with self.assertRaises(ValueError):
            generate_simulated_datasets(
                self.ref_data, self.tmpdir, 2, read_length=None,
                levelrange=range(6, 5, -1), truncate=True)

    def test_generate_simulated_datasets_invalid_simulation_method(self):
        tmp = mkdtemp()
        try:
            copy(join(self.tmpdir, 'ref1.txt'), join(tmp, 'ref1.txt'))
            name = 'B1-REF-L6-iter0'
            qdir = join(tmp, name)
            makedirs(qdir)
            copy(self.query_fp, join(qdir, QUERY_TAXA_TSV))
            databases = {
                'B1-REF': [
                    join(tmp, 'ref1.txt'),
                    join(qdir, QUERY_TAXA_TSV),
                    'ref1',
                    'GTGCCAGCMGCCGCGGTAA',
                    'GGACTACHVGGGTWTCTAAT',
                    '515f',
                    '806r',
                ],
            }
            ref_data = pd.DataFrame.from_dict(databases, orient='index')
            ref_data.columns = [
                'Reference file path',
                'Reference tax path',
                'Reference id',
                'Fwd primer',
                'Rev primer',
                'Fwd primer id',
                'Rev primer id',
            ]
            with self.assertRaises(ValueError):
                generate_simulated_datasets(
                    ref_data, tmp, 2, read_length=100,
                    levelrange=range(6, 5, -1), trim_primers=False,
                    simulation_method='not-a-method')
            with self.assertRaises(ValueError):
                generate_simulated_datasets(
                    ref_data, tmp, 2, read_length=100,
                    levelrange=range(6, 5, -1), trim_primers=False,
                    simulation_method=['cross-validated-taxa',
                                       'not-a-method'])
        finally:
            rmtree(tmp)

    def test_generate_simulated_datasets_cross_validated_trad(self):
        tmp = mkdtemp()
        try:
            copy(join(self.tmpdir, 'ref1.txt'), join(tmp, 'ref1.txt'))
            name = 'B1-REF-L6-iter0'
            qdir = join(tmp, name)
            makedirs(qdir)
            copy(self.query_fp, join(qdir, QUERY_TAXA_TSV))
            databases = {
                'B1-REF': [
                    join(tmp, 'ref1.txt'),
                    join(qdir, QUERY_TAXA_TSV),
                    'ref1',
                    'GTGCCAGCMGCCGCGGTAA',
                    'GGACTACHVGGGTWTCTAAT',
                    '515f',
                    '806r',
                ],
            }
            ref_data = pd.DataFrame.from_dict(databases, orient='index')
            ref_data.columns = [
                'Reference file path',
                'Reference tax path',
                'Reference id',
                'Fwd primer',
                'Rev primer',
                'Fwd primer id',
                'Rev primer id',
            ]
            generate_simulated_datasets(
                ref_data, tmp, 2, read_length=100,
                levelrange=range(6, 5, -1), trim_primers=False,
                simulation_method='cross-validated-trad')
            trad = cross_validated_trad_root(tmp)
            clean_taxa_path = join(
                ref_dbs_root(tmp), 'ref1', 'query_taxa_clean.tsv')
            expected = import_taxonomy_to_dict(clean_taxa_path)
            sim_reads = simulated_reads_filepath(
                join(ref_dbs_root(tmp), 'ref1', 'ref1_clean.fasta'),
                '515f', '806r', trim_primers=False)
            for i in range(2):
                fold = join(trad, 'B1-REF-iter{}'.format(i))
                qt = import_taxonomy_to_dict(join(fold, QUERY_TAXA_TSV))
                for sid, tax in qt.items():
                    self.assertEqual(tax, expected[sid])
                self.assertTrue(islink(join(fold, REF_SEQS_FASTA)))
                self.assertTrue(islink(join(fold, REF_TAXA_TSV)))
                self.assertEqual(
                    realpath(join(fold, REF_SEQS_FASTA)), realpath(sim_reads))
                self.assertEqual(
                    realpath(join(fold, REF_TAXA_TSV)),
                    realpath(clean_taxa_path))
                qids = set(import_to_list(join(fold, QUERY_TAXA_TSV), field=0))
                rids = set(import_to_list(join(fold, REF_TAXA_TSV), field=0))
                self.assertTrue(qids.issubset(rids))
            self.assertFalse(exists(novel_taxa_simulations_root(tmp)))
            self.assertFalse(exists(cross_validated_root(tmp)))
        finally:
            rmtree(tmp)

    def _singleton_first_rank_ref_data(self, tmp):
        '''Frame whose taxonomy has one sequence as the sole holder of its
        first rank, so whichever fold tests it has none of that rank in
        training.'''
        copy(join(self.tmpdir, 'ref1.txt'), join(tmp, 'ref1.txt'))
        lines = [
            line for line in
            open(self.query_fp, encoding='utf-8').read().split('\n') if line
        ]
        sid, taxon = lines[0].split('\t')
        lines[0] = '\t'.join(
            [sid, taxon.replace('k__Bacteria', 'k__Archaea', 1)])
        taxa_fp = join(tmp, 'taxa_singleton.tsv')
        with open(taxa_fp, 'w', encoding='utf-8') as out:
            out.write('\n'.join(lines))
        ref_data = pd.DataFrame.from_dict(
            {'B1-REF': [
                join(tmp, 'ref1.txt'), taxa_fp, 'ref1',
                'GTGCCAGCMGCCGCGGTAA', 'GGACTACHVGGGTWTCTAAT', '515f', '806r',
            ]},
            orient='index')
        ref_data.columns = [
            'Reference file path', 'Reference tax path', 'Reference id',
            'Fwd primer', 'Rev primer', 'Fwd primer id', 'Rev primer id',
        ]
        return ref_data, sid

    def test_cv_drops_query_whose_first_rank_is_absent_from_training(self):
        '''A singleton first rank is dropped from the fold, not fatal.

        Previously this raised RuntimeError('unknown kingdom in query set') and
        killed the whole run over one sequence.
        '''
        tmp = mkdtemp()
        try:
            ref_data, orphan_sid = self._singleton_first_rank_ref_data(tmp)
            buf = StringIO()
            with redirect_stdout(buf):
                generate_simulated_datasets(
                    ref_data, tmp, 2, read_length=100,
                    levelrange=range(6, 5, -1), trim_primers=False,
                    simulation_method='cross-validated-taxa')
            output = buf.getvalue()
            cvdir = cross_validated_root(tmp)

            folds = [join(cvdir, 'B1-REF-iter{0}'.format(i)) for i in range(2)]
            for fold in folds:
                self.assertTrue(exists(join(fold, QUERY_TAXA_TSV)))

            # The orphan is a query in exactly one fold, and that fold must have
            # dropped it from BOTH the expected taxonomy and the query FASTA --
            # a sequence with no expected taxonomy would be classified and then
            # have nothing to score against.
            dropped_somewhere = False
            for fold in folds:
                q_ids = set(import_to_list(
                    join(fold, QUERY_TAXA_TSV), field=0))
                ref_ids = set(import_to_list(join(fold, REF_TAXA_TSV), field=0))
                fasta_ids = {
                    line[1:].split()[0]
                    for line in open(join(fold, QUERY_FASTA),
                                     encoding='utf-8')
                    if line.startswith('>')
                }
                # query_taxa.tsv and query.fasta must always agree
                self.assertEqual(q_ids, fasta_ids)
                if orphan_sid not in ref_ids and orphan_sid not in q_ids:
                    dropped_somewhere = True
            self.assertTrue(
                dropped_somewhere,
                'the singleton-first-rank query was never dropped, so this '
                'fixture no longer reproduces the condition')
            self.assertIn('dropped', output)
            self.assertIn('k__Archaea', output)
        finally:
            rmtree(tmp)

    def test_trad_cv_never_drops_queries(self):
        '''cross-validated-trad keeps every query: its reference is the full DB.

        The taxonomy-aware builder has to truncate and sometimes drop queries;
        trad cannot, because every query's lineage is in the reference by
        construction. Same fixture that forces a drop above.
        '''
        tmp = mkdtemp()
        try:
            ref_data, orphan_sid = self._singleton_first_rank_ref_data(tmp)
            buf = StringIO()
            with redirect_stdout(buf):
                generate_simulated_datasets(
                    ref_data, tmp, 2, read_length=100,
                    levelrange=range(6, 5, -1), trim_primers=False,
                    simulation_method='cross-validated-trad')
            self.assertNotIn('dropped', buf.getvalue())

            trad = cross_validated_trad_root(tmp)
            all_queries = set()
            for i in range(2):
                fold = join(trad, 'B1-REF-iter{0}'.format(i))
                q = import_taxonomy_to_dict(join(fold, QUERY_TAXA_TSV))
                ref_ids = set(import_to_list(join(fold, REF_TAXA_TSV), field=0))
                # every query is in its own reference, untrimmed
                self.assertTrue(set(q).issubset(ref_ids))
                all_queries.update(q)
            # including the sequence the taxonomy-aware builder had to drop
            self.assertIn(orphan_sid, all_queries)
        finally:
            rmtree(tmp)

    def test_novel_taxa_ignores_stale_fold_from_another_database(self):
        '''A half-written fold left by a failed run must not break this one.

        generate_novel_sequence_sets runs once per database but used to glob
        every fold in cross-validated/, so a directory holding ref_taxa.tsv but
        not yet query_taxa.tsv -- exactly what an interrupted run leaves -- made
        it die with "unknown file type in .../query_taxa.tsv".
        '''
        tmp = mkdtemp()
        try:
            # Leftover from a previous run on a DIFFERENT database: the fold dir
            # and ref_taxa.tsv exist, query_taxa.tsv does not.
            stale = join(cross_validated_root(tmp), 'otherdb-iter1')
            makedirs(stale)
            with open(join(stale, REF_TAXA_TSV), 'w') as out:
                out.write('x\tk__Bacteria; p__Firmicutes\n')

            buf = StringIO()
            with redirect_stdout(buf):
                generate_simulated_datasets(
                    self.ref_data, tmp, 2, read_length=100,
                    levelrange=range(6, 5, -1), trim_primers=False,
                    simulation_method=['cross-validated-taxa', 'novel-taxa'])

            # This database's novel folds were still built...
            ntdir = novel_taxa_simulations_root(tmp)
            for i in range(2):
                self.assertTrue(exists(
                    join(ntdir, 'B1-REF-L6-iter{0}'.format(i))))
            # ...and nothing was built for the stale database.
            self.assertEqual(glob(join(ntdir, 'otherdb*')), [])
            # Scoping means the stale fold is never even looked at, so no
            # warning is needed; the run simply succeeds.
            self.assertTrue(exists(join(stale, REF_TAXA_TSV)))
        finally:
            rmtree(tmp)

    def test_novel_taxa_skips_incomplete_fold_of_same_database(self):
        '''An incomplete fold of the database being processed is skipped loudly.'''
        tmp = mkdtemp()
        try:
            generate_simulated_datasets(
                self.ref_data, tmp, 2, read_length=100,
                levelrange=range(6, 5, -1), trim_primers=False,
                simulation_method='cross-validated-taxa')
            # Simulate an interrupted write: drop one fold's query taxonomy.
            fold = join(cross_validated_root(tmp), 'B1-REF-iter1')
            remove(join(fold, QUERY_TAXA_TSV))

            buf = StringIO()
            with redirect_stdout(buf):
                generate_novel_sequence_sets(
                    cross_validated_root(tmp),
                    novel_taxa_simulations_root(tmp),
                    levelrange=range(6, 5, -1), index='B1-REF')
            output = buf.getvalue()
            self.assertIn('skipping', output)
            self.assertIn('B1-REF-iter1', output)
            # the intact fold was still processed
            self.assertTrue(exists(join(
                novel_taxa_simulations_root(tmp), 'B1-REF-L6-iter0')))
        finally:
            rmtree(tmp)

    def _trad_ref_data(self, tmp):
        '''Build the single-database frame used by the trad-CV tests.'''
        copy(join(self.tmpdir, 'ref1.txt'), join(tmp, 'ref1.txt'))
        qdir = join(tmp, 'B1-REF-L6-iter0')
        makedirs(qdir)
        copy(self.query_fp, join(qdir, QUERY_TAXA_TSV))
        ref_data = pd.DataFrame.from_dict(
            {'B1-REF': [
                join(tmp, 'ref1.txt'),
                join(qdir, QUERY_TAXA_TSV),
                'ref1',
                'GTGCCAGCMGCCGCGGTAA',
                'GGACTACHVGGGTWTCTAAT',
                '515f',
                '806r',
            ]},
            orient='index')
        ref_data.columns = [
            'Reference file path',
            'Reference tax path',
            'Reference id',
            'Fwd primer',
            'Rev primer',
            'Fwd primer id',
            'Rev primer id',
        ]
        return ref_data

    def _run_trad(self, tmp, iterations=2, query_size=None):
        '''Generate trad-CV folds; return (query id sets, n simulated reads).'''
        generate_simulated_datasets(
            self._trad_ref_data(tmp), tmp, iterations, read_length=100,
            levelrange=range(6, 5, -1), trim_primers=False,
            simulation_method='cross-validated-trad',
            trad_cv_query_size=query_size)
        trad = cross_validated_trad_root(tmp)
        fold_queries = [
            set(import_to_list(
                join(trad, 'B1-REF-iter{0}'.format(i), QUERY_TAXA_TSV),
                field=0))
            for i in range(iterations)]
        n_reads = seq_count(simulated_reads_filepath(
            join(ref_dbs_root(tmp), 'ref1', 'ref1_clean.fasta'),
            '515f', '806r', trim_primers=False))
        return fold_queries, n_reads

    def test_trad_query_size_default_pools_whole_database(self):
        """query_size=None divides the whole database between the folds."""
        tmp = mkdtemp()
        try:
            folds, n_reads = self._run_trad(tmp, iterations=2)
            # Folds partition the database: disjoint and covering it exactly.
            self.assertEqual(folds[0] & folds[1], set())
            self.assertEqual(len(folds[0]) + len(folds[1]), n_reads)
            for fold in folds:
                self.assertEqual(len(fold), n_reads // 2)
        finally:
            rmtree(tmp)

    def test_trad_query_size_float_is_fraction_of_total_pool(self):
        """A float is the TOTAL fraction queried, split between the folds."""
        tmp = mkdtemp()
        try:
            folds, n_reads = self._run_trad(tmp, iterations=2, query_size=0.5)
            pool = folds[0] | folds[1]
            # Half the database in total...
            self.assertEqual(len(pool), round(0.5 * n_reads))
            # ...divided between the two folds, so a quarter each.
            self.assertEqual(folds[0] & folds[1], set())
            for fold in folds:
                self.assertEqual(len(fold), round(0.5 * n_reads) // 2)
        finally:
            rmtree(tmp)

    def test_trad_query_size_int_is_absolute_total_count(self):
        """An int is the TOTAL number of sequences, not a per-fold count."""
        tmp = mkdtemp()
        try:
            total = 4
            folds, n_reads = self._run_trad(
                tmp, iterations=2, query_size=total)
            pool = folds[0] | folds[1]
            self.assertEqual(len(pool), total)
            self.assertEqual(folds[0] & folds[1], set())
            for fold in folds:
                self.assertEqual(len(fold), total // 2)
            # Smaller than the whole database, which the default would use.
            self.assertLess(len(pool), n_reads)
        finally:
            rmtree(tmp)

    def test_trad_query_size_one_point_zero_matches_default(self):
        """query_size=1.0 means the whole database, i.e. the default."""
        tmp_default = mkdtemp()
        tmp_explicit = mkdtemp()
        try:
            default, _ = self._run_trad(tmp_default, iterations=2)
            explicit, _ = self._run_trad(
                tmp_explicit, iterations=2, query_size=1.0)
            self.assertEqual(default, explicit)
        finally:
            rmtree(tmp_default)
            rmtree(tmp_explicit)

    def test_trad_query_size_keeps_full_reference(self):
        """The reference stays the whole database regardless of query_size."""
        tmp = mkdtemp()
        try:
            self._run_trad(tmp, iterations=2, query_size=4)
            trad = cross_validated_trad_root(tmp)
            clean_taxa_path = join(
                ref_dbs_root(tmp), 'ref1', 'query_taxa_clean.tsv')
            sim_reads = simulated_reads_filepath(
                join(ref_dbs_root(tmp), 'ref1', 'ref1_clean.fasta'),
                '515f', '806r', trim_primers=False)
            for i in range(2):
                fold = join(trad, 'B1-REF-iter{0}'.format(i))
                self.assertEqual(
                    realpath(join(fold, REF_SEQS_FASTA)), realpath(sim_reads))
                self.assertEqual(
                    realpath(join(fold, REF_TAXA_TSV)),
                    realpath(clean_taxa_path))
                # queries are present in the reference they are classified against
                qids = set(import_to_list(join(fold, QUERY_TAXA_TSV), field=0))
                rids = set(import_to_list(join(fold, REF_TAXA_TSV), field=0))
                self.assertTrue(qids.issubset(rids))
        finally:
            rmtree(tmp)

    def test_trad_query_size_larger_than_database_uses_whole_database(self):
        """An int bigger than the database warns and falls back to all of it."""
        tmp = mkdtemp()
        try:
            buf = StringIO()
            with redirect_stdout(buf):
                folds, n_reads = self._run_trad(
                    tmp, iterations=2, query_size=9999)
            output = buf.getvalue()
            # warned, naming the oversized request and the real database size
            self.assertIn('WARNING', output)
            self.assertIn('9999', output)
            self.assertIn(str(n_reads), output)
            # and fell back to the whole database, as query_size=None would
            pool = folds[0] | folds[1]
            self.assertEqual(len(pool), n_reads)
            self.assertEqual(folds[0] & folds[1], set())
            default, _ = self._run_trad(mkdtemp(), iterations=2)
            self.assertEqual(folds, default)
        finally:
            rmtree(tmp)

    def test_trad_query_size_rejects_invalid_values(self):
        """Bad query_size values raise, naming both the float and int reading."""
        cases = [
            (1.5, 'at most 1'),            # float may not exceed the database
            (0.0, 'greater than 0'),
            (-0.5, 'greater than 0'),
            (0, 'at least 1'),             # int must be >= 1
            ('0.1', 'must be a float, an int, or None'),
            (True, 'must be a float or an int'),
            # a pool too small to give every fold at least one sequence
            (1, 'cannot be divided between'),
            (0.05, 'cannot be divided between'),
        ]
        for bad, expected in cases:
            tmp = mkdtemp()
            try:
                with self.assertRaises(ValueError) as ctx:
                    self._run_trad(tmp, iterations=2, query_size=bad)
                message = str(ctx.exception)
                self.assertIn(expected, message)
                # every message spells out both readings
                self.assertIn('FRACTION', message)
                self.assertIn('ABSOLUTE NUMBER', message)
                self.assertIn('TOTAL pool', message)
            finally:
                rmtree(tmp)

    def test_generate_simulated_datasets_novel_taxa_only(self):
        tmp = mkdtemp()
        try:
            copy(join(self.tmpdir, 'ref1.txt'), join(tmp, 'ref1.txt'))
            name = 'B1-REF-L6-iter0'
            qdir = join(tmp, name)
            makedirs(qdir)
            copy(self.query_fp, join(qdir, QUERY_TAXA_TSV))
            databases = {
                'B1-REF': [
                    join(tmp, 'ref1.txt'),
                    join(qdir, QUERY_TAXA_TSV),
                    'ref1',
                    'GTGCCAGCMGCCGCGGTAA',
                    'GGACTACHVGGGTWTCTAAT',
                    '515f',
                    '806r',
                ],
            }
            ref_data = pd.DataFrame.from_dict(databases, orient='index')
            ref_data.columns = [
                'Reference file path',
                'Reference tax path',
                'Reference id',
                'Fwd primer',
                'Rev primer',
                'Fwd primer id',
                'Rev primer id',
            ]
            generate_simulated_datasets(
                ref_data, tmp, 2, read_length=100,
                levelrange=range(6, 5, -1), trim_primers=False,
                simulation_method='novel-taxa')
            self.assertTrue(exists(join(
                novel_taxa_simulations_root(tmp), 'B1-REF-L6-iter0')))
            self.assertFalse(exists(cross_validated_root(tmp)))
            self.assertFalse(exists(cross_validated_trad_root(tmp)))
        finally:
            rmtree(tmp)

    def test_simulated_reads_filepath_matches_generate(self):
        tmp = mkdtemp()
        try:
            copy(join(self.tmpdir, 'ref1.txt'), join(tmp, 'ref1.txt'))
            name = 'B1-REF-L6-iter0'
            qdir = join(tmp, name)
            makedirs(qdir)
            copy(self.query_fp, join(qdir, QUERY_TAXA_TSV))
            databases = {
                'B1-REF': [
                    join(tmp, 'ref1.txt'),
                    join(qdir, QUERY_TAXA_TSV),
                    'ref1',
                    'GTGCCAGCMGCCGCGGTAA',
                    'GGACTACHVGGGTWTCTAAT',
                    '515f',
                    '806r',
                ],
            }
            ref_data = pd.DataFrame.from_dict(databases, orient='index')
            ref_data.columns = [
                'Reference file path',
                'Reference tax path',
                'Reference id',
                'Fwd primer',
                'Rev primer',
                'Fwd primer id',
                'Rev primer id',
            ]
            generate_simulated_datasets(
                ref_data, tmp, 2, read_length=100,
                levelrange=range(6, 5, -1))
            db_dir = join(ref_dbs_root(tmp), 'ref1')
            clean_fasta = join(db_dir, 'ref1_clean.fasta')
            expected = simulated_reads_filepath(
                clean_fasta, '515f', '806r', min_read_length=80,
                trim_primers=True, truncate=True)
            paths = glob(join(db_dir, '*_515f-806r_trunc.fasta'))
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0], expected)
        finally:
            rmtree(tmp)

    def test_find_last_common_ancestor(self):
        taxa1 = import_to_list(self.query_fp)
        taxa2 = import_to_list(self.obs_taxa_fp)
        t1 = extract_taxa_names(taxa1, level=slice(None), field=1)
        t2 = extract_taxa_names(taxa2, level=slice(None), field=1)
        for i, n in zip(range(8), [7, 6, 5, 4, 7, 3, 6, 6]):
            self.assertEqual(find_last_common_ancestor(t2[i], t1[i]), n)

    def test_evaluate_classification_na_ranks(self):
        exp = ('Eukaryota;Chordata;Actinopteri;NA;Lutjanidae;Lutjanus;'
               'Lutjanus griseus')
        self.assertEqual(evaluate_classification(exp, exp), 'match')
        self.assertEqual(
            evaluate_classification('Eukaryota;Chordata;Actinopteri', exp),
            'underclassification')
        # the internal NA rank must line up, not be skipped
        self.assertEqual(
            evaluate_classification(
                'Eukaryota;Chordata;Actinopteri;Lutjanidae;Lutjanus;'
                'Lutjanus griseus', exp),
            'misclassification')
        # an internal gap is not a truncation
        self.assertEqual(
            evaluate_classification(
                'Eukaryota;Chordata;Actinopteri;;Lutjanidae', exp),
            'misclassification')
        self.assertEqual(evaluate_classification('Unassigned', exp),
                         'underclassification')

    def test_evaluate_classification_trailing_na_and_empty_ranks(self):
        self.assertEqual(
            evaluate_classification('A;B;C;D;E;F;NA', 'A;B;C;D;E;F'), 'match')
        self.assertEqual(
            evaluate_classification('A;B;C;D;E;;', 'A;B;C;D;E;F;G'),
            'underclassification')
        self.assertEqual(
            evaluate_classification('A;B;C;D;E;F;G', 'A;B;C;D;E;F;NA'),
            'overclassification')
        # ranks are compared whole, not as string prefixes
        self.assertEqual(
            evaluate_classification('A;B;C;D;E;Lut', 'A;B;C;D;E;Lutjanus'),
            'misclassification')

    def test_clean_database_drops_all_na_lineages(self):
        db_dir = mkdtemp()
        self.addCleanup(rmtree, db_dir)
        taxa_fp = join(db_dir, 'ref_taxa.tsv')
        seqs_fp = join(db_dir, 'ref_seqs.fasta')
        with open(taxa_fp, 'w') as f:
            f.write('\n'.join([
                'Feature ID\tTaxon',
                'gadus\tEukaryota;Chordata;Actinopteri;Gadiformes;Gadidae;'
                'Gadus;NA',
                'allna\tNA;NA;NA;NA;NA;NA;NA',
                'shallow\tEukaryota;NA;NA;NA;NA;NA;NA',
                'blank\t',
            ]))
        with open(seqs_fp, 'w') as f:
            f.write('>gadus\nACGT\n>allna\nACGA\n>shallow\nACGC\n'
                    '>blank\nACGG\n')

        clean_taxa, clean_fasta = clean_database(taxa_fp, seqs_fp, db_dir)

        kept = import_taxonomy_to_dict(clean_taxa)
        # lineages that place a sequence somewhere survive, even if shallow or
        # unresolved at the tip
        self.assertEqual(sorted(kept), ['gadus', 'shallow'])
        # and the sequences with no usable taxonomy go with them
        self.assertEqual(sorted(extract_fasta_ids(clean_fasta)),
                         ['gadus', 'shallow'])

    def test_novel_taxa_classification_evaluation(self):
        # test novel taxa evaluation
        results = novel_taxa_classification_evaluation(
            [self.paramdir], self.tmpdir, join(self.tmpdir, 'summary.txt'),
            test_type='novel-taxa')
        self.assertEqual(results.iloc[0]['Dataset'], 'B1-REF')
        self.assertEqual(int(results.iloc[0]['level']), 6)
        self.assertEqual(int(results.iloc[0]['iteration']), 0)
        self.assertEqual(results.iloc[0]['Method'], 'method1')
        self.assertEqual(results.iloc[0]['Parameters'], 'param1')
        self.assertAlmostEqual(results.iloc[0]['match_ratio'], 0.25)
        self.assertAlmostEqual(results.iloc[0]['overclassification_ratio'], 0.)
        self.assertAlmostEqual(
            results.iloc[0]['underclassification_ratio'], 0.625)
        self.assertAlmostEqual(
            results.iloc[0]['misclassification_ratio'], 0.125)
        self.assertAlmostEqual(results.iloc[0]['Precision'], 0.666666667)
        self.assertAlmostEqual(results.iloc[0]['Recall'], 0.25)
        self.assertAlmostEqual(
            results.iloc[0]['F-measure'], 0.363636364)
        self.assertEqual(results.iloc[0]['mismatch_level_list'], self.exp_m)
        # test cross-validated evaluation
        results = novel_taxa_classification_evaluation(
            [self.paramdir], self.tmpdir, join(self.tmpdir, 'summary.txt'),
            test_type='cross-validated')
        self.assertEqual(results.iloc[0]['Dataset'], 'B1-REF-L6')
        self.assertEqual(int(results.iloc[0]['iteration']), 0)
        self.assertEqual(results.iloc[0]['Method'], 'method1')
        self.assertEqual(results.iloc[0]['Parameters'], 'param1')
        self.assertAlmostEqual(results.iloc[0]['match_ratio'], 0.25)
        self.assertEqual(results.iloc[0]['overclassification_ratio'], 0)
        self.assertAlmostEqual(
            results.iloc[0]['underclassification_ratio'], 0.625)
        self.assertAlmostEqual(
            results.iloc[0]['misclassification_ratio'], 0.125)
        assert_level_lists_equal(
            results.iloc[0]['Precision'], self.exp_p)
        assert_level_lists_equal(results.iloc[0]['Recall'], self.exp_r)
        assert_level_lists_equal(results.iloc[0]['F-measure'], self.exp_f)
        assert_level_lists_equal(
            results.iloc[0]['Recall_original'], self.exp_r_org)
        assert_level_lists_equal(
            results.iloc[0]['F-measure_original'], self.exp_f_org)
        self.assertEqual(results.iloc[0]['mismatch_level_list'], self.exp_m)

    def test_extract_per_level_accuracy(self):
        results = novel_taxa_classification_evaluation(
            [self.paramdir], self.tmpdir, join(self.tmpdir, 'summary.txt'),
            test_type='cross-validated')
        pla = extract_per_level_accuracy(results)
        # confirm that method/dataset data propagate properly
        self.assertEqual(pla['Dataset'].unique(), ['B1-REF-L6'])
        self.assertEqual(pla['iteration'].unique(), ['0'])
        self.assertEqual(pla['Method'].unique(), ['method1'])
        self.assertEqual(pla['Parameters'].unique(), ['param1'])
        # confirm that level and per-level results are correct
        for i in range(6):
            self.assertEqual(pla.iloc[i]['level'], i + 1)
            self.assertEqual(pla.iloc[i]['Precision'], self.exp_p[i + 1])
            self.assertEqual(pla.iloc[i]['Recall'], self.exp_r[i + 1])
            self.assertEqual(pla.iloc[i]['F-measure'], self.exp_f[i + 1])
            self.assertEqual(
                pla.iloc[i]['Recall_original'], self.exp_r_org[i + 1])
        # match_ratio is a classification ratio, not a score: it is no longer
        # derived here, because it no longer equals Recall
        self.assertNotIn('match_ratio', pla.columns)

    @classmethod
    def setUpClass(self):
        _ref1 = '\n'.join([
            '>179419',
            'TGAGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCATGCCTAATACATGCAAGTCGAACGAG'
            'CTTCCGTTGAATGACGTGCTTGCACTGATTTCAACAATGAAGCGAGTGGCGAACTGGTGAGTAAC'
            'ACGTGGGGAATCTGCCCAGAAGCAGGGGATAACACTTGGAAACAGGTGCTAATACCGTATAACAA'
            'CAAAATCCGCATGGATCTTGTTTGAAAGGTGGCTTCGGCTATCACTTCTGGATGATCCCGCGGCG'
            'TATTAGTTAGTTGGTGAGGTAAAGGCCCACCAAGACGATGATACGTAGCCGACCTGAGAGGGTAA'
            'TCGGCCACATTGGGACTGAGACACGGCCCAAACTCCTACGGGAGGCAGCAGTAGGGAATCTTCCA'
            'CAATGGACGAAAGTCTGATGGAGCAATGCCGCGTGAGTGAAGAAGGGTTTCGGCTCGTAAAACTC'
            'TGTTGTTAAAGAAGAACACCTTTGAGAGTAACTGTTCAAGGGTTGACGGTATTTAACCAGAAAGC'
            'CACGGCTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCAAGCGTTGTCCGGATTTATTG'
            'GGCGTAAAGCGAGCGCAGGCGGTTTTTTAAGTCTGATGTGAAAGCCTTCGGCTTAACCGGAGAAG'
            'TGCATCGGAAACTGGGAGACTTGAGTGCAGAAGAGGACAGTGGAACTCCATGTGTAGCGGTGGAA'
            'TGCGTAGATATATGGAAGAACACCAGTGGCGAAGGCGGCTGTCTAGTCTGTAACTGACGCTGAGG'
            'CTCGAAAGCATGGGTAGCGAACAGGATTAGATACCCTGGTAGTCCATGCCGTAAACGATGAGTGC'
            'TAAGTGTTGGAGGGTTTCCGCCCTTCAGTGCTGCAGCTAACGCATTAAGCACTCCGCCTGGGGAG'
            'TACGACCGCAAGGTTGAAACTCAAAGGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGT'
            'TTAATTCGAAGCTACGCGAAGAACCTTACCAGGTCTTGACATCTTCTGCCAATCTTAGAGATAAG'
            'ACGTTCCCTTCGGGGACAGAATGACAGGTGGTGCATGGTTGTCGTCAGCTCGTGTCGTGAGATGT'
            'TGGGTTAAGTCCCGCAACGAGCGCAACCCTTATTATCAGTTGCCAGCATTCAGTTGGGCACTCTG'
            'GTGAGACTGCCGGTGACAAACCGGAGGAAGGTGGGGATGACGTCAAATCATCATGCCCCTTATGA'
            'CCTGGGCTACACACGTGCTACAATGGACGGTACAACGAGTCGCGAAGTCGTGAGGCTAAGCTAAT'
            'CTCTTAAAGCCGTTCTCAGTTCGGATTGTAGGCTGCAACTCGCCTACATGAAGTTGGAATCGCTA'
            'GTAATCGCGGATCAGCATGCCGCGGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACAC'
            'CATGAGAGTTTGTAACACCCAAAGCCGGTGAGATAACCTTCGGGAGTCAGCCGTCTAAGGTGGGA'
            'CAGATGATTAGGGTGAAGTCGTAACAAGGTAGCCGTAGGAGAACCTGCGGCTGGATCACCTCCTT'
            'TCT',
            '>1117026',
            'GAGTGGCGAACTGGTGAGTAACACGTGGGAAATCTGCCCAGAAGCAGGGGATAACACTTGGAAAC'
            'AGGTGCTAATACCGTATAACAACAAGAACCGCATGGTTCTTGTTTGAAAGGTGGTTTCGGCTATC'
            'ACTTCTGGATGATCCCGCGGCGTATTAGTTAGTTGGTGAGGTAAAGGCCCACCAAGACAATGATA'
            'CGTAGCCGACCTGAGAGGGTAATCGGCCACATTGGGACTGAGACACGGCCCAAACTCCTACGGGA'
            'GGCAGCAGTAGGGAATCTTCCACAATGGACGAAAGTCTGATGGAGCAATGCCGCGTGAGTGAAGA'
            'AGGGTTTCGGCTCGTAAAACTCTGTTGTTAAAGAAGAACACCTCTGAGAGTAACTGTTCAGGGGT'
            'TGACGGTATTTAACCAGAAAGCCACGGCTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGG'
            'CAAGCGTTGTCCGGATTTATTGGGCGTAAAGCGAGCGCAGGCGGTTTCTTAAGTCTGATGTGAAA'
            'GCCTTCGGCTTAACCGGAGAAGTGCATCGGAAACTGGGTAACTTGAGTGCAGAAGAGGACAGTGG'
            'AACTCCATGTGTAGCGGTGGAATGCGTAGATATATGGAAGAACACCAGTGGCGAAGGCGGCTGTC'
            'TAGTCTGTAACTGACGCTGAGGCTCGAAAGCATGGGTAGCGAACAGGATTAGATACCCTGGTAGT'
            'CCATGCCGTAAACGATGAGTGCTAGGTGTTGGAGGGTTTCCGCCCTTCAGTGCCGCAGCTAACGC'
            'ATTAAGCACTCCGCCTGGGGAGTACGACCGCAAGGTTGAAACTCAAAGGAATTGACGGGGGCCCG'
            'CACAAGCGGTGGAGCATGTGGTTTAATTCGAAGCTACGCGAAGAACCTTACCAGGTCTTGACATA'
            'CTATGCAAACCTAAGAGATTAGGCGTTCCCTTCGGGGACATGGATACAGGTGGTGCATGGTTGTC'
            'GTCAGCTCGTGTCGTGAGATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTATTATCAGTTGC'
            'CAGCATTCAGTTGGGCACTCTGGTGAGACTGCCGGTGACAAACCGGAGGAAGGTGGGGATGACGT'
            'CAAATCATCATGCCCCTTATGACCTGGGCTACACACGTGCTACAATGGACGGTACAACGAGTTGC'
            'GAAGTCGTGAGGCTAAGCTAATCTCTTAAAGCCGTTCTCAGTTCGGATTGTAGGCTGCAACTCGC'
            'CTACATGAAGTTGGAATCGCTAGTAATCGCGGATCAGCATGCCGCGGTGAATACGTTCCCGGGCC'
            'TTGTACACACCGCCCGTCACACCATGAGAGTTTGTAACACCCAAAGCCGGTGAGATAACCTTCGG'
            'GAGTCAGCCGTCTATAGTG',
            '>192680',
            'AGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATACATGCAAGTCGAACGAAGC'
            'CTTCTTTCACCGAATGTTTGCATTCACCGAAAGAAGCTTAGTGGCGAACGGGTGAGTAACACGTA'
            'GGCAACCTGCCCAAAAGAGGGGGATAACACTTGGAAACAGGTGCTAATACCGCATAACCATGAAC'
            'ACCGCATGATGTTCATGTAAAAGGCGGCTTTTGCTGTCACTTTTGGATGGGCCTGCGGCGTATTA'
            'ACTTGTTGGTGGGGTAACGGCCTACCAAGGTGATGATACGTAGCCGAACTGAGAGGTTGATCGGC'
            'CACATTGGGACTGAGACACGGCCCAAACTCCTACGGGAGGCAGCAGTAGGGAATCTTCCACAATG'
            'GACGAAAGTCTGATGGAGCAACGCCGCGTGAATGAAGAAGGCCTTCGGGTCGTAAAATTCTGTTG'
            'TCAGAGAAGAACGTGCGTGAGAGTAACTGTTCACGTATTGACGGTATCTGATCAGAAAGCCACGG'
            'CTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCGAGCGTTGTCCGGATTTATTGGGCGT'
            'AAAGGGAACGCAGGCGGTCTTTTAAGTCTGATGTGAAAGCCTTCGGCTTAACCGAAGTAGTGCAT'
            'TGGAAACTGGAAGACTTGAGTGCAGAAGAGGAGAGTGGAACTCCATGTGTAGCGGTGAAATGCGT'
            'AGATATATGGAAGAACACCAGTGGCGAAAGCGGCTCTCTGGTCTGTAACTGACGCTGAGGTTCGA'
            'AAGCGTGGGTAGCAAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGAGTGCTAAGT'
            'GTTGGAGGGTTTCCGCCCTTCAGTGCTGCAGCTAACGCATTAAGCACTCCGCCTGGGGAGTACGG'
            'TCGCAAGACTGAAACTCAAAGGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAAT'
            'TCGAAGCAACGCGAAGAACCTTACCAGGTCTTGACATCTTCTGACAATTCTAGAGATGGAACGTT'
            'CCCTTCGGGGACAGAATGACAGGTGGTGCATGGTTGTCGTCAGCTCGTGTCGTGAGATGTTGGGT'
            'TAAGTCCCGCAACGAGCGCAACCCTTATTGTCAGTTGCCATCATTAAGTTGGGCACTCTGGCGAG'
            'ACTGCCGGTGACAAACCGGAGGAAGGTGGGGATGACGTCAAATCATCATGCCCCTTATGACCTGG'
            'GCTACACACGTGCTACAATGGACGGTACAACGAGTCGCTAACTCGCGAGGGCAAGCTAATCTCTT'
            'AAAGCCGTTCTCAGTTCGGACTGCAGGCTGCAACCCGCCTGCACGAAGTTGGAATTGCTAGTAAT'
            'CGCGGATCAGCATGCCGCGGTGAATACGTTCCCGGGTCTTGCACTCACCGCCCGTCA',
            '>2562098',
            'AGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATACATGCAAGTCGAGCGGATC'
            'ATCGGGAGCTTGCTCCCGATGATCAGCGGCGGACGGGTGAGTAACACGTGGGCAACCTGCCTGTA'
            'AGACTGGGATAACTCCGGGAAACCGGGGCTAATACCGGATAATTCATCTCCTCTCATGAGGGGAT'
            'GCTGAAAGACGGTTTCGGCTGTCACTTACAGATGGGCCCGCGGCGCATTAGCTAGTTGGTGAGGT'
            'AACGGCTCACCAAGGCAACGATGCGTAGCCGACCTGAGAGGGTGATCGGCCACACTGGGACTGAG'
            'ACACGGCCCAGACTCCTACGGGAGGCAGCAGTAGGGAATCTTCCGCAATGGACGAAAGTCTGACG'
            'GAGCAACGCCGCGTGAGCGAAGAAGGCCTTCGGGTCGTAAAGCTCTGTTGTCAGGGAAGAACAAG'
            'TACCGGAGTAACTGCCGGTACCTTGACGGTACCTGACCAGAAAGCCACGGCTAACTACGTGCCAG'
            'CAGCCGCGGTAATACGTAGGTGGCAAGCGTTGTCCGGAATTATTGGGCGTAAAGCGCGCGCAGGC'
            'GGTCCTTTAAGTCTGATGTGAAAGCCCACGGCTCAACCGTGGAGGGTCATTGGAAACTGGGGGAC'
            'TTGAGTGCAGAAGAGGAGAGCGGAATTCCACGTGTAGCGGTGAAATGCGTAGAGATGTGGGGGAA'
            'CACCAGTGGCGAAGGCGGCTCTCTGGTCTGTAACTGACGCTGAGGCGCGAAAGCGTGGGGAGCGA'
            'ACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGAGTGCTAAGTGTTAGAGGGTTTCCG'
            'CCCTTTAGTGCTGCAGCAAACGCATTAAGCACTCCGCCTGGGGAGTACGGCCGCAAGGCTGAAAC'
            'TCAAAGGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGAAGCAACGCGAA'
            'GAACCTTACCAGGTCTTGACATCCTCTGCCACTCCTGGAGACAGGACGTTCCCCTTCGGGGGACA'
            'GAGTGACAGGTGGTGCATGGTTGTCGTCAGCTCGTGTCGTGAGATGTTGGGTTAAGTCCCGCAAC'
            'GAGCGCAACCCTTGTTCTTAGTTGCCAGCATTCAGTTGGGCGCTCTAAGGAGACTGCCGGTGACA'
            'AACCGGAGGAAGGTGGGGATGACGTCAAATCATCATGCCCCTTATGACCTGGGCTACACACGTGC'
            'TGCAATGGATAGAACAAAGGGCAGCGAAGCCGCGAGGTGAAGCCAATCCCATAAATCTATTCTCA'
            'GTTCGGATTGCAGGCTGCAACTCGCCTGCATGAAGCCGGAATCGCTAGTAATCGCGGATCAGCAT'
            'GCCGCGGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACACCACGAGAGTTTGTAACAC'
            'CCGAAGTCGGTGGGGTAACCTTTTGGAGCCAGCCGCCTAAGGTGGGACAGATGATTGGGGTGAAG'
            'TCGTAACAAGGTA',
            '>4308624',
            'CTTTATTGGAGAGTTTGATCCTGGCTCAGGATGAACGCTGGCGGCGTGCCTAATACATGCAAGTC'
            'GAGCGAATGGATTGAGAGCTTGCTCTCAAGAAGTTAGCGGCGGACGGGTGAGTAACACGTGGGTA'
            'ACCTGCCCATAAGACTGGGATAACTCCGGGAAACCGGGGCTAATACCGGATAACATTTTGAACCG'
            'CATGGTTCGAAATTGAAAGGCGGCTTTGGCTGTCACTTATGGATGGACCCGCGTCGCATTAGCTA'
            'GTTGGTGAGGTAACGGCTCACCAAGGCAACGATGCGTAGCCGACCTGAGAGGGTGATCGGCCACA'
            'CTGGGACTGAGACACGGCCCAGACTCCTACGGGAGGCAGCAGTAGGGAATCTTCCGCAATGGACG'
            'AAAGTCTGACGGAGCAACGCCGCGTGAGTGATGAAGGCTTTCGGGTCGTAAAACTCTGTTGTTAG'
            'GGAAGAACAAGTGCTAGTTGAATAAGCTGGCACCTTGACGGTACCTAACCAGAAAGCCACGGCTA'
            'ACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCAAGCGTTATCCGGAATTATTGGGCGTAAA'
            'GCGCGCGCAGGTGGTTTCTTAAGTCTGATGTGAAAGCCCACGGCTCAACCGTGGAGGGTCATTGG'
            'AAACTGGGAGACTTGAGTGCAGAAGAGGAAAGTGGAATTCCATGTGTAGCGGTGAAATGCGTAGA'
            'GATATGGAGGAACACCAGTGGCGAAGGCGACTTTCTGGTCTGTAACTGACACTGAGGCGCGAAAG'
            'CGTGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGAGTGCTAAGTGTT'
            'AGAGGGTTTCCGCCCTTTAGTGCTGAAGTTAACGCATTAAGCACTCCGCCTGGGGAGTACGGCCG'
            'CAAGGCTGAAACTCAAAGGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCG'
            'AAGCAACGCGAAGAACCTTACCAGGTCTTGACATCCTCTGAAAACCCTAGAGATAGGGCTTCTCC'
            'TTCGGGAGCAGAGTGACAGGTGGTGCATGGTTGTCGTCAGCTCGTGTCGTGAGATGTTGGGTTAA'
            'GTCCCGCAACGAGCGCAACCCTTGATCTTAGTTGCCATCATTAAGTTGGGCACTCTAAGGTGACT'
            'GCCGGTGACAAACCGGAGGAAGGTGGGGATGACGTCAAATCATCATGCCCCTTATGACCTGGGCT'
            'ACACACGTGCTACAATGGACGGTACAAAGAGCTGCAAGACCGCGAGGTGGAGCTAATCTCATAAA'
            'ACCGTTCTCAGTTCGGATTGTAGGCTGCAACTCGCCTACATGAAGCTGGAATCGCTAGTAATCGC'
            'GGATCAGCATGCCGCGGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACACCACGAGAG'
            'TTTGTAACACCCGAAGTCGGTGGGGTAACCTTTATGGAGCCAGCCGCCTAAGGTGGGACAGATGA'
            'TTGGGGTGAAGTCGTAACAAGGTAGCCGTATCGGAAGGTGCGGCTGGATCACCTCCTTTCT',
            '>102222',
            'GCGAACGGGTGAGTAACAGGTGGGTACCTGCCCAGAAGCAGGGGATAACACTTGGAAACAGATGT'
            'TAATCCCGTATAACAAAAGAAACCCGCTTGTTTTTCTTTAAAAAGATGGTTGTGCTTATCACTTT'
            'TGATGGACCCGGGGCGCATTAGCTAGTTGGTGAGGTAACGGCTCACCAAGGCAATGATGCGTAGC'
            'CGACTTGAGAGGGTAATCGCCACATTGGGATTGAGACACGGCCCAGACTCTACGGGAGCAGCAGT'
            'AGGGAATCTTCCACCAATGGACGCAAGTCTGATGGAGCAACGCCGCGTGAGTGAAGAAGGGTTTC'
            'GGCTCGTAAAGCTCTGTTGTTAAAGAAGAACGTGGGTGAGAGTAACTGTTCACCCAGTGACGGTA'
            'TTTAACCAGAAAGCCACGGCTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCAAGCGTT'
            'ATCCGGATTTATTGGGCGTAAAGCGAGCGCAGGCGGTCTTTTAAGTCTAATGTGAAAGCCTTCGG'
            'CTCAACCGAAGAAGTGCATTGGAAACTGGGAGACTTGAGTGCAGAAGAGGACAGTGGAACTCCAT'
            'GTGTAGCGGTGAAATGCGTAGATATATGGAAGAACACCAGTGGCGAAGGCGGCTGTCTGGTCTGT'
            'AACTGACGCTGAGGCTCGAAAGCATGGGTAGCGAACAGGATTAGATACCCTGGTAGTCCATGCCG'
            'TAAACGATGATTACTAAGTGTTGGAGGGTTTCCGCCCTTCAGTGCTGCAGCTAACGCATTAAGTA'
            'ATCCGCCTGGGGAGTACGACCGCAAGGTTGAAACTCAAAAGAATTGACGGGGGCCCGCACAAGCG'
            'GTGGAGCATGTGGTTTAATTCGAAGCTACGCGAAGAACCTTACCAGGTCTTGACATCTTCTGCCA'
            'ACCTAAGAGATTAGGCGTTCCCTTCGGGGACAGAATGACAGGTGGTGCATGGTTGTCTTCAGCTC'
            'GTGTCGTGAGATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTATTACTAGTTGCCAGCATTC'
            'AGTTGGGCACTCTAATGAGACTGCCGGTGACAAACCGGAGGAAGGTGGGGACGACGTCAAATCAT'
            'CATGCCCCTTATGACCTGGGCTACACACGTGCTACAATGGATGGGCAACGAGTCCCGAAACCGCG'
            'AGGTTAACTAATCTCTTAAAACCATTCTCAATTCGGACTGTAGGCTGCAACTCGCCTACACGAAG'
            'TCGGAATCGCTAGTAATCGCGGATCAACATGCCGCGGGGAATACGTTCCCGGGCCTTGTACACAC'
            'CGCCGTCACACCATGAGAATTTGTACACCCAAAGCCGGTGGGGTACCTTTTAGACTACCGGCTAA'
            'AGTGGGGACAAATGATTAAGGTGAAGTCGTA',
            '>27815',
            'TAACCCGTGGGCACCTTGCCAAGAAGCAGGGGATAACCCTTGGAAAAGGTTGCTAATTCCGTATA'
            'ACAGAGAAAACCGCCTTGGTTTTCTTTTAAAAGGATGGTTCTGCTATCACTTCTGGATGGACCCG'
            'CGGCGCATTAGCTAGTTGGTGAGGTAACGGCTCACCAAGGCGATGATGCGTAGCCGACCTGAGAG'
            'GTAACCGCCACATTGGACTGAGACACGGGCCAAGACCTCTACGGGAGGCAGAAGTAGGGAATCTT'
            'CGACAATGGACGAAAGTCTGATGGAGCAACGCCGCGTGAGTGAAGAAGGGTTTCGGATCGTAAAG'
            'CTCTGTTGTTAAAGAAGAACGTGGGTGAGAGTAACTGTTCACCCATTGACGGTATTTAACCAGAA'
            'AGCCACGGCTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCAACCGTTGATCCGGGATT'
            'TATTGGGCGTAAAGCGAGCGCAGGCGGTCTTTTAAGTCTAATGTTGAAAACTTCGGCTCAACCGA'
            'AGAAGTGCATTGGAAACTGGGAGACTTGAGTGCAGAAGAGGATAGTGGAACTCCATGTGTAGCGG'
            'TGAAATGCGTAGATATATGGAGGAACACCAGTGGCGAAGGCGGCTGCTCTGGTCTGTAACTGACG'
            'CTGAGGCTCGAAAGCATGGGTAGCGAACAGGATTAGATACCCTGGTAGTCCATGCCGTAAACGAT'
            'GATTACTAAGTGTTGGAGGGTTTCCGCCCTTCAGTGCTGCAGCTAACGCATTAAGTAATCCGCCT'
            'GGGGAGTACGACCGCAAGGTTGAAACTCAAAAGAATTGACGGGGGCCCGCACAAGCGGTGGAGCA'
            'TGTGGTTTAATTCGAAGCTACGCGAAGAACCTTACCAGGTCTTGACATCTTCTGCCAACCTAGAG'
            'ATTAGGCGTTCCCTTCGGGGACAGAATGACAGGTGGTGCATGGTTGTCGTCAGCTCGTGTCGTGA'
            'GATGTTGGGTTAAGTCCCGCAACGGCGCAACCCTTATTACTAGTTGCCAGCATTCAGTTGGGCAC'
            'TCTAGTGAGACTGCCGGTGACAAACCGGAGGAAGGTGGGGACGACGTCAAATCATCATGCCCCTT'
            'ATGACCTGGGCTACACACGTGCTACAATGGGTGGTACAACGAGTTGCGAAACCGCGAGGTTTAAG'
            'CTAATCTCTTAAAACCATTCTCAGTTCGGACTGTAGGCTGCAACTCGCCTACACGAAGTCGGAAT'
            'CGCTAGTAATCGCGGATCAGCATGCCGCGGTGAATACGTTCCCGGGCCTTGTACACACCGGCCGT'
            'CACACCATGAGAGTTTGTAACAACCCAAGGCGGTTGGGTAACCTTTTAGGGGCTAGCCGTTTAAG'
            'GTGGGACAAATTATTAGGGGTGAAGTCGTAACAAGGTAACC',
            '>1136710',
            'GATGAACGCTGGCGGCGTGCCTAATACATGCAAGTCGGACGCACTTTCGTTGATTGAATTAGAGA'
            'TGCTTGCATCGAAGATGATTTCAACTATAAAGTGAGTGGCGAACGGGTGAGTAACACGTGGGTAA'
            'CCTGCCCAGAAGTGGGGGATAACACCTGGAAACAGATGCTAATACCGCATAATAAAATGAACCGC'
            'ATGGTTTATTTTTAAAAGATGGCTTCGGCTATCACTTCTGGATGGACCCGCGGCGTATTAGCTAG'
            'TTGGTGAGATAAAGGCTCACCAAGGCTGTGATACGTAGCCGACCTGAGAGGGTAATCGGCCACAT'
            'TGGGACTGAGACACGGCCCAGACTCCTACGGGAGGCAGCAGTAGGGAATCTTCCACAATGGACGA'
            'AAGTCTGATGGAGCAACGCCGCGTGAGTGATGAAGGCTTTAGGGTCGTAAAACTCTGTTGTTGGA'
            'GAAGAACGTGTGTGAGAGTAACTGCTCATGCAGTGACGGTATCCAACCAGAAAGCCACGGCTAAC'
            'TACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCAAGCGTTATCCGGATTTATTGGGCGTAAAGC'
            'GAGCGCAGGCGGTTTTTTAAGTCTAATGTGAAAGCCTTCGGCTTAACCGAAGAAGTGCATTGGAA'
            'ACTGGGAAACTTGAGTGCAGAAGAGGACAGTGGAACTCCATGTGTAGCGGTGAAATGCGTAGATA'
            'TATGGAAGAACACCAGTGGCGAAGGCGGCTGTCTGGTCTGTAACTGACGCTGAGGCTCGAAAGCA'
            'TGGGTAGCGAACAGGATTAGATACCCTGGTAGTCCATGCCGTAAACGATGAATGCTAAGTGTTGG'
            'AGGGTTTCCGCCCTTCAGTGCTGCAGCTAACGCATTAAGCATTCCGCCTGGGGAGTACGACCGCA'
            'AGGTTGAAACTCAAAAGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGAA'
            'GCTACGCGAAGAACCTTACCAGGTCTTGACATCTTCTGCTAACCTAAGAGATTAGGCGTTCCCTT'
            'CGGGGACAGAATGACAGGTGGTGCATGGTTGTCGTCAGCTCGTGTCGTGAGATGTTGGGTTAAGT'
            'CCCGCAACGAGCGCAACCCCTATTATTAGTTGCCAGCATTAAGTTGGGCACTCTAGTGAGACTGC'
            'CGGTGACAAACCGGAGGAAGGTGGGGACGACGTCAAATCATCATGCCCCTTATGACCTGGGCTAC'
            'ACACGTGCTACAATGGACGGTACAACGAGTTGCGAGACCGCGAGGTTAAGCTAATCTCTTAAAAC'
            'CGTTCTCAGTTCGGACTGCAGGCTGCAACTCGCCTGCACGAAGTTGGAATCGCTAGTAATCGCGG'
            'ATCAGCATGCCGCGGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACACCATGAGAGTT'
            'TGTAACACCCAAAGCCGGTGGAGTAACCTTCGGGAGCTAGCCGTCTAAGGTGGGACAGATGATTG'
            'GGGTGAAGTCGTAACAAGGTAGCCGTAGGAGAACCTGCGGCTGGATCACCTCCTTTCT'])

        _taxa1 = '\n'.join([
            '179419	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales'
            '; f__Lactobacillaceae; g__Lactobacillus; s__brevis',
            '1117026	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacil'
            'lales; f__Lactobacillaceae; g__Lactobacillus; s__brevis',
            '192680	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales'
            '; f__Lactobacillaceae; g__Lactobacillus; s__ruminis',
            '2562098	k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales;'
            ' f__Bacillaceae; g__Bacillus; s__foraminis',
            '4308624	k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales;'
            ' f__Bacillaceae; g__Bacillus; s__cereus',
            '102222	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales'
            '; f__Lactobacillaceae; g__Pediococcus; s__acidilactici',
            '27815	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales'
            '; f__Lactobacillaceae; g__Pediococcus; s__acidilactici',
            '1136710	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacil'
            'lales; f__Lactobacillaceae; g__Pediococcus; s__damnosus'])

        _taxa2 = '\n'.join([
            '179419	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales'
            '; f__Lactobacillaceae; g__Lactobacillus; s__brevis',
            '1117026	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacil'
            'lales; f__Lactobacillaceae; g__Lactobacillus',
            '192680	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales'
            '; f__Lactobacillaceae',
            '2562098	k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales',
            '4308624	k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales;'
            ' f__Bacillaceae; g__Bacillus; s__cereus',
            '102222	k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales;'
            ' f__Bacillaceae; g__Bacillus; s__cereus',
            '27815	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales'
            '; f__Lactobacillaceae; g__Pediococcus',
            '1136710	k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacil'
            'lales; f__Lactobacillaceae; g__Pediococcus'])

        self.tmpdir = mkdtemp()
        name = 'B1-REF-L6-iter0'
        self.query_fp = join(self.tmpdir, name, QUERY_TAXA_TSV)
        self.paramdir = join(self.tmpdir, name, name, 'method1', 'param1')
        self.obs_taxa_fp = join(self.paramdir, QUERY_TAX_ASSIGNMENTS_TXT)
        refpath = join(self.tmpdir, 'ref1.txt')
        self.cvdir = cross_validated_root(self.tmpdir)
        self.ntdir = novel_taxa_simulations_root(self.tmpdir)
        if not exists(self.paramdir):
            makedirs(self.paramdir)
        with open(refpath, 'w') as out:
            out.write(_ref1)
        with open(self.query_fp, 'w') as out:
            out.write(_taxa1)
        with open(self.obs_taxa_fp, 'w') as out:
            out.write(_taxa2)

        self.databases = {'B1-REF': [refpath, self.query_fp,
                                     "ref1", "GTGCCAGCMGCCGCGGTAA",
                                     "GGACTACHVGGGTWTCTAAT", "515f", "806r"]}
        self.ref_data = pd.DataFrame.from_dict(self.databases, orient="index")
        self.ref_data.columns = ["Reference file path", "Reference tax path",
                                 "Reference id", "Fwd primer", "Rev primer",
                                 "Fwd primer id", "Rev primer id"]

        # level 0 (kingdom) is never scored, so it is NaN rather than 0
        self.exp_p = [nan, 1.0, 1.0, 0.875, 0.8571428571428571,
                      0.83333333333333337, 0.66666666666666663]
        self.exp_r = [nan, 1.0, 1.0, 0.875, 0.75, 0.625, 0.25]
        self.exp_f = [nan, 1.0, 1.0, 0.875, 0.79999999999999993,
                      0.7142857142857143, 0.36363636363636365]
        # no overclassification anywhere in this fixture, so the original
        # scoring coincides with the new scoring at every level
        self.exp_r_org = list(self.exp_r)
        self.exp_f_org = list(self.exp_f)
        self.exp_m = [0, 0, 0, 1, 1, 1, 3, 2]

    @classmethod
    def tearDownClass(self):
        rmtree(self.tmpdir)


if __name__ == "__main__":
    main()
