#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2014--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------


from unittest import TestCase, main
from tempfile import mkdtemp
from os.path import join
from shutil import rmtree
from tax_credit.taxa_manipulator import (string_search,
                                         unique_lines,
                                         trim_taxonomy_strings,
                                         branching_taxa,
                                         extract_taxa_names,
                                         compile_reference_taxa,
                                         extract_rownames,
                                         extract_fasta_ids,
                                         filter_sequences,
                                         stratify_taxonomy_subsets,
                                         accept_list_or_file,
                                         normalize_taxon,
                                         reference_lineage_prefixes,
                                         truncate_taxa_to_reference,
                                         is_unassigned_taxon,
                                         strip_taxonomy_header,
                                         drop_unassigned_taxonomies)


class EvalFrameworkTests(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.op11 = \
            '203525\tk__Bacteria; p__OP11; c__OP11-1; o__; f__; g__; s__'
        cls.op11_t = 'k__Bacteria; p__OP11; c__OP11-1; o__; f__; g__; s__'

        cls.table1 = [
            '229854\tk__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; '
            'o__Legionellales; f__Legionellaceae; g__Legionella; s__',
            '367523\tk__Bacteria; p__Bacteroidetes; c__Flavobacteriia; o__Flav'
            'obacteriales; f__Flavobacteriaceae; g__Flavobacterium; s__',
            '239330\tk__Bacteria; p__Proteobacteria; c__Deltaproteobacteria; '
            'o__Desulfuromonadales; f__Geobacteraceae; g__Geobacter; s__',
            '203525\tk__Bacteria; p__OP11; c__OP11-1; o__; f__; g__; s__']

        cls.table2 = [
            '229854\tk__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; '
            'o__Legionellales; f__Legionellaceae; g__Legionella; s__',
            '229854\tk__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; '
            'o__Legionellales; f__Legionellaceae; g__Legionella; s__',
            '367523\tk__Bacteria; p__Bacteroidetes; c__Flavobacteriia; o__Flav'
            'obacteriales; f__Flavobacteriaceae; g__Flavobacterium; s__',
            '367523\tk__Bacteria; p__Bacteroidetes; c__Flavobacteriia; o__Flav'
            'obacteriales; f__Flavobacteriaceae; g__Flavobacterium; s__',
            '239330\tk__Bacteria; p__Proteobacteria; c__Deltaproteobacteria; '
            'o__Desulfuromonadales; f__Geobacteraceae; g__Geobacter; s__',
            '203525\tk__Bacteria; p__OP11; c__OP11-1; o__; f__; g__; s__',
            '239330\tk__Bacteria; p__Proteobacteria; c__Deltaproteobacteria; '
            'o__Desulfuromonadales; f__Geobacteraceae; g__Geobacter; s__']

        cls.table3 = [
            '229854\tk__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; '
            'o__Legionellales; f__Legionellaceae; g__Legionella; s__',
            '367523\tk__Bacteria; p__Bacteroidetes; c__Flavobacteriia; o__Flav'
            'obacteriales; f__Flavobacteriaceae; g__Flavobacterium; s__',
            '239330\tk__Bacteria; p__Proteobacteria; c__Deltaproteobacteria; '
            'o__Desulfuromonadales; f__Geobacteraceae; g__Geobacter; s__']

        cls.seqs1 = '\n'.join(['>229854',
                               'ACTAGTAGTTGAC',
                               '>367523',
                               'ATCGATGCATGCA',
                               '>239330',
                               'TGTGTGCTGGTAGTTAC',
                               '>203525',
                               'TGTATGCTGATGC\n'])

        cls.seqs2 = '\n'.join(['>229854',
                               'ACTAGTAGTTGAC',
                               '>367523',
                               'ATCGATGCATGCA',
                               '>239330',
                               'TGTGTGCTGGTAGTTAC\n'])

        cls.tmpdir = mkdtemp()

    def test_string_search(self):
        self.assertEqual(string_search(self.table1, '203525'), [self.op11])
        self.assertEqual(string_search(self.table1, '229854|367523|239330',
                                       discard=True), [self.op11])
        self.assertEqual(string_search(self.table1, 'OP11-1'), [self.op11])
        self.assertEqual(string_search(self.table1, 'OP11-1'), [self.op11])

    def test_unique_lines(self):
        self.assertEqual(unique_lines(self.table2), self.table1)
        self.assertEqual(unique_lines(self.table2, 'u'), [self.op11])
        self.assertEqual(unique_lines(self.table2, 'u', field=1,
                                      printfield=True), [self.op11_t])
        self.assertEqual(set(unique_lines(self.table2, 'd')), set(self.table3))

    def test_trim_taxonomy_strings(self):
        self.assertEqual(trim_taxonomy_strings(
            [self.op11], level=6), [self.op11])
        self.assertEqual(trim_taxonomy_strings(
            [self.op11], level=0), ['203525\tk__Bacteria'])

    def test_normalize_taxon(self):
        na_taxon = ('Eukaryota;Chordata;Actinopteri;NA;Lutjanidae;Lutjanus;'
                    'Lutjanus griseus')
        # internal NA ranks keep their position
        self.assertEqual(normalize_taxon(na_taxon), na_taxon)
        # trailing NA and empty ranks are removed
        self.assertEqual(normalize_taxon('A;B;C;NA;E;F;NA'), 'A;B;C;NA;E;F')
        self.assertEqual(normalize_taxon('A;B;C;D;E;NA;NA'), 'A;B;C;D;E')
        self.assertEqual(normalize_taxon('A;B;C;D;E;;'), 'A;B;C;D;E')
        self.assertEqual(normalize_taxon('A;B;C; NA'), 'A;B;C')
        self.assertEqual(normalize_taxon('NA'), '')
        self.assertEqual(normalize_taxon(';;'), '')
        self.assertEqual(normalize_taxon('Unassigned'), 'Unassigned')
        # names that only start with NA are not ranks to strip
        self.assertEqual(normalize_taxon('A;B;NAxx'), 'A;B;NAxx')

    def test_reference_lineage_prefixes(self):
        ref = ['r1\tEukaryota;Chordata;Actinopteri',
               'r2\tEukaryota;Chordata;Chondrichthyes;NA;Hexanchidae']
        self.assertEqual(
            reference_lineage_prefixes(ref),
            {'Eukaryota',
             'Eukaryota;Chordata',
             'Eukaryota;Chordata;Actinopteri',
             'Eukaryota;Chordata;Chondrichthyes',
             'Eukaryota;Chordata;Chondrichthyes;NA',
             'Eukaryota;Chordata;Chondrichthyes;NA;Hexanchidae'})

    def test_truncate_taxa_to_reference(self):
        ref = ['r1\tA;B;C;D;Nemichthys',
               'r2\tA;B;C;OtherFamily;OtherGenus']
        query = [
            # genus Avocettina is gone with its only species: expect family
            'q1\tA;B;C;D;Avocettina',
            # every rank present: unchanged
            'q2\tA;B;C;D;Nemichthys',
            # family and genus both absent: expect the order
            'q3\tA;B;C;GoneFamily;GoneGenus',
        ]
        self.assertEqual(
            truncate_taxa_to_reference(query, ref),
            ['q1\tA;B;C;D', 'q2\tA;B;C;D;Nemichthys', 'q3\tA;B;C'])

    def test_truncate_taxa_to_reference_drops_unreachable(self):
        # nothing in common with the reference: the query cannot be evaluated
        self.assertEqual(
            truncate_taxa_to_reference(['q1\tBacteria;Firmicutes'],
                                       ['r1\tEukaryota;Chordata']),
            [])

    def test_truncate_taxa_to_reference_keeps_internal_na(self):
        ref = ['r1\tEukaryota;Chordata;Actinopteri;NA;Centropomidae;Lates']
        query = ['q1\tEukaryota;Chordata;Actinopteri;NA;Centropomidae;Gone']
        self.assertEqual(
            truncate_taxa_to_reference(query, ref),
            ['q1\tEukaryota;Chordata;Actinopteri;NA;Centropomidae'])

    def test_is_unassigned_taxon(self):
        # no rank assigned anywhere
        self.assertTrue(is_unassigned_taxon('NA;NA;NA;NA;NA;NA;NA'))
        self.assertTrue(is_unassigned_taxon('NA; NA ;NA'))
        self.assertTrue(is_unassigned_taxon('NA'))
        self.assertTrue(is_unassigned_taxon(';;'))
        self.assertTrue(is_unassigned_taxon(''))
        self.assertTrue(is_unassigned_taxon('   '))
        # placeholders a classifier emits in place of a call
        self.assertTrue(is_unassigned_taxon('Unassigned'))
        self.assertTrue(is_unassigned_taxon(' No blast hit '))
        # non-strings: None, and the NaN pandas reads a bare 'NA' field as
        self.assertTrue(is_unassigned_taxon(None))
        self.assertTrue(is_unassigned_taxon(float('nan')))
        # one assigned rank is enough, however shallow or deep it sits
        self.assertFalse(is_unassigned_taxon('Eukaryota;NA;NA;NA;NA;NA;NA'))
        self.assertFalse(is_unassigned_taxon('NA;NA;Actinopteri;NA;NA;NA;NA'))
        self.assertFalse(is_unassigned_taxon(
            'Eukaryota;Chordata;Actinopteri;Gadiformes;Gadidae;Gadus;NA'))
        # names that merely start with NA are real ranks
        self.assertFalse(is_unassigned_taxon('NAxx;NA;NA'))

    def test_strip_taxonomy_header(self):
        gadus = ('gadus\tEukaryota;Chordata;Actinopteri;Gadiformes;Gadidae;'
                 'Gadus;NA')
        other = 'other\tEukaryota;Chordata;Actinopteri;NA;NA;NA;NA'
        # QIIME 2's own header, and the other spellings databases ship with
        for header in ['Feature ID\tTaxon', 'feature-id\tTaxon',
                       'featureid\tTaxon', 'id\tTaxon', '#OTU ID\tTaxon']:
            self.assertEqual(strip_taxonomy_header([header, gadus, other]),
                             [gadus, other])
        # a headerless database is passed through untouched
        self.assertEqual(strip_taxonomy_header([gadus, other]), [gadus, other])
        self.assertEqual(strip_taxonomy_header([]), [])
        # only the first line is a candidate, so a record that happens to sit
        # below a header-like string is kept
        self.assertEqual(strip_taxonomy_header([gadus, 'id\tTaxon']),
                         [gadus, 'id\tTaxon'])
        # only the ID column is examined: a real record is never mistaken for
        # a header because its taxonomy mentions something header-like
        self.assertEqual(strip_taxonomy_header(['seq1\tid;Taxon']),
                         ['seq1\tid;Taxon'])

    def test_drop_unassigned_taxonomies(self):
        gadus = ('gadus\tEukaryota;Chordata;Actinopteri;Gadiformes;Gadidae;'
                 'Gadus;NA')
        kingdom_only = 'shallow\tEukaryota;NA;NA;NA;NA;NA;NA'
        lines = [gadus,
                 'empty\tNA;NA;NA;NA;NA;NA;NA',
                 kingdom_only,
                 'blank\t',
                 'unassigned\tUnassigned',
                 'no_tax_field']
        # only the records that place a sequence somewhere survive
        self.assertEqual(drop_unassigned_taxonomies(lines),
                         [gadus, kingdom_only])
        # unchanged when every record is usable
        self.assertEqual(drop_unassigned_taxonomies([gadus]), [gadus])
        self.assertEqual(drop_unassigned_taxonomies([]), [])

    def test_branching_taxa(self):
        self.assertEqual(branching_taxa(self.table1, field=6), [])
        self.assertEqual(set(branching_taxa(self.table1, field=1)),
                         set(self.table1))
        self.assertEqual(set(branching_taxa(self.table1, field=2)),
                         set(['229854\tk__Bacteria; p__Proteobacteria; '
                              'c__Gammaproteobacteria; o__Legionellales; '
                              'f__Legionellaceae; g__Legionella; s__',
                              '239330\tk__Bacteria; p__Proteobacteria; '
                              'c__Deltaproteobacteria; o__Desulfuromonadales; '
                              'f__Geobacteraceae; g__Geobacter; s__']))

    def test_extract_taxa_names(self):
        self.assertEqual(extract_taxa_names(
            self.table1), ['s__', 's__', 's__', 's__'])
        self.assertEqual(extract_taxa_names(
            self.table1, level=slice(0, 1), field=1),
            ['k__Bacteria', 'k__Bacteria', 'k__Bacteria', 'k__Bacteria'])

    def test_compile_reference_taxa(self):
        self.assertEqual(compile_reference_taxa(
            [self.op11]), {self.op11_t: ['203525']})
        self.assertEqual(compile_reference_taxa(
            self.table1)[self.op11_t], ['203525'])

    def test_extract_rownames(self):
        self.assertEqual(extract_rownames(self.table1),
                         {'229854', '367523', '239330', '203525'})

    def test_extract_fasta_ids(self):
        with open(join(self.tmpdir, 'seqs1.tmp'), 'w') as out:
            out.write(self.seqs1)
        self.assertEqual(extract_fasta_ids(join(self.tmpdir, 'seqs1.tmp')),
                         {'229854', '367523', '239330', '203525'})

    def test_filter_sequences(self):
        filter_sequences(join(self.tmpdir, 'seqs1.tmp'),
                         join(self.tmpdir, 'seqs.tmp'), self.table3)
        with open(join(self.tmpdir, 'seqs.tmp'), 'r') as sq:
            self.assertEqual(sq.read(), self.seqs2)
        filter_sequences(join(self.tmpdir, 'seqs1.tmp'),
                         join(self.tmpdir, 'seqs.tmp'),
                         [self.op11], keep=False)
        with open(join(self.tmpdir, 'seqs.tmp'), 'r') as sq:
            self.assertEqual(sq.read(), self.seqs2)

    def test_stratify_taxonomy_subsets(self):
        stratify_taxonomy_subsets(self.table2, 2, self.tmpdir, 'test', level=5)
        iter0 = accept_list_or_file(
            join(self.tmpdir, 'test-iter0/query_taxa.tsv'))
        iter1 = accept_list_or_file(
            join(self.tmpdir, 'test-iter1/query_taxa.tsv'))
        self.assertEqual(set(iter0) & set(iter1), set(self.table3))

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmpdir)


if __name__ == "__main__":
    main()
