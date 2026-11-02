#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2014--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase

from tax_credit.simulation_names import (
    DIR_CROSS_VALIDATED,
    DIR_CROSS_VALIDATED_TRAD,
    DIR_NOVEL_TAXA_SIMULATIONS,
    DIR_REF_DBS,
    DIR_SELF_VALIDATED,
    cross_validated_root,
    cross_validated_trad_root,
    format_cv_fold_dirname,
    format_novel_fold_dirname,
    novel_taxa_simulations_root,
    parse_cv_dataset_id,
    parse_novel_dataset_id,
    parse_self_validated_dataset_id,
    ref_dbs_root,
    self_validated_root,
)


class SimulationNamesTests(TestCase):

    def test_roots(self):
        base = "/proj/data"
        self.assertEqual(
            cross_validated_root(base),
            "/proj/data/{}".format(DIR_CROSS_VALIDATED),
        )
        self.assertEqual(
            cross_validated_trad_root(base),
            "/proj/data/{}".format(DIR_CROSS_VALIDATED_TRAD),
        )
        self.assertEqual(
            novel_taxa_simulations_root(base),
            "/proj/data/{}".format(DIR_NOVEL_TAXA_SIMULATIONS),
        )
        self.assertEqual(
            self_validated_root(base),
            "/proj/data/{}".format(DIR_SELF_VALIDATED),
        )
        self.assertEqual(
            ref_dbs_root(base),
            "/proj/data/{}".format(DIR_REF_DBS),
        )

    def test_format_cv_fold_dirname(self):
        self.assertEqual(format_cv_fold_dirname("silva", 2), "silva-iter2")

    def test_parse_cv_dataset_id(self):
        p = parse_cv_dataset_id("silva-iter0")
        self.assertEqual(p.database, "silva")
        self.assertEqual(p.iteration, "0")

    def test_format_novel_fold_dirname(self):
        self.assertEqual(
            format_novel_fold_dirname("silva", 3, 1),
            "silva-L3-iter1",
        )

    def test_parse_novel_dataset_id(self):
        p = parse_novel_dataset_id("silva-L3-iter0")
        self.assertEqual(p.database, "silva")
        self.assertEqual(p.level, 3)
        self.assertEqual(p.iteration, "0")

    def test_parse_novel_dataset_id_hyphenated_db(self):
        p = parse_novel_dataset_id("B1-REF-L6-iter0")
        self.assertEqual(p.database, "B1-REF")
        self.assertEqual(p.level, 6)
        self.assertEqual(p.iteration, "0")

    def test_parse_self_validated_dataset_id(self):
        p = parse_self_validated_dataset_id("silva")
        self.assertEqual(p.database, "silva")
        self.assertEqual(p.iteration, "0")
