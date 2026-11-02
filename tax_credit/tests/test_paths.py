#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2014--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from tax_credit.paths import (
    EXPECTED_SUBDIR,
    QUERY_TAX_ASSIGNMENTS_TXT,
    assignment_result_leaf_glob,
    expected_tables_glob,
    list_assignment_result_dirs,
    mock_observed_tables_glob,
    parse_assignment_results_dir,
    parse_expected_table_path,
    parse_mock_result_table_path,
    parse_result_leaf_dir_to_parts,
    parse_taxonomy_map_path_to_dataset_id,
)


class PathsTests(TestCase):

    def test_mock_observed_tables_glob(self):
        self.assertEqual(
            mock_observed_tables_glob("/results", "table*biom"),
            "/results/*/*/*/*/table*biom",
        )

    def test_expected_tables_glob(self):
        self.assertEqual(
            expected_tables_glob("/data", "table.L6-taxa.biom"),
            "/data/*/*/{}/table.L6-taxa.biom".format(EXPECTED_SUBDIR),
        )

    def test_parse_mock_result_table_path(self):
        fp = "/analyses/mock1/greengenes/nb/0.7:table/table.biom"
        p = parse_mock_result_table_path(fp)
        self.assertEqual(
            p, ("mock1", "greengenes", "nb", "0.7:table")
        )

    def test_parse_expected_table_path(self):
        fp = "/data/mock1/greengenes/expected/table.L6-taxa.biom"
        p = parse_expected_table_path(fp)
        self.assertEqual(p.dataset_id, "mock1")
        self.assertEqual(p.reference_id, "greengenes")

    def test_parse_assignment_results_dir(self):
        fp = "/out/silva-L3-iter0/naive-bayes/p1"
        p = parse_assignment_results_dir(fp)
        self.assertEqual(p.dataset_id, "silva-L3-iter0")
        self.assertEqual(p.method_id, "naive-bayes")
        self.assertEqual(p.params_id, "p1")

    def test_parse_taxonomy_map_path_to_dataset_id(self):
        fp = "/tmp/ds1/ref1/m1/p1/rep_set_tax_assignments.txt"
        self.assertEqual(
            parse_taxonomy_map_path_to_dataset_id(fp), "ds1"
        )

    def test_parse_result_leaf_dir_to_parts(self):
        d = "/tmp/ds1/ref1/m1/p1"
        p = parse_result_leaf_dir_to_parts(d)
        self.assertEqual(p.dataset_id, "ds1")
        self.assertEqual(p.reference_id, "ref1")
        self.assertEqual(p.method_id, "m1")
        self.assertEqual(p.parameter_id, "p1")

    def test_assignment_result_leaf_glob(self):
        self.assertEqual(
            assignment_result_leaf_glob("/out"),
            "/out/*/*/*/*",
        )

    def test_list_assignment_result_dirs_filters_and_sorts(self):
        with TemporaryDirectory() as tmp:
            good = "{}/ds/ds/method/p1".format(tmp)
            empty_leaf = "{}/ds/ds/method/p2".format(tmp)
            clf_only = "{}/db/db/method/fitparams".format(tmp)
            for d in (good, empty_leaf, clf_only):
                os.makedirs(d)
            with open("{}/{}".format(good, QUERY_TAX_ASSIGNMENTS_TXT), "w") as f:
                f.write("x\n")
            found = list_assignment_result_dirs(tmp)
            self.assertEqual(found, [good])
