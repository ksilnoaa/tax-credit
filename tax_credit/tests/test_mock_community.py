#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2014--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Tests for ``tax_credit.mock_community``.

The shared fixture is one mock sample with three ASVs on a four-rank
backbone (k, p, g, s). Expected values below are worked out by hand in the
comments of each test.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from tax_credit import mock_community as mc

RANKS = ["k", "p", "g", "s"]


def _fixture():
    counts = pd.DataFrame({"s1": [50, 30, 20], "other": [0, 0, 7]},
                          index=["asv1", "asv2", "asv3"])
    # asv1 correct, asv2 stops at genus, asv3 wrong genus and species
    assignments = pd.Series({"asv1": "K;P;G1;S1", "asv2": "K;P;G1",
                             "asv3": "K;P;G9;S9"})
    # the third expected taxon is only known to genus
    composition = pd.DataFrame(
        {"s1": [0.4, 0.4, 0.2]}, index=["K;P;G1;S1", "K;P;G1;S2", "K;P;G2"])
    asv_taxonomy = pd.Series({"asv1": "K;P;G1;S1", "asv2": "K;P;G1;S2",
                              "asv3": "K;P;G2;S3"})
    return counts, assignments, composition, asv_taxonomy


def _row(metrics, rank):
    return metrics[metrics["rank"] == rank].iloc[0]


class LineageTests(unittest.TestCase):

    def test_normalize_lineage(self):
        self.assertEqual(mc.normalize_lineage(" K ; P ;G;NA;"), "K;P;G")
        self.assertEqual(mc.normalize_lineage("K;NA;G"), "K;NA;G")
        for unassigned in ("Unassigned", "", "No blast hit", None, np.nan):
            self.assertEqual(mc.normalize_lineage(unassigned), "")

    def test_truncate_lineage(self):
        self.assertEqual(mc.truncate_lineage("K;P;G;S", 2), ("K;P;G", True))
        self.assertEqual(mc.truncate_lineage("K;P", 2), ("K;P", False))
        self.assertEqual(mc.truncate_lineage("", 2), ("Unassigned", False))
        # an internal NA at the evaluated rank is not a name at that rank
        self.assertEqual(mc.truncate_lineage("K;P;NA;S", 2), ("K;P", False))
        self.assertEqual(mc.truncate_lineage("K;NA;G;S", 2), ("K;NA;G", True))

    def test_rank_levels(self):
        self.assertEqual(mc.rank_levels(RANKS, ["g", "s"]), {"g": 2, "s": 3})
        with self.assertRaises(ValueError):
            mc.rank_levels(RANKS, ["genus"])


class MetricTests(unittest.TestCase):

    def setUp(self):
        self.counts, self.assignments, self.composition, self.asv_taxonomy = (
            _fixture())

    def _evaluate(self, **kwargs):
        metrics, _ = mc.evaluate_mock_samples(
            self.counts, self.assignments, RANKS, ["g", "s"], ["s1"],
            composition=self.composition, **kwargs)
        return metrics

    def test_species_taxon_rates(self):
        # observed resolved at species: G1;S1 (0.5), G9;S9 (0.2)
        # expected resolved at species: G1;S1, G1;S2   -> 1 shared
        row = _row(self._evaluate(), "s")
        self.assertAlmostEqual(row["Taxon Accuracy Rate"], 0.5)
        self.assertAlmostEqual(row["Taxon Detection Rate"], 0.5)
        self.assertEqual(row["tot_val_obs"], 2)
        self.assertEqual(row["tot_val_exp"], 2)
        # TAR = tp_pa / (tp_pa + fp_pa), TDR = tp_pa / (tp_pa + fn_pa)
        self.assertEqual(row["tp_pa"], 1)
        self.assertEqual(row["fp_pa"], 1)
        self.assertEqual(row["fn_pa"], 1)
        self.assertAlmostEqual(
            row["Taxon Accuracy Rate"],
            row["tp_pa"] / (row["tp_pa"] + row["fp_pa"]))
        self.assertAlmostEqual(
            row["Taxon Detection Rate"],
            row["tp_pa"] / (row["tp_pa"] + row["fn_pa"]))

    def test_species_taxon_rates_legacy(self):
        # unresolved labels count: observed adds K;P;G1, expected adds K;P;G2
        row = _row(self._evaluate(legacy_unresolved_taxa=True), "s")
        self.assertAlmostEqual(row["Taxon Accuracy Rate"], 1 / 3)
        self.assertAlmostEqual(row["Taxon Detection Rate"], 1 / 3)

    def test_min_relative_abundance(self):
        # G9;S9 at 0.2 falls below the threshold
        row = _row(self._evaluate(min_relative_abundance=0.25), "s")
        self.assertAlmostEqual(row["Taxon Accuracy Rate"], 1.0)
        self.assertAlmostEqual(row["Taxon Detection Rate"], 0.5)

    def test_bray_curtis(self):
        metrics = self._evaluate()
        # species: |.5-.4| + .3 + .2 + .4 + .2 = 1.2, over 2
        self.assertAlmostEqual(_row(metrics, "s")["Bray-Curtis"], 0.6)
        # genus: G1 .8 vs .8, G9 .2 vs 0, G2 0 vs .2 -> .4 / 2
        self.assertAlmostEqual(_row(metrics, "g")["Bray-Curtis"], 0.2)
        self.assertAlmostEqual(_row(metrics, "g")["Taxon Accuracy Rate"], 0.5)

    def test_bray_curtis_identical_is_zero(self):
        a = pd.Series({"x": 0.3, "y": 0.7})
        self.assertAlmostEqual(mc.bray_curtis(a, a), 0.0)
        self.assertTrue(np.isnan(mc.bray_curtis(pd.Series(dtype=float),
                                                pd.Series(dtype=float))))

    def test_precision_recall_not_computed_without_asv_taxonomy(self):
        row = _row(self._evaluate(), "s")
        self.assertTrue(np.isnan(row["Precision"]))
        self.assertTrue(np.isnan(row["match_ratio"]))

    def test_precision_recall_weighted_by_reads(self):
        # species: asv1 match (tp 50), asv2 under (fn 30), asv3 mis (fp 20, fn 20)
        row = _row(self._evaluate(asv_taxonomy=self.asv_taxonomy), "s")
        self.assertAlmostEqual(row["Precision"], 50 / 70)
        self.assertAlmostEqual(row["Recall"], 0.5)
        p, r = 50 / 70, 0.5
        self.assertAlmostEqual(row["F-measure"], 2 * p * r / (p + r))
        # unweighted: tp 1, fp 1, fn 2
        self.assertAlmostEqual(row["ASV Precision"], 0.5)
        self.assertAlmostEqual(row["ASV Recall"], 1 / 3)
        self.assertAlmostEqual(row["match_ratio"], 0.5)
        self.assertAlmostEqual(row["underclassification_ratio"], 0.3)
        self.assertAlmostEqual(row["overclassification_ratio"], 0.0)
        self.assertAlmostEqual(row["misclassification_ratio"], 0.2)
        self.assertEqual(row["n_asvs_scored"], 3)
        self.assertAlmostEqual(row["reads_scored_fraction"], 1.0)
        # read tallies reconcile with the read-weighted scores
        self.assertAlmostEqual(row["TP"], 50.)
        self.assertAlmostEqual(row["FP"], 20.)
        self.assertAlmostEqual(row["FN"], 50.)
        self.assertAlmostEqual(row["FN_original"], 50.)
        self.assertAlmostEqual(row["Precision"],
                               row["TP"] / (row["TP"] + row["FP"]))
        self.assertAlmostEqual(row["Recall"],
                               row["TP"] / (row["TP"] + row["FN"]))
        # ASV tallies reconcile with the ASV scores
        self.assertEqual(row["n_match"], 1)
        self.assertEqual(row["n_under"], 1)
        self.assertEqual(row["n_over"], 0)
        self.assertEqual(row["n_mis"], 1)
        self.assertAlmostEqual(
            row["ASV Precision"],
            row["n_match"] / (row["n_match"] + row["n_over"] + row["n_mis"]))
        # no overclassification here, so the original scoring coincides
        self.assertAlmostEqual(row["Recall_original"], row["Recall"])
        self.assertAlmostEqual(row["F-measure_original"], row["F-measure"])

    def test_overclassification_separates_new_and_original_recall(self):
        # asv2's reference stops at genus but it is assigned down to species,
        # so at species rank it is an overclassification: a false positive, but
        # not a false negative under the new scoring
        self.assignments["asv2"] = "K;P;G1;S5"
        asv_taxonomy = pd.Series(
            {"asv1": "K;P;G1;S1", "asv2": "K;P;G1", "asv3": "K;P;G2;S3"},
            name="Taxon")
        row = _row(self._evaluate(asv_taxonomy=asv_taxonomy), "s")
        self.assertEqual(row["n_over"], 1)
        # reads: asv1 match 50, asv2 over 30, asv3 mis 20
        self.assertAlmostEqual(row["TP"], 50.)
        self.assertAlmostEqual(row["FP"], 50.)       # over 30 + mis 20
        self.assertAlmostEqual(row["FN"], 20.)       # mis only
        self.assertAlmostEqual(row["FN_original"], 50.)   # + over 30
        self.assertAlmostEqual(row["Recall"], 50 / 70)
        self.assertAlmostEqual(row["Recall_original"], 0.5)
        self.assertLess(row["Recall_original"], row["Recall"])
        # precision is unaffected by the redefinition
        self.assertAlmostEqual(row["Precision"], 0.5)
        # and match_ratio tracks the original scoring, not the new Recall
        self.assertAlmostEqual(row["match_ratio"], row["Recall_original"])

    def test_precision_recall_genus(self):
        # genus: asv1 and asv2 match (80), asv3 G9 vs G2 mis (20)
        row = _row(self._evaluate(asv_taxonomy=self.asv_taxonomy), "g")
        self.assertAlmostEqual(row["Precision"], 0.8)
        self.assertAlmostEqual(row["Recall"], 0.8)

    def test_unknown_asvs_left_out_of_precision(self):
        asv_taxonomy = self.asv_taxonomy.drop("asv3")
        row = _row(self._evaluate(asv_taxonomy=asv_taxonomy), "s")
        # tp 50, fn 30, no fp
        self.assertAlmostEqual(row["Precision"], 1.0)
        self.assertAlmostEqual(row["Recall"], 50 / 80)
        self.assertEqual(row["n_asvs_scored"], 2)
        self.assertAlmostEqual(row["reads_scored_fraction"], 0.8)

    def test_composition_from_asv_taxonomy(self):
        composition = mc.composition_from_asv_taxonomy(
            self.counts, self.asv_taxonomy)
        self.assertAlmostEqual(composition.loc["K;P;G1;S1", "s1"], 0.5)
        self.assertAlmostEqual(composition.loc["K;P;G2;S3", "other"], 1.0)
        metrics, _ = mc.evaluate_mock_samples(
            self.counts, self.assignments, RANKS, ["s"], ["s1"],
            asv_taxonomy=self.asv_taxonomy)
        # expected G1;S1 .5, G1;S2 .3, G2;S3 .2 ; observed G1;S1 .5, G1 .3, G9;S9 .2
        # |diffs| 0 + .3 + .3 + .2 + .2 = 1.0, over 2
        self.assertAlmostEqual(metrics.loc[0, "Bray-Curtis"], 0.5)
        self.assertAlmostEqual(metrics.loc[0, "Taxon Detection Rate"], 1 / 3)

    def test_unassigned_and_missing_assignments(self):
        assignments = pd.Series({"asv1": "Unassigned"})
        observed, resolved = mc.collapse_observed(
            self.counts["s1"], assignments, 3)
        self.assertEqual(list(observed.index), ["Unassigned"])
        self.assertAlmostEqual(observed["Unassigned"], 1.0)
        self.assertFalse(resolved["Unassigned"])

    def test_composition_table(self):
        _, table = mc.evaluate_mock_samples(
            self.counts, self.assignments, RANKS, ["s"], ["s1"],
            composition=self.composition)
        row = table[table["taxon"] == "K;P;G1;S1"].iloc[0]
        self.assertAlmostEqual(row["expected"], 0.4)
        self.assertAlmostEqual(row["observed"], 0.5)
        self.assertAlmostEqual(table["expected"].sum(), 1.0)
        self.assertAlmostEqual(table["observed"].sum(), 1.0)

    def test_empty_observations_give_nan_accuracy(self):
        tar, tdr, n_obs, n_exp, _ = mc.taxon_accuracy_detection(
            pd.Series(dtype=float), pd.Series(dtype=bool),
            pd.Series({"a": 1.0}), pd.Series({"a": True}))
        self.assertTrue(np.isnan(tar))
        self.assertAlmostEqual(tdr, 0.0)


class SampleSelectionTests(unittest.TestCase):

    def setUp(self):
        self.counts, _, self.composition, self.asv_taxonomy = _fixture()

    def test_composition_samples(self):
        composition = self.composition.assign(missing=[1.0, 0, 0])
        samples, excluded = mc.select_mock_samples(
            self.counts, composition=composition)
        self.assertEqual(samples, ["s1"])
        self.assertEqual(excluded, [
            {"sample_id": "missing", "reason": mc.EXCLUDED_NOT_IN_FEATURE_TABLE},
            {"sample_id": "other", "reason": mc.EXCLUDED_NOT_IN_COMPOSITION},
        ])

    def test_asv_taxonomy_samples(self):
        counts = self.counts.assign(blank=[0, 0, 0])
        samples, excluded = mc.select_mock_samples(
            counts, asv_taxonomy=self.asv_taxonomy)
        self.assertEqual(samples, ["s1", "other"])
        self.assertEqual(excluded, [
            {"sample_id": "blank", "reason": mc.EXCLUDED_NO_KNOWN_READS}])

    def test_requested_samples(self):
        samples, excluded = mc.select_mock_samples(
            self.counts, asv_taxonomy=self.asv_taxonomy, requested=["other"])
        self.assertEqual(samples, ["other"])
        self.assertEqual(excluded, [
            {"sample_id": "s1", "reason": mc.EXCLUDED_NOT_REQUESTED}])

    def test_requested_sample_missing(self):
        with self.assertRaises(ValueError):
            mc.select_mock_samples(self.counts, composition=self.composition,
                                   requested=["other"])

    def test_unscored_asvs(self):
        asv_taxonomy = pd.Series({"asv1": "K;P;G1;S1", "asv2": "Unassigned"})
        table = mc.unscored_asvs(self.counts, asv_taxonomy, ["s1"])
        self.assertEqual(list(table["Feature ID"]), ["asv2", "asv3"])
        self.assertEqual(list(table["reason"]), [
            "empty or Unassigned in asv_taxonomy", "not in asv_taxonomy"])
        self.assertEqual(list(table["reads_in_mock_samples"]), [30, 20])
        self.assertEqual(list(table["reads_total"]), [30, 27])


class ReaderTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write(self, name, text):
        fp = os.path.join(self.tmp, name)
        with open(fp, "w") as fh:
            fh.write(text)
        return fp

    def test_read_composition(self):
        fp = self._write(
            "composition.tsv",
            "Taxonomy\tm1\tm2\n"
            "K;P;G1;S1\t2\t1\n"
            "K;P;G1;S1;NA\t2\t\n"
            "K;P;G2\t4\t1\n")
        table = mc.read_composition(fp)
        # duplicate lineages after normalization are summed, columns rescaled
        self.assertEqual(list(table.index), ["K;P;G1;S1", "K;P;G2"])
        self.assertAlmostEqual(table.loc["K;P;G1;S1", "m1"], 0.5)
        self.assertAlmostEqual(table.loc["K;P;G2", "m2"], 0.5)

    def test_read_composition_rejects_unassigned_rows(self):
        fp = self._write("c.tsv", "Taxonomy\tm1\nUnassigned\t1\n")
        with self.assertRaises(ValueError):
            mc.read_composition(fp)

    def test_read_taxonomy_requires_headers(self):
        fp = self._write(
            "tax.tsv",
            "Feature ID\tTaxon\tConfidence\n"
            "#q2:types\tcategorical\tnumeric\n"
            "asv1\tK;P;G1;S1\t0.9\n"
            "asv2\tUnassigned\t1\n")
        taxonomy = mc.read_taxonomy(fp)
        self.assertEqual(taxonomy["asv1"], "K;P;G1;S1")
        self.assertEqual(taxonomy["asv2"], "")
        bad = self._write("bad.tsv", "id\ttaxonomy\nasv1\tK\n")
        with self.assertRaises(ValueError):
            mc.read_taxonomy(bad)

    def test_read_feature_table_biom_convert_tsv(self):
        fp = self._write(
            "table.tsv",
            "# Constructed from biom file\n"
            "#OTU ID\ts1\ts2\n"
            "asv1\t10.0\t0.0\n"
            "asv2\t5\t3\n")
        table = mc.read_feature_table(fp)
        self.assertEqual(list(table.columns), ["s1", "s2"])
        self.assertEqual(table.loc["asv2", "s2"], 3)

    def test_read_reference_taxonomy(self):
        fp = self._write("ref.tsv", "r1\tK;P;G1;S1\nr2\tK;P;NA\n")
        taxonomy = mc.read_reference_taxonomy(fp)
        self.assertEqual(taxonomy["r2"], "K;P")


class BackboneTests(unittest.TestCase):

    def test_statuses(self):
        reference = ["K;P;G1;S1", "K;P;GX;S2", "K;P;G2;S4"]
        expected = ["K;P;G1;S1", "K;P;G1;S2", "K;P;G2", "K;P;G3;S5"]
        report, messages = mc.check_backbone(
            expected, reference, RANKS, ["g", "s"])
        status = {(r.rank, r.taxon): r.status for r in report.itertuples()}
        self.assertEqual(status, {
            ("g", "K;P;G1"): "found",
            ("g", "K;P;G2"): "found",
            ("g", "K;P;G3"): "not_in_database",
            ("s", "K;P;G1;S1"): "found",
            ("s", "K;P;G1;S2"): "different_lineage",
            ("s", "K;P;G3;S5"): "not_in_database",
        })
        row = report[report["taxon"] == "K;P;G1;S2"].iloc[0]
        self.assertEqual(row["database_lineages"], "K;P;GX;S2")
        self.assertEqual(messages, [])

    def test_prefix_and_depth_messages(self):
        reference = ["k__K;p__P;c__C;g__G;s__S"]
        _, messages = mc.check_backbone(["K;P;G;S"], reference, RANKS, ["s"])
        self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()
