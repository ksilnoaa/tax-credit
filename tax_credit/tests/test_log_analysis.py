#!/usr/bin/env python

from unittest import TestCase

import pandas as pd

from tax_credit.log_analysis import (
    RANK_TO_LEVEL,
    _rank_at_depth,
    expected_rank_name,
    filter_sensitivity_to_rank,
    load_classification_accuracy_logs,
    select_top_sensitivity_taxa,
    summarize_confusion_pairs,
    summarize_method_parameter_sensitivity,
    summarize_taxon_errors,
    taxonomy_depth,
    truncate_to_rank,
)

# reference lineage with an unknown (NA) order above assigned lower ranks
NA_TAXON = "Eukaryota;Chordata;Actinopteri;NA;Lutjanidae;Lutjanus;Lutjanus griseus"


class LogAnalysisTests(TestCase):
    def _sample_df(self):
        return pd.DataFrame([
            {
                "dataset": "db1",
                "level": 6,
                "iteration": 0,
                "method": "consensus-vsearch",
                "parameters": "pi0.8",
                "observed_taxonomy": "A;B;C;D;E;F",
                "expected_taxonomy": "A;B;C;D;E;F;G",
                "result": "underclassification",
                "mismatch_level": 6,
            },
            {
                "dataset": "db1",
                "level": 6,
                "iteration": 0,
                "method": "consensus-vsearch",
                "parameters": "pi0.8",
                "observed_taxonomy": "A;B;C;D;E;H",
                "expected_taxonomy": "A;B;C;D;E;F;G",
                "result": "misclassification",
                "mismatch_level": 5,
            },
            {
                "dataset": "db1",
                "level": 6,
                "iteration": 0,
                "method": "consensus-vsearch",
                "parameters": "pi0.8",
                "observed_taxonomy": "A;B;C;D;E;F;G",
                "expected_taxonomy": "A;B;C;D;E;F;G",
                "result": "match",
                "mismatch_level": 7,
            },
        ])

    def test_truncate_to_rank(self):
        taxon = "A;B;C;D;E;F;G"
        self.assertEqual(truncate_to_rank(taxon, "genus"), "A;B;C;D;E;F")
        self.assertEqual(truncate_to_rank(taxon, "species"), taxon)

    def test_truncate_to_rank_keeps_internal_na(self):
        self.assertEqual(truncate_to_rank(NA_TAXON, "species"), NA_TAXON)
        self.assertEqual(
            truncate_to_rank(NA_TAXON, "genus"),
            "Eukaryota;Chordata;Actinopteri;NA;Lutjanidae;Lutjanus",
        )
        self.assertEqual(
            truncate_to_rank(NA_TAXON, "family"),
            "Eukaryota;Chordata;Actinopteri;NA;Lutjanidae",
        )

    def test_truncate_to_rank_strips_trailing_na(self):
        self.assertEqual(truncate_to_rank("A;B;C;D;E;F;NA", "species"), "A;B;C;D;E;F")
        self.assertEqual(truncate_to_rank("A;B;C;D;E;NA;NA", "genus"), "A;B;C;D;E")
        self.assertEqual(truncate_to_rank("A;B;C;D;E;;", "species"), "A;B;C;D;E")

    def test_rank_at_depth_keeps_internal_na(self):
        self.assertEqual(_rank_at_depth(NA_TAXON, 3), "NA")
        self.assertEqual(_rank_at_depth(NA_TAXON, 4), "Lutjanidae")
        self.assertEqual(_rank_at_depth(NA_TAXON, 5), "Lutjanus")
        self.assertEqual(_rank_at_depth("A;B;C;D;E;F;NA", 6), "")

    def test_taxonomy_depth_na(self):
        self.assertEqual(taxonomy_depth(NA_TAXON), 7)
        self.assertEqual(taxonomy_depth("A;B;C;D;E;F;NA"), 6)
        self.assertEqual(taxonomy_depth("NA"), 0)
        self.assertEqual(taxonomy_depth("Unassigned"), 0)

    def test_summaries_keep_internal_na(self):
        df = pd.DataFrame([{
            "dataset": "db1",
            "level": 6,
            "iteration": 0,
            "method": "bt2-blca",
            "parameters": "pi0.8",
            "observed_taxonomy":
                "Eukaryota;Chordata;Actinopteri;NA;Lutjanidae;Lutjanus;Lutjanus analis",
            "expected_taxonomy": NA_TAXON,
            "result": "misclassification",
            "mismatch_level": 6,
        }])
        df["expected_depth"] = df["expected_taxonomy"].map(taxonomy_depth)
        df["observed_depth"] = df["observed_taxonomy"].map(taxonomy_depth)
        df["ranks_short"] = df["expected_depth"] - df["observed_depth"]

        summary = summarize_taxon_errors(df, rank="species", min_obvs=1)
        self.assertEqual(summary.iloc[0]["expected_taxonomy"], NA_TAXON)

        pairs = summarize_confusion_pairs(df, rank="species", min_count=1)
        self.assertEqual(pairs.iloc[0]["expected_taxonomy"], NA_TAXON)
        self.assertEqual(pairs.iloc[0]["expected_genus"], "Lutjanus")
        self.assertTrue(pairs.iloc[0]["same_genus"])
        self.assertTrue(pairs.iloc[0]["same_family"])

    def test_summarize_taxon_errors(self):
        df = self._sample_df()
        df["expected_depth"] = df["expected_taxonomy"].str.count(";") + 1
        df["observed_depth"] = df["observed_taxonomy"].str.count(";") + 1
        df["ranks_short"] = df["expected_depth"] - df["observed_depth"]
        summary = summarize_taxon_errors(df, rank="species", min_obvs=1)
        self.assertEqual(summary.iloc[0]["n_obvs"], 3)
        self.assertEqual(summary.iloc[0]["n_match"], 1)
        self.assertEqual(summary.iloc[0]["n_under"], 1)
        self.assertEqual(summary.iloc[0]["n_mis"], 1)

    def test_summarize_confusion_pairs(self):
        df = self._sample_df()
        pairs = summarize_confusion_pairs(df, rank="species", min_count=1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0]["count"], 1)

    def test_load_empty(self):
        self.assertTrue(load_classification_accuracy_logs([]).empty)

    def test_rank_to_level(self):
        self.assertEqual(RANK_TO_LEVEL["family"], 4)
        self.assertEqual(RANK_TO_LEVEL["genus"], 5)
        self.assertEqual(RANK_TO_LEVEL["species"], 6)
        self.assertEqual(truncate_to_rank("A;B;C;D;E;F;G", "order"), "A;B;C;D")

    def test_expected_rank_name(self):
        self.assertEqual(expected_rank_name("A;B;C;D;E;F;G"), "species")
        self.assertEqual(expected_rank_name("A;B;C;D;E;F;NA"), "genus")
        self.assertEqual(expected_rank_name(NA_TAXON), "species")
        self.assertEqual(expected_rank_name("Unassigned"), "unassigned")

    def test_sensitivity_keeps_rank_and_filters(self):
        df = self._sample_df()
        # a reference resolved only to genus
        df.loc[len(df)] = {
            "dataset": "db1", "level": 6, "iteration": 0,
            "method": "consensus-vsearch", "parameters": "pi0.8",
            "observed_taxonomy": "A;B;C;D;E;F",
            "expected_taxonomy": "A;B;C;D;E;F",
            "result": "match", "mismatch_level": 6,
        }
        df["expected_depth"] = df["expected_taxonomy"].map(taxonomy_depth)
        df["observed_depth"] = df["observed_taxonomy"].map(taxonomy_depth)
        df["ranks_short"] = df["expected_depth"] - df["observed_depth"]

        summary = summarize_taxon_errors(df, rank="species", min_obvs=1)
        self.assertEqual(
            dict(zip(summary["expected_taxonomy"], summary["expected_rank"])),
            {"A;B;C;D;E;F;G": "species", "A;B;C;D;E;F": "genus"},
        )

        pivot = summarize_method_parameter_sensitivity(df, rank="species", min_obvs=1)
        self.assertEqual(len(pivot), 2)
        at_species, n_excluded = filter_sensitivity_to_rank(pivot, "species")
        self.assertEqual(list(at_species.index), [("db1", "A;B;C;D;E;F;G")])
        self.assertEqual(n_excluded, 1)

    def test_select_top_sensitivity_taxa_per_dataset(self):
        index = pd.MultiIndex.from_tuples(
            [("db1", "A"), ("db1", "B"), ("db1", "C"), ("db2", "D"), ("db2", "E")],
            names=["dataset", "expected_taxonomy"],
        )
        pivot = pd.DataFrame(
            {
                "db1 / m / p": [0.2, 0.9, None, None, None],
                "db2 / m / p": [None, None, None, 0.1, 0.5],
            },
            index=index,
        )
        top = select_top_sensitivity_taxa(pivot, top_n=2)
        # worst taxon first within each dataset; C has no values and is dropped
        self.assertEqual(
            list(top.index),
            [("db1", "B"), ("db1", "A"), ("db2", "E"), ("db2", "D")],
        )
