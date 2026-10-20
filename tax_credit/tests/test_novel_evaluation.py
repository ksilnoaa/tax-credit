#!/usr/bin/env python

import unittest
from os import makedirs
from os.path import join
from tempfile import TemporaryDirectory

import pandas as pd

from tax_credit import framework_functions
from tax_credit import novel_evaluation
from tax_credit.novel_evaluation import (
    _truncate_taxonomy_at_level,
    extract_per_level_classification_ratios,
    extract_per_level_classification_ratios_by_fold,
    per_level_classification_ratios_from_log,
    select_best_runs,
)

# reference lineage with an unknown (NA) order above assigned lower ranks
NA_TAXON = "Eukaryota;Chordata;Actinopteri;NA;Lutjanidae;Lutjanus;Lutjanus griseus"


class NovelEvaluationModuleTests(unittest.TestCase):
    """Phase 2: novel/CV assignment evaluation lives in ``novel_evaluation``."""

    def test_reexports_match_module(self):
        self.assertIs(
            framework_functions.novel_taxa_classification_evaluation,
            novel_evaluation.novel_taxa_classification_evaluation,
        )
        self.assertIs(
            framework_functions.extract_per_level_accuracy,
            novel_evaluation.extract_per_level_accuracy,
        )

    def test_truncate_taxonomy_at_level_keeps_internal_na(self):
        self.assertEqual(
            _truncate_taxonomy_at_level(NA_TAXON, 4),
            "Eukaryota;Chordata;Actinopteri;NA;Lutjanidae",
        )
        # truncating onto the NA rank leaves it trailing, so it is dropped
        self.assertEqual(
            _truncate_taxonomy_at_level(NA_TAXON, 3),
            "Eukaryota;Chordata;Actinopteri",
        )
        self.assertEqual(
            _truncate_taxonomy_at_level("A;B;C;D;E;F;NA", 6), "A;B;C;D;E;F"
        )

    def test_per_level_ratios_with_internal_na(self):
        # Wrong genus under an NA order: the error belongs at level 5 (genus).
        # Dropping the NA rank would shift it up to level 4 (family).
        observed = "Eukaryota;Chordata;Actinopteri;NA;Lutjanidae;Ocyurus;Ocyurus chrysurus"
        header = ["dataset", "level", "iteration", "method", "parameters",
                  "observed_taxonomy", "expected_taxonomy", "result",
                  "mismatch_level"]
        row = ["db1", "6", "0", "bt2-blca", "pi0.8", observed, NA_TAXON,
               "misclassification", "5"]
        with TemporaryDirectory() as tmp:
            log_fp = join(tmp, "classification_accuracy_log.tsv")
            with open(log_fp, "w") as f:
                f.write("\t".join(header) + "\n" + "\t".join(row) + "\n")
            rows, _ = per_level_classification_ratios_from_log(log_fp)

        by_level = {r["level"]: r for r in rows}
        self.assertEqual(by_level[3]["match_ratio"], 1.0)
        self.assertEqual(by_level[4]["match_ratio"], 1.0)
        self.assertEqual(by_level[5]["misclassification_ratio"], 1.0)
        self.assertEqual(by_level[6]["misclassification_ratio"], 1.0)

    def test_extract_per_level_accuracy_parses_stringified_lists(self):
        # a summary read back from CSV holds its per-level scores as reprs
        df = pd.DataFrame([{
            "Dataset": "db1", "level": 6, "iteration": "0",
            "Method": "m", "Parameters": "p",
            "mismatch_level_list": "[0, 0, 0, 0, 0, 0, 1, 1]",
            "Precision": "[nan, 1, 1, 1, 1, 1, 0.5]",
            "Recall": "[nan, 1, 1, 1, 1, 1, 0.8]",
            "F-measure": "[nan, 1, 1, 1, 1, 1, 0.6]",
            "Recall_original": "[nan, 1, 1, 1, 1, 1, 0.4]",
            "F-measure_original": "[nan, 1, 1, 1, 1, 1, 0.3]",
        }])
        pla = novel_evaluation.extract_per_level_accuracy(df)
        self.assertEqual(list(pla["level"]), list(range(1, 7)))
        self.assertEqual(pla.iloc[5]["Recall"], 0.8)
        self.assertEqual(pla.iloc[5]["Recall_original"], 0.4)
        self.assertEqual(pla.iloc[5]["F-measure_original"], 0.3)

    def test_extract_per_level_accuracy_does_not_derive_match_ratio(self):
        # match_ratio used to be taken from Recall, which was only valid while
        # every non-match counted as a false negative
        df = pd.DataFrame([{
            "Dataset": "db1", "level": 6, "iteration": "0",
            "Method": "m", "Parameters": "p",
            "mismatch_level_list": "[0, 0, 0, 0, 0, 0, 1, 1]",
            "Precision": "[nan, 1, 1, 1, 1, 1, 1]",
            "Recall": "[nan, 1, 1, 1, 1, 1, 1]",
            "F-measure": "[nan, 1, 1, 1, 1, 1, 1]",
        }])
        pla = novel_evaluation.extract_per_level_accuracy(df)
        self.assertNotIn("match_ratio", pla.columns)

    def test_per_level_metrics_separate_new_and_original_scoring(self):
        # one match and one overclassification at species: the new Recall
        # forgives the overclassification, the original does not
        header = ["dataset", "level", "iteration", "method", "parameters",
                  "observed_taxonomy", "expected_taxonomy", "result",
                  "mismatch_level"]
        matched = "A;B;C;D;E;F;G"
        over_obs = "A;B;C;D;E;F;Z"
        over_exp = "A;B;C;D;E;F"
        rows = [
            ["db1", "6", "0", "m", "p", matched, matched, "match", "7"],
            ["db1", "6", "0", "m", "p", over_obs, over_exp,
             "overclassification", "6"],
        ]
        with TemporaryDirectory() as tmp:
            log_fp = join(tmp, "classification_accuracy_log.tsv")
            with open(log_fp, "w") as f:
                f.write("\t".join(header) + "\n")
                for row in rows:
                    f.write("\t".join(row) + "\n")
            level_rows, _ = per_level_classification_ratios_from_log(log_fp)

        species = {row["level"]: row for row in level_rows}[6]
        self.assertEqual(species["n_match"], 1)
        self.assertEqual(species["n_over"], 1)
        self.assertEqual(species["FN"], 0)
        self.assertEqual(species["FN_original"], 1)
        self.assertEqual(species["Recall"], 1.0)
        self.assertEqual(species["Recall_original"], 0.5)
        self.assertEqual(species["Precision"], 0.5)
        # match_ratio tracks the original scoring, not the new Recall
        self.assertEqual(species["match_ratio"], 0.5)

    def test_classification_ratios_keep_novel_levels(self):
        header = ["dataset", "level", "iteration", "method", "parameters",
                  "observed_taxonomy", "expected_taxonomy", "result",
                  "mismatch_level"]
        # L5 read matches; L6 read is misclassified at class
        logs = {
            "db1-L5-iter0": ("A;B;C;D;E", "A;B;C;D;E", "match", "5"),
            "db1-L6-iter0": ("A;B;X;D;E;F", "A;B;C;D;E;F", "misclassification", "2"),
        }
        with TemporaryDirectory() as tmp:
            dirs = []
            for dataset_id, (obs, exp, result, mismatch) in logs.items():
                results_dir = join(tmp, dataset_id, "method1", "param1")
                makedirs(results_dir)
                row = ["db1", dataset_id[5], "0", "method1", "param1",
                       obs, exp, result, mismatch]
                with open(join(results_dir, "classification_accuracy_log.tsv"), "w") as f:
                    f.write("\t".join(header) + "\n" + "\t".join(row) + "\n")
                dirs.append(results_dir)

            by_fold = extract_per_level_classification_ratios_by_fold(dirs)
            averaged = extract_per_level_classification_ratios(dirs)

        self.assertEqual(set(by_fold["iteration"]), {"0"})
        self.assertEqual(sorted(averaged["novel_level"].unique()), [5, 6])
        class_level = averaged[averaged["level"] == 2].set_index("novel_level")
        self.assertEqual(class_level.loc[5, "match_ratio"], 1.0)
        self.assertEqual(class_level.loc[6, "misclassification_ratio"], 1.0)

    def test_select_best_runs(self):
        df = pd.DataFrame([
            {"Dataset": "db1", "Method": "a", "Parameters": "p1",
             "F-measure": 0.5, "misclassification_ratio": 0.1},
            {"Dataset": "db1", "Method": "a", "Parameters": "p1",
             "F-measure": 0.7, "misclassification_ratio": 0.3},
            {"Dataset": "db1", "Method": "b", "Parameters": "p1",
             "F-measure": 0.4, "misclassification_ratio": 0.2},
            {"Dataset": "db1", "Method": "b", "Parameters": "p2",
             "F-measure": 0.65, "misclassification_ratio": 0.2},
            {"Dataset": "db2", "Method": "b", "Parameters": "p1",
             "F-measure": 0.9, "misclassification_ratio": 0.0},
        ])
        best = select_best_runs(df, ["F-measure", "misclassification_ratio"])
        db1 = best[best["Dataset"] == "db1"].set_index("metric")

        # highest mean F-measure: b/p2 (0.65) beats a/p1 averaged over 2 folds (0.6)
        self.assertEqual(tuple(db1.loc["F-measure", ["Method", "Parameters"]]), ("b", "p2"))
        self.assertEqual(db1.loc["F-measure", "direction"], "highest")
        self.assertEqual(db1.loc["F-measure", "n_tied"], 1)
        # lowest misclassification: all three runs average 0.2; first sorted run wins
        self.assertEqual(tuple(db1.loc["misclassification_ratio", ["Method", "Parameters"]]), ("a", "p1"))
        self.assertEqual(db1.loc["misclassification_ratio", "direction"], "lowest")
        self.assertEqual(db1.loc["misclassification_ratio", "n_folds"], 2)
        self.assertEqual(db1.loc["misclassification_ratio", "n_tied"], 3)
        self.assertEqual(len(best[best["Dataset"] == "db2"]), 2)


if __name__ == '__main__':
    unittest.main()
