#!/usr/bin/env python

"""Smoke tests for the evaluation metric plots: each builds a figure."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from tax_credit.log_plotting import (  # noqa: E402
    method_parameter_sensitivity_heatmap_from_data_frame,
)
from tax_credit.plotting_functions import (  # noqa: E402
    faceted_boxplot_from_data_frame,
    heatmap_from_data_frame,
    pointplot_from_data_frame,
    stacked_classification_barplot_from_data_frame,
    stacked_classification_panels_from_data_frames,
)


def _per_level_table():
    rows = []
    for dataset in ("db1", "db2"):
        for method, params in (("naive-bayes", "p1"), ("bt2-blca", "p2")):
            for iteration in ("0", "1"):
                for level, rank in ((5, "genus"), (6, "species")):
                    rows.append({
                        "Dataset": dataset, "Method": method, "Parameters": params,
                        "iteration": iteration, "level": level, "rank": rank,
                        "F-measure": 0.02 + 0.001 * level,
                        "match_ratio": 0.5, "underclassification_ratio": 0.3,
                        "overclassification_ratio": 0.1,
                        "misclassification_ratio": 0.1,
                    })
    return pd.DataFrame(rows)


class MetricPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_pointplot_zooms_small_values(self):
        fig = pointplot_from_data_frame(
            _per_level_table(), x="rank", metric="F-measure",
            x_order=["genus", "species"], title="t",
        )
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.legends), 1)
        self.assertLess(fig.axes[0].get_ylim()[1], 1.0)
        self.assertIn("zoomed", fig.axes[0].get_ylabel())

    def test_faceted_boxplot(self):
        fig = faceted_boxplot_from_data_frame(
            _per_level_table(), x="Dataset", metric="match_ratio",
            col="rank", col_order=["genus", "species"],
        )
        self.assertIsInstance(fig, Figure)
        self.assertEqual([ax.get_title() for ax in fig.axes], ["genus", "species"])
        self.assertEqual(fig.axes[0].get_ylim(), (0.0, 1.0))

    def test_heatmap_annotates_small_grids(self):
        fig = heatmap_from_data_frame(
            _per_level_table(), metric="misclassification_ratio",
            rows=["Method", "Parameters"], cols=["Dataset", "rank"],
        )
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes[0].texts), 8)  # one value per cell

    def test_stacked_barplot_grid(self):
        ratios = _per_level_table()
        fig = stacked_classification_barplot_from_data_frame(
            ratios[~((ratios["Dataset"] == "db2") & (ratios["Method"] == "bt2-blca"))],
        )
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 4)  # 2 runs x 2 datasets
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        self.assertIn("no data", texts)

    def test_panels_keep_row_labels(self):
        run = _per_level_table().query("Dataset == 'db1' and iteration == '0' and Method == 'bt2-blca'")
        fig = stacked_classification_panels_from_data_frames(
            [("a", run), ("b", None)], ncols=2, row_labels=["db1"],
        )
        self.assertEqual(fig.axes[0].get_ylabel(), "db1\nratio")

    def test_sensitivity_heatmap_panels_per_dataset(self):
        index = pd.MultiIndex.from_tuples(
            [("db1", "A;B;Sp1"), ("db1", "A;B;Sp2"), ("db2", "A;B;Sp3")],
            names=["dataset", "expected_taxonomy"],
        )
        pivot = pd.DataFrame(
            {
                "db1 / naive-bayes / p1": [0.5, None, None],
                "db1 / bt2-blca / p2": [0.1, 0.2, None],
                "db2 / naive-bayes / p1": [None, None, 0.9],
            },
            index=index,
        )
        fig = method_parameter_sensitivity_heatmap_from_data_frame(pivot, title="t")
        heatmaps = [ax for ax in fig.axes if ax.get_title() in ("db1", "db2")]
        self.assertEqual(len(heatmaps), 2)
        self.assertEqual(
            [t.get_text() for t in heatmaps[0].get_xticklabels()],
            ["naive-bayes · p1", "bt2-blca · p2"],
        )
        self.assertEqual(
            [t.get_text() for t in heatmaps[0].get_yticklabels()], ["Sp1", "Sp2"],
        )


if __name__ == "__main__":
    unittest.main()
