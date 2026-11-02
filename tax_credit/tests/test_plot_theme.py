#!/usr/bin/env python

import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib import rcParams  # noqa: E402

from tax_credit.plot_theme import (  # noqa: E402
    FONT_FAMILY,
    METHOD_COLORS,
    apply_tax_credit_theme,
    eval_method_label,
    method_palette,
    metric_cmap,
    metric_label,
    metric_limits,
)


class PlotThemeTests(unittest.TestCase):
    def test_metric_limits_full_range(self):
        self.assertEqual(metric_limits([0.4, 0.6]), (0.0, 1.0, False))
        self.assertEqual(metric_limits([0.0, 0.0]), (0.0, 1.0, False))
        self.assertEqual(metric_limits([1.0, 1.0]), (0.0, 1.0, False))
        self.assertEqual(metric_limits([np.nan]), (0.0, 1.0, False))

    def test_metric_limits_zooms_near_zero_and_one(self):
        low, high, zoomed = metric_limits([0.01, 0.03, np.nan])
        self.assertTrue(zoomed)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.04)

        low, high, zoomed = metric_limits([0.97, 0.99])
        self.assertTrue(zoomed)
        self.assertAlmostEqual(low, 0.96)
        self.assertEqual(high, 1.0)

    def test_metric_cmap(self):
        self.assertEqual(metric_cmap("F-measure"), "mako_r")
        self.assertEqual(metric_cmap("misclassification_ratio"), "rocket_r")

    def test_labels(self):
        self.assertEqual(metric_label("misclassification_ratio"), "Misclassification ratio")
        self.assertEqual(metric_label("F-measure"), "F-measure")
        self.assertEqual(eval_method_label("novel-taxa"), "Novel taxa")

    def test_method_palette(self):
        palette = method_palette(["naive-bayes", "new-method", "naive-bayes"])
        self.assertEqual(palette["naive-bayes"], METHOD_COLORS["naive-bayes"])
        self.assertNotIn(palette["new-method"], METHOD_COLORS.values())

        overridden = method_palette(["naive-bayes"], {"naive-bayes": "#123456"})
        self.assertEqual(overridden["naive-bayes"], "#123456")

        named = method_palette(["a", "b"], "Set2")
        self.assertEqual(sorted(named), ["a", "b"])

    def test_theme_embeds_truetype_fonts(self):
        apply_tax_credit_theme()
        self.assertEqual(rcParams["pdf.fonttype"], 42)
        self.assertEqual(rcParams["font.sans-serif"][: len(FONT_FAMILY)], FONT_FAMILY)
        self.assertTrue(rcParams["figure.constrained_layout.use"])


if __name__ == "__main__":
    unittest.main()
