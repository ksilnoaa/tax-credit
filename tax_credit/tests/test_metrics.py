#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main

from numpy import isnan

from tax_credit.metrics import (
    CLASSIFICATION_METRIC_COLUMNS,
    RATIO_COLUMNS,
    metrics_from_outcomes,
)


class MetricDefinitionTests(TestCase):
    """The table of metric definitions, asserted directly."""

    def test_definitions_one_of_each_outcome(self):
        # 4 records, one of each outcome: every count is unambiguous
        m = metrics_from_outcomes(match=1, underclassification=1,
                                  overclassification=1, misclassification=1)
        self.assertEqual(m['n_scored'], 4)
        self.assertEqual(m['TP'], 1)                     # match
        self.assertEqual(m['FP'], 2)                     # over + mis
        self.assertEqual(m['FN'], 2)                     # under + mis
        self.assertEqual(m['FN_original'], 3)            # under + over + mis
        self.assertAlmostEqual(m['Precision'], 1 / 3)    # tp / (tp + fp)
        self.assertAlmostEqual(m['Recall'], 1 / 3)       # tp / (tp + fn)
        self.assertAlmostEqual(m['Recall_original'], 1 / 4)
        self.assertAlmostEqual(m['F-measure'], 1 / 3)
        self.assertAlmostEqual(m['F-measure_original'], 2 / 7)
        for column in RATIO_COLUMNS:
            self.assertAlmostEqual(m[column], 0.25)

    def test_overclassification_costs_precision_but_not_recall(self):
        # the case that separates the new scoring from the original
        m = metrics_from_outcomes(match=3, overclassification=1)
        self.assertEqual(m['FN'], 0)
        self.assertEqual(m['FN_original'], 1)
        self.assertAlmostEqual(m['Precision'], 0.75)
        self.assertEqual(m['Recall'], 1.0)               # forgiven
        self.assertAlmostEqual(m['Recall_original'], 0.75)
        self.assertAlmostEqual(m['F-measure'], 2 * 0.75 / 1.75)
        self.assertAlmostEqual(m['F-measure_original'], 0.75)

    def test_underclassification_costs_recall_but_not_precision(self):
        m = metrics_from_outcomes(match=3, underclassification=1)
        self.assertEqual(m['FP'], 0)
        self.assertEqual(m['Precision'], 1.0)
        self.assertAlmostEqual(m['Recall'], 0.75)
        self.assertAlmostEqual(m['Recall_original'], 0.75)

    def test_weights_follow_the_same_formulas(self):
        # tallies may be read counts; scaling every tally cannot change a score
        unweighted = metrics_from_outcomes(2, 5, 1, 1)
        weighted = metrics_from_outcomes(20., 50., 10., 10.)
        for column in CLASSIFICATION_METRIC_COLUMNS:
            if column.startswith('n_') or column in ('TP', 'FP', 'FN',
                                                     'FN_original'):
                continue
            self.assertAlmostEqual(weighted[column], unweighted[column],
                                   msg=column)
        self.assertEqual(weighted['TP'], 20.)

    def test_every_metric_column_is_reported(self):
        m = metrics_from_outcomes(1, 1, 1, 1)
        self.assertEqual(sorted(m), sorted(CLASSIFICATION_METRIC_COLUMNS))


class MetricIdentityTests(TestCase):
    """Relationships that downstream code may rely on, or must not."""

    tallies = (2, 5, 1, 3)

    def setUp(self):
        self.m = metrics_from_outcomes(*self.tallies)

    def test_recall_original_is_the_match_ratio(self):
        # this identity held for the old Recall and still holds for the original
        self.assertAlmostEqual(self.m['Recall_original'], self.m['match_ratio'])

    def test_recall_is_not_the_match_ratio(self):
        # ... and no longer holds for the new Recall, which rescales it by the
        # share of records that were not overclassified
        self.assertNotAlmostEqual(self.m['Recall'], self.m['match_ratio'])
        self.assertAlmostEqual(
            self.m['Recall'],
            self.m['match_ratio'] / (1 - self.m['overclassification_ratio']))

    def test_ratios_sum_to_one(self):
        self.assertAlmostEqual(sum(self.m[c] for c in RATIO_COLUMNS), 1.0)

    def test_original_false_negatives_account_for_every_record(self):
        self.assertEqual(self.m['TP'] + self.m['FN_original'],
                         self.m['n_scored'])

    def test_new_false_negatives_do_not(self):
        # Recall is deliberately not a microaveraged recall any more
        self.assertNotEqual(self.m['TP'] + self.m['FN'], self.m['n_scored'])


class MetricEdgeCaseTests(TestCase):
    """NaN means "no observations"; 0.0 means "observed and wrong"."""

    def assertNaN(self, value, label):
        self.assertTrue(isnan(value), f'{label} should be NaN, got {value}')

    def test_nothing_scored_is_all_nan(self):
        m = metrics_from_outcomes()
        self.assertEqual(m['n_scored'], 0)
        for column in ('Precision', 'Recall', 'Recall_original', 'F-measure',
                       'F-measure_original', *RATIO_COLUMNS):
            self.assertNaN(m[column], column)

    def test_wrong_but_observed_is_zero_not_nan(self):
        m = metrics_from_outcomes(underclassification=1, overclassification=1,
                                  misclassification=1)
        self.assertEqual(m['Precision'], 0.0)
        self.assertEqual(m['Recall'], 0.0)
        self.assertEqual(m['F-measure'], 0.0)

    def test_all_underclassification_leaves_precision_undefined(self):
        # nothing was predicted at this rank, so there is no precision to take
        m = metrics_from_outcomes(underclassification=5)
        self.assertNaN(m['Precision'], 'Precision')
        self.assertEqual(m['Recall'], 0.0)
        self.assertNaN(m['F-measure'], 'F-measure')

    def test_all_overclassification_leaves_new_recall_undefined(self):
        # the new FN forgives overclassification, so no record counts against
        # recall and its denominator is empty
        m = metrics_from_outcomes(overclassification=5)
        self.assertEqual(m['Precision'], 0.0)
        self.assertNaN(m['Recall'], 'Recall')
        self.assertNaN(m['F-measure'], 'F-measure')
        # the original scoring still has a denominator here
        self.assertEqual(m['Recall_original'], 0.0)
        self.assertEqual(m['F-measure_original'], 0.0)

    def test_all_match_is_one(self):
        m = metrics_from_outcomes(match=10)
        for column in ('Precision', 'Recall', 'Recall_original', 'F-measure',
                       'F-measure_original', 'match_ratio'):
            self.assertEqual(m[column], 1.0, column)


if __name__ == "__main__":
    main()
