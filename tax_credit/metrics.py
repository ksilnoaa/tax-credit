#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Definitions of the per-sequence classification metrics.

Every metric tax-credit reports about individual sequence assignments is a
function of four tallies -- match, underclassification, overclassification and
misclassification -- so the arithmetic lives here and nowhere else. This module
deliberately imports nothing from the rest of the package, so both
``framework_functions`` and ``novel_evaluation`` can use it.
"""

from numpy import isnan, nan


# Classification outcomes from evaluate_classification, in stack order: correct
# first, then increasingly serious errors.
CLASSIFICATION_OUTCOMES = ('match', 'underclassification',
                           'overclassification', 'misclassification')

OUTCOME_COUNT_COLUMNS = ['n_scored', 'n_match', 'n_under', 'n_over', 'n_mis']
SCORE_COUNT_COLUMNS = ['TP', 'FP', 'FN', 'FN_original']
SCORE_COLUMNS = ['Precision', 'Recall', 'Recall_original',
                 'F-measure', 'F-measure_original']
RATIO_COLUMNS = ['match_ratio', 'underclassification_ratio',
                 'overclassification_ratio', 'misclassification_ratio']

# Every key returned by metrics_from_outcomes, in output order.
CLASSIFICATION_METRIC_COLUMNS = (OUTCOME_COUNT_COLUMNS + SCORE_COUNT_COLUMNS
                                 + SCORE_COLUMNS + RATIO_COLUMNS)

# Audit-only columns: reported so the original scoring stays visible, but never
# plotted or used to rank runs.
ORIGINAL_SCORE_COLUMNS = frozenset({'Recall_original', 'F-measure_original'})


def _ratio(numerator, denominator):
    '''Fraction, or NaN when the denominator holds no observations.'''
    if denominator <= 0:
        return nan
    return numerator / denominator


def _fmeasure(precision, recall):
    '''Harmonic mean of precision and recall.

    NaN if either is undefined, 0.0 if both are a real zero.
    '''
    if isnan(precision) or isnan(recall):
        return nan
    if precision + recall <= 0:
        return 0.
    return 2. * precision * recall / (precision + recall)


def metrics_from_outcomes(match=0., underclassification=0.,
                          overclassification=0., misclassification=0.):
    '''Scores and counts for one set of tallied classification outcomes.

    Every per-sequence metric tax-credit reports is a function of these four
    tallies, so this is the only place the definitions live. The tallies may be
    read counts or other weights as well as sequence counts; the formulas are
    the same either way.

        TP          = match
        FP          = overclassification + misclassification
        FN          = underclassification + misclassification
        FN_original = underclassification + overclassification
                      + misclassification

    ``FN`` forgives overclassification: assigning deeper than the reference
    knows does not count against sensitivity. ``FN_original`` is the
    microaveraged count upstream tax-credit used, where every non-match costs a
    false negative; it feeds ``Recall_original`` and ``F-measure_original``.
    Overclassification still counts in ``FP``, so ``Precision`` penalises it
    and ``F-measure`` therefore does too, half as heavily as before.

    Two consequences worth knowing. ``TP + FN`` is no longer the number of
    records scored, so ``Recall`` is not a microaveraged recall --
    ``Recall_original`` is. And ``Recall`` no longer equals ``match_ratio``;
    the two differ by exactly the overclassification rate.

    A score is NaN when its denominator holds no observations at all, and 0.0
    when there were observations but no true positives, so "nothing was scored"
    stays distinguishable from "everything was wrong". ``F-measure`` is 0.0
    when precision and recall are both a real zero.
    '''
    n_scored = (match + underclassification + overclassification
                + misclassification)
    tp = match
    fp = overclassification + misclassification
    fn = underclassification + misclassification
    fn_org = underclassification + overclassification + misclassification

    p = _ratio(tp, tp + fp)
    r = _ratio(tp, tp + fn)
    r_org = _ratio(tp, tp + fn_org)

    metrics = {
        'n_scored': n_scored,
        'n_match': match,
        'n_under': underclassification,
        'n_over': overclassification,
        'n_mis': misclassification,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'FN_original': fn_org,
        'Precision': p,
        'Recall': r,
        'Recall_original': r_org,
        'F-measure': _fmeasure(p, r),
        'F-measure_original': _fmeasure(p, r_org),
    }
    outcome_counts = (match, underclassification, overclassification,
                      misclassification)
    for column, count in zip(RATIO_COLUMNS, outcome_counts):
        metrics[column] = _ratio(count, n_scored)
    return metrics
