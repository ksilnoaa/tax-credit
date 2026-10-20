#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Shared style, colours, labels and axis limits for tax-credit plots."""

import numpy as np
import seaborn as sns

from tax_credit.novel_evaluation import LOWER_IS_BETTER_METRICS

# Arial is a plain TrueType font on macOS and Windows, so PDFs keep editable
# text; DejaVu Sans ships with matplotlib and covers Linux.
FONT_FAMILY = ["Arial", "DejaVu Sans"]

# Okabe & Ito (2008) colourblind-safe palette.
OKABE_ITO = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
)

# Fixed colour per classify method, so a method looks the same in every plot.
METHOD_COLORS = {
    "naive-bayes": "#0072B2",
    "consensus-vsearch": "#E69F00",
    "bt2-blca": "#CC79A7",
    "consensus-blast": "#009E73",
}

# Stack order for classification ratios: correct at the bottom, then
# increasingly serious errors.
RATIO_STACK_ORDER = (
    "match_ratio",
    "underclassification_ratio",
    "overclassification_ratio",
    "misclassification_ratio",
)

CLASSIFICATION_RATIO_COLORS = {
    "match_ratio": "#009E73",
    "underclassification_ratio": "#56B4E9",
    "overclassification_ratio": "#F0E442",
    "misclassification_ratio": "#D55E00",
}

_METRIC_LABELS = {
    "match_ratio": "Match ratio",
    "underclassification_ratio": "Underclassification ratio",
    "overclassification_ratio": "Overclassification ratio",
    "misclassification_ratio": "Misclassification ratio",
    "Bray-Curtis": "Bray-Curtis dissimilarity",
    "Taxon Accuracy Rate": "Taxon accuracy rate (precision_pa)",
    "Taxon Detection Rate": "Taxon detection rate (recall_pa)",
    "Recall_original": "Recall (original scoring)",
    "F-measure_original": "F-measure (original scoring)",
    "ASV Recall_original": "ASV recall (original scoring)",
    "ASV F-measure_original": "ASV F-measure (original scoring)",
}

# Alternative spellings accepted in config files. The mock-community taxon
# rates are a precision and a recall over presence/absence of taxa, so both
# names refer to the same column.
METRIC_ALIASES = {
    "precision_pa": "Taxon Accuracy Rate",
    "recall_pa": "Taxon Detection Rate",
    "TAR": "Taxon Accuracy Rate",
    "TDR": "Taxon Detection Rate",
}

# Reported for audit, never plotted or used to rank runs: the original scoring
# and the raw tallies behind the scores.
NON_PLOT_METRICS = frozenset({
    "Recall_original",
    "F-measure_original",
    "ASV Recall_original",
    "ASV F-measure_original",
    "TP", "FP", "FN", "FN_original",
    "n_scored", "n_match", "n_under", "n_over", "n_mis",
    "tp_pa", "fp_pa", "fn_pa", "tot_val_exp", "tot_val_obs",
    "n_asvs_scored", "reads_scored_fraction",
})

_EVAL_METHOD_LABELS = {
    "cross-validated": "Cross-validated",
    "cross-validated-taxa": "Cross-validated",
    "cross-validated-trad": "Cross-validated (traditional)",
    "novel-taxa": "Novel taxa",
    "self-validated": "Self-validated",
    "mock-community": "Mock community",
}

_THEME_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 9,
    "legend.frameon": False,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.6,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    # embed TrueType so PDF text stays editable instead of outlined paths
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_tax_credit_theme():
    """Apply the shared tax-credit plot style. Safe to call repeatedly."""
    sns.set_theme(context="paper", style="ticks", rc=_THEME_RC)


def resolve_metric(metric):
    """Canonical column name for a metric, resolving config aliases.

    ``precision_pa`` and ``recall_pa`` name the mock-community taxon accuracy
    and detection rates; every other name is returned unchanged.
    """
    return METRIC_ALIASES.get(str(metric).strip(), metric)


def plottable_metrics(metrics, available=None):
    """Metrics to plot: aliases resolved, audit-only columns dropped.

    *metrics* is what the config asked for, in order. *available*, when given,
    restricts the result to columns that exist. Duplicates introduced by
    aliasing are collapsed, keeping the first occurrence.
    """
    resolved, seen = [], set()
    for metric in metrics:
        metric = resolve_metric(metric)
        if metric in NON_PLOT_METRICS or metric in seen:
            continue
        if available is not None and metric not in available:
            continue
        seen.add(metric)
        resolved.append(metric)
    return resolved


def metric_label(metric):
    """Readable metric name, e.g. ``misclassification_ratio`` -> ``Misclassification ratio``."""
    return _METRIC_LABELS.get(resolve_metric(metric), metric)


def ratio_label(ratio_col):
    """Short legend label for a classification ratio, e.g. ``match``."""
    return ratio_col.replace("_ratio", "")


def eval_method_label(eval_method):
    """Readable evaluation method name, e.g. ``novel-taxa`` -> ``Novel taxa``."""
    return _EVAL_METHOD_LABELS.get(eval_method, eval_method)


def method_palette(methods, override=None):
    """Colour for each method.

    Methods in ``METHOD_COLORS`` keep their colour; others take the unused
    Okabe-Ito colours in sorted order (cycling past four extra methods).
    *override* may be a dict (method -> colour, merged over those defaults) or
    a seaborn palette name (applied to the sorted methods instead).
    """
    methods = sorted(set(methods))
    if isinstance(override, str) and override:
        return dict(zip(methods, sns.color_palette(override, len(methods))))
    spare = [color for color in OKABE_ITO if color not in METHOD_COLORS.values()]
    palette, n_spare = {}, 0
    for method in methods:
        if method in METHOD_COLORS:
            palette[method] = METHOD_COLORS[method]
        else:
            palette[method] = spare[n_spare % len(spare)]
            n_spare += 1
    if isinstance(override, dict):
        palette.update({m: c for m, c in override.items() if m in palette})
    return palette


def metric_cmap(metric):
    """Sequential colormap where darker means more of *metric*.

    ``mako_r`` for scores (higher is better), ``rocket_r`` for the error ratios.
    """
    return "rocket_r" if metric in LOWER_IS_BETTER_METRICS else "mako_r"


def annotate_heatmap_cells(ax, values, cmap, vmin, vmax, fmt=".2f", fontsize=6.5):
    """Write each non-missing value on a heatmap drawn with ``seaborn.heatmap``.

    Use instead of seaborn's ``annot``, which labels only part of the grid with
    seaborn 0.12 and matplotlib 3.8. Text is white on dark cells.
    """
    import matplotlib as mpl

    colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for (row, col), value in np.ndenumerate(np.asarray(values, dtype=float)):
        if not np.isfinite(value):
            continue
        red, green, blue, _ = colormap(norm(value))
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        ax.text(
            col + 0.5, row + 0.5, format(value, fmt), ha="center", va="center",
            fontsize=fontsize, color="white" if luminance < 0.45 else "0.15",
        )


def _round_up(value):
    step = 10 ** np.floor(np.log10(value))
    return float(np.ceil(round(value / step, 9)) * step)


def metric_limits(values, zoom_fraction=0.25):
    """Axis or colour limits for a metric on a 0-1 scale.

    Uses 0-1 unless every value sits in the bottom (or top) *zoom_fraction* of
    that range; then the limits zoom to the data so small differences stay
    visible. Returns ``(low, high, zoomed)``.
    """
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0, False
    low, high = float(finite.min()), float(finite.max())
    if 0 < high <= zoom_fraction:
        return 0.0, min(_round_up(high * 1.1), 1.0), True
    if 1 - zoom_fraction <= low < 1:
        return max(1.0 - _round_up((1 - low) * 1.1), 0.0), 1.0, True
    return 0.0, 1.0, False
