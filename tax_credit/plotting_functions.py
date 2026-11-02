#!/usr/bin/env python


# ----------------------------------------------------------------------------
# Copyright (c) 2016--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import seaborn as sns
import numpy as np
from seaborn import violinplot, heatmap
import matplotlib.pyplot as plt
from scipy.stats import (kruskal,
                         linregress,
                         mannwhitneyu,
                         wilcoxon,
                         ttest_ind,
                         ttest_rel)
from statsmodels.sandbox.stats.multicomp import multipletests
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
from skbio.stats.distance import anosim
from biom import load_table
from glob import glob
from os.path import join, split
from itertools import combinations
from IPython.display import display, Markdown

from tax_credit.plot_theme import (
    CLASSIFICATION_RATIO_COLORS,
    RATIO_STACK_ORDER,
    annotate_heatmap_cells,
    apply_tax_credit_theme,
    method_palette,
    metric_cmap,
    metric_label,
    metric_limits,
    ratio_label,
)


def _rotate_long_labels(ax, labels, max_chars=4):
    """Rotate x tick labels when any label is longer than *max_chars*."""
    if len(labels) and max(len(str(label)) for label in labels) > max_chars:
        ax.tick_params(axis="x", labelrotation=35)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")


def _figure_legend(fig, axes, title=None):
    """Replace per-axes legends with one figure legend outside the right edge."""
    handles, labels = [], []
    for ax in np.ravel(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    if handles:
        fig.legend(handles, labels, title=title, loc="outside right upper")


def _join_label(key):
    if isinstance(key, tuple):
        return " · ".join(str(part) for part in key)
    return str(key)


def _add_group_separators(ax, index, axis):
    """Draw white lines between runs of the first level of a MultiIndex."""
    if getattr(index, "nlevels", 1) < 2:
        return
    first = index.get_level_values(0)
    draw = ax.axhline if axis == "y" else ax.axvline
    for pos in range(1, len(first)):
        if first[pos] != first[pos - 1]:
            draw(pos, color="white", linewidth=2)


def pointplot_from_data_frame(df, x, metric, hue="Method", col="Dataset",
                              x_order=None, col_order=None, palette=None,
                              x_label=None, title=None):
    """Mean *metric* by *x*, one line per *hue*, one panel per *col*.

    Error bars span the minimum to maximum of the rows behind each point (e.g.
    folds and parameter sets). The y axis zooms when all values are near 0 or
    1 (see ``metric_limits``). *palette* maps hue values to colours and
    defaults to ``method_palette``. Returns the figure.
    """
    apply_tax_credit_theme()
    if x_order is None:
        x_order = sorted(df[x].unique(), key=lambda v: (isinstance(v, str), v))
    x_order = list(x_order)
    col_order = list(col_order) if col_order is not None else sorted(df[col].unique())
    hue_order = sorted(df[hue].unique())
    palette = palette or method_palette(hue_order)
    low, high, zoomed = metric_limits(df[metric])

    panel_width = max(2.8, 0.45 * len(x_order) + 1.0)
    fig, axes = plt.subplots(
        1, len(col_order), figsize=(panel_width * len(col_order) + 1.8, 3.2),
        sharey=True, squeeze=False,
    )
    for ax, panel in zip(axes[0], col_order):
        sns.pointplot(
            data=df[df[col] == panel], x=x, y=metric, hue=hue, order=x_order,
            hue_order=hue_order, palette=palette, errorbar=("pi", 100),
            dodge=0.3, scale=0.7, errwidth=1.0, capsize=0.08, ax=ax,
        )
        ax.set_title(str(panel))
        ax.set_xlabel(x_label or x)
        ax.set_ylabel("")
        ax.set_ylim(low, high)
        _rotate_long_labels(ax, x_order)
    axes[0, 0].set_ylabel(metric_label(metric) + (" (zoomed axis)" if zoomed else ""))
    _figure_legend(fig, axes, title=hue)
    if title:
        fig.suptitle(title)
    return fig


def heatmap_from_data_frame(df, metric, rows=("Method", "Parameters"),
                            cols=("Dataset",), cmap=None, vmin=None, vmax=None,
                            annotate=None, title=None):
    """Heatmap of the mean *metric* for each *rows* x *cols* combination.

    *cmap* defaults to ``metric_cmap``; *vmin* / *vmax* default to
    ``metric_limits`` of the plotted values. Cells show values when *annotate*
    is true (default: 120 cells or fewer). White lines separate groups of the
    first row and column levels; grey cells have no data. Returns the figure.
    """
    apply_tax_credit_theme()
    rows, cols = list(rows), list(cols)
    # observed=True keeps ordered categorical columns (e.g. rank names) in order
    pivot = df.pivot_table(index=rows, columns=cols, values=metric, observed=True)
    n_rows, n_cols = max(len(pivot.index), 1), max(len(pivot.columns), 1)
    zoomed = False
    if vmin is None or vmax is None:
        low, high, zoomed = metric_limits(pivot.to_numpy())
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
    if annotate is None:
        annotate = n_rows * n_cols <= 120

    cmap = cmap or metric_cmap(metric)

    cell_w, cell_h = (0.5, 0.32) if annotate else (0.32, 0.22)
    fig, ax = plt.subplots(figsize=(n_cols * cell_w + 3.4, n_rows * cell_h + 2.0))
    heatmap(
        pivot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        xticklabels=[_join_label(column) for column in pivot.columns],
        yticklabels=[_join_label(row) for row in pivot.index],
        cbar_kws={
            "label": metric_label(metric) + (" (zoomed scale)" if zoomed else ""),
            "format": "%.2f",
            "shrink": 0.8,
        },
        ax=ax,
    )
    if annotate:
        annotate_heatmap_cells(ax, pivot.to_numpy(dtype=float), cmap, vmin, vmax)
    ax.grid(False)
    ax.set_facecolor("0.92")
    _add_group_separators(ax, pivot.index, axis="y")
    _add_group_separators(ax, pivot.columns, axis="x")
    ax.set_xlabel(" · ".join(cols))
    ax.set_ylabel(" · ".join(rows))
    ax.tick_params(length=0)
    # seaborn turns row labels vertical in tall cells; keep them readable
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="x", labelrotation=40)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    if title:
        ax.set_title(title)
    return fig


def stacked_classification_panels_from_data_frames(
    panels,
    ncols,
    level_col="level",
    ratio_cols=RATIO_STACK_ORDER,
    colors=None,
    level_labels=None,
    level_axis_label="taxonomic level",
    row_labels=None,
    col_titles=None,
    panel_size=(2.6, 2.6),
    title=None,
):
    """Grid of stacked barplots of classification ratios, one run per panel.

    panels: list of ``(panel_title, df)`` in row-major order, *ncols* per row.
        Each df holds one run's per-level ratios (*level_col* plus
        *ratio_cols*); one stacked bar is drawn per level. A ``None`` or empty
        df draws a "no data" panel so the grid stays aligned; a ``None`` title
        draws none.
    level_labels: optional mapping of level to tick label.
    row_labels / col_titles: optional label for each row (y axis of the first
        column) and title for each column (top row).
    panel_size: ``(width, height)`` of each panel in inches.

    Returns the figure.
    """
    from matplotlib.patches import Patch

    apply_tax_credit_theme()
    colors = colors or CLASSIFICATION_RATIO_COLORS
    ratio_cols = list(ratio_cols)
    ncols = max(1, min(ncols, len(panels) or 1))
    nrows = int(np.ceil(max(len(panels), 1) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size[0] * ncols + 1.6, panel_size[1] * nrows + 0.9),
        sharey=True, squeeze=False,
    )

    for idx, ax in enumerate(axes.flat):
        panel_title, panel_df = panels[idx] if idx < len(panels) else (None, None)
        if panel_title:
            ax.set_title(panel_title, fontsize=8)
        if panel_df is None or panel_df.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    va="center", color="0.5")
            ax.grid(False)
            ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        panel_df = panel_df.sort_values(level_col)
        levels = list(panel_df[level_col])
        x = np.arange(len(levels))
        bottoms = np.zeros(len(levels))
        for col in ratio_cols:
            heights = panel_df[col].to_numpy(dtype=float)
            ax.bar(x, heights, 0.8, bottom=bottoms, color=colors[col],
                   edgecolor="white", linewidth=0.4)
            bottoms += heights
        ax.set_xticks(
            x,
            [str(level_labels.get(level, level)) if level_labels
             else str(int(level)) for level in levels],
        )
        ax.set_ylim(0, 1)

    for row, ax in enumerate(axes[:, 0]):
        label = row_labels[row] if row_labels and row < len(row_labels) else None
        ax.set_ylabel(f"{label}\nratio" if label else "ratio")
    if col_titles:
        for ax, col_title in zip(axes[0], col_titles):
            existing = ax.get_title()
            ax.set_title(f"{col_title}\n{existing}" if existing else col_title)

    fig.legend(
        handles=[
            Patch(facecolor=colors[col], edgecolor="0.7", linewidth=0.5,
                  label=ratio_label(col))
            for col in reversed(ratio_cols)
        ],
        loc="outside right upper",
    )
    fig.supxlabel(level_axis_label, fontsize=9)
    if title:
        fig.suptitle(title)
    return fig


def stacked_classification_barplot_from_data_frame(
    df,
    run_cols=("Method", "Parameters"),
    col="Dataset",
    level_col="level",
    level_labels=None,
    level_axis_label="taxonomic level",
    title=None,
):
    """Stacked barplots of classification ratios by level for every run.

    One row per run (unique *run_cols* combination) and one column per value of
    *col* (e.g. reference database). Each panel stacks match, under-, over- and
    misclassification ratios for each level. Returns the figure.
    """
    run_cols = list(run_cols)
    runs = [
        tuple(run)
        for run in df[run_cols].drop_duplicates().sort_values(run_cols).to_numpy()
    ]
    columns = sorted(df[col].unique())
    panels = []
    for run in runs:
        run_df = df
        for run_col, value in zip(run_cols, run):
            run_df = run_df[run_df[run_col] == value]
        for column in columns:
            panels.append((None, run_df[run_df[col] == column]))
    return stacked_classification_panels_from_data_frames(
        panels,
        ncols=len(columns),
        level_col=level_col,
        level_labels=level_labels,
        level_axis_label=level_axis_label,
        row_labels=["\n".join(str(part) for part in run) for run in runs],
        col_titles=[str(column) for column in columns],
        panel_size=(2.6, 1.9),
        title=title,
    )


def _short_taxon_label(taxon, resolved):
    """Last named rank of a lineage, marked when not resolved at the plotted rank."""
    names = [name for name in str(taxon).split(";") if name and name != "NA"]
    name = names[-1] if names else str(taxon)
    return name if resolved else f"{name} (unresolved)"


def composition_barplot_from_data_frame(df, run_order, sample_col="SampleID",
                                        run_col="run", row_col=None,
                                        taxon_col="taxon",
                                        resolved_col="resolved",
                                        value_col="abundance", top_n=12,
                                        title=None):
    """Stacked relative-abundance bars: one column per sample, one bar per run.

    *df* has one row per sample, run and taxon. *run_order* lists the runs left
    to right (e.g. ``Expected`` then one best run per method). With *row_col*
    (e.g. the reference database) the figure gets one row of panels per value
    of that column, and *run_order* may be a dict mapping each row value to its
    own run list, since the best run differs per row.

    Up to *top_n* taxa get their own colour, chosen across the whole figure so
    every panel uses the same colours: first the taxa of the first run (by
    abundance), then those with the highest abundance in any other bar. The
    rest are pooled as ``Other``, and ``Unassigned`` is dark grey. Taxa are
    labelled by their last named rank. Returns the figure.
    """
    from matplotlib.patches import Patch

    apply_tax_credit_theme()
    df = df.copy()
    df["label"] = [
        "Unassigned" if taxon == "Unassigned" else _short_taxon_label(taxon, res)
        for taxon, res in zip(df[taxon_col], df[resolved_col])
    ]
    rows = list(dict.fromkeys(df[row_col])) if row_col else [None]
    if isinstance(run_order, dict):
        runs_by_row = {row: list(run_order.get(row, [])) for row in rows}
    else:
        runs_by_row = {row: list(run_order) for row in rows}
    first_runs = [runs[0] for runs in runs_by_row.values() if runs]

    first = (df[df[run_col].isin(first_runs)].groupby("label")[value_col]
             .max().sort_values(ascending=False))
    peak = df.groupby("label")[value_col].max().sort_values(ascending=False)
    ordered = list(dict.fromkeys(list(first.index) + list(peak.index)))
    named = [label for label in ordered if label != "Unassigned"][:top_n]
    df.loc[~df["label"].isin(named + ["Unassigned"]), "label"] = "Other"
    stack = named + [label for label in ("Other", "Unassigned")
                     if label in set(df["label"])]
    colors = dict(zip(named, sns.color_palette("tab20", max(len(named), 1))))
    colors.update({"Other": "0.82", "Unassigned": "0.45"})

    samples = list(dict.fromkeys(df[sample_col]))
    widest = max((len(runs) for runs in runs_by_row.values()), default=1)
    width = max(2.4, 0.45 * widest + 0.8)
    fig, axes = plt.subplots(
        len(rows), max(len(samples), 1),
        figsize=(width * max(len(samples), 1) + 2.6, 3.4 * len(rows)),
        sharey=True, squeeze=False,
    )
    for row_index, row in enumerate(rows):
        row_df = df[df[row_col] == row] if row_col else df
        runs = [run for run in runs_by_row[row] if run in set(row_df[run_col])]
        for ax, sample in zip(axes[row_index], samples):
            panel = row_df[row_df[sample_col] == sample]
            pivot = (panel.pivot_table(index=run_col, columns="label",
                                       values=value_col, aggfunc="sum")
                     .reindex(index=runs, columns=stack).fillna(0.0))
            x = np.arange(len(runs))
            bottoms = np.zeros(len(runs))
            for label in stack:
                heights = pivot[label].to_numpy(dtype=float)
                ax.bar(x, heights, 0.8, bottom=bottoms, color=colors[label],
                       edgecolor="white", linewidth=0.3)
                bottoms += heights
            ax.set_xticks(x, runs)
            _rotate_long_labels(ax, runs)
            ax.set_ylim(0, 1)
            if row_index == 0:
                ax.set_title(str(sample), fontsize=9)
        label = "relative abundance"
        axes[row_index, 0].set_ylabel(
            f"{row}\n{label}" if row_col else label)
    fig.legend(
        handles=[Patch(facecolor=colors[label], label=label)
                 for label in reversed(stack)],
        # centred, so a two-line suptitle has the top right corner to itself
        loc="outside right", fontsize=7,
    )
    if title:
        fig.suptitle(title)
    return fig


def boxplot_from_data_frame(df,
                            group_by="Method",
                            metric="Precision",
                            hue=None,
                            y_min=None,
                            y_max=None,
                            plotf=violinplot,
                            color=None,
                            color_palette=None,
                            label_rotation=45,
                            title=None):
    """Boxplot or violinplot of *metric* by *group_by* (and *hue*).

    To draw boxplots instead of violin plots, pass ``plotf=seaborn.boxplot``.
    *hue*, *color* and *color_palette* pass to the seaborn function. *y_min*
    and *y_max* default to matplotlib's autoscaling. Returns the figure.
    """
    apply_tax_credit_theme()
    n_groups = df[group_by].nunique()
    fig, ax = plt.subplots(figsize=(max(5, n_groups * 0.8 + 1.5), 4))
    plot_kwargs = {
        "x": group_by,
        "y": metric,
        "hue": hue,
        "data": df,
        "palette": color_palette,
        "order": sorted(df[group_by].unique()),
        "ax": ax,
    }
    if hue is None:
        plot_kwargs["color"] = color if color is not None else "0.6"
    plotf(**plot_kwargs)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_ylabel(metric_label(metric))
    ax.set_xlabel(group_by)
    if title:
        ax.set_title(title)
    for lab in ax.get_xticklabels():
        lab.set_rotation(label_rotation)
        lab.set_ha("right")
    if hue is not None and ax.get_legend() is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title=hue)
    return fig


def faceted_boxplot_from_data_frame(df, x, metric, hue="Method", col=None,
                                    col_order=None, palette=None, title=None):
    """Boxplots of *metric* by *x* and *hue*, one panel per value of *col*.

    Each underlying row (e.g. fold) is drawn as a point over its box. The y
    axis zooms when all values are near 0 or 1 (see ``metric_limits``).
    *palette* maps hue values to colours and defaults to ``method_palette``.
    Returns the figure.
    """
    apply_tax_credit_theme()
    col_order = list(col_order) if col_order is not None else sorted(df[col].unique())
    x_order = sorted(df[x].unique())
    hue_order = sorted(df[hue].unique())
    palette = palette or method_palette(hue_order)
    low, high, zoomed = metric_limits(df[metric])

    panel_width = max(2.6, 0.32 * len(x_order) * len(hue_order) + 1.0)
    fig, axes = plt.subplots(
        1, len(col_order), figsize=(panel_width * len(col_order) + 1.8, 3.4),
        sharey=True, squeeze=False,
    )
    for ax, panel in zip(axes[0], col_order):
        panel_df = df[df[col] == panel]
        sns.boxplot(
            data=panel_df, x=x, y=metric, hue=hue, order=x_order,
            hue_order=hue_order, palette=palette, fliersize=0, linewidth=0.8,
            saturation=1, boxprops={"alpha": 0.55}, ax=ax,
        )
        sns.stripplot(
            data=panel_df, x=x, y=metric, hue=hue, order=x_order,
            hue_order=hue_order, palette=palette, dodge=True, size=3.5,
            edgecolor="0.2", linewidth=0.5, legend=False, ax=ax,
        )
        ax.set_title(str(panel))
        ax.set_xlabel(x)
        ax.set_ylabel("")
        ax.set_ylim(low, high)
        _rotate_long_labels(ax, x_order, max_chars=16)
    axes[0, 0].set_ylabel(metric_label(metric) + (" (zoomed axis)" if zoomed else ""))
    _figure_legend(fig, axes, title=hue)
    if title:
        fig.suptitle(title)
    return fig


def calculate_linear_regress(df, x, y, group_by):
    '''Calculate slope, intercept from series of lines
    df: pandas.DataFrame
    x: str
        x axis variable
    y: str
        y axis variable
    group_by: str
        df variable to use for separating data subsets
    '''
    results = []
    for group in df[group_by].unique():
        df_mod = df[df[group_by] == group]
        slope, intercept, r_value, p_value, std_err = linregress(df_mod[x],
                                                                 df_mod[y])
        results.append((group, slope, intercept, r_value, p_value, std_err))
    result = pd.DataFrame(results, columns=[group_by, "Slope", "Intercept",
                                            "R", "P-val", "Std Error"])
    return result


def per_level_kruskal_wallis(df,
                             y_vars,
                             group_by,
                             dataset_col='Dataset',
                             level_name="level",
                             levelrange=range(1, 7),
                             alpha=0.05,
                             pval_correction='fdr_bh'):

    '''Test whether 2+ population medians are different.

    Due to the assumption that H has a chi square distribution, the number of
    samples in each group must not be too small. A typical rule is that each
    sample must have at least 5 measurements.

    df = pandas.DataFrame
    y_vars = LIST of variables (df column names) to test
    group_by = df variable to use for separating subgroups to compare
    dataset_col = df variable to use for separating individual datasets to test
    level_name = df variable name that specifies taxonomic level
    levelrange = range of taxonomic levels to test.
    alpha = level of alpha significance for test
    pval_correction = type of p-value correction to use
    '''
    dataset_list = []
    p_list = []
    for dataset in df[dataset_col].unique():
        df1 = df[df[dataset_col] == dataset]
        for var in y_vars:
            dataset_list.append((dataset, var))
            for level in levelrange:
                level_subset = df1[level_name] == level

                # group data by groups
                group_list = []
                for group in df1[group_by].unique():
                    group_data = df1[group_by] == group
                    group_results = df1[level_subset & group_data][var]
                    group_list.append(group_results)

                # kruskal-wallis tests
                try:
                    h_stat, p_val = kruskal(*group_list, nan_policy='omit')
                # default to p=1.0 if all values = 0
                # this is not technically correct, from the standpoint of p-val
                # correction below makes p-vals very slightly less significant
                # than they should be
                except ValueError:
                    h_stat, p_val = ('na', 1)  # noqa

                p_list.append(p_val)

    # correct p-values
    rej, pval_corr, alphas, alphab = multipletests(np.array(p_list),
                                                   alpha=alpha,
                                                   method=pval_correction)

    range_len = len([i for i in levelrange])
    results = [(dataset_list[i][0], dataset_list[i][1],
                *[pval_corr[i * range_len + n] for n in range(0, range_len)])
               for i in range(0, len(dataset_list))]
    result = pd.DataFrame(results, columns=[dataset_col, "Variable",
                                            *[n for n in levelrange]])
    return result


def seek_tables(expected_results_dir, table_fn='merged_table.biom'):
    '''Find and deliver merged biom tables'''
    table_fps = glob(join(expected_results_dir, '*', '*', table_fn))
    for table in table_fps:
        reference_dir, _ = split(table)
        dataset_dir, reference_id = split(reference_dir)
        _, dataset_id = split(dataset_dir)
        yield table, dataset_id, reference_id


def batch_beta_diversity(expected_results_dir, method="braycurtis",
                         permutations=99, col='method', dim=2,
                         colormap={'expected': 'red', 'rdp': 'seagreen',
                                   'sortmerna': 'gray', 'uclust': 'blue',
                                   'blast': 'purple'}):

    '''Find merged biom tables and run beta_diversity_through_plots'''
    for table, dataset_id, reference_id in seek_tables(expected_results_dir):
        display(Markdown('## {0} {1}'.format(dataset_id, reference_id)))
        s, r, pc, dm = beta_diversity_pcoa(table, method=method, col=col,
                                           permutations=permutations, dim=dim,
                                           colormap=colormap)
        plt.show()
        plt.clf()


def make_distance_matrix(biom_fp, method="braycurtis"):
    '''biom.Table --> skbio.DistanceMatrix'''
    table = load_table(biom_fp)

    # extract sample metadata from table, put in df
    table_md = {s_id: dict(table.metadata(s_id)) for s_id in table.ids()}
    s_md = pd.DataFrame.from_dict(table_md, orient='index')

    # extract data from table and multiply, assuming that table contains
    # relative abundances (which cause beta_diversity to fail)
    table_data = [[int(num * 100000) for num in table.data(s_id)]
                  for s_id in table.ids()]

    # beta diversity
    dm = beta_diversity(method, table_data, table.ids())

    return dm, s_md


def beta_diversity_pcoa(biom_fp, method="braycurtis", permutations=99, dim=2,
                        col='method', colormap={'expected': 'red',
                                                'rdp': 'seagreen',
                                                'sortmerna': 'gray',
                                                'uclust': 'blue',
                                                'blast': 'purple'}):

    '''From biom table, compute Bray-Curtis distance; generate PCoA plot;
    and calculate adonis differences.

    biom_fp: path
        Path to biom.Table containing sample metadata.
    method: str
        skbio.Diversity method to use for ordination.
    permutations: int
        Number of permutations to perform for anosim tests.
    dim: int
        Number of dimensions to plot. Currently supports only 2-3 dimensions.
    col: str
        metadata name to use for distinguishing groups for anosim tests and
        pcoa plots.
    colormap: dict
        map groups names (must be group names in col) to colors used for plots.
    '''

    dm, s_md = make_distance_matrix(biom_fp, method=method)

    # pcoa
    pc = pcoa(dm)

    # anosim tests
    results = anosim(dm, s_md, column=col, permutations=permutations)
    print('R = ', results['test statistic'], '; P = ', results['p-value'])

    if dim == 2:
        pc123 = pc.samples.loc[:, ["PC1", "PC2", "PC3"]]
        smd_merge = s_md.merge(pc123, left_index=True, right_index=True)
        smd_merge['Color'] = [colormap[x] for x in smd_merge['method']]
        title = smd_merge['reference'][0]
        labels = ['PC {0} ({1:.2f})'.format(d + 1, pc.proportion_explained[d])
                  for d in range(0, 2)]
        circle_plot_from_dataframe(smd_merge, "PC1", "PC2", title,
                                   columns=["method", "sample_id", "params"],
                                   color="Color", labels=labels)

    else:
        # skbio pcoa plots
        pcoa_plot_skbio(pc, s_md, col='method')

    return s_md, results, pc, dm


def circle_plot_from_dataframe(df, x, y, title=None, color="Color",
                               columns=None, labels=None, plot_width=400,
                               plot_height=400, fill_alpha=0.2, size=10,
                               output_fn=None):
    '''2D scatter from dataframe (matplotlib). Uses *color* column for face colors.

    df: pandas.DataFrame
        Containing all sample data, including color categories.
    x, y: str
        Column names for coordinates.
    title: str
        Figure title.
    color: str
        Column of matplotlib color names/hex for each point.
    columns: list or None
        Columns used for legend grouping (first column must exist in *df*).
    labels: list or None
        Axis labels for x and y; defaults to *x* and *y* column names.
    output_fn: path or None
        If set, save figure to this path (e.g. ``.png``).
    '''
    if columns is None:
        columns = ["method", "sample_id", "params"]
    if labels is None:
        labels = [x, y]

    legend_col = columns[0] if columns and columns[0] in df.columns else None

    # figsize in inches (~ pixel size / 100 for notebook-friendly scale)
    w_in = max(plot_width / 100.0, 3.0)
    h_in = max(plot_height / 100.0, 3.0)
    fig, ax = plt.subplots(figsize=(w_in, h_in))

    area = (size * 12) ** 2 / 100.0
    for _, row in df.iterrows():
        ax.scatter(
            row[x],
            row[y],
            color=row[color],
            alpha=fill_alpha,
            s=area,
            edgecolors="0.35",
            linewidths=0.4,
        )

    if legend_col is not None:
        seen = set()
        for _, row in df.iterrows():
            lab = row[legend_col]
            if lab in seen:
                continue
            seen.add(lab)
            ax.scatter(
                [],
                [],
                color=row[color],
                alpha=fill_alpha,
                s=area,
                edgecolors="0.35",
                linewidths=0.4,
                label=str(lab),
            )
        ax.legend(
            title=legend_col,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize="small",
        )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    if output_fn is not None:
        fig.savefig(output_fn, bbox_inches="tight", dpi=150)

    plt.show()
    plt.close(fig)


def pcoa_plot_skbio(pc, s_md, col='method'):
    '''Input principal coordinates, display figure.

    pc: skbio.OrdinationResults
        Sample coordinates.
    s_md: pandas.DataFrame
        Sample metadata.
    col: str
        Category in s_md to use for coloring groups.
    '''

    # make labels for PCoA plot
    pcl = ['PC {0} ({1:.2f})'.format(d + 1, pc.proportion_explained[d])
           for d in range(0, 3)]
    fig = pc.plot(s_md, col, axis_labels=(pcl[0], pcl[1], pcl[2]),
                  cmap='jet', s=50)
    fig


def average_distance_boxplots(expected_results_dir, group_by="method",
                              standard='expected', metric="distance",
                              params='params', beta="braycurtis",
                              reference_filter=True, reference_col='reference',
                              references=['gg_13_8_otus',
                                          'unite_20.11.2016_clean_fullITS'],
                              paired=True, use_best=True, parametric=True,
                              plotf=violinplot, label_rotation=45,
                              color_palette=None, y_min=0.0, y_max=1.0,
                              color=None, hue=None):

    '''Distance boxplots that aggregate and average results across multiple
    mock community datasets.

    reference_filter: bool
        Filter by reference dataset to only include specific references in
        results?
    reference_col: str
        df column header containing reference set information.
    references: list
        List of strings containing names of reference datasets to include.
    paired: bool
        Perform paired or unpaired comparisons?
    parametric: bool
        Perform parametric or non-parametric statistical tests?
    use_best: bool
        Compare average distance distributions across all methods (False) or
        only the best parameter configuration for each method? (True)
    '''
    box = dict()
    best = dict()
    # Aggregate all distance matrix data
    archive = pd.DataFrame()
    for table, dataset_id, reference_id in seek_tables(expected_results_dir):
        dm, sample_md = make_distance_matrix(table, method=beta)
        per_method = per_method_distance(dm, sample_md, group_by=group_by,
                                         standard=standard, metric=metric)
        archive = pd.concat([archive, per_method])

    # filter out auxiliary reference database results
    if reference_filter is True:
        archive = archive[archive[reference_col].isin(references)]

    # plot results for each reference db separately
    for reference in archive[reference_col].unique():
        display(Markdown('## {0}'.format(reference)))
        archive_subset = archive[archive[reference_col] == reference]

        # for each method find best average method/parameter config
        if use_best:
            best[reference], param_report = isolate_top_params(
                archive_subset, group_by, params, metric)
            # display(pd.DataFrame(param_report, columns=[group_by, params]))

            method_rank = _show_method_rank(
                best[reference], group_by, params, metric,
                [group_by, params, metric], ascending=False)

        else:
            best[reference] = archive_subset

        results = per_method_pairwise_tests(best[reference], group_by=group_by,
                                            metric=metric, paired=paired,
                                            parametric=parametric)

        box[reference] = boxplot_from_data_frame(
            best[reference], group_by=group_by, color=color, hue=hue,
            y_min=None, y_max=None, plotf=plotf, label_rotation=label_rotation,
            metric=metric, color_palette=color_palette)

        if use_best:
            _add_significance_to_boxplots(
                results, method_rank, box[reference].axes[0], method='method')

        plt.show()
        plt.clf()

        display(results)

    return box, best


def _add_significance_to_boxplots(pairwise, rankings, ax, method='Method'):

    x_labels = [a.get_text() for a in ax.get_xticklabels()]
    methods = [m for m in rankings[method]]

    ranks = []
    # iterate range instead of methods, so that we can use pop for comparisons
    # against shrinking list of methods
    methods_copy = methods.copy()
    for n in range(len(methods_copy)):
        method = methods_copy.pop(0)
        inner_rank = {method}
        for other_method in methods_copy:
            if method in pairwise.index.levels[0] and \
                    other_method in pairwise.loc[method].index:
                if pairwise.loc[method].loc[other_method]['FDR P'] > 0.05:
                    inner_rank.add(other_method)
            elif other_method in pairwise.index.levels[0] and \
                    method in pairwise.loc[other_method].index:
                if pairwise.loc[other_method].loc[method]['FDR P'] > 0.05:
                    inner_rank.add(other_method)
        # only add new set of equalities if it contains unique items
        if len(ranks) == 0 or not inner_rank.issubset(ranks[-1]):
            ranks.append(inner_rank)

    # provide unique letters for each significance group
    letters = 'abcdefghijklmnopqrstuvwxyz'

    sig_groups = {}
    for method in methods:
        sig_groups[method] = []
        for rank, letter in zip(ranks, letters):
            if method in rank:
                sig_groups[method].append(letter)
        sig_groups[method] = ''.join(sig_groups[method])

    # add significance labels above plot
    pos = range(len(x_labels))
    for tick, label in zip(pos, x_labels):
        ax.text(tick, ax.get_ybound()[1], sig_groups[label], size='medium',
                horizontalalignment='center', color='k', weight='semibold')

    return ax


def _show_method_rank(best, group_by, params, metric, display_fields,
                      ascending=False):
    '''Find the best param configuration for each method and show those
    configs, along with the parameters and metric scores.
    '''
    avg_best = best.groupby([group_by, params]).mean(numeric_only=True).reset_index()
    avg_best_sorted = avg_best.sort_values(by=metric, ascending=ascending)
    method_rank = avg_best_sorted.loc[:, display_fields]
    display(method_rank)
    return method_rank


def fastlane_boxplots(expected_results_dir, group_by="method",
                      standard='expected', metric="distance", hue=None,
                      plotf=violinplot, label_rotation=45,
                      y_min=0.0, y_max=1.0, color=None, beta="braycurtis"):

    '''per_method_boxplots for those who don't have time to wait.'''

    for table, dataset_id, reference_id in seek_tables(expected_results_dir):
        display(Markdown('## {0} {1}'.format(dataset_id, reference_id)))

        dm, sample_md = make_distance_matrix(table, method=beta)

        per_method_boxplots(dm, sample_md, group_by=group_by, metric=metric,
                            standard=standard, hue=hue, y_min=y_min,
                            y_max=y_max, plotf=plotf, color=color,
                            label_rotation=label_rotation)


def per_method_boxplots(dm, sample_md, group_by="method", standard='expected',
                        metric="distance", hue=None, y_min=0.0, y_max=1.0,
                        plotf=violinplot, label_rotation=45, color=None,
                        color_palette=None):
    '''Generate distance boxplots and Mann-Whitney U tests on distance matrix.

    dm: skbio.DistanceMatrix
    sample_md: pandas.DataFrame
        containing sample metadata
    group_by: str
        df category to use for grouping samples
    standard: str
        group name in group_by category to which all other groups are compared.
    metric: str
        name of distance column in output.

    To generate boxplots instead of violin plots, pass plotf=seaborn.boxplot

    hue, color variables all pass directly to equivalently named variables in
        seaborn.violinplot().
    '''
    box = dict()
    within_between = within_between_category_distance(dm, sample_md, 'method')

    per_method = per_method_distance(dm, sample_md, group_by=group_by,
                                     standard=standard, metric=metric)

    for d, g, s in [(within_between, 'Comparison', '1: Within- vs. Between-'),
                    (per_method, group_by, '2: Pairwise ')]:

        display(Markdown('## Comparison {0} Distance'.format(s + group_by)))
        box[g] = boxplot_from_data_frame(
            d, group_by=g, color=color, metric=metric, y_min=None, y_max=None,
            hue=hue, plotf=plotf, label_rotation=label_rotation,
            color_palette=color_palette)

        results = per_method_pairwise_tests(d, group_by=g, metric=metric)

        plt.show()
        plt.clf()
        display(results)

    return box


def per_method_distance(dm, md, group_by='method', standard='expected',
                        metric='distance', sample='sample_id'):
    '''Compile list of distances between groups of samples in distance matrix.
    returns dataframe of distances and group metadata.

    dm: skbio.DistanceMatrix
    md: pandas.DataFrame
        containing sample metadata
    group_by: str
        df category to use for grouping samples
    standard: str
        group name in group_by category to which all other groups are compared.
    metric: str
        name of distance column in output.
    sample: str
        df category containing sample_id names.
    '''
    results = []
    expected = md[md[group_by] == standard]
    observed = md[md[group_by] != standard]
    for group in observed[group_by].unique():
        group_md = observed[observed[group_by] == group]
        for i in list(expected.index.values):
            for j in list(group_md.index.values):
                if group_md.loc[j][sample] == expected.loc[i][sample]:
                    results.append((*[n for n in group_md.loc[j]], dm[i, j]))
    return pd.DataFrame(results, columns=[*[n for n in md.columns.values],
                                          metric])


def within_between_category_distance(dm, md, md_category, distance='distance'):
    '''Compile list of distances between groups of samples and within groups
    of samples.

    dm: skbio.DistanceMatrix
    md: pandas.DataFrame
        containing sample metadata
    md_category: str
        df category to use for grouping samples
    '''
    distances = []
    for i, sample_id1 in enumerate(dm.ids):
        sample_md1 = md[md_category][sample_id1]
        for sample_id2 in dm.ids[:i]:
            sample_md2 = md[md_category][sample_id2]
            if sample_md1 == sample_md2:
                comp = 'within'
                group = sample_md1
            else:
                comp = 'between'
                group = sample_md1 + '_' + sample_md2
            distances.append((comp, group, dm[sample_id1, sample_id2]))
    return pd.DataFrame(distances, columns=["Comparison", md_category,
                                            distance])


def per_method_pairwise_tests(df, group_by='method', metric='distance',
                              paired=False, parametric=True):
    '''Perform mann whitney U tests between group distance distributions,
    followed by FDR correction. Returns pandas dataframe of p-values.
    df: pandas.DataFrame
        results from per_method_distance()
    group_by: str
        df category to use for grouping samples
    metric: str
        df category to use as variable for comparison.
    paired: bool
        Perform Wilcoxon signed rank test instead of Mann Whitney U. df must be
        ordered such that paired samples will appear in same order in subset
        dataframes when df is subset by term f[df[group_by] == a[0]][metric].
    '''
    pvals = []
    groups = [group for group in df[group_by].unique()]
    combos = [a for a in combinations(groups, 2)]
    for a in combos:
        try:
            if paired is False and parametric is False:
                u, p = mannwhitneyu(df[df[group_by] == a[0]][metric],
                                    df[df[group_by] == a[1]][metric],
                                    alternative='two-sided')
            elif paired is False and parametric is True:
                u, p = ttest_ind(df[df[group_by] == a[0]][metric],
                                 df[df[group_by] == a[1]][metric],
                                 nan_policy='raise')
            elif paired is True and parametric is False:
                u, p = wilcoxon(df[df[group_by] == a[0]][metric],
                                df[df[group_by] == a[1]][metric])
            else:
                u, p = ttest_rel(df[df[group_by] == a[0]][metric],
                                 df[df[group_by] == a[1]][metric],
                                 nan_policy='raise')
        except ValueError:
            # default to p=1.0 if all values = 0
            # this is not technically correct, from the standpoint of p-val
            # correction below makes p-vals very slightly less significant
            # than they should be
            u, p = 0.0, 1.0
        pvals.append((a[0], a[1], u, p))

    result = pd.DataFrame(pvals, columns=["Method A", "Method B", "stat", "P"])
    result.set_index(['Method A', 'Method B'], inplace=True)
    try:
        result['FDR P'] = multipletests(result['P'], method='fdr_bh')[1]
    except ZeroDivisionError:
        pass

    return result


def isolate_top_params(df, group_by="Method", params="Parameters",
                       metric="F-measure", ascending=True):
    '''For each method in df, find top params for each method and filter df to
    contain only those parameters.

    df: pandas df
    group_by: str
        df category name to use for segregating groups from which top param is
        chosen.
    params: str
        df category name indicating parameters column.
    '''
    best = pd.DataFrame()
    param_report = []
    for group in df[group_by].unique():
        subset = df[df[group_by] == group]
        avg = subset.groupby(params).mean(numeric_only=True).reset_index()
        sorted_avg = avg.sort_values(by=metric, ascending=ascending)
        top_param = sorted_avg.reset_index()[params][0]
        param_report.append((group, top_param))
        best = pd.concat([best, subset[subset[params] == top_param]])
    return best, param_report


def rank_optimized_method_performance_by_dataset(df,
                                                 dataset="Dataset",
                                                 method="Method",
                                                 params="Parameters",
                                                 metric="F-measure",
                                                 level="Level",
                                                 level_range=range(5, 7),
                                                 display_fields=["Method",
                                                                 "Parameters",
                                                                 "Precision",
                                                                 "Recall",
                                                                 "F-measure"],
                                                 ascending=False,
                                                 paired=True,
                                                 parametric=True,
                                                 hue=None,
                                                 y_min=0.0,
                                                 y_max=1.0,
                                                 plotf=violinplot,
                                                 label_rotation=45,
                                                 color=None,
                                                 color_palette=None):

    '''Rank the performance of methods using optimized parameter configuration
    within each dataset in dataframe. Optimal methods are computed from the
    mean performance of each method/param configuration across all datasets
    in df.

    df: pandas df
    dataset: str
        df category to use for grouping samples (by dataset)
    method: str
        df category to use for grouping samples (by method); these groups are
        compared in plots and pairwise statistical testing.
    params: str
        df category containing parameter configurations for each method. Best
        method configurations are computed by grouping method groups on this
        category value, then finding the best average metric value.
    metric: str
        df category containing metric to use for ranking and statistical
        comparisons between method groups.
    level: str
        df category containing taxonomic level information.
    level_range: range
        Perform plotting and testing at each level in range.
    display_fields: list
        List of columns in df to display in results.
    ascending: bool
        Rank methods my metric score in ascending or descending order?
    paired: bool
        Perform paired statistical test? See per_method_pairwise_tests()
    parametric: bool
        Perform parametric statistical test? See per_method_pairwise_tests()

    To generate boxplots instead of violin plots, pass plotf=seaborn.boxplot

    hue, color variables all pass directly to equivalently named variables in
        seaborn.violinplot(). See boxplot_from_data_frame() for more
        information.
    '''
    box = dict()
    for d in df[dataset].unique():
        for lv in level_range:
            display(Markdown("## {0} level {1}".format(d, lv)))
            df_l = df[df[level] == lv]
            best, param_report = isolate_top_params(df_l[df_l[dataset] == d],
                                                    method, params, metric,
                                                    ascending=ascending)

            method_rank = _show_method_rank(
                best, method, params, metric, display_fields,
                ascending=ascending)

            results = per_method_pairwise_tests(best, group_by=method,
                                                metric=metric, paired=paired,
                                                parametric=parametric)
            display(results)

            box[d] = boxplot_from_data_frame(
                best, group_by=method, color=color, metric=metric, y_min=y_min,
                y_max=y_max, label_rotation=label_rotation, hue=hue,
                plotf=plotf, color_palette=color_palette)

            _add_significance_to_boxplots(
                results, method_rank, box[d].axes[0])

            plt.show()

    return box
