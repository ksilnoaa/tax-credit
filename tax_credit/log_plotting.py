#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Plotting helpers for classification log analysis."""

import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap

from tax_credit.plot_theme import (
    annotate_heatmap_cells,
    apply_tax_credit_theme,
    metric_limits,
)


def _short_taxon_label(taxon: str, max_len: int = 40) -> str:
    label = str(taxon).split(";")[-1] if ";" in str(taxon) else str(taxon)
    if len(label) > max_len:
        return label[: max_len - 3] + "..."
    return label


def _run_label(column, dataset) -> str:
    """Run column label without its dataset, e.g. ``naive-bayes · nb-conf0.7``."""
    parts = [part.strip() for part in str(column).split(" / ")]
    if parts and parts[0] == str(dataset):
        parts = parts[1:]
    return " · ".join(parts)


def method_parameter_sensitivity_heatmap_from_data_frame(
    pivot_df,
    title=None,
    value_label="fraction of reads misclassified",
    annotate_max_cells=120,
):
    """Heatmap of a taxon-level metric per method/parameter run, one panel per dataset.

    pivot_df: indexed by ``(dataset, expected_taxonomy)`` with one column per
        run named ``"<dataset> / <method> / <parameters>"``, as returned by
        ``summarize_method_parameter_sensitivity`` and
        ``filter_sensitivity_to_rank``. Row order is kept, so rank rows first
        (``select_top_sensitivity_taxa``).

    Each panel shows only its dataset's runs. Hatched cells have no reads for
    that taxon in that run. Cell values are printed when a panel has at most
    *annotate_max_cells* cells. Returns the figure.
    """
    apply_tax_credit_theme()
    blocks = []
    for dataset in dict.fromkeys(pivot_df.index.get_level_values(0)):
        block = pivot_df.xs(dataset, level=0)
        block = block.loc[:, [c for c in block.columns if str(c).startswith(f"{dataset} / ")]]
        block = block.dropna(axis=1, how="all")
        if block.empty:
            continue
        block.index = [_short_taxon_label(taxon) for taxon in block.index]
        block.columns = [_run_label(column, dataset) for column in block.columns]
        blocks.append((dataset, block))
    if not blocks:
        raise ValueError("No sensitivity values to plot.")

    low, high, zoomed = metric_limits(
        np.concatenate([block.to_numpy(dtype=float).ravel() for _, block in blocks])
    )
    n_rows = max(len(block) for _, block in blocks)
    fig, axes = plt.subplots(
        1,
        len(blocks),
        figsize=(sum(0.45 * block.shape[1] + 2.2 for _, block in blocks) + 1.4,
                 0.26 * n_rows + 2.8),
        width_ratios=[block.shape[1] for _, block in blocks],
        squeeze=False,
    )
    for ax, (dataset, block) in zip(axes[0], blocks):
        # hatched background shows through cells with no data
        ax.patch.set_facecolor("0.97")
        ax.patch.set_hatch("////")
        ax.patch.set_edgecolor("0.8")
        heatmap(
            block,
            cmap="rocket_r",
            vmin=low,
            vmax=high,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        if block.size <= annotate_max_cells:
            annotate_heatmap_cells(ax, block.to_numpy(dtype=float), "rocket_r", low, high)
        ax.grid(False)
        ax.set_title(str(dataset))
        ax.set_xlabel("method · parameters")
        ax.set_ylabel("")
        ax.tick_params(length=0)
        # seaborn turns row labels vertical in tall cells; keep them readable
        ax.tick_params(axis="y", labelrotation=0)
        ax.tick_params(axis="x", labelrotation=35)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    axes[0, 0].set_ylabel("expected taxon")
    fig.colorbar(
        axes[0, -1].collections[0],
        ax=axes[0].tolist(),
        shrink=0.6,
        format="%.2f",
        label=(
            f"{value_label}{' (zoomed scale)' if zoomed else ''}\n"
            "hatched: no reads for taxon in run"
        ),
    )
    if title:
        fig.suptitle(title)
    return fig
