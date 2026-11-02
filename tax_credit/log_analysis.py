#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Per-read classification log aggregation and taxon-level error summaries."""

from __future__ import annotations

from os.path import exists, join

import pandas as pd

from tax_credit.paths import (
    CLASSIFICATION_ACCURACY_LOG_TSV,
    parse_assignment_results_dir,
)
from tax_credit.taxa_manipulator import is_unassigned_taxon, normalize_taxon

# Rank names by taxonomy depth index (0 = kingdom ... 6 = species).
RANK_NAMES = ("kingdom", "phylum", "class", "order", "family", "genus", "species")

RANK_TO_LEVEL = {
    rank: level for level, rank in enumerate(RANK_NAMES) if level > 0
}

_LOG_COLUMNS = [
    "dataset",
    "level",
    "iteration",
    "method",
    "parameters",
    "observed_taxonomy",
    "expected_taxonomy",
    "result",
    "mismatch_level",
]


def _truncate_taxonomy_at_level(taxon: str, level: int) -> str:
    """Return taxonomy truncated through rank ``level`` (1–6).

    Internal NA ranks keep their position; trailing NA ranks are dropped.
    """
    return normalize_taxon(";".join(taxon.split(";")[:level + 1]))


def taxonomy_depth(taxon: str) -> int:
    """Number of assigned ranks, counting internal NA ranks but not trailing ones."""
    if is_unassigned_taxon(taxon):
        return 0
    return len(normalize_taxon(taxon).split(";"))


def expected_rank_name(taxon: str) -> str:
    """Deepest rank named in a taxonomy string, e.g. ``genus``; ``unassigned`` if none."""
    depth = taxonomy_depth(taxon)
    if depth == 0:
        return "unassigned"
    return RANK_NAMES[min(depth, len(RANK_NAMES)) - 1]


def truncate_to_rank(taxon: str, rank: str) -> str:
    """Truncate a taxonomy string to family, genus, or species rank."""
    rank = rank.lower()
    if rank not in RANK_TO_LEVEL:
        raise ValueError(
            f"rank must be one of {sorted(RANK_TO_LEVEL)}; got {rank!r}"
        )
    if rank == "species":
        return normalize_taxon(taxon)
    return _truncate_taxonomy_at_level(taxon, RANK_TO_LEVEL[rank])


def _rank_at_depth(taxon: str, depth: int) -> str:
    """Return the rank name at a given 0-based depth index, or empty string."""
    parts = normalize_taxon(taxon).split(";")
    if depth < 0 or depth >= len(parts):
        return ""
    return parts[depth]


def load_classification_accuracy_logs(results_dirs) -> pd.DataFrame:
    """Load per-read classification logs from assignment result directories."""
    frames = []
    for results_dir in results_dirs:
        log_fp = join(results_dir, CLASSIFICATION_ACCURACY_LOG_TSV)
        if not exists(log_fp):
            continue
        df = pd.read_csv(log_fp, sep="\t")
        df.columns = [str(c).strip() for c in df.columns]
        drop_cols = [c for c in df.columns if c not in _LOG_COLUMNS]
        df = df.drop(columns=drop_cols, errors="ignore")
        dataset_id, method_id, params_id = parse_assignment_results_dir(results_dir)
        df["results_dir"] = results_dir
        df["dataset_id"] = dataset_id
        if "method" not in df.columns:
            df["method"] = method_id
        if "parameters" not in df.columns:
            df["parameters"] = params_id
        df["method"] = df["method"].astype(str).str.strip()
        df["parameters"] = df["parameters"].astype(str).str.strip()
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["mismatch_level"] = pd.to_numeric(out["mismatch_level"], errors="coerce")
    out["expected_depth"] = out["expected_taxonomy"].map(taxonomy_depth)
    out["observed_depth"] = out["observed_taxonomy"].map(taxonomy_depth)
    out["ranks_short"] = out["expected_depth"] - out["observed_depth"]
    return out


def _add_expected_at_rank(df: pd.DataFrame, rank: str) -> pd.DataFrame:
    out = df.copy()
    out["expected_at_rank"] = out["expected_taxonomy"].map(
        lambda t: truncate_to_rank(t, rank)
    )
    return out


def _result_counts(group: pd.DataFrame) -> pd.Series:
    counts = group["result"].value_counts()
    n_obvs = len(group)
    row = {
        "n_obvs": n_obvs,
        "n_match": int(counts.get("match", 0)),
        "n_under": int(counts.get("underclassification", 0)),
        "n_over": int(counts.get("overclassification", 0)),
        "n_mis": int(counts.get("misclassification", 0)),
    }
    for count_key, count in list(row.items()):
        if count_key == "n_obvs":
            continue
        pct_key = count_key.replace("n_", "pct_")
        row[pct_key] = count / n_obvs if n_obvs else 0.0
    non_match = {
        "underclassification": row["n_under"],
        "overclassification": row["n_over"],
        "misclassification": row["n_mis"],
    }
    row["dominant_error"] = max(non_match, key=non_match.get) if sum(non_match.values()) else "match"
    row["mean_mismatch_level"] = group["mismatch_level"].mean()
    row["median_observed_depth"] = group["observed_depth"].median()
    row["median_expected_depth"] = group["expected_depth"].median()
    return pd.Series(row)


def summarize_taxon_errors(
    df: pd.DataFrame,
    rank: str = "species",
    min_obvs: int = 3,
    group_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Per-expected-taxon error profile table."""
    if df.empty:
        return pd.DataFrame()

    group_cols = group_cols or ("dataset", "method", "parameters")
    work = _add_expected_at_rank(df, rank)
    keys = [*group_cols, "expected_at_rank"]
    rows = []
    for key, group in work.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row.update(_result_counts(group).to_dict())
        row["expected_taxonomy"] = row.pop("expected_at_rank")
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    summary["expected_rank"] = summary["expected_taxonomy"].map(expected_rank_name)
    summary = summary[summary["n_obvs"] >= min_obvs]
    return summary.sort_values(
        ["n_mis", "n_under", "n_obvs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def summarize_confusion_pairs(
    df: pd.DataFrame,
    rank: str = "species",
    min_count: int = 1,
) -> pd.DataFrame:
    """Confusion pairs for misclassifications only."""
    if df.empty:
        return pd.DataFrame()

    work = _add_expected_at_rank(df, rank)
    mis = work[work["result"] == "misclassification"].copy()
    if mis.empty:
        return pd.DataFrame()

    mis["observed_at_rank"] = mis["observed_taxonomy"].map(
        lambda t: truncate_to_rank(t, rank)
    )
    mis["expected_genus"] = mis["expected_taxonomy"].map(lambda t: _rank_at_depth(t, 5))
    mis["observed_genus"] = mis["observed_taxonomy"].map(lambda t: _rank_at_depth(t, 5))
    mis["expected_family"] = mis["expected_taxonomy"].map(lambda t: _rank_at_depth(t, 4))
    mis["observed_family"] = mis["observed_taxonomy"].map(lambda t: _rank_at_depth(t, 4))
    mis["same_genus"] = mis["expected_genus"] == mis["observed_genus"]
    mis["same_family"] = mis["expected_family"] == mis["observed_family"]

    group_cols = [
        "dataset",
        "method",
        "parameters",
        "expected_at_rank",
        "observed_at_rank",
    ]
    pairs = (
        mis.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "count", "expected_at_rank": "expected_taxonomy",
                         "observed_at_rank": "observed_taxonomy"})
    )
    exp_counts = (
        work.groupby(["dataset", "method", "parameters", "expected_at_rank"], as_index=False)
        .size()
        .rename(columns={"size": "expected_obvs", "expected_at_rank": "expected_taxonomy"})
    )
    pairs = pairs.merge(
        exp_counts,
        on=["dataset", "method", "parameters", "expected_taxonomy"],
        how="left",
    )
    pairs["pct_of_expected"] = pairs["count"] / pairs["expected_obvs"]
    pairs = pairs[pairs["count"] >= min_count]

    meta_cols = ["expected_genus", "observed_genus", "same_genus", "same_family"]
    meta = (
        mis.groupby(
            ["dataset", "method", "parameters", "expected_at_rank", "observed_at_rank"],
            as_index=False,
        )[meta_cols]
        .first()
        .rename(columns={"expected_at_rank": "expected_taxonomy",
                         "observed_at_rank": "observed_taxonomy"})
    )
    pairs = pairs.merge(
        meta,
        on=["dataset", "method", "parameters", "expected_taxonomy", "observed_taxonomy"],
        how="left",
    )
    pairs["expected_rank"] = pairs["expected_taxonomy"].map(expected_rank_name)
    return pairs.sort_values("count", ascending=False).reset_index(drop=True)


def summarize_method_parameter_sensitivity(
    df: pd.DataFrame,
    rank: str = "species",
    min_obvs: int = 3,
    metric: str = "pct_mis",
) -> pd.DataFrame:
    """Pivot table of error metric by taxon and method/parameter combo.

    Indexed by ``dataset``, ``expected_taxonomy`` and ``expected_rank``; use
    ``filter_sensitivity_to_rank`` to keep taxa resolved to *rank*.
    """
    taxon_summary = summarize_taxon_errors(
        df, rank=rank, min_obvs=min_obvs,
        group_cols=("dataset", "method", "parameters"),
    )
    if taxon_summary.empty:
        return pd.DataFrame()

    taxon_summary["run_combo"] = (
        taxon_summary["dataset"].astype(str)
        + " / "
        + taxon_summary["method"].astype(str)
        + " / "
        + taxon_summary["parameters"].astype(str)
    )
    if metric not in taxon_summary.columns:
        raise ValueError(f"metric must be one of taxon summary columns; got {metric!r}")

    pivot = taxon_summary.pivot_table(
        index=["dataset", "expected_taxonomy", "expected_rank"],
        columns="run_combo",
        values=metric,
        aggfunc="first",
    )
    pivot = pivot.sort_index()
    return pivot


def select_top_sensitivity_taxa(pivot_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Keep the *top_n* taxa per dataset with the highest value in any run.

    Groups by the first index level (dataset) when the index has several
    levels. Rows are ordered by dataset, then by descending maximum value (ties
    by taxon), so the worst taxa come first.
    """
    if pivot_df.empty or top_n <= 0:
        return pivot_df

    def _ranked(scores: pd.Series) -> list:
        frame = scores.fillna(-1.0).rename("score").to_frame()
        frame["label"] = [str(key) for key in frame.index]
        frame = frame.sort_values(["score", "label"], ascending=[False, True])
        return list(frame.index[:top_n])

    scores = pivot_df.max(axis=1, skipna=True)
    if pivot_df.index.nlevels < 2:
        return pivot_df.loc[_ranked(scores)]
    keep = []
    for _, dataset_scores in scores.groupby(level=0, sort=True):
        keep.extend(_ranked(dataset_scores))
    return pivot_df.loc[keep]


def filter_sensitivity_to_rank(
    pivot_df: pd.DataFrame, rank: str,
) -> tuple[pd.DataFrame, int]:
    """Keep sensitivity rows whose expected taxonomy is resolved to *rank*.

    Drops the ``expected_rank`` index level. Returns ``(filtered, n_excluded)``,
    where *n_excluded* counts rows whose expected taxonomy stops at another rank.
    """
    if pivot_df.empty or "expected_rank" not in (pivot_df.index.names or []):
        return pivot_df, 0
    keep = pivot_df.index.get_level_values("expected_rank") == rank
    filtered = pivot_df[keep].droplevel("expected_rank")
    return filtered, int((~keep).sum())


def summarize_cross_fold_stability(
    df: pd.DataFrame,
    rank: str = "species",
    min_obvs: int = 1,
) -> pd.DataFrame:
    """Cross-fold error stability per expected taxon."""
    if df.empty or "iteration" not in df.columns:
        return pd.DataFrame()

    work = _add_expected_at_rank(df, rank)
    rows = []
    group_keys = ["dataset", "method", "parameters", "iteration", "expected_at_rank"]
    for key, group in work.groupby(group_keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_keys, key))
        row.update(_result_counts(group).to_dict())
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    fold_summary = pd.DataFrame(rows)
    fold_summary["any_error"] = fold_summary["n_match"] < fold_summary["n_obvs"]
    fold_summary["has_misclass"] = fold_summary["n_mis"] > 0
    fold_summary["error_rate"] = 1 - fold_summary["pct_match"]

    stability = (
        fold_summary.groupby(
            ["dataset", "method", "parameters", "expected_at_rank"],
            as_index=False,
        )
        .agg(
            n_folds=("iteration", "nunique"),
            folds_with_any_error=("any_error", "sum"),
            folds_with_misclass=("has_misclass", "sum"),
            mean_error_rate=("error_rate", "mean"),
            mean_pct_mis=("pct_mis", "mean"),
            total_obvs=("n_obvs", "sum"),
        )
        .rename(columns={"expected_at_rank": "expected_taxonomy"})
    )
    stability["expected_rank"] = stability["expected_taxonomy"].map(expected_rank_name)
    stability = stability[stability["total_obvs"] >= min_obvs]
    return stability.sort_values("folds_with_misclass", ascending=False).reset_index(drop=True)
