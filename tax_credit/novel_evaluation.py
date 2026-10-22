#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Novel-taxa and cross-validated assignment evaluation from query tax files.

Consumes paths laid out under ``tax_credit.paths`` and dataset ids from
``tax_credit.simulation_names``; PRF and taxonomy helpers stay in
``framework_functions``.
"""

from collections import Counter
from os.path import exists, join

import pandas as pd

from tax_credit.paths import (
    CLASSIFICATION_ACCURACY_LOG_TSV,
    CLASSIFICATION_METRICS_TSV,
    QUERY_TAX_ASSIGNMENTS_TXT,
    QUERY_TAXA_TSV,
    parse_assignment_results_dir,
)
from tax_credit.simulation_names import (
    parse_cv_dataset_id,
    parse_novel_dataset_id,
    parse_self_validated_dataset_id,
)
from tax_credit.metrics import (
    CLASSIFICATION_METRIC_COLUMNS,
    CLASSIFICATION_OUTCOMES,
    metrics_from_outcomes,
)
from tax_credit.taxa_manipulator import export_list_to_file, normalize_taxon



# Summary columns, in order. The first thirteen are the historical layout; the
# rest report the new scoring alongside the original.
SUMMARY_COLUMNS = [
    "Dataset", "level", "iteration", "Method", "Parameters",
    "match_ratio", "overclassification_ratio", "underclassification_ratio",
    "misclassification_ratio", "mismatch_level_list",
    "Precision", "Recall", "F-measure",
    "Recall_original", "F-measure_original",
    "TP", "FP", "FN", "FN_original",
]

# Per-read accuracy log columns, in order.
ACCURACY_LOG_COLUMNS = [
    "dataset", "level", "iteration", "method", "parameters",
    "observed_taxonomy", "expected_taxonomy", "result", "mismatch_level",
    "Precision", "Recall", "F-measure",
    "Recall_original", "F-measure_original",
]


def write_classification_metrics(per_level, meta, metrics_fp):
    '''Write per-rank classification metrics for one assignment result.

    per_level: dict
        ``compute_prf`` output for a cross-validated-style run: metric name to
        a list of seven values indexed by taxonomic level.
    meta: dict
        Identifying columns written on every row, e.g. ``Dataset``,
        ``novel_level``, ``iteration``, ``Method``, ``Parameters``.
    metrics_fp: path
        Output file.

    One row per rank 1-6, so every metric is reported at the same grain. This
    is the tidy counterpart to the summary CSV, which stores per-rank scores as
    lists inside single cells.
    '''
    from tax_credit.framework_functions import CLASSIFICATION_METRIC_COLUMNS
    from tax_credit.log_analysis import RANK_NAMES

    rows = []
    for level in range(1, 7):
        rows.append({
            **meta,
            "level": level,
            "rank": RANK_NAMES[level],
            **{column: per_level[column][level]
               for column in CLASSIFICATION_METRIC_COLUMNS},
        })
    table = pd.DataFrame(rows)
    table.to_csv(metrics_fp, sep="\t", index=False)
    return table


def novel_taxa_classification_evaluation(results_dirs, expected_results_dir,
                                         summary_fp, test_type='novel-taxa',
                                         rank_positions=None):
    '''Input glob of novel taxa results, receive a summary of accuracy results.
    results_dirs = list or glob of novel taxa observed results in format:
                    precomputed_results_dir/dataset_id/method_id/params_id/
                    For the usual four-level sweep tree under a single root, use
                    ``tax_credit.paths.list_assignment_result_dirs`` instead of
                    hand-written globs.
    expected_results_dir = directory containing expected novel-taxa results in
                    format:
                    expected_results_dir/dataset_id/method_id/params_id/
    summary_fp = filepath to contain summary of results
    test_type = one of 'novel-taxa', 'cross-validated',
        'cross-validated-trad', or 'self-validated'
    rank_positions = optional dict of database name to the seven rank
        positions to score, for databases whose taxonomy is not the standard
        seven ranks (see ``framework_functions.load_prf``). Databases not
        listed keep the first seven ranks.

    Returns results as df, in addition to printing summary_fp.

    Writes two files into each result directory: the per-read
    ``classification_accuracy_log.tsv`` and the per-rank
    ``classification_metrics.tsv``. Scores follow
    ``framework_functions.metrics_from_outcomes``; ``Recall_original`` and
    ``F-measure_original`` carry the original tax-credit scoring, in which
    overclassification also counts as a false negative.
    '''
    from tax_credit.framework_functions import (
        compute_prf,
        evaluate_classification,
        find_last_common_ancestor,
        load_prf,
    )

    results = []

    for results_dir in results_dirs:
        res_parts = parse_assignment_results_dir(results_dir)
        dataset_id, method_id, params_id = res_parts

        if test_type == 'novel-taxa':
            novel_parts = parse_novel_dataset_id(dataset_id)
            index, level, iteration = (
                novel_parts.database,
                novel_parts.level,
                novel_parts.iteration,
            )
        elif test_type in ('cross-validated', 'cross-validated-trad'):
            cv_parts = parse_cv_dataset_id(dataset_id)
            index, iteration = cv_parts.database, cv_parts.iteration
            level = 6
        elif test_type == 'self-validated':
            sv_parts = parse_self_validated_dataset_id(dataset_id)
            index, iteration = sv_parts.database, sv_parts.iteration
            level = 6
        else:
            raise ValueError(
                'test_type must be "novel-taxa", "cross-validated", '
                '"cross-validated-trad", or "self-validated"')

        obs_fp = join(results_dir, QUERY_TAX_ASSIGNMENTS_TXT)
        exp_fp = join(expected_results_dir, dataset_id, QUERY_TAXA_TSV)
        exp_taxa, obs_taxa = load_prf(
            obs_fp, exp_fp,
            rank_positions=(rank_positions or {}).get(index))

        # per-rank metrics for every test type, for the sidecar and the plots
        per_level = compute_prf(exp_taxa, obs_taxa,
                                test_type='cross-validated')
        if test_type == 'novel-taxa':
            # novel-taxa summarises one full-depth score, not a per-rank list
            scores = compute_prf(exp_taxa, obs_taxa, test_type='novel-taxa')
            full_depth = scores
        else:
            scores = per_level
            full_depth = {column: values[6]
                          for column, values in per_level.items()}

        mismatch_level_list = [0] * 8
        log = ['\t'.join(ACCURACY_LOG_COLUMNS)]

        for obs, exp in zip(obs_taxa, exp_taxa):
            mismatch_level = find_last_common_ancestor(obs, exp)
            mismatch_level_list[mismatch_level] += 1

            result = evaluate_classification(obs, exp)

            log.append('\t'.join(map(str, [
                index, level, iteration, method_id, params_id,
                obs, exp, result, mismatch_level,
                scores['Precision'], scores['Recall'], scores['F-measure'],
                scores['Recall_original'], scores['F-measure_original']])))

        log_fp = join(results_dir, CLASSIFICATION_ACCURACY_LOG_TSV)
        export_list_to_file(log, log_fp)

        write_classification_metrics(
            per_level,
            {
                "Dataset": index,
                "novel_level": level if test_type == 'novel-taxa' else None,
                "iteration": iteration,
                "Method": method_id,
                "Parameters": params_id,
            },
            join(results_dir, CLASSIFICATION_METRICS_TSV),
        )

        # the four ratios describe the full-depth comparison, as they always
        # have; match_ratio is no longer equal to Recall
        results.append((index, level, iteration, method_id, params_id,
                        full_depth['match_ratio'],
                        full_depth['overclassification_ratio'],
                        full_depth['underclassification_ratio'],
                        full_depth['misclassification_ratio'],
                        mismatch_level_list,
                        scores['Precision'], scores['Recall'],
                        scores['F-measure'],
                        scores['Recall_original'],
                        scores['F-measure_original'],
                        scores['TP'], scores['FP'], scores['FN'],
                        scores['FN_original']))

    result = pd.DataFrame(results, columns=SUMMARY_COLUMNS)
    result.to_csv(summary_fp)
    return result


def _as_level_list(value):
    '''Per-level metric list, parsed from a summary CSV cell when needed.'''
    if isinstance(value, str):
        return [float(part) for part in value.strip('[]').split(',')]
    return value


PER_LEVEL_METRIC_COLUMNS = ('Precision', 'Recall', 'F-measure',
                            'Recall_original', 'F-measure_original',
                            'TP', 'FP', 'FN', 'FN_original')


def extract_per_level_accuracy(df, columns=PER_LEVEL_METRIC_COLUMNS):
    '''Split per-rank metric lists into one dataframe row per rank.

    df: dataframe
        Summary from ``novel_taxa_classification_evaluation`` for a
        cross-validated or self-validated run, in which each metric column
        holds a list of seven values indexed by taxonomic level.
    columns: iterable
        Metric columns to split out. Columns absent from *df* are skipped, so a
        summary written before a metric existed still loads.

    Classification ratios are deliberately *not* derived here. ``match_ratio``
    used to be taken from ``Recall``, which held only while every non-match
    counted as a false negative; the two now differ by the overclassification
    rate. Use ``extract_per_level_classification_ratios_by_fold`` for ratios.
    '''
    columns = [column for column in columns if column in df.columns]
    results = []

    for _, data in df.iterrows():
        parsed = {column: _as_level_list(data[column]) for column in columns}
        for level in range(1, 7):
            results.append((data['Dataset'], level, data['iteration'],
                            data['Method'], data['Parameters'],
                            *[parsed[column][level] for column in columns]))

    return pd.DataFrame(results, columns=["Dataset", "level", "iteration",
                                          "Method", "Parameters", *columns])


CLASSIFICATION_RATIO_COLS = [
    "match_ratio",
    "misclassification_ratio",
    "overclassification_ratio",
    "underclassification_ratio",
]

# Every per-rank metric the accuracy log and metrics sidecar can supply.
CLASSIFICATION_METRIC_COLS = list(CLASSIFICATION_METRIC_COLUMNS)


def _truncate_taxonomy_at_level(taxon, level):
    """Return taxonomy truncated through rank ``level`` (1–6, matching PRF indices).

    Internal NA ranks keep their position; trailing NA ranks are dropped.
    """
    return normalize_taxon(";".join(taxon.split(";")[:level + 1]))


def _dataset_parts_from_dataset_id(dataset_id):
    """Return ``(database, novel_level, iteration)`` for an assignment dataset id.

    ``novel_level`` is ``None`` for cross-validated and self-validated ids.
    """
    try:
        parts = parse_novel_dataset_id(dataset_id)
        return parts.database, parts.level, str(parts.iteration)
    except (ValueError, TypeError):
        pass
    for parser in (parse_cv_dataset_id, parse_self_validated_dataset_id):
        try:
            parts = parser(dataset_id)
            return parts.database, None, str(parts.iteration)
        except (ValueError, TypeError):
            continue
    return dataset_id.split("-", 1)[0], None, "0"


def per_level_classification_ratios_from_log(log_fp):
    """Compute classification metrics at ranks 1-6 from a per-read accuracy log.

    Returns ``(rows, meta)``: one row per rank holding ``level`` and every key
    of ``framework_functions.CLASSIFICATION_METRIC_COLUMNS``, and the dataset /
    method / parameters values read off the log.

    Each rank is scored independently, by re-classifying every read's observed
    and expected taxonomy truncated to that rank, so trailing ``NA`` ranks in a
    reference are handled the same way at every level.
    """
    from tax_credit.framework_functions import evaluate_classification

    outcomes = {level: dict.fromkeys(CLASSIFICATION_OUTCOMES, 0)
                for level in range(1, 7)}
    meta = {}

    with open(log_fp) as f:
        header = [col.strip() for col in f.readline().strip().split("\t")]
        col_idx = {name: idx for idx, name in enumerate(header)}
        obs_idx = col_idx["observed_taxonomy"]
        exp_idx = col_idx["expected_taxonomy"]
        for key in ("dataset", "method", "parameters"):
            if key in col_idx:
                meta[key] = None

        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            obs = fields[obs_idx]
            exp = fields[exp_idx]
            if meta:
                for key in meta:
                    meta[key] = fields[col_idx[key]]
            for level in range(1, 7):
                obs_t = _truncate_taxonomy_at_level(obs, level)
                exp_t = _truncate_taxonomy_at_level(exp, level)
                result = evaluate_classification(obs_t, exp_t)
                outcomes[level][result] += 1

    rows = [{"level": level, **metrics_from_outcomes(**outcomes[level])}
            for level in range(1, 7)]
    return rows, meta


def _per_level_rows(results_dir):
    """Per-rank metric rows for one result directory, or None if unavailable.

    Prefers the ``classification_metrics.tsv`` sidecar and falls back to
    recomputing from the per-read log. Identifying columns are always taken
    from the directory path rather than the file, so both sources yield the
    same dtypes.
    """
    metrics_fp = join(results_dir, CLASSIFICATION_METRICS_TSV)
    if exists(metrics_fp):
        table = pd.read_csv(metrics_fp, sep="\t")
        keep = ["level"] + [column for column in CLASSIFICATION_METRIC_COLUMNS
                            if column in table.columns]
        return table[keep].to_dict("records")
    log_fp = join(results_dir, CLASSIFICATION_ACCURACY_LOG_TSV)
    if exists(log_fp):
        return per_level_classification_ratios_from_log(log_fp)[0]
    return None


def extract_per_level_classification_ratios_by_fold(results_dirs):
    """Per-level classification ratios, one row per result directory and level.

    Reads ``classification_metrics.tsv`` from each assignment result
    directory, falling back to recomputing from
    ``classification_accuracy_log.tsv`` for results written before the sidecar
    existed. Columns: ``Dataset``, ``novel_level`` (novel-taxa simulation
    level, ``<NA>`` otherwise), ``iteration``, ``Method``, ``Parameters``,
    ``level`` and ``CLASSIFICATION_METRIC_COLS``.
    """
    rows = []
    for results_dir in results_dirs:
        level_rows = _per_level_rows(results_dir)
        if level_rows is None:
            continue
        dataset_id, method_id, params_id = parse_assignment_results_dir(results_dir)
        database, novel_level, iteration = _dataset_parts_from_dataset_id(dataset_id)
        for level_row in level_rows:
            rows.append({
                "Dataset": database,
                "novel_level": novel_level,
                "iteration": iteration,
                "Method": method_id,
                "Parameters": params_id,
                **level_row,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["novel_level"] = df["novel_level"].astype("Int64")
    return df


def extract_per_level_classification_ratios(results_dirs):
    """Per-level classification metrics averaged across iterations.

    Groups by Dataset, novel_level, Method, Parameters and level, so novel-taxa
    simulations at different novel levels are never averaged together.
    """
    df = extract_per_level_classification_ratios_by_fold(results_dirs)
    if df.empty:
        return df
    return df.groupby(
        ["Dataset", "novel_level", "Method", "Parameters", "level"],
        as_index=False,
        dropna=False,
    )[[column for column in CLASSIFICATION_METRIC_COLS
        if column in df.columns]].mean()


# Error ratios and dissimilarities, where the best run has the lowest value.
LOWER_IS_BETTER_METRICS = frozenset({
    "misclassification_ratio",
    "overclassification_ratio",
    "underclassification_ratio",
    "Bray-Curtis",
})


def select_best_runs(df, metrics, group_cols=("Dataset",),
                     run_cols=("Method", "Parameters"), tolerance=1e-9):
    """Pick the best method + parameter run per group for each metric.

    *df* holds one row per fold at a single taxonomic level (or novel level).
    Each run's metric is averaged across its rows; metrics in
    ``LOWER_IS_BETTER_METRICS`` are minimised and all others maximised. Ties
    within *tolerance* go to the first run sorted by *run_cols*. Metric names
    are resolved through ``plot_theme.METRIC_ALIASES``; audit-only columns
    raise ``ValueError``.

    Returns one row per group and metric with the group columns, ``metric``,
    ``direction`` (``highest`` / ``lowest``), the run columns, ``value``
    (mean score), ``n_folds`` and ``n_tied`` (runs sharing the best value).
    """
    from tax_credit.plot_theme import NON_PLOT_METRICS, resolve_metric

    group_cols, run_cols = list(group_cols), list(run_cols)
    metrics = [resolve_metric(metric) for metric in metrics]
    audit_only = [metric for metric in metrics if metric in NON_PLOT_METRICS]
    if audit_only:
        raise ValueError(
            "These are reported for audit only and cannot rank runs: "
            f"{audit_only}. The original scoring (*_original) and the raw "
            "tallies are not comparable across runs with different numbers of "
            "sequences or reads.")
    grouped = df.groupby(group_cols + run_cols, dropna=False)
    means = grouped[metrics].mean()
    means["n_folds"] = grouped.size()
    means = means.reset_index()

    rows = []
    for group_key, group in means.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for metric in metrics:
            scored = group.dropna(subset=[metric])
            if scored.empty:
                continue
            lower = metric in LOWER_IS_BETTER_METRICS
            best_value = scored[metric].min() if lower else scored[metric].max()
            tied = scored[(scored[metric] - best_value).abs() <= tolerance]
            best = tied.sort_values(run_cols).iloc[0]
            rows.append({
                **dict(zip(group_cols, group_key)),
                "metric": metric,
                "direction": "lowest" if lower else "highest",
                **{col: best[col] for col in run_cols},
                "value": float(best_value),
                "n_folds": int(best["n_folds"]),
                "n_tied": len(tied),
            })
    return pd.DataFrame(rows)
