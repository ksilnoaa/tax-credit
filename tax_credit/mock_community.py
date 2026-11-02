#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2014--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Mock-community evaluation of taxonomy assignments.

Every function here works on in-memory tables, with no directory layout or
file-name conventions. It replaces the BIOM-on-disk workflow of the upstream
``mock_evaluation`` / ``eval_framework`` modules, which are not part of this
Tourmaline-integrated fork.

Inputs
------
counts
    ``DataFrame`` of read counts, one row per ASV (index = feature id) and one
    column per sample.
assignments
    ``Series`` of observed lineages from a classifier, indexed by feature id.
composition
    ``DataFrame`` of expected relative abundance, one row per unique lineage
    (index) and one column per mock sample.
asv_taxonomy
    ``Series`` of known (true) lineages, indexed by feature id.
ranks
    Rank names for positions in a lineage, e.g.
    ``["kingdom", ..., "species"]``. The *level* of a rank is its 0-based
    position, so with the default seven ranks species is level 6.

Lineage handling
----------------
``normalize_lineage`` is applied to every lineage, expected and observed:
whitespace around each rank is stripped, trailing ``NA`` / empty ranks are
dropped (internal ``NA`` ranks keep their position), and ``Unassigned``,
``Unclassified`` and ``No blast hit`` become the empty lineage.

At a given rank, a lineage is *resolved* if it has a name (not ``NA``) at
that rank. ``A;B;C`` is resolved at level 2 but not at level 3.

Metrics (per sample, per rank)
------------------------------
Taxon Accuracy Rate (TAR, also reported as ``precision_pa``)
    ``tp_pa / (tp_pa + fp_pa)``, i.e. |observed ∩ expected| / |observed|. NaN
    when no taxa are observed.
Taxon Detection Rate (TDR, also reported as ``recall_pa``)
    ``tp_pa / (tp_pa + fn_pa)``, i.e. |observed ∩ expected| / |expected|. NaN
    when no taxa are expected.
    Observed taxa are those above ``min_relative_abundance`` of the sample's
    reads; expected taxa are those with expected abundance > 0. By default only
    lineages resolved at the rank are counted on either side. With
    ``legacy_unresolved_taxa=True`` unresolved lineages count as taxa too
    (truncated lineages and ``Unassigned``), as in the original tax-credit.
    The underlying counts are reported too: ``tp_pa`` (shared taxa), ``fp_pa``
    (observed but not expected), ``fn_pa`` (expected but not observed), and
    the totals ``tot_val_obs`` (= tp_pa + fp_pa) and ``tot_val_exp``
    (= tp_pa + fn_pa). These are presence/absence counts of taxa in the
    community, not of ASVs.
Bray-Curtis
    Dissimilarity between expected and observed relative abundance at the
    rank. Unresolved lineages keep their truncated label (``Unassigned`` for
    none), so an ASV assigned only to genus matches an expected taxon only
    known to genus. No abundance threshold is applied.
Precision, Recall, F-measure (needs ``asv_taxonomy``)
    Each ASV's observed lineage is scored against its known lineage with
    ``framework_functions.evaluate_classification``, and the tallies are turned
    into scores by ``metrics.metrics_from_outcomes``, which defines them:
    match = true positive, underclassification = false negative, and
    over- and misclassification = false positive, with misclassification also a
    false negative. Overclassification is *not* a false negative -- assigning
    deeper than the reference knows does not count against recall -- but it is
    still a false positive, so precision penalises it.

    ``Recall_original`` and ``F-measure_original`` report the original
    tax-credit scoring, in which overclassification was a false negative too.
    Precision is unchanged by this, so there is no ``Precision_original``.

    These columns are weighted by the ASV's reads in the sample, and the
    tallies ``TP`` / ``FP`` / ``FN`` / ``FN_original`` are read sums that
    reconcile with them. ``ASV Precision`` / ``ASV Recall`` /
    ``ASV F-measure`` / ``ASV Recall_original`` /
    ``ASV F-measure_original`` weight every ASV equally, and the tallies
    ``n_match`` / ``n_under`` / ``n_over`` / ``n_mis`` are ASV counts that
    reconcile with those. ``*_ratio`` columns give the read-weighted fraction
    of each classification outcome. ASVs without a known lineage are left out.
"""

import re

import numpy as np
import pandas as pd

from tax_credit.framework_functions import score_classifications
from tax_credit.metrics import CLASSIFICATION_OUTCOMES  # noqa: F401
from tax_credit.taxa_manipulator import normalize_taxon

UNASSIGNED_LABELS = frozenset({"", "Unassigned", "Unclassified", "No blast hit"})
UNASSIGNED = "Unassigned"
DEFAULT_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus",
                 "species")

# Composition metrics: presence/absence of taxa, not per-ASV scoring.
COMPOSITION_METRIC_COLUMNS = [
    "Taxon Accuracy Rate",
    "Taxon Detection Rate",
    "Bray-Curtis",
]

# Per-ASV scoring weighted by each ASV's reads in the sample. The tallies are
# read sums, so TP / (TP + FP) reproduces Precision.
READ_WEIGHTED_SCORE_COLUMNS = [
    "Precision",
    "Recall",
    "F-measure",
    "Recall_original",
    "F-measure_original",
    "TP",
    "FP",
    "FN",
    "FN_original",
]

# The same scoring with every ASV weighted equally. The tallies are ASV
# counts, so n_match / (n_match + n_over + n_mis) reproduces ASV Precision.
ASV_SCORE_COLUMNS = [
    "ASV Precision",
    "ASV Recall",
    "ASV F-measure",
    "ASV Recall_original",
    "ASV F-measure_original",
    "n_match",
    "n_under",
    "n_over",
    "n_mis",
]

# Read-weighted fraction of each classification outcome.
CLASSIFICATION_RATIO_COLUMNS = [
    "match_ratio",
    "underclassification_ratio",
    "overclassification_ratio",
    "misclassification_ratio",
]

# Everything classification_scores fills in.
CLASSIFICATION_SCORE_COLUMNS = (READ_WEIGHTED_SCORE_COLUMNS
                                + ASV_SCORE_COLUMNS
                                + CLASSIFICATION_RATIO_COLUMNS)

METRIC_COLUMNS = COMPOSITION_METRIC_COLUMNS + CLASSIFICATION_SCORE_COLUMNS

COUNT_COLUMNS = [
    "tp_pa",
    "fp_pa",
    "fn_pa",
    "tot_val_exp",
    "tot_val_obs",
    "n_asvs_scored",
    "reads_scored_fraction",
]

_RANK_PREFIX = re.compile(r"^[a-zA-Z]__")


# --- Lineages --------------------------------------------------------------

def normalize_lineage(taxon, delim=";"):
    """Canonical lineage string; ``''`` when nothing is assigned."""
    if taxon is None or (isinstance(taxon, float) and np.isnan(taxon)):
        return ""
    ranks = [rank.strip() for rank in str(taxon).strip().split(delim)]
    lineage = normalize_taxon(delim.join(ranks), delim)
    return "" if lineage in UNASSIGNED_LABELS else lineage


def truncate_lineage(lineage, level, delim=";"):
    """Return ``(label, resolved)`` for *lineage* cut after rank *level*.

    *label* is ``Unassigned`` for an empty lineage. *resolved* is True when
    the lineage has a name at *level*; an internal ``NA`` at *level* counts as
    unresolved (``A;B;NA;D`` at level 2 gives ``("A;B", False)``).
    """
    ranks = lineage.split(delim) if lineage else []
    label = normalize_taxon(delim.join(ranks[:level + 1]), delim)
    resolved = bool(label) and len(label.split(delim)) > level
    return (label or UNASSIGNED), resolved


def rank_levels(ranks, eval_ranks):
    """Map each rank in *eval_ranks* to its level in *ranks*."""
    ranks = list(ranks)
    missing = [rank for rank in eval_ranks if rank not in ranks]
    if missing:
        raise ValueError(
            f"Evaluation ranks {missing} are not in the rank list {ranks}")
    return {rank: ranks.index(rank) for rank in eval_ranks}


# --- Readers ---------------------------------------------------------------

def read_feature_table(fp):
    """Read ASV read counts from a TSV or BIOM file.

    TSV: first column holds feature ids, the header row holds sample ids. A
    leading ``# Constructed from biom file`` line (``biom convert`` output) is
    skipped.
    """
    if str(fp).endswith(".biom"):
        from biom import load_table
        table = load_table(fp).to_dataframe(dense=True)
    else:
        with open(fp, encoding="utf-8") as fh:
            first = fh.readline()
        skip = 1 if first.startswith("# Constructed from biom file") else 0
        table = pd.read_csv(fp, sep="\t", skiprows=skip, index_col=0,
                            dtype={0: str})
    table.index = table.index.astype(str).str.strip()
    table.columns = table.columns.astype(str).str.strip()
    table.index.name = "Feature ID"
    if table.index.duplicated().any():
        dups = sorted(set(table.index[table.index.duplicated()]))[:5]
        raise ValueError(f"{fp}: duplicate feature ids, e.g. {dups}")
    try:
        table = table.apply(pd.to_numeric).fillna(0)
    except ValueError as exc:
        raise ValueError(f"{fp}: read counts must be numeric ({exc})") from exc
    if (table < 0).any().any():
        raise ValueError(f"{fp}: read counts cannot be negative")
    return table


def read_composition(fp):
    """Read expected relative abundances: lineages as rows, samples as columns.

    Lineages are normalized and rows with the same normalized lineage are
    summed. Missing values are 0. Each sample column is rescaled to sum to 1.
    """
    table = pd.read_csv(fp, sep="\t", index_col=0, dtype={0: str})
    table.columns = table.columns.astype(str).str.strip()
    lineages = table.index.map(normalize_lineage)
    if (lineages == "").any():
        raise ValueError(
            f"{fp}: every row needs a taxonomy; empty or Unassigned rows are "
            "not allowed")
    try:
        table = table.apply(pd.to_numeric).fillna(0.0)
    except ValueError as exc:
        raise ValueError(
            f"{fp}: relative abundances must be numeric ({exc})") from exc
    if (table < 0).any().any():
        raise ValueError(f"{fp}: relative abundances cannot be negative")
    table = table.groupby(lineages).sum()
    table.index.name = "Taxon"
    totals = table.sum()
    empty = list(totals[totals <= 0].index)
    if empty:
        raise ValueError(f"{fp}: samples with no expected abundance: {empty}")
    return table / totals


def read_taxonomy(fp):
    """Read a ``Feature ID`` / ``Taxon`` table (other columns are ignored).

    Used for known ASV taxonomy and for classifier output (QIIME 2
    ``taxonomy.tsv``). Returns normalized lineages indexed by feature id.
    """
    table = pd.read_csv(fp, sep="\t", dtype=str, comment=None)
    table.columns = table.columns.str.strip()
    missing = [col for col in ("Feature ID", "Taxon") if col not in table]
    if missing:
        raise ValueError(
            f"{fp}: missing column(s) {missing}; the header must include "
            "'Feature ID' and 'Taxon'")
    table = table[~table["Feature ID"].astype(str).str.startswith("#q2:")]
    ids = table["Feature ID"].astype(str).str.strip()
    if ids.duplicated().any():
        dups = sorted(set(ids[ids.duplicated()]))[:5]
        raise ValueError(f"{fp}: duplicate Feature IDs, e.g. {dups}")
    return pd.Series(
        [normalize_lineage(t) for t in table["Taxon"]], index=ids,
        name="Taxon")


def read_reference_taxonomy(fp):
    """Read a reference database taxonomy (headerless or with a header)."""
    table = pd.read_csv(fp, sep="\t", header=None, dtype=str, usecols=[0, 1])
    if str(table.iloc[0, 0]).strip() in ("Feature ID", "Feature-ID", "id"):
        table = table.iloc[1:]
    return pd.Series(
        [normalize_lineage(t) for t in table[1]],
        index=table[0].astype(str).str.strip(), name="Taxon")


# --- Expected composition and samples --------------------------------------

def composition_from_asv_taxonomy(counts, asv_taxonomy):
    """Expected composition from known ASV lineages and their read counts.

    Relative abundance is computed over ASVs with a known lineage only.
    Samples without reads from such ASVs are dropped.
    """
    known = asv_taxonomy[asv_taxonomy != ""]
    ids = counts.index.intersection(known.index)
    table = counts.loc[ids].groupby(known.loc[ids]).sum()
    table.index.name = "Taxon"
    totals = table.sum()
    table = table.loc[:, totals > 0]
    return table / totals[totals > 0]


EXCLUDED_NOT_IN_COMPOSITION = "not in composition"
EXCLUDED_NOT_IN_FEATURE_TABLE = "in composition but not in feature table"
EXCLUDED_NO_KNOWN_READS = "no reads from ASVs with known taxonomy"
EXCLUDED_NOT_REQUESTED = "not in datasets.samples"


def select_mock_samples(counts, composition=None, asv_taxonomy=None,
                        requested=None):
    """Samples to evaluate, and the samples left out with the reason.

    With a composition, samples are its columns that are also in *counts*.
    Without one, samples are those in *counts* with reads from ASVs with a
    known lineage. *requested* restricts either set; requested samples that are
    not available raise ``ValueError``.

    Returns ``(samples, excluded)``; *excluded* is a list of
    ``{"sample_id", "reason"}`` dicts using the ``EXCLUDED_*`` reasons. It
    includes composition samples missing from the feature table.
    """
    table_samples = list(counts.columns)
    excluded = []
    if composition is not None:
        in_table = set(table_samples)
        available = [s for s in composition.columns if s in in_table]
        excluded.extend(
            {"sample_id": s, "reason": EXCLUDED_NOT_IN_FEATURE_TABLE}
            for s in composition.columns if s not in in_table)
        unavailable_reason = EXCLUDED_NOT_IN_COMPOSITION
    elif asv_taxonomy is not None:
        derived = composition_from_asv_taxonomy(counts, asv_taxonomy)
        available = list(derived.columns)
        unavailable_reason = EXCLUDED_NO_KNOWN_READS
    else:
        raise ValueError("A composition or an ASV taxonomy is required.")

    if requested:
        requested = [str(s) for s in requested]
        unavailable = [s for s in requested if s not in available]
        if unavailable:
            raise ValueError(
                f"Requested mock samples not available: {unavailable}. "
                f"Available: {available}")
        samples = requested
    else:
        samples = available
    chosen, available_set = set(samples), set(available)
    for sample in table_samples:
        if sample in chosen:
            continue
        reason = (EXCLUDED_NOT_REQUESTED if sample in available_set
                  else unavailable_reason)
        excluded.append({"sample_id": sample, "reason": reason})
    return samples, excluded


def unscored_asvs(counts, asv_taxonomy, samples):
    """ASVs in *counts* that precision / recall cannot score.

    These are ASVs missing from *asv_taxonomy* or whose known lineage is empty
    or ``Unassigned``. Returns a DataFrame with ``Feature ID``, ``reason``,
    ``reads_in_mock_samples`` (summed over *samples*) and ``reads_total``,
    sorted by reads in mock samples.
    """
    asv_taxonomy = asv_taxonomy.map(normalize_lineage)
    known = asv_taxonomy[asv_taxonomy != ""]
    rows = []
    mock = counts[list(samples)].sum(axis=1) if len(samples) else counts.sum(axis=1) * 0
    total = counts.sum(axis=1)
    for feature_id in counts.index:
        if feature_id in known.index:
            continue
        reason = ("empty or Unassigned in asv_taxonomy"
                  if feature_id in asv_taxonomy.index else "not in asv_taxonomy")
        rows.append({
            "Feature ID": feature_id,
            "reason": reason,
            "reads_in_mock_samples": mock[feature_id],
            "reads_total": total[feature_id],
        })
    table = pd.DataFrame(
        rows, columns=["Feature ID", "reason", "reads_in_mock_samples",
                       "reads_total"])
    return table.sort_values(
        ["reads_in_mock_samples", "Feature ID"], ascending=[False, True],
        ignore_index=True)


# --- Collapsing ------------------------------------------------------------

def collapse_observed(sample_counts, assignments, level):
    """Observed relative abundance per truncated lineage at *level*.

    Returns ``(abundance, resolved)``: two Series indexed by label. ASVs
    missing from *assignments* are ``Unassigned``. Abundance is relative to
    all reads in the sample.
    """
    sample_counts = sample_counts[sample_counts > 0]
    total = sample_counts.sum()
    if total <= 0:
        return pd.Series(dtype=float), pd.Series(dtype=bool)
    lineages = assignments.reindex(sample_counts.index).fillna("")
    return _collapse(sample_counts / total, lineages, level)


def collapse_expected(sample_composition, level):
    """Expected relative abundance per truncated lineage at *level*."""
    sample_composition = sample_composition[sample_composition > 0]
    lineages = pd.Series(sample_composition.index, index=sample_composition.index)
    return _collapse(sample_composition, lineages, level)


def _collapse(abundance, lineages, level):
    labels, resolved = {}, {}
    for key, lineage in lineages.items():
        label, is_resolved = truncate_lineage(lineage, level)
        labels[key] = label
        resolved[label] = is_resolved
    collapsed = abundance.groupby(pd.Series(labels)).sum()
    return collapsed, pd.Series(resolved).reindex(collapsed.index)


# --- Metrics ---------------------------------------------------------------

def taxon_accuracy_detection(observed, observed_resolved, expected,
                             expected_resolved, min_relative_abundance=0.0,
                             legacy_unresolved_taxa=False):
    """Return ``(TAR, TDR, n_observed, n_expected, n_shared)``."""
    obs = observed[observed > min_relative_abundance].index
    exp = expected[expected > 0].index
    if not legacy_unresolved_taxa:
        obs = [t for t in obs if observed_resolved[t]]
        exp = [t for t in exp if expected_resolved[t]]
    obs, exp = set(obs), set(exp)
    shared = len(obs & exp)
    tar = shared / len(obs) if obs else np.nan
    tdr = shared / len(exp) if exp else np.nan
    return tar, tdr, len(obs), len(exp), shared


def bray_curtis(observed, expected):
    """Bray-Curtis dissimilarity between two abundance Series."""
    labels = observed.index.union(expected.index)
    x = observed.reindex(labels, fill_value=0.0).astype(float)
    y = expected.reindex(labels, fill_value=0.0).astype(float)
    denom = x.sum() + y.sum()
    if denom <= 0:
        return np.nan
    return float((x - y).abs().sum() / denom)


def classification_scores(sample_counts, assignments, asv_taxonomy, level):
    """Per-ASV scores and classification ratios for one sample at one rank.

    Returns a dict with every ``CLASSIFICATION_SCORE_COLUMNS`` entry plus
    ``n_asvs_scored`` and ``reads_scored_fraction``, all NaN when no ASV in the
    sample has a known lineage.

    Each ASV is scored twice, by ``framework_functions.score_classifications``:
    once weighted by its reads in the sample, giving ``Precision`` / ``Recall``
    / ``F-measure`` and the read tallies ``TP`` / ``FP`` / ``FN`` /
    ``FN_original``; and once with every ASV weighted equally, giving the
    ``ASV *`` scores and the ASV tallies ``n_match`` / ``n_under`` / ``n_over``
    / ``n_mis``. Each set of tallies reconciles with the scores beside it.
    ``Recall_original`` and ``F-measure_original`` carry the original
    tax-credit scoring, in which overclassification also counts as a false
    negative.
    """
    present = sample_counts[sample_counts > 0]
    known = asv_taxonomy.reindex(present.index).fillna("")
    known = known[known != ""]
    scores = {col: np.nan for col in CLASSIFICATION_SCORE_COLUMNS}
    scores["n_asvs_scored"] = len(known)
    total = present.sum()
    scores["reads_scored_fraction"] = (
        float(present[known.index].sum() / total) if total > 0 else np.nan)
    if known.empty:
        return scores

    exp, obs = [], []
    for feature_id, lineage in known.items():
        exp.append(truncate_lineage(lineage, level)[0])
        obs.append(truncate_lineage(assignments.get(feature_id, ""), level)[0])
    weights = [float(present[f]) for f in known.index]

    # read-weighted: the headline scores, the read tallies, and the ratios
    weighted = score_classifications(exp, obs, sample_weight=weights)
    for column in READ_WEIGHTED_SCORE_COLUMNS + CLASSIFICATION_RATIO_COLUMNS:
        scores[column] = weighted[column]

    # every ASV weighted equally
    per_asv = score_classifications(exp, obs)
    for column in ("Precision", "Recall", "F-measure", "Recall_original",
                   "F-measure_original"):
        scores[f"ASV {column}"] = per_asv[column]
    for column in ("n_match", "n_under", "n_over", "n_mis"):
        scores[column] = per_asv[column]
    return scores


def evaluate_mock_samples(counts, assignments, ranks, eval_ranks, samples,
                          composition=None, asv_taxonomy=None,
                          min_relative_abundance=0.0,
                          legacy_unresolved_taxa=False):
    """Score one set of taxonomy assignments against the expected mock data.

    Returns ``(metrics, composition_table)``:

    metrics
        One row per sample and rank: ``SampleID``, ``rank``, ``level``, the
        ``METRIC_COLUMNS`` (NaN when not computable) and ``COUNT_COLUMNS``.
    composition_table
        Expected and observed relative abundance per sample, rank and taxon
        label (columns ``SampleID``, ``rank``, ``level``, ``taxon``,
        ``resolved``, ``expected``, ``observed``).

    If *composition* is None it is built from *asv_taxonomy* and *counts*.
    Precision / recall columns are only filled when *asv_taxonomy* is given.
    """
    if composition is None:
        if asv_taxonomy is None:
            raise ValueError("A composition or an ASV taxonomy is required.")
        composition = composition_from_asv_taxonomy(counts, asv_taxonomy)
    levels = rank_levels(ranks, eval_ranks)

    metric_rows, composition_rows = [], []
    for sample in samples:
        sample_counts = counts[sample]
        sample_expected = composition[sample]
        for rank, level in levels.items():
            observed, observed_resolved = collapse_observed(
                sample_counts, assignments, level)
            expected, expected_resolved = collapse_expected(
                sample_expected, level)
            tar, tdr, n_obs, n_exp, n_shared = taxon_accuracy_detection(
                observed, observed_resolved, expected, expected_resolved,
                min_relative_abundance, legacy_unresolved_taxa)
            row = {
                "SampleID": sample,
                "rank": rank,
                "level": level,
                "Taxon Accuracy Rate": tar,
                "Taxon Detection Rate": tdr,
                "Bray-Curtis": bray_curtis(observed, expected),
                "tp_pa": n_shared,
                "fp_pa": n_obs - n_shared,
                "fn_pa": n_exp - n_shared,
                "tot_val_exp": n_exp,
                "tot_val_obs": n_obs,
            }
            if asv_taxonomy is not None:
                row.update(classification_scores(
                    sample_counts, assignments, asv_taxonomy, level))
            metric_rows.append(row)

            resolved = pd.concat([observed_resolved, expected_resolved])
            resolved = resolved[~resolved.index.duplicated()]
            for taxon in observed.index.union(expected.index):
                composition_rows.append({
                    "SampleID": sample,
                    "rank": rank,
                    "level": level,
                    "taxon": taxon,
                    "resolved": bool(resolved[taxon]),
                    "expected": float(expected.get(taxon, 0.0)),
                    "observed": float(observed.get(taxon, 0.0)),
                })

    metrics = pd.DataFrame(metric_rows).reindex(
        columns=["SampleID", "rank", "level"] + METRIC_COLUMNS + COUNT_COLUMNS)
    composition_table = pd.DataFrame(
        composition_rows,
        columns=["SampleID", "rank", "level", "taxon", "resolved", "expected",
                 "observed"])
    return metrics, composition_table


# --- Backbone check --------------------------------------------------------

BACKBONE_FOUND = "found"
BACKBONE_DIFFERENT_LINEAGE = "different_lineage"
BACKBONE_NOT_IN_DATABASE = "not_in_database"


def check_backbone(expected_lineages, reference_lineages, ranks, eval_ranks,
                   max_candidates=3):
    """Compare expected lineages with a reference database's taxonomy.

    For every expected lineage resolved at each evaluation rank, reports
    whether the truncated lineage exists in the database:

    ``found``
        The lineage exists in the database.
    ``different_lineage``
        The name at that rank exists in the database, but under a different
        lineage (``database_lineages`` lists up to *max_candidates*). This is
        the signature of expected taxa written against another backbone.
    ``not_in_database``
        The name is not in the database at that rank at all. The database
        may simply lack that taxon, or the names differ between backbones.

    Returns ``(report, messages)``. *messages* flags differences in rank count
    and in rank-prefix style (``g__Name`` vs ``Name``).
    """
    expected = sorted({normalize_lineage(t) for t in expected_lineages} - {""})
    reference = [t for t in (normalize_lineage(t) for t in reference_lineages)
                 if t]
    levels = rank_levels(ranks, eval_ranks)
    messages = []

    ref_depth = max((len(t.split(";")) for t in reference), default=0)
    exp_depth = max((len(t.split(";")) for t in expected), default=0)
    if ref_depth and ref_depth != len(ranks):
        messages.append(
            f"database lineages have up to {ref_depth} ranks but the rank "
            f"list has {len(ranks)} ({','.join(ranks)})")
    if exp_depth > len(ranks):
        messages.append(
            f"expected lineages have up to {exp_depth} ranks but the rank "
            f"list has only {len(ranks)}")
    ref_prefixed = _prefixed_fraction(reference)
    exp_prefixed = _prefixed_fraction(expected)
    if abs(ref_prefixed - exp_prefixed) > 0.5:
        messages.append(
            f"rank prefixes differ: {ref_prefixed:.0%} of database ranks and "
            f"{exp_prefixed:.0%} of expected ranks look like 'g__Name'")

    rows = []
    for rank, level in levels.items():
        ref_lineages, ref_by_name = set(), {}
        for lineage in reference:
            label, resolved = truncate_lineage(lineage, level)
            if not resolved:
                continue
            ref_lineages.add(label)
            ref_by_name.setdefault(label.split(";")[-1], set()).add(label)
        seen = set()
        for lineage in expected:
            label, resolved = truncate_lineage(lineage, level)
            if not resolved or label in seen:
                continue
            seen.add(label)
            name = label.split(";")[-1]
            candidates = []
            if label in ref_lineages:
                status = BACKBONE_FOUND
            elif name != "NA" and name in ref_by_name:
                status = BACKBONE_DIFFERENT_LINEAGE
                candidates = sorted(ref_by_name[name])[:max_candidates]
            else:
                status = BACKBONE_NOT_IN_DATABASE
            rows.append({
                "rank": rank,
                "taxon": label,
                "status": status,
                "database_lineages": " | ".join(candidates),
            })
    report = pd.DataFrame(
        rows, columns=["rank", "taxon", "status", "database_lineages"])
    return report, messages


def _prefixed_fraction(lineages):
    parts = [rank for lineage in lineages for rank in lineage.split(";")
             if rank and rank != "NA"]
    if not parts:
        return 0.0
    return sum(bool(_RANK_PREFIX.match(rank)) for rank in parts) / len(parts)
