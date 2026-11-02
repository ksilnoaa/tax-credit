# Python API reference

This page documents the main public entry points of this Tourmaline-integrated fork. In normal use you do not import these yourself — Tourmaline's `scripts/run_tax_credit.py` calls them for you. They are documented here for extending the framework or for interactive work of the kind shown in [`examples/`](../examples/).

> **Removed in this fork.** The legacy notebook-only modules `eval_framework`, `mock_evaluation`, `biom_cache`, `process_mocks`, `mock_denoise`, `mock_transport`, `mock_quality`, `mockrobiota_extract` and `simulated_communities` are gone. Mock-community scoring now lives in `tax_credit.mock_community`.

**Suggested imports**

```python
from tax_credit import mock_community
from tax_credit.novel_evaluation import (
    novel_taxa_classification_evaluation,
    extract_per_level_accuracy,
    select_best_runs,
)
from tax_credit.framework_functions import (
    generate_simulated_datasets,
    recall_simulated_taxa_dirs,
    parameter_sweep,
)
from tax_credit import paths
from tax_credit import simulation_names
```

---

## `tax_credit.mock_community`

Mock-community evaluation: observed feature tables and per-ASV assignments versus the expected composition and/or per-ASV "trueish" taxonomies. Replaces the removed `mock_evaluation` / `eval_framework` scoring path and works directly from TSV/BIOM inputs rather than mounted BIOM objects.

| Function | Purpose |
|----------|---------|
| `read_feature_table(fp)`, `read_composition(fp)`, `read_taxonomy(fp)`, `read_reference_taxonomy(fp)` | Input readers for counts, expected composition, assignments, and reference taxonomy. |
| `select_mock_samples(counts, composition=None, asv_taxonomy=None, ...)` | Pick the samples scoreable against the expected data available. |
| `collapse_observed(...)`, `collapse_expected(...)` | Collapse abundance to a taxonomic level for comparison. |
| `taxon_accuracy_detection(observed, observed_resolved, expected, ...)` | Taxon detection rate, false positives, and resolution-aware variants. |
| `classification_scores(sample_counts, assignments, asv_taxonomy, level)` | Per-ASV scores at one level, twice: read-weighted (`Precision`, `Recall`, `F-measure`, `Recall_original`, `F-measure_original`, tallies `TP` / `FP` / `FN` / `FN_original` as read sums, and the four ratios) and with every ASV weighted equally (`ASV *` scores, tallies `n_match` / `n_under` / `n_over` / `n_mis` as ASV counts). Built on `framework_functions.score_classifications`. |
| `bray_curtis(observed, expected)` | Composition dissimilarity between observed and expected. |
| `evaluate_mock_samples(counts, assignments, ranks, eval_ranks, samples, ...)` | Top-level driver returning one row of metrics per sample and level. |
| `check_backbone(expected_lineages, reference_lineages, ranks, eval_ranks, ...)` | Report expected taxa that the reference database could never recover. |

Tourmaline's `scripts/tax_credit_mock.py` wraps these for config validation, input staging and summary writing.

Output columns are grouped in module constants: `COMPOSITION_METRIC_COLUMNS` (`Taxon Accuracy Rate`, `Taxon Detection Rate`, `Bray-Curtis`), `READ_WEIGHTED_SCORE_COLUMNS`, `ASV_SCORE_COLUMNS` and `CLASSIFICATION_RATIO_COLUMNS` (together `CLASSIFICATION_SCORE_COLUMNS`), with `METRIC_COLUMNS` the union. `COUNT_COLUMNS` holds the presence/absence counts behind the taxon rates — `tp_pa`, `fp_pa`, `fn_pa`, `tot_val_exp`, `tot_val_obs` (these **replace** `n_shared_taxa`, `n_observed_taxa` and `n_expected_taxa`) — plus `n_asvs_scored` and `reads_scored_fraction`. `Taxon Accuracy Rate` is `tp_pa / tot_val_obs` and `Taxon Detection Rate` is `tp_pa / tot_val_exp`, so the two are also known as `precision_pa` and `recall_pa` (see `plot_theme.METRIC_ALIASES`).

---

## `tax_credit.novel_evaluation`

Text-based assignment evaluation for **novel-taxa**, **cross-validated**, **cross-validated-trad**, and **self-validated** layouts (not BIOM composition tables). Uses `framework_functions.load_prf`, `compute_prf`, `evaluate_classification`, `find_last_common_ancestor`, and `paths.QUERY_*` filenames.

### `novel_taxa_classification_evaluation(results_dirs, expected_results_dir, summary_fp, test_type='novel-taxa', rank_positions=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `results_dirs` | iterable of `str` | Each path must end with `dataset_id/method_id/params_id` (`parse_assignment_results_dir`). Must contain `query_tax_assignments.txt`. Build with `paths.list_assignment_result_dirs(results_root)` when outputs use the standard four-level sweep tree. |
| `expected_results_dir` | `str` | Must contain `join(expected_results_dir, dataset_id, query_taxa.tsv)` for each dataset. |
| `summary_fp` | `str` | Where to write the summary CSV (pandas default comma separator). |
| `test_type` | `str` | `'novel-taxa'`, `'cross-validated'`, `'cross-validated-trad'`, or `'self-validated'`. Selects how `dataset_id` is parsed (`parse_novel_dataset_id`, `parse_cv_dataset_id`, or `parse_self_validated_dataset_id`). Other values raise `ValueError`. |
| `rank_positions` | `dict`, optional | Database name → the seven rank positions to score, for a database whose taxonomy is not the standard seven ranks (passed to `load_prf`). Databases not listed keep the first seven ranks. |

**Per directory:** writes two files under that results dir — the per-read `classification_accuracy_log.tsv` (`CLASSIFICATION_ACCURACY_LOG_TSV`) and the per-rank `classification_metrics.tsv` (`CLASSIFICATION_METRICS_TSV`) — appends one summary row, and returns the full `DataFrame`.

**Output columns:** `Dataset`, `level`, `iteration`, `Method`, `Parameters`, `match_ratio`, `overclassification_ratio`, `underclassification_ratio`, `misclassification_ratio`, `mismatch_level_list`, `Precision`, `Recall`, `F-measure`, `Recall_original`, `F-measure_original`, `TP`, `FP`, `FN`, `FN_original`.

For the cross-validated and self-validated types every score column holds a length-7 list indexed by taxonomic level; index 0 (kingdom) is never scored and is `NaN`. The four ratio columns describe the full-depth comparison only. For novel-taxa every score is a single full-depth value.

`Recall_original` and `F-measure_original` report the original tax-credit scoring, in which overclassification also counted as a false negative. See `metrics.metrics_from_outcomes` for the definitions. **`match_ratio` no longer equals `Recall`** — it equals `Recall_original`; the new `Recall` is `match_ratio / (1 − overclassification_ratio)`.

`classification_accuracy_log.tsv` has one line per query (`dataset`, `level`, `iteration`, `method`, `parameters`, `observed_taxonomy`, `expected_taxonomy`, `result`, `mismatch_level`) followed by the fold's `Precision`, `Recall`, `F-measure`, `Recall_original` and `F-measure_original`. `classification_metrics.tsv` has one row per rank 1–6 (`level`, `rank`) with the identifying columns (`Dataset`, `novel_level`, `iteration`, `Method`, `Parameters`) and every `CLASSIFICATION_METRIC_COLUMNS` entry; it is written for every `test_type`, so novel-taxa results get per-rank scores too.

**Loading saved summaries:** `pd.read_csv(summary_fp, index_col=0)`.

---

### `extract_per_level_accuracy(df, columns=[...])`

Expands summary rows into **per-level** rows for plotting (levels `1..6` in the implementation).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `DataFrame` | (required) | Typically the return value of `novel_taxa_classification_evaluation`. |
| `columns` | `iterable` | `PER_LEVEL_METRIC_COLUMNS` (`Precision`, `Recall`, `F-measure`, `Recall_original`, `F-measure_original`, `TP`, `FP`, `FN`, `FN_original`) | Which columns to expand. Columns absent from `df` are skipped, so summaries written before a metric existed still load. Values stored as stringified lists (as read back from CSV) are parsed. |

**Returns:** `DataFrame` with columns `Dataset`, `level`, `iteration`, `Method`, `Parameters`, plus one column per expanded metric.

This function no longer derives `match_ratio`. It used to take it from `Recall`, which was valid only while every non-match counted as a false negative; use `extract_per_level_classification_ratios_by_fold` for ratios.

### `extract_per_level_classification_ratios_by_fold(results_dirs)` / `extract_per_level_classification_ratios(results_dirs)`

Per-rank classification metrics at levels `1..6` (phylum..species), read from each results directory's `classification_metrics.tsv` and recomputed from `classification_accuracy_log.tsv` when that sidecar is absent (results written before it existed). Every rank is scored independently, by re-classifying each read's observed and expected taxonomy truncated to that rank.

The `_by_fold` form returns one row per directory and level with `Dataset`, `novel_level` (novel-taxa simulation level, `<NA>` otherwise), `iteration`, `Method`, `Parameters`, `level` and every column in `CLASSIFICATION_METRIC_COLS` (the four ratios, the four outcome counts, `n_scored`, the five scores and the four tallies). The other form averages across iterations, grouped by `Dataset`, `novel_level`, `Method`, `Parameters` and `level`.

### `select_best_runs(df, metrics, group_cols=("Dataset",), run_cols=("Method", "Parameters"), tolerance=1e-9)`

For each group and metric, returns the run (method + parameters) with the best mean score across its rows. Metrics in `LOWER_IS_BETTER_METRICS` (mis-, over- and underclassification ratio) are minimised; all others are maximised. Metric names are resolved through `plot_theme.METRIC_ALIASES`, and audit-only columns (`plot_theme.NON_PLOT_METRICS` — the `*_original` scores and the raw tallies) raise `ValueError`, because they are not comparable across runs with different numbers of sequences or reads. Ties within `tolerance` go to the first run sorted by `run_cols`. Output columns: the group columns, `metric`, `direction`, the run columns, `value`, `n_folds`, `n_tied`. Filter `df` to a single level first.

---

## `tax_credit.metrics`

The definitions of every per-sequence classification metric. Imports nothing
else from the package, so both `framework_functions` and `novel_evaluation` use
it.

### `metrics_from_outcomes(match, underclassification, overclassification, misclassification)`

Takes four tallies and returns every metric derived from them, as a dict keyed
by `CLASSIFICATION_METRIC_COLUMNS`. The tallies may be sequence counts or
weights (e.g. read counts); the formulas are identical either way.

```
TP          = match
FP          = overclassification + misclassification
FN          = underclassification + misclassification
FN_original = underclassification + overclassification + misclassification

Precision          = TP / (TP + FP)
Recall             = TP / (TP + FN)
Recall_original    = TP / (TP + FN_original)
F-measure          = 2PR / (P + R)          using Recall
F-measure_original = 2PR / (P + R)          using Recall_original
match_ratio, underclassification_ratio, overclassification_ratio,
misclassification_ratio = each tally / n_scored
```

`FN` forgives overclassification: assigning deeper than the reference knows
does not count against sensitivity. Overclassification is still a false
positive, so `Precision` penalises it and `F-measure` therefore does too, half
as heavily as before. `FN_original` is the microaveraged count the original
tax-credit used.

Two consequences:

- `TP + FN` is **not** the number of records scored, so `Recall` is not a
  microaveraged recall. `Recall_original` is.
- `Recall` **no longer equals `match_ratio`**. `Recall_original` does; `Recall`
  is `match_ratio / (1 − overclassification_ratio)`, so the two agree only when
  nothing was overclassified.

Column groups, in output order: `OUTCOME_COUNT_COLUMNS` (`n_scored`, `n_match`,
`n_under`, `n_over`, `n_mis`), `SCORE_COUNT_COLUMNS` (`TP`, `FP`, `FN`,
`FN_original`), `SCORE_COLUMNS` (`Precision`, `Recall`, `Recall_original`,
`F-measure`, `F-measure_original`) and `RATIO_COLUMNS`; together
`CLASSIFICATION_METRIC_COLUMNS`. `ORIGINAL_SCORE_COLUMNS` names the two
audit-only scores. All of these are re-exported from `framework_functions`.

A score is `NaN` when its denominator holds no observations at all, and `0.0`
when there were observations but no true positives, so "nothing was scored"
stays distinguishable from "everything was wrong". `F-measure` is `0.0` when
precision and recall are both a real zero.

| situation | `Precision` | `Recall` | `F-measure` |
|---|---|---|---|
| nothing scored (`n_scored == 0`) | NaN | NaN | NaN |
| kingdom rank (never scored) | NaN | NaN | NaN |
| `TP == 0`, some FP and FN | 0 | 0 | 0 |
| all underclassification (`FP == 0`) | NaN | 0 | NaN |
| all overclassification (`FN == 0`) | 0 | NaN | NaN |

## `tax_credit.framework_functions` (selected)

Large module: simulation generation, parameter sweeps, PRF utilities, QIIME helpers, runtime benchmarking. Functions below are the ones **novel evaluation** depends on.

| Function | Signature highlights | Role |
|----------|---------------------|------|
| `load_prf(obs_fp, exp_fp, level=slice(0,7), sort=True, rank_positions=None)` | Paths or list-like inputs accepted via `load_taxa`. `rank_positions` (seven 0-based positions) replaces `level` for a database with extra ranks, via `taxa_manipulator.project_taxon`; raises `ValueError` if an expected taxonomy is deeper than the positions declare. | Align expected/observed taxon lists. |
| `compute_prf(exp, obs, test_type='cross-validated', l_range=range(1,7), sample_weight=None, exclude=None)` | `test_type` in `mock`, `novel-taxa`, `cross-validated`, `cross-validated-trad`, `self-validated`. | Returns a dict keyed by `CLASSIFICATION_METRIC_COLUMNS`; CV modes fill length-7 lists by level, with unscored levels `NaN`. |
| `score_classifications(exp, obs, sample_weight=None, exclude=None)` | Used by `compute_prf` and `mock_community`. | Tallies the four outcomes and returns the full metric dict. |
| `precision_recall_fscore(exp, obs, sample_weight=None, exclude=None)` | Thin wrapper on `score_classifications`. | The three headline scores only. *Recall* is the new definition. |
| `evaluate_classification(obs_taxon, exp_taxon)` | String taxonomies. | `'match'`, `'underclassification'`, `'overclassification'`, `'misclassification'`. |
| `find_last_common_ancestor(obs, exp)` | Taxonomy strings. | Index of shallowest mismatch (used for mismatch histograms). |

Simulation and sweep entry points (e.g. `generate_simulated_datasets`, `parameter_sweep`, `recall_simulated_taxa_dirs`) remain here; see below and source docstrings for full parameter lists.

### `recall_simulated_taxa_dirs(data_dir, databases, iterations, ref_seqs=..., ref_taxa=..., max_level=6, min_level=0, multilevel=True)`

Builds ``(dataset_reference_combinations, reference_dbs)`` for taxonomy-assignment parameter sweeps: fold directory names under ``data_dir`` (novel-taxa ``<db>-L<level>-iter<n>`` when ``multilevel=True``, or cross-validated ``<db>-iter<n>`` when ``multilevel=False``) mapped to ``ref_seqs`` / ``ref_taxa`` paths. Pass the appropriate root (e.g. ``novel_taxa_simulations_root(data_dir)`` or ``cross_validated_root(data_dir)``) as the first argument so paths resolve to the simulated tree you generated.

### `recall_self_validated_dirs(data_dir, databases, ref_seqs=..., ref_taxa=...)`

Same return shape as ``recall_simulated_taxa_dirs``, but for **self-validated** datasets: one ``(database, database)`` pair per reference database under ``self_validated_root(data_dir)`` (no CV fold iterations).

### `generate_self_validated_datasets(dataframe, data_dir, ...)`

Builds one self-validation dataset per reference database: every sequence is classified against the full database (no held-out folds, no ID removal from the reference). Writes under ``self-validated/<database>/``.

### `trad_cv_shared_reference_qzas(project_data_dir, reference_id)`

Returns ``(ref_seqs_qza, ref_taxa_qza)`` paths under ``ref_dbs/<reference_id>/`` for **cross-validated-trad** shared training artifacts (``_trad_cv_shared_ref_seqs.qza``, ``_trad_cv_shared_ref_taxa.qza``).

### `trad_cv_naive_bayes_commands_single_classifier(trad_sim_data_dir, project_data_dir, results_dir, database_names, iterations, method_parameters_combinations, ...)`

For **cross-validated-trad**, builds two shell-command lists: **fit** naive Bayes once per (database, **fit** parameter combo) into ``results_dir/<db>/<db>/<method>/<fit-params>/classifier.qza``, then **classify** each fold’s ``query.qza`` into ``results_dir/<fold-id>/<fold-id>/<method>/<run-id>/`` (same depth as ``parameter_sweep`` with ``multilevel=False``). Run all fit commands before classify. See [`examples/cross-validated-and-novel-taxa.ipynb`](../examples/cross-validated-and-novel-taxa.ipynb) (cross-validated assignment section).

**``method_parameters_combinations``:** per method, either a **flat** dict (all keys go to ``fit-classifier-naive-bayes``; classify uses only ``confidence`` and ``classify_n_jobs``), or ``{'fit': {...}, 'classify': {...}}`` where each inner dict maps QIIME flag stems to lists (Cartesian product). Classify flags (e.g. ``p-confidence``, ``p-n-jobs``, ``p-reads-per-batch``) are passed to ``classify-sklearn``; omitted ``p-confidence`` / ``p-n-jobs`` default from the function kwargs. When both sides sweep, result dirs use ``<fit-id>__cls__<classify-id>``.

`generate_simulated_datasets(..., simulation_method=...)` accepts a single value or any combination of:
- `cross-validated-taxa` (original taxonomy-aware CV output under `cross-validated/`)
- `cross-validated-trad` (random CV under `cross-validated-trad/`; each fold keeps only test sequences in `query.fasta` / `query_taxa.tsv`, while `ref_seqs.fasta` and `ref_taxa.tsv` are **symlinks** to the full simulated-reads FASTA and cleaned taxonomy TSV, so **the reference is the whole database in every fold, queries included**. Per-fold `ref_seqs.qza` and `ref_taxa.qza` symlink to shared QIIME artifacts in the ref database directory (`_trad_cv_shared_ref_seqs.qza` and `_trad_cv_shared_ref_taxa.qza`) to save disk space.)
- `novel-taxa` (novel-taxa output under `novel-taxa-simulations/`)

Default behavior generates **all three** simulation types.  
Backward compatibility: `cross-validated` is treated as an alias of `cross-validated-taxa`.

### Dropped queries in `cross-validated-taxa`

`generate_cross_validated_sequences` truncates each query's expected taxonomy to the deepest prefix the fold's training set still holds. A query with no prefix in training at all is **dropped from the fold** (it used to raise `RuntimeError('unknown kingdom in query set')` and abort the run), and the counts are printed per fold:

```
<db>-iter3: expected taxonomy truncated to the training set for 113 of 200 queries; 1 dropped (first rank absent from the training set: 'k__Archaea')
```

A fold left with no queries at all still raises, since it cannot be evaluated. `cross-validated-trad` and `self-validated` never drop anything: their reference is the whole database, so every query's full lineage is present by construction.

### `trad_cv_query_size` (cross-validated-trad query pool)

`generate_simulated_datasets(..., trad_cv_query_size=None)` forwards to `generate_cross_validated_trad_sequences(..., query_size=...)`. It sizes the **total query pool**, which is then divided evenly between the `iterations` folds — it is *not* the size of one fold. Other simulation methods ignore it.

| Value | Total pool | Per fold (with `iterations=8`) |
|---|---|---|
| `None` (default) | The whole database. | `n_sequences / 8`, i.e. 12.5% of the database. |
| float in `(0, 1]` | That **fraction** of the database — `0.2` is 20% of sequences. | 20%/8 = **2.5%** of the database. |
| int | That **absolute number** of sequences — `8000` is 8000 sequences. | 8000/8 = **1000** sequences. |

`1.0` is accepted and means the whole database, so it behaves exactly like the default.

An **int larger than a given database** is not an error: a `WARNING:` is printed and that whole database is used as the pool. This is what makes one int usable across databases of different sizes — every database big enough supplies the same query count, and smaller ones contribute everything they have (the warning says so, since their query counts will not match).

In every case a random pool of the requested size is drawn with a fixed seed, then split with `KFold(n_splits=iterations, shuffle=True, random_state=0)`. Folds are therefore always **disjoint** and together cover the pool exactly once, differing by at most one sequence when the pool does not divide evenly.

`ValueError` is raised, with a message naming both the float and int readings, for: a float at or below 0 or above 1; an int below 1; a non-numeric type; and any value whose resolved pool would be smaller than `iterations` (some fold would get no sequences) — including a database with fewer sequences than `iterations`.

Fold contents are rewritten on every call, so a changed `query_size` takes effect with no cache flag; `force` only rebuilds the shared full-database reference artifacts. Under Tourmaline, snakemake skips `tax_credit_prepare_datasets` on a run that already finished — delete `.datasets.done` or pass `--forcerun tax_credit_prepare_datasets`. `force_regenerate: true` does neither.

---

## `tax_credit.paths` and `tax_credit.simulation_names`

- **`paths`:** filename constants (`FEATURE_TABLE_BIOM`, `QUERY_TAXA_TSV`, …), default glob patterns, and `parse_*` helpers for path segments. Prefer these over hard-coded strings or `split(sep)[-5]` indexing.

### `assignment_result_leaf_glob(results_root)`

Returns the glob string `join(results_root, '*', '*', '*', '*')` for sweep leaves (`dataset/reference/method/parameters`).

### `list_assignment_result_dirs(results_root, assignments_filename=QUERY_TAX_ASSIGNMENTS_TXT, sort=True)`

Lists directories under *results_root* at that depth that contain the assignments file. Use this instead of a raw `glob` when **cross-validated-trad** (or similar) also has classifier-only directories at the same depth.
- **`simulation_names`:** constants such as `DIR_CROSS_VALIDATED`, `DIR_CROSS_VALIDATED_TRAD`, `DIR_NOVEL_TAXA_SIMULATIONS`, `DIR_REF_DBS`; helpers `cross_validated_root`, `cross_validated_trad_root`, `novel_taxa_simulations_root`, `ref_dbs_root`; `format_*` / `parse_*` for fold IDs (`parse_cv_dataset_id` applies to both CV trees; novel IDs support hyphenated DB names).

See [directory-layout.md](directory-layout.md).

---

## `tax_credit.plot_theme`

Shared plot style. `apply_tax_credit_theme()` sets fonts (Arial, falling back to DejaVu Sans), editable TrueType PDF text, `constrained_layout`, and trimmed axes; every plotting function below calls it. Also provides `METHOD_COLORS` / `method_palette(methods, override=None)` (a fixed Okabe-Ito colour per classify method), `CLASSIFICATION_RATIO_COLORS` and `RATIO_STACK_ORDER`, `metric_cmap(metric)` (`mako_r` for scores, `rocket_r` for error ratios), `metric_limits(values)` (0-1 unless every value is within 0.25 of 0 or 1, then zoomed; returns `(low, high, zoomed)`), and `metric_label` / `eval_method_label` / `ratio_label` for readable names.

Metric names: `METRIC_ALIASES` maps config spellings to columns (`precision_pa` / `TAR` → `Taxon Accuracy Rate`, `recall_pa` / `TDR` → `Taxon Detection Rate`), resolved by `resolve_metric(metric)`. `NON_PLOT_METRICS` lists the audit-only columns — the `*_original` scores and every raw count (`TP`, `FP`, `FN`, `FN_original`, `n_*`, `tp_pa`, `fp_pa`, `fn_pa`, `tot_val_*`, `reads_scored_fraction`). `plottable_metrics(metrics, available=None)` resolves aliases, drops audit-only columns and duplicates, and (given `available`) keeps only columns that exist; Tourmaline filters `plot_metrics` through it.

## `tax_credit.plotting_functions`

Seaborn/matplotlib helpers (boxplots, heatmaps, PCoA, etc.); dependencies match the QIIME amplicon environment described in [installation.md](installation.md). Every plotting function returns a matplotlib `Figure` and never calls `plt.show()`.

Evaluation metric plots used by the Tourmaline tax-credit step:

| Function | Draws |
|----------|-------|
| `pointplot_from_data_frame(df, x, metric, hue="Method", col="Dataset", x_order=None, col_order=None, palette=None, x_label=None, title=None)` | Mean metric per `x`, one line per `hue`, one panel per `col`; error bars span min-max. |
| `faceted_boxplot_from_data_frame(df, x, metric, hue="Method", col=None, col_order=None, palette=None, title=None)` | Boxplots with each row drawn as a point, one panel per `col`. |
| `heatmap_from_data_frame(df, metric, rows=("Method", "Parameters"), cols=("Dataset",), cmap=None, vmin=None, vmax=None, annotate=None, title=None)` | Mean metric per row/column group; colour map and limits default from the metric; values printed for 120 cells or fewer. |
| `stacked_classification_barplot_from_data_frame(df, run_cols=("Method", "Parameters"), col="Dataset", level_col="level", level_labels=None, level_axis_label=..., title=None)` | Classification ratios by level; one row per run, one column per dataset. |
| `stacked_classification_panels_from_data_frames(panels, ncols, ..., row_labels=None, col_titles=None, panel_size=(2.6, 2.6), title=None)` | Grid of `(title, df)` panels, each one run's ratios by level. |

`tax_credit.log_plotting.method_parameter_sensitivity_heatmap_from_data_frame(pivot_df, title=None, value_label=..., annotate_max_cells=120)` draws one heatmap panel per dataset from a `(dataset, expected_taxonomy)`-indexed pivot, keeping its row order; hatched cells have no data. Rank rows first with `log_analysis.select_top_sensitivity_taxa(pivot_df, top_n)`, which keeps the `top_n` worst taxa per dataset.

**API changes relative to upstream tax-credit:**

- `show=` removed from every plotting function; figures are returned instead. Call `plt.show()` or display the figure in notebooks.
- `pointplot_from_data_frame` takes one `metric` (was a `y_vars` list), uses `x` / `hue` / `col` (were `x_axis` / `color_by` / `group_by`) and returns a `Figure` (was a dict of `FacetGrid`s).
- `heatmap_from_data_frame`, `boxplot_from_data_frame` and `method_parameter_sensitivity_heatmap_from_data_frame` return a `Figure` (were `Axes`). `boxplot_from_data_frame` no longer fixes the y axis to 0-1 by default.
- `faceted_boxplot_from_data_frame` returns a `Figure` (was a `FacetGrid`) and takes `palette` (was `color_palette`).
- `stacked_classification_barplot_from_data_frame` draws a grid of runs x datasets (was one axis of nested clusters) and returns a `Figure`.
- `select_top_sensitivity_taxa` keeps `top_n` taxa per dataset (was across all datasets) in ranked order.
- Removed: `lmplot_from_data_frame` (did not run on seaborn 0.12) and `DEFAULT_CLASSIFICATION_RATIO_COLORS` (use `plot_theme.CLASSIFICATION_RATIO_COLORS`).

---

## Related

- [Overview](overview.md) — scientific modes (mock / CV / novel).
- [Directory layout](directory-layout.md) — on-disk contracts.
- [Examples](../examples/) — interactive notebooks using this API.
- [README](../README.md) — running a benchmark from Tourmaline.
