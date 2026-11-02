# Overview

tax-credit supports **systematic benchmarking** of marker-gene taxonomic classifiers. It standardizes where results live on disk, how metrics are computed, and how plots summarize comparisons across methods and parameters.

## Evaluation modes

The framework stresses three complementary designs:

### Mock communities

**Known composition.** Sequences come from mixtures of organisms with known taxonomy and relative abundance (e.g. [Mockrobiota](http://caporasolab.us/mockrobiota/)). Observed BIOM tables are compared to **expected** composition tables. This measures performance under realistic sequencing error and community complexity.

Typical outputs: precision, recall, F-measure, taxon detection, and optional per-sequence metrics when `trueish-taxonomies.tsv` and per-sequence assignment files are available.

### Cross-validated reference classification

Simulations split a reference database into **query** (test) and **reference** (training) folds. Labels remain known, so classic **precision and recall** apply at multiple taxonomic levels. `framework_functions.generate_simulated_datasets` can produce one or more layouts via **`simulation_method`** (default: all of the following):

- **`cross-validated-taxa`** (maps to on-disk `cross-validated/`): stratified folds by taxonomic strata; query labels may be trimmed so each expected taxonomy prefix appears somewhere in the training taxonomies. Queries are **not** present in the per-fold reference FASTA / taxonomy tables.

  A query whose expected taxonomy has **no** prefix in the fold's training set — not even its first rank — is **dropped from that fold**, and the per-fold trimmed and dropped counts are printed during dataset generation. This happens when every sequence sharing a first rank lands in the same test fold, which is guaranteed when only one sequence holds it. Note the split runs on reads that survived in-silico amplicon extraction and the minimum-length filter, so a clade can be plentiful in the database and a singleton here. Dropped queries are left out of `query.fasta` too, keeping it aligned with `query_taxa.tsv`. If a first rank is rare enough to be dropped often, consider cleaning or relabelling it in the reference rather than reading the scores as-is.

- **`cross-validated-trad`** (`cross-validated-trad/`): random splits by sequence ID. Query FASTA and `query_taxa.tsv` contain only the test fold, but **`ref_seqs.fasta`** and **`ref_taxa.tsv`** are **symbolic links** to the full simulated-reads FASTA and cleaned taxonomy for that database.

  **Only the query list is held out — the reference is not.** Every fold is classified against the entire database, queries included, so each query sequence has an exact self-match in the reference it is searched against. Query taxonomies are also left untrimmed, with no check that test taxa appear in the reference. Evaluation compares assignments to the query labels as usual.

  Read this mode as a **ceiling on a random subset**, closer to `self-validated` than to a held-out cross-validation: scores are optimistically biased, and since a single classifier is fitted per database and reused for every fold, fold-to-fold spread understates real variance. Use `cross-validated-taxa` when you want queries genuinely absent from the fold's reference.

  **`query_size`** (Tourmaline: `trad_cv_query_size`) sets the size of the **total query pool**, which is then divided evenly between the `iterations` folds. A float in `(0, 1]` is a fraction of the database; an int is an absolute number of sequences. Both describe the total across all folds, not the size of one fold — with `query_size=0.2` and `iterations=8`, 20% of the database is queried in total and each fold holds 20%/8 = 2.5% of it. Left unset, the pool is the whole database (equivalent to `query_size=1.0`), giving the historical `n_sequences / iterations` per fold. An int larger than a given database warns and falls back to that whole database, so a single int can be shared across databases of different sizes.

  A random pool of the requested size is drawn with a fixed seed, then split with `KFold(n_splits=iterations, shuffle=True)`, so folds are always disjoint and together cover the pool exactly once; they differ by at most one sequence when the pool does not divide evenly. Every fold's query files are rewritten on each call, so a changed `query_size` takes effect without any cache flag — `force` only rebuilds the shared full-database reference artifacts. Under Tourmaline the catch is one level up: snakemake skips the datasets phase on a finished run, so delete `.datasets.done` or pass `--forcerun tax_credit_prepare_datasets`.

  Shared **`ref_seqs.qza`** / **`ref_taxa.qza`** artifacts under `ref_dbs/` reduce duplication across folds (see [directory-layout.md](directory-layout.md)).

The legacy alias **`cross-validated`** means **`cross-validated-taxa`**.

### Novel-taxa simulations

**Queries with no exact match in the reference.** Reference sequences that share the query taxonomy are removed; correct behavior is often assignment to the **last common ancestor** (LCA). Metrics emphasize match vs. overclassification vs. underclassification vs. misclassification, summarized in classification logs and aggregate tables.

## Scoring

Every assignment, whether a simulated query or a mock-community ASV, is one of four outcomes: *match*, *underclassification* (correct but shallower than expected), *overclassification* (deeper than the expected taxonomy goes) or *misclassification* (a wrong name at some rank). Precision, recall and F-measure are computed from those tallies in one place, `metrics.metrics_from_outcomes`:

- match is a true positive;
- underclassification is a false negative;
- misclassification is both a false positive and a false negative;
- overclassification is a false positive **only**.

The last point departs from the original tax-credit, which also counted overclassification as a false negative. Assigning deeper than the reference knows now lowers precision, and so F-measure, but not recall. The original scoring is still reported as `Recall_original` and `F-measure_original`, together with the raw tallies (`TP`, `FP`, `FN`, `FN_original`), for comparison and audit; those columns are never plotted or used to rank runs. `match_ratio` equals `Recall_original`; the new `Recall` is `match_ratio / (1 − overclassification_ratio)`.

A score whose denominator is empty is NaN rather than 0, so "nothing to score" (for example precision when every assignment was underclassified) is distinguishable from "everything was wrong". Scores produced before this revision are not comparable with new ones; re-score old results with the current code rather than reading their old summaries.

## Data flow (high level)

1. **Simulate or acquire** communities and reference data (see `framework_functions.generate_simulated_datasets`).
2. **Run classifiers** (QIIME 2 naive-bayes, BLAST, VSEARCH consensus, BLCA, etc.) and place outputs under the expected directory depth (see [directory-layout.md](directory-layout.md)).
3. **Evaluate** with `mock_community.evaluate_mock_samples` (mock communities) or `novel_evaluation.novel_taxa_classification_evaluation` (novel / CV / CV-trad / self-validated text assignments; set `test_type` accordingly).
4. **Visualize** with `plotting_functions` and `log_plotting`.

In this fork, Tourmaline's `scripts/run_tax_credit.py` performs all four steps from `config_04_tax_credit.yaml`.

## BIOM and QIIME 2

Mock-community evaluation consumes feature tables (BIOM or TSV), ASV sequences, and expected composition and/or per-ASV taxonomies. Simulated modes consume QIIME 2 artifacts (`.qza`) and exported FASTA/TSV produced by `framework_functions`.

## Hardware expectations

Moderate laptops can run small benchmarks, but a full matrix of databases x methods x parameter sets x folds is hundreds of assignment jobs and generally wants **cluster** resources. Start with a reduced matrix; Tourmaline ships an sbatch wrapper (`scripts/sbatch_tourmaline2_step4.sh`) for SLURM.
