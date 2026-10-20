# tour2-tax-credit (Tourmaline fork)

**A modified version of [tax-credit](https://github.com/caporaso-lab/tax-credit) (TAXonomic ClassifieR Evaluation Tool), adapted to run as the reference-database benchmarking step of [Tourmaline 2](https://github.com/aomlomics/tourmaline).**

This is **not** the upstream tax-credit repository. It is a fork maintained for use with Tourmaline, and it is not intended to reproduce the analyses of the original tax-credit paper. If you want the original framework and its supplementary notebooks, use [caporaso-lab/tax-credit](https://github.com/caporaso-lab/tax-credit).

---

The original tax-credit is a notebook-driven framework: you open a Jupyter notebook per analysis, edit paths and parameter sweeps, and run the cells. This fork keeps the scientific core of that framework — the simulation, evaluation and plotting code — but designed to run through [Tourmaline](https://github.com/aomlomics/tourmaline/tree/V2), which drives everything from a YAML config.

Tourmaline's `scripts/run_tax_credit.py` imports it and runs the whole benchmark: it builds the simulated datasets, emits and executes the classification jobs, scores the assignments, and writes summary tables and plots.

## How it differs from the original tax-credit

- **Trimmed to what Tourmaline uses.** The legacy notebook-only modules (`eval_framework`, `mock_evaluation`, `biom_cache`, `process_mocks`, `mock_denoise`, `mock_transport`, `mock_quality`, `mockrobiota_extract`, `simulated_communities`) have been removed. See [Package layout](#package-layout) for what remains.
- **New evaluation modes.** Random-split cross-validation (`cross-validated-trad`, with a configurable query-set size) and running a full database against itself (`self-validated`) have been added alongside the original taxonomy-stratified folds, with shared reference artifacts to avoid duplicating a database per fold.
- **New mock-community implementation.** `tax_credit.mock_community` replaces the old `mock_evaluation` / `eval_framework` scoring path, and works from feature tables, ASV sequences and expected composition or per-ASV "trueish" taxonomies.
- **Log analysis and plot theming.** `tax_credit.log_analysis`, `tax_credit.log_plotting` and `tax_credit.plot_theme` were added to summarize per-taxon classifier behaviour and to give the generated figures a consistent look.
- **Modernized environment.** Targets QIIME 2 amplicon 2024.10 (Python 3.10). 

## Installation

This package expects the scientific stack from a QIIME 2 amplicon environment; it declares no PyPI dependencies of its own.

```bash
conda env create -n qiime2-amplicon-2024.10 \
  --file https://data.qiime2.org/distro/amplicon/qiime2-amplicon-2024.10-py310-osx-conda.yml
conda activate qiime2-amplicon-2024.10

git clone https://github.com/ksil-NOAA/tour2-tax-credit.git tax-credit
cd tour2-tax-credit
pip install -e .
```

Clone it as a sibling of your Tourmaline directory — Tourmaline's `tax_credit_package_dir` defaults to `../tour2-tax-credit`.

See [docs/installation.md](docs/installation.md) for platform-specific environment files and the non-conda fallback.

Five evaluation modes are supported. The first four are simulated from the reference database itself; `mock-community` requires real sequencing data you supply.

| Mode | What it does |
|---|---|
| `cross-validated` | Taxonomy-aware K-fold splits; classify held-out sequences. |
| `cross-validated-trad` | Random splits of the query list; the reference keeps every sequence (see caveat below). |
| `novel-taxa` | Hold out whole taxa, so a query's own taxon is absent from the reference. |
| `self-validated` | Classify the full database against itself; a best-case ceiling. |
| `mock-community` | Classify real reads from communities of known composition. |

Benchmark runs are large — a full matrix of databases × methods × parameter sets × folds is hundreds of assignment jobs. Start with a reduced matrix.

### A caveat on `cross-validated-trad`

This mode holds out the query *list* but **not the reference**. Each fold's `ref_seqs.fasta` and `ref_taxa.tsv` are symlinks to the full database, so every query is classified against a reference that still contains it — an exact self-match. A single classifier is also fitted per database and reused across all folds.

Read its scores as a **ceiling measured on a random subset** rather than as cross-validated performance: they are optimistically biased, and fold-to-fold spread understates real variance. Use `cross-validated` (taxonomy-aware) when you want queries genuinely absent from the fold's reference.

By default the whole database is divided between the folds, so each queries `n_sequences / iterations` sequences. Tourmaline's `trad_cv_query_size` shrinks the **total** query pool instead, which is then divided the same way — a **float** in (0, 1] is a **fraction** of the database, an **int** is an **absolute count**, and both describe the total across all folds rather than one fold. With `trad_cv_query_size: 0.2` and `iterations: 8`, 20% of the database is queried in total and each fold holds 2.5% of it. An int larger than a given database warns and uses that whole database instead, so one int can be shared across databases of different sizes. Folds stay disjoint either way. Changing it takes effect the next time Tourmaline's datasets phase runs; on a run that already finished, delete `.datasets.done` or pass `--forcerun tax_credit_prepare_datasets`.

## Package layout

| Module | Role |
|---|---|
| `framework_functions` | Dataset simulation, parameter sweeps, classification command generation |
| `novel_evaluation` | Scoring for novel-taxa and cross-validated assignments; best-run selection |
| `mock_community` | Mock-community scoring (precision/recall, taxon detection, Bray-Curtis) |
| `log_analysis`, `log_plotting` | Per-taxon classifier behaviour summaries and their figures |
| `plotting_functions`, `plot_theme` | Metric plots and shared figure styling |
| `paths`, `simulation_names` | On-disk layout conventions and simulation naming |
| `taxa_manipulator` | Taxonomy string parsing, normalization and filtering |

Full API documentation is in [docs/python-api.md](docs/python-api.md); on-disk conventions are in [docs/directory-layout.md](docs/directory-layout.md).

## Examples

The notebooks in [`examples/`](examples/) are reference material, not part of the Tourmaline workflow:

- [`cross-validated-and-novel-taxa.ipynb`](examples/cross-validated-and-novel-taxa.ipynb) — comparing rCRUX and CRABS reference databases interactively, using the current API.
- [`mock-community-legacy.ipynb`](examples/mock-community-legacy.ipynb) — **historical reference only.** Its evaluation cells import modules that were removed from this fork and will raise `ModuleNotFoundError`. Use Tourmaline's mock-community mode instead.

## Documentation

| Document | Description |
|---|---|
| [docs/index.md](docs/index.md) | Documentation index |
| [docs/installation.md](docs/installation.md) | Environment setup |
| [docs/overview.md](docs/overview.md) | Evaluation modes and how they fit together |
| [docs/directory-layout.md](docs/directory-layout.md) | On-disk conventions |
| [docs/python-api.md](docs/python-api.md) | Python API reference |

## Citation

This fork is derived from tax-credit. **If you use this software, please cite the original paper:**

> Bokulich NA, Kaehler BD, Rideout JR, Dillon M, Bolyen E, Knight R, Huttley GA, Caporaso JG. Optimizing taxonomic classification of marker-gene amplicon sequences with QIIME 2's q2-feature-classifier plugin. *Microbiome*. 2018;6(1):90. doi:[10.1186/s40168-018-0470-z](https://doi.org/10.1186/s40168-018-0470-z)

```bibtex
@article{bokulich2018optimizing,
  title   = {Optimizing taxonomic classification of marker-gene amplicon
             sequences with {QIIME} 2's q2-feature-classifier plugin},
  author  = {Bokulich, Nicholas A. and Kaehler, Benjamin D. and
             Rideout, Jai Ram and Dillon, Matthew and Bolyen, Evan and
             Knight, Rob and Huttley, Gavin A. and Caporaso, J. Gregory},
  journal = {Microbiome},
  volume  = {6},
  number  = {1},
  pages   = {90},
  year    = {2018},
  doi     = {10.1186/s40168-018-0470-z}
}
```

## License

BSD 3-Clause, unchanged from upstream — see [COPYING.txt](COPYING.txt). Copyright (c) 2014--, tax-credit development team.

## Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
