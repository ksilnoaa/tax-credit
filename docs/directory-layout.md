# Directory layout and path helpers

tax-credit assumes **fixed-depth** directory trees for discovering results. Central definitions live in `tax_credit/paths.py` and `tax_credit/simulation_names.py` so Tourmaline and library code share one contract.

## Mock-style results (observed)

Observed mock (or mock-like) BIOM tables are expected under:

```text
<root>/<dataset_id>/<reference_id>/<method_id>/<parameter_id>/<table files>
```

Default observed table glob pattern: `table*biom` (see `DEFAULT_MOCK_RESULT_TABLE_PATTERN`).

## Expected composition

Expected tables live under:

```text
<root>/<dataset_id>/<reference_id>/expected/<file>
```

Default expected filename pattern: `table.L{level}-taxa.biom` (see `DEFAULT_EXPECTED_TABLE_PATTERN`).

## Per-sequence and sidecar files

Common filenames (constants in `paths.py`) include:

| Constant | Typical use |
|----------|-------------|
| `FEATURE_TABLE_BIOM` | Mock feature table without taxonomy (`feature_table.biom`) |
| `TRUEISH_TAXONOMIES_TSV` | Per-sequence expected labels |
| `REP_SEQS_TAX_ASSIGNMENTS_TXT` / `TAXONOMY_TSV` | Observed taxonomy for rep seqs |
| `QUERY_TAX_ASSIGNMENTS_TXT` / `QUERY_TAXA_TSV` | Novel/CV query vs expected lists |
| `CLASSIFICATION_ACCURACY_LOG_TSV` | Per-read classification log |

## Novel and cross-validated simulations

Under the data directory, simulation roots use names exposed in `simulation_names.py`:

| Constant | Folder | Contents |
|----------|--------|----------|
| `DIR_CROSS_VALIDATED` | `cross-validated/` | **`cross-validated-taxa`** folds — stratified CV; per-fold `ref_seqs.fasta` / `ref_taxa.tsv` exclude query IDs. |
| `DIR_CROSS_VALIDATED_TRAD` | `cross-validated-trad/` | **`cross-validated-trad`** folds — random KFold; `query.fasta` / `query_taxa.tsv` are the test split only; `ref_seqs.fasta` and `ref_taxa.tsv` are **symlinks** to the full simulated-reads FASTA and cleaned taxonomy under `ref_dbs/`. |
| `DIR_NOVEL_TAXA_SIMULATIONS` | `novel-taxa-simulations/` | **Novel-taxa** derived datasets (from CV-taxa folds). |
| `DIR_SELF_VALIDATED` | `self-validated/` | **Self-validated** datasets — one directory per reference database; query and reference both contain every sequence (no CV folds). |
| `DIR_REF_DBS` | `ref_dbs/` | Per–reference-id cleaned FASTA, taxonomy, extracted simulated reads, and (for trad) shared QIIME artifacts: `_trad_cv_shared_ref_seqs.qza`, `_trad_cv_shared_ref_taxa.qza`. |

`generate_simulated_datasets(..., simulation_method=...)` controls which of these trees are written. By default it builds **all** three simulation types. Requesting **`novel-taxa`** without **`cross-validated-taxa`** still runs the CV-taxa generator internally to feed novel-taxa construction, then removes the temporary fold directories (see `framework_functions` source).

Fold directory names are generated and parsed with helpers such as:

- `format_cv_fold_dirname` / `parse_cv_dataset_id` — e.g. `<db>-iter<n>` (used for both `cross-validated` and `cross-validated-trad` trees)
- `format_novel_fold_dirname` / `parse_novel_dataset_id` — e.g. `<db>-L<level>-iter<n>` (supports hyphenated database IDs like `B1-REF`)

Path roots: `cross_validated_root`, `cross_validated_trad_root`, `novel_taxa_simulations_root`, `ref_dbs_root` in `simulation_names.py`.

## Assignment result directories (novel / CV)

Paths ending in:

```text
.../<dataset_id>/<method_id>/<params_id>
```

are parsed by `parse_assignment_results_dir`.

Sweeps write one level higher: `results_root/<dataset_id>/<reference_id>/<method_id>/<params_id>/` (with `dataset_id == reference_id` for fold-based sims). To collect leaves that contain `query_tax_assignments.txt` (and skip trad **fit** dirs that only hold `classifier.qza`), use `list_assignment_result_dirs(results_root)` in `paths.py`.

## Using the API

Prefer **named helpers** over manual `split`/`join`:

- `mock_observed_tables_glob`, `expected_tables_glob`
- `assignment_result_leaf_glob`, `list_assignment_result_dirs`
- `parse_mock_result_table_path`, `parse_expected_table_path`
- `parse_result_leaf_dir_to_parts`, `parse_taxonomy_map_path_to_dataset_id`

This makes renames and validation errors easier to manage than hard-coded path indices.
