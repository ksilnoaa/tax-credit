# tax-credit documentation

**TAX CREdiT** (TAXonomic ClassifieR Evaluation Tool) is a framework for comparing taxonomic classifiers on mock communities, cross-validated reference data (taxonomy-stratified or traditional random folds), novel-taxa simulations, and self-validation.

This is the **Tourmaline-integrated fork** of [caporaso-lab/tax-credit](https://github.com/caporaso-lab/tax-credit). Benchmarks are driven by Tourmaline's tax-credit step rather than by the upstream Jupyter notebooks. See the [README](../README.md) for what differs from upstream and how to cite the original work.

## Guides

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | QIIME 2 amplicon conda setup, `pip install`, optional PyPI extras |
| [Overview](overview.md) | Evaluation modes, scientific goals, and how analyses fit together |
| [Directory layout](directory-layout.md) | On-disk conventions, `paths` and `simulation_names` helpers |
| [Python API](python-api.md) | Modules and import patterns in this fork |

## Quick start

1. Create a [QIIME 2 amplicon](https://docs.qiime2.org/) environment (2024.10, Python 3.10).
2. From the repository root: `pip install -e .`
3. Point `tax_credit_package_dir` in Tourmaline's `config_04_tax_credit.yaml` at this clone.
4. From the Tourmaline directory: `./tourmaline.sh --step tax-credit --configfile config_04_tax_credit.yaml --cores 8`

## Further reading

- Repository [README](../README.md) — differences from upstream, citation, license
- Example notebooks in [`examples/`](../examples/) — interactive use of the Python API
- Tourmaline's [tax-credit step docs](https://github.com/aomlomics/tourmaline/blob/main/docs/steps/tax_credit.md)
- License: [COPYING.txt](../COPYING.txt)
