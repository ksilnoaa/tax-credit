#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2014--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Distribution metadata for tax-credit.

QIIME 2 amplicon conda environments supply the scientific Python stack; this
package declares no extra PyPI dependencies so ``pip install -e .`` only
registers tax-credit in the environment.
"""

from setuptools import find_packages, setup

# Reference: qiime2-amplicon 2024.10 (released) — see
# https://raw.githubusercontent.com/qiime2/distributions/dev/2024.10/amplicon/released/qiime2-amplicon-ubuntu-latest-conda.yml
# Plotting uses matplotlib / seaborn (already in that environment); no bokeh.

INSTALL_REQUIRES = []

# Optional: recreate a PyPI scientific stack when *not* using the QIIME 2 amplicon
# conda env (QIIME 2 APIs will still be missing).
EXTRAS_FULL_PYPI = [
    "numpy>=1.26,<2",
    "pandas>=2.2",
    "scipy>=1.13",
    "biom-format>=2.1.14,<3",
    "scikit-bio>=0.5.0",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "seaborn>=0.12",
    "statsmodels>=0.14",
    "ipython>=8.12",
    "jupyter",
    "notebook",
    "ipywidgets",
    "pytest>=8",
]

LONG_DESCRIPTION = """# tax-credit (Tourmaline fork)

A modified version of [tax-credit](https://github.com/caporaso-lab/tax-credit)
(TAXonomic ClassifieR Evaluation Tool), adapted to run as the reference-database
benchmarking step of [Tourmaline 2](https://github.com/aomlomics/tourmaline).

Benchmarks are driven by Tourmaline's `scripts/run_tax_credit.py` from a single
YAML config; the upstream supplementary notebooks and the modules used only by
them have been removed. See the repository README for the full list of
differences from upstream, and please cite the original paper:

  Bokulich NA, Kaehler BD, Rideout JR, Dillon M, Bolyen E, Knight R,
  Huttley GA, Caporaso JG. Optimizing taxonomic classification of marker-gene
  amplicon sequences with QIIME 2's q2-feature-classifier plugin.
  Microbiome. 2018;6(1):90. doi:10.1186/s40168-018-0470-z

## Install (QIIME 2 amplicon 2024.10)

```bash
conda env create -n qiime2-amplicon-2024.10 \
  --file https://data.qiime2.org/distro/amplicon/qiime2-amplicon-2024.10-py310-osx-conda.yml
conda activate qiime2-amplicon-2024.10
pip install -e .
```

No additional PyPI packages are required: numpy, pandas, scipy, biom-format,
scikit-bio, matplotlib, seaborn, statsmodels and pytest all come from the
QIIME 2 amplicon environment. Without that conda stack, `qiime2` / `q2-*`
imports will not work; for a non-conda venv you can try
`pip install -e ".[full-pypi]"` (still no QIIME 2).

Clone this repository as a sibling of your Tourmaline directory — Tourmaline's
`tax_credit_package_dir` config key defaults to `../tax-credit`.
"""

setup(
    name="tax-credit",
    version="0.0.0-dev",
    license="BSD-3-Clause",
    python_requires=">=3.10,<3.11",
    packages=find_packages(exclude=("*.tests", "*.tests.*", "tests.*", "tests")),
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "full-pypi": EXTRAS_FULL_PYPI,
    },
    author="tax-credit development team",
    description=(
        "Systematic benchmarking of taxonomic classification methods; "
        "Tourmaline-integrated fork of caporaso-lab/tax-credit"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/ksil-NOAA/tax-credit-v2",
    project_urls={
        "Upstream tax-credit": "https://github.com/caporaso-lab/tax-credit",
        "Tourmaline": "https://github.com/aomlomics/tourmaline",
        "Original paper (please cite)": "https://doi.org/10.1186/s40168-018-0470-z",
        "QIIME 2": "https://docs.qiime2.org/",
        "QIIME 2 install": "https://docs.qiime2.org/2024.10/install/native/#install-qiime-2-within-a-miniconda-or-anaconda-distribution",
        "QIIME 2 amplicon 2024.10 (conda deps)": "https://raw.githubusercontent.com/qiime2/distributions/dev/2024.10/amplicon/released/qiime2-amplicon-ubuntu-latest-conda.yml",
    },
)
