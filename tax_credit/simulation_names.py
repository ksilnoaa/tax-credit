#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2014--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Directory names and string encodings for cross-validated and novel-taxa sims."""

from __future__ import annotations

import re
from os.path import join
from typing import NamedTuple

_NOVEL_DATASET_ID_RE = re.compile(r"^(.*)-L(\d+)-iter(\d+)$")

# --- Top-level folders under a *data_dir* ----------------------------------
DIR_CROSS_VALIDATED = "cross-validated"
DIR_CROSS_VALIDATED_TRAD = "cross-validated-trad"
DIR_NOVEL_TAXA_SIMULATIONS = "novel-taxa-simulations"
DIR_SELF_VALIDATED = "self-validated"
DIR_REF_DBS = "ref_dbs"

# --- Glob fragments for discovering fold directories -----------------------
GLOB_CV_FOLD_DIRS = "*-iter*"
GLOB_NOVEL_FOLD_DIRS = "*-L*-iter*"


class NovelDatasetParts(NamedTuple):
    """Parsed ``<database>-L<level>-iter<n>`` fold / dataset id."""

    database: str
    level: int
    iteration: str


class CrossValidatedDatasetParts(NamedTuple):
    """Parsed ``<database>-iter<n>`` dataset id."""

    database: str
    iteration: str


class SelfValidatedDatasetParts(NamedTuple):
    """Parsed self-validated dataset id (reference database name only)."""

    database: str
    iteration: str


def cross_validated_root(data_dir: str) -> str:
    return join(data_dir, DIR_CROSS_VALIDATED)


def cross_validated_trad_root(data_dir: str) -> str:
    return join(data_dir, DIR_CROSS_VALIDATED_TRAD)


def novel_taxa_simulations_root(data_dir: str) -> str:
    return join(data_dir, DIR_NOVEL_TAXA_SIMULATIONS)


def self_validated_root(data_dir: str) -> str:
    return join(data_dir, DIR_SELF_VALIDATED)


def ref_dbs_root(data_dir: str) -> str:
    return join(data_dir, DIR_REF_DBS)


def format_cv_fold_dirname(database: str, iteration: int) -> str:
    """Build cross-validation fold directory name ``{db}-iter{n}``."""
    return "{0}-iter{1}".format(database, iteration)


def parse_cv_fold_dirname(basename: str) -> CrossValidatedDatasetParts:
    """Parse a CV fold directory basename into database and iteration string."""
    if "-iter" not in basename:
        raise ValueError(
            "CV fold basename must contain '-iter'; got {!r}".format(basename)
        )
    database, iteration = basename.split("-iter", 1)
    return CrossValidatedDatasetParts(database, iteration)


def format_novel_fold_dirname(database: str, level: int, iteration: int) -> str:
    """Build novel-taxa fold name ``{db}-L{level}-iter{n}``."""
    return "{0}-L{1}-iter{2}".format(database, level, iteration)


def parse_novel_dataset_id(dataset_id: str) -> NovelDatasetParts:
    """Parse ``dataset_id`` matching ``{database}-L{level}-iter{n}``.

    *database* may contain hyphens (e.g. ``B1-REF-L6-iter0``); the level
    segment is always ``-L<digits>-iter<digits>`` at the end.
    """
    m = _NOVEL_DATASET_ID_RE.match(dataset_id)
    if not m:
        raise ValueError(
            "Novel-taxa dataset_id must match {{db}}-L{{level}}-iter{{n}}; "
            "got {!r}".format(dataset_id)
        )
    database, level_s, iteration_s = m.group(1), m.group(2), m.group(3)
    return NovelDatasetParts(database, int(level_s), iteration_s)


def parse_cv_dataset_id(dataset_id: str) -> CrossValidatedDatasetParts:
    """Parse a cross-validated *dataset_id* (``<db>-iter<n>``)."""
    return parse_cv_fold_dirname(dataset_id)


def parse_self_validated_dataset_id(dataset_id: str) -> SelfValidatedDatasetParts:
    """Parse a self-validated *dataset_id* (the reference database name)."""
    if "-iter" in dataset_id:
        raise ValueError(
            "Self-validated dataset_id must not contain '-iter'; got {!r}".format(
                dataset_id
            )
        )
    return SelfValidatedDatasetParts(dataset_id, "0")
