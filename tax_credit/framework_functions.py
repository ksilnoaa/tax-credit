#!/usr/bin/env python


# ----------------------------------------------------------------------------
# Copyright (c) 2016--, tax-credit development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from os.path import (
    abspath,
    dirname,
    exists,
    expandvars,
    basename,
    isfile,
    join,
    lexists,
    split,
    splitext,
)
from os import makedirs, remove, symlink, system, stat
from glob import glob
from itertools import product
from shutil import rmtree, move
from random import sample
from time import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from seaborn import regplot
from biom import load_table
from biom.cli.util import write_biom_table
from biom.parse import MetadataMap
from numpy import random
from skbio import io
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from tax_credit.taxa_manipulator import (accept_list_or_file,
                                         normalize_taxon,
                                         import_to_list,
                                         import_taxonomy_to_dict,
                                         export_list_to_file,
                                         filter_sequences,
                                         string_search,
                                         drop_unassigned_taxonomies,
                                         strip_taxonomy_header,
                                         is_unassigned_taxon,
                                         trim_taxonomy_strings,
                                         truncate_taxa_to_reference,
                                         extract_taxa_names)
from qiime2 import Artifact
from qiime2.plugins import feature_classifier
from q2_types.feature_data import DNAIterator

from tax_credit.paths import (
    QUERY_FASTA,
    QUERY_TAXA_TSV,
    REF_SEQS_FASTA,
    REF_TAXA_TSV,
    parse_result_leaf_dir_to_parts,
    parse_taxonomy_map_path_to_dataset_id,
)
from tax_credit.simulation_names import (
    GLOB_CV_FOLD_DIRS,
    GLOB_NOVEL_FOLD_DIRS,
    cross_validated_root,
    cross_validated_trad_root,
    format_cv_fold_dirname,
    format_novel_fold_dirname,
    novel_taxa_simulations_root,
    parse_cv_dataset_id,
    parse_novel_dataset_id,
    ref_dbs_root,
    self_validated_root,
)


def gen_param_sweep(data_dir, results_dir, reference_dbs,
                    dataset_reference_combinations,
                    method_parameters_combinations,
                    output_name='rep_set_tax_assignments.txt', force=False):
    '''Create list of commands from input dictionaries of method_parameter and
    dataset_reference_combinations
    '''
    for dataset, reference in dataset_reference_combinations:
        input_dir = join(data_dir, dataset)
        reference_seqs, reference_tax = reference_dbs[reference]
        for method, parameters in method_parameters_combinations.items():
            parameter_ids = sorted(parameters.keys())
            for parameter_combination in product(*[parameters[id_]
                                                 for id_ in parameter_ids]):
                parameter_comb_id = ':'.join(map(str, parameter_combination))
                parameter_output_dir = join(results_dir, dataset, reference,
                                            method, parameter_comb_id)
                if not exists(join(parameter_output_dir, output_name)) \
                        or force:
                    params = dict(zip(parameter_ids, parameter_combination))
                    yield (parameter_output_dir, input_dir, reference_seqs,
                           reference_tax, method, params)


def parameter_sweep(data_dir, results_dir, reference_dbs,
                    dataset_reference_combinations,
                    method_parameters_combinations,
                    command_template="mkdir -p {0} ; assign_taxonomy.py -v -i "
                                     "{1} -o {0} -r {2} -t {3} -m {4} {5} "
                                     "--rdp_max_memory 16000",
                    infile='rep_set.fna',
                    output_name='rep_set_tax_assignments.txt', force=False):
    '''Create list of commands from input dictionaries of method_parameter and
    dataset_reference_combinations
    '''
    sweep = gen_param_sweep(data_dir, results_dir, reference_dbs,
                            dataset_reference_combinations,
                            method_parameters_combinations,
                            output_name, force)
    commands = []
    for assignment in sweep:
        (parameter_output_dir, input_dir, reference_seqs, reference_tax,
         method, params) = assignment
        query_seqs = join(input_dir, infile)
        parameter_str = ' '.join(['--%s %s' % e for e in params.items()])
        command = command_template.format(parameter_output_dir, query_seqs,
                                          reference_seqs, reference_tax,
                                          method, parameter_str)
        commands.append(command)
    return commands


def add_metadata_to_biom_table(biom_input_fp, taxonomy_map_fp, biom_output_fp):
    '''Load biom, add metadata, write to new table'''
    newbiom = load_table(biom_input_fp)
    if stat(taxonomy_map_fp).st_size == 0:
        metadata = {}
    else:
        metadata = MetadataMap.from_file(
            taxonomy_map_fp, header=['Sample ID', 'taxonomy', 'c'])
    newbiom.add_metadata(metadata, 'observation')
    write_biom_table(newbiom, 'json', biom_output_fp)


def generate_per_method_biom_tables(taxonomy_glob, data_dir,
                                    biom_input_fn='feature_table.biom',
                                    biom_output_fn='table.biom'):
    '''Create biom tables from glob of taxonomy assignment results'''
    taxonomy_map_fps = glob(expandvars(taxonomy_glob))
    for taxonomy_map_fp in taxonomy_map_fps:
        dataset_id = parse_taxonomy_map_path_to_dataset_id(taxonomy_map_fp)
        biom_input_fp = join(data_dir, dataset_id, biom_input_fn)
        output_dir = split(taxonomy_map_fp)[0]
        biom_output_fp = join(output_dir, biom_output_fn)
        if exists(biom_output_fp):
            remove(biom_output_fp)
        add_metadata_to_biom_table(biom_input_fp, taxonomy_map_fp,
                                   biom_output_fp)


def move_results_to_repository(method_dirs, precomputed_results_dir):
    '''Move new taxonomy assignment results to git repository'''
    for method_dir in method_dirs:
        leaf = parse_result_leaf_dir_to_parts(method_dir)
        dataset_id, db_id, method_id, param_id = leaf

        new_location = join(precomputed_results_dir, dataset_id, db_id,
                            method_id, param_id)
        if exists(new_location):
            rmtree(new_location)
        move(method_dir, new_location)


def clean_database(taxa_in, seqs_in, db_dir,
                   junk='__;|__$|_sp$|unknown|unidentified'):

    '''Remove ambiguous and empty taxonomies from reference seqs/taxonomy.

    taxa_in: path
        File containing taxonomy strings in tab-separated format:
        <SequenceID>    <taxonomy string>

    seqs_in: path
        File containing sequences corresponding to taxa_in, in fasta format.

    db_dir: dir path
        Output directory.

    junk: str
        '|'-separated list of search terms. Taxonomies containing these terms
        will be removed from the database.

    A leading 'Feature ID<tab>Taxon' header row is dropped if present, so
    databases that ship with one and databases that do not both clean the same
    way.

    The junk patterns are written for Greengenes-style prefixed ranks
    ('k__Bacteria;p__;c__'), so they do not match databases that spell an
    unresolved rank 'NA'. Records whose every rank is 'NA' or blank are
    therefore dropped separately, by taxonomy content rather than by pattern:
    they place a sequence nowhere, but still get scored against whatever a
    classifier returns for them (a non-call scores as a match, any real call as
    a misclassification), which is noise in both directions. Lineages that are
    merely unresolved at the tip ('...;Gadus;NA') are kept.
    '''

    clean_taxa = join(
        db_dir, '{0}_clean.tsv'.format(basename(splitext(taxa_in)[0])))
    clean_fasta = join(
        db_dir, '{0}_clean.fasta'.format(basename(splitext(seqs_in)[0])))

    # Drop a 'Feature ID<tab>Taxon' header row, if this database has one
    taxa = strip_taxonomy_header(taxa_in)
    # Remove empty taxa from ref taxonomy
    taxa = string_search(taxa, junk, discard=True)
    # Remove records with no assigned rank ('NA;NA;...', '', 'Unassigned')
    taxa = drop_unassigned_taxonomies(taxa)
    # Remove brackets (and other special characters causing problems)
    clean_list = [line.translate(str.maketrans('', '', '[]()'))
                  for line in taxa]
    export_list_to_file(clean_list, clean_taxa)
    # Remove corresponding seqs from ref fasta
    filter_sequences(seqs_in, clean_fasta, clean_taxa, keep=True)

    return clean_taxa, clean_fasta


def seq_count(infile):
    '''Count sequences in fasta file'''
    with open(infile, "r") as f:
        count = sum(1 for line in f) / 2
    return count


class Node(object):
    def __init__(self):
        self.tips = set()
        self.children = defaultdict(Node)

    def add(self, sid, taxon):
        self.tips.add(sid)
        if taxon:
            self.children[taxon[0]].add(sid, taxon[1:])

    def __lt__(self, other):
        return len(self.tips) < len(other.tips)


def build_tree(taxonomy, sequences):
    tree = Node()
    filtered_ids = {s.metadata['id'] for s in sequences}
    for sid, taxon in taxonomy.view(pd.Series).items():
        if sid not in filtered_ids:
            continue
        tree.add(sid, taxon.split(';'))
    return tree


def get_strata(tree, k, taxon=[]):
    if not tree.children:
        return [(';'.join(taxon), tree.tips)]
    sorted_children = map(list, map(reversed, tree.children.items()))
    sorted_children = iter(sorted(sorted_children))
    misc = set()
    # aggregate children with small tip sets
    for child, level in sorted_children:
        if len(child.tips) >= k:
            break
        misc.update(child.tips)
    else:  # all the tips are in misc
        return [(';'.join(taxon), misc)]
    # grab the tips from the child that's not in misc
    strata = get_strata(child, k, taxon+[level])
    # get the rest
    for child, level in sorted_children:
        strata.extend(get_strata(child, k, taxon+[level]))
    # if there were more than k misc, make a stratum for them
    if len(misc) > k:
        strata = [(';'.join(taxon), misc)] + strata
    else:  # randomly add them to the other strata
        for i, tip in enumerate(misc):
            strata[i % len(strata)][1].add(tip)
    return strata


def distances(A, B, n):
    iA = random.choice(range(len(A)), size=n)
    iB = random.choice(range(len(B)), size=n)
    d = [0]*n
    for i, (a, b) in enumerate(zip(iA, iB)):
        d[i] = sum(x != y for x, y in zip(A[a], B[b]))
    return d


def distance_comparison(dataframe, data_dir, test_name, samples=10000):
    simulated_dir = join(data_dir, test_name)
    for index, data in dataframe.iterrows():
        lengths = Counter()
        inner = []
        outer = []
        trainsets = glob(join(simulated_dir, index+'*', REF_SEQS_FASTA))
        testsets = glob(join(simulated_dir, index+'*', QUERY_FASTA))
        for train_fp, test_fp in zip(trainsets, testsets):
            train = list(map(str, io.read(train_fp, format='fasta')))
            test = list(map(str, io.read(test_fp, format='fasta')))
            if not train or not test:
                continue
            lengths.update(map(len, test))
            inner.extend(distances(train, train, samples))
            outer.extend(distances(train, test, samples))
        inner.sort()
        outer.sort()
        df = pd.DataFrame(
            {'train/train': inner, 'train/test': outer})

        plt.figure()  # figsize=(width, height))
        ax = regplot(x='train/train', y='train/test', data=df, fit_reg=False)
        ax.set_title(index, fontsize=20)
        maxval = max((inner[-1], outer[-1]))
        plt.plot([0, maxval], [0, maxval], linewidth=2)
        plt.show()


def _simulated_reads_min_length_tag(min_read_length):
    '''Filename fragment so simulated-read caches differ when min-length
    filtering changes. Default 80 matches historical filenames (empty tag).
    '''
    if min_read_length is None:
        return '_nominlen'
    if min_read_length == 80:
        return ''
    return '_min{0}'.format(min_read_length)


def simulated_reads_filepath(clean_fasta, fwd_primer_id, rev_primer_id,
                             min_read_length=80, trim_primers=True,
                             truncate=True):
    '''Path to the simulated-reads FASTA that ``generate_simulated_datasets``
    writes for the same arguments (excluding ``read_length``, which is not
    part of the filename).

    clean_fasta : str
        Full path to the ``*_clean.fasta`` file for this reference (as produced
        under ``ref_dbs_root``).
    '''
    base, ext = splitext(clean_fasta)
    minlen_tag = _simulated_reads_min_length_tag(min_read_length)
    if trim_primers:
        primer_pair = '{0}-{1}'.format(fwd_primer_id, rev_primer_id)
        if truncate:
            return join('{0}_{1}_trunc{2}{3}'.format(
                base, primer_pair, minlen_tag, ext))
        return join('{0}_{1}_notrunc{2}{3}'.format(
            base, primer_pair, minlen_tag, ext))
    if truncate:
        return join('{0}_noprimer_trunc{1}{2}'.format(
            base, minlen_tag, ext))
    return join('{0}_noprimer_full{1}{2}'.format(base, minlen_tag, ext))


def generate_simulated_datasets(dataframe, data_dir, iterations,
                                read_length=None,
                                levelrange=range(6, 0, -1), force=False,
                                min_read_length=80, trim_primers=True,
                                truncate=True,
                                simulation_method=(
                                    'cross-validated-taxa',
                                    'cross-validated-trad',
                                    'novel-taxa',
                                ),
                                trad_cv_query_size=None):
    '''From a dataframe of sequence reference databases, build training/test
    sets of "novel taxa" queries, taxonomies, and reference seqs/taxonomies.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Rows define reference DB paths, taxonomy paths, and primer metadata.
    iterations : int
        Number of cross-validation folds (must be >= 2).
    read_length : int or None, optional
        When ``truncate`` is True, maximum length of each simulated read:
        passed as ``trunc_len`` to ``extract_reads`` (with primers) or used to
        slice from the 5' end (without primers). Required if ``truncate`` is
        True; omit or set to None when ``truncate`` is False.
    min_read_length : int or None, optional
        Sequences shorter than this (after extraction or 5' truncation) are
        dropped. Default ``80``. If ``None``, no minimum-length filter is
        applied. Non-default values add a ``_min<N>`` or ``_nominlen`` suffix
        to the simulated-reads filename so results do not overwrite the
        default cache.
    trim_primers : bool, optional
        If True (default), extract in-silico amplicons with
        ``feature_classifier.methods.extract_reads`` using the forward and
        reverse primer sequences. If False, take each cleaned reference
        sequence from the 5' end up to ``read_length`` (no primer matching).
        Output filenames differ between the two modes so cached files do not
        collide.
    truncate : bool, optional
        If True (default), cap read length at ``read_length``. If False, keep
        the full in-silico amplicon (with primers) or the full cleaned
        reference sequence (without primers). With primers, ``trunc_len=0``
        is passed to ``extract_reads`` (QIIME 2: no truncation). Simulated-read
        filenames use ``_trunc``, ``_notrunc``, ``_noprimer_trunc``, or
        ``_noprimer_full`` (plus optional min-length tags); they never embed
        ``read_length``.
    simulation_method : str or iterable of str, optional
        Any combination of simulation builders to run. Supported values are
        ``cross-validated-taxa`` (original cross-validated behavior with
        taxonomy-aware trimming), ``cross-validated-trad`` (random split with
        untrimmed test taxonomies), and ``novel-taxa``. Default runs all
        three. For backward compatibility, ``cross-validated`` is accepted as
        an alias of ``cross-validated-taxa``.
    trad_cv_query_size : float, int, or None, optional
        Size of the **total** ``cross-validated-trad`` query pool, divided
        evenly between the ``iterations`` folds. A **float** in ``(0, 1]`` is
        a **fraction** of the database; an **int** is an **absolute number of
        sequences**. Both describe the total across all folds: with
        ``trad_cv_query_size=0.2`` and ``iterations=8``, 20% of the database
        is queried in total and each fold holds 2.5% of it. ``None``
        (default) uses the whole database as the pool, so each fold holds
        ``n_sequences / iterations`` sequences. An int larger than a given
        database warns and falls back to that whole database, so one int can
        be shared across databases of different sizes. Ignored unless
        ``cross-validated-trad`` is in ``simulation_method``. See
        :func:`generate_cross_validated_trad_sequences`.

        Fold contents are rewritten on every call, so a changed value takes
        effect without ``force``; see
        :func:`generate_cross_validated_trad_sequences`.
    '''
    allowed_methods = {
        'cross-validated-taxa',
        'cross-validated-trad',
        'novel-taxa',
    }
    if isinstance(simulation_method, str):
        requested_methods = [
            m.strip() for m in simulation_method.split(',') if m.strip()]
    else:
        requested_methods = list(simulation_method)
    normalized_methods = set()
    for method in requested_methods:
        if method == 'cross-validated':
            method = 'cross-validated-taxa'
        normalized_methods.add(method)
    if not normalized_methods:
        raise ValueError('simulation_method cannot be empty.')
    invalid_methods = sorted(normalized_methods - allowed_methods)
    if invalid_methods:
        raise ValueError(
            'simulation_method must contain only "cross-validated-taxa", '
            '"cross-validated-trad", or "novel-taxa"; got {!r}'.format(
                invalid_methods))

    make_cv_taxa = 'cross-validated-taxa' in normalized_methods
    make_cv_trad = 'cross-validated-trad' in normalized_methods
    make_novel = 'novel-taxa' in normalized_methods
    if iterations < 2:
        raise ValueError('Must perform two or more iterations for '
                         'construction of cross-validated datasets.')
    if truncate and read_length is None:
        raise ValueError(
            'read_length is required when truncate=True; use truncate=False '
            'if you do not need a truncation length.')

    cv_dir = cross_validated_root(data_dir)
    cv_trad_dir = cross_validated_trad_root(data_dir)
    novel_dir = novel_taxa_simulations_root(data_dir)
    novel_cv_tmp_dir = join(data_dir, '.novel-tmp-cross-validated-taxa')
    if make_novel and not make_cv_taxa and exists(novel_cv_tmp_dir):
        rmtree(novel_cv_tmp_dir)

    for index, data in dataframe.iterrows():
        db_dir = join(ref_dbs_root(data_dir), data['Reference id'])
        if not exists(db_dir):
            makedirs(db_dir)

        # Clean taxonomy/sequences, to remove empty/ambiguous taxonomies
        clean_fasta = join(db_dir, '{0}_clean.fasta'.format(
            basename(splitext(data['Reference file path'])[0])))
        clean_taxa = join(db_dir, '{0}_clean.tsv'.format(
            basename(splitext(data['Reference tax path'])[0])))
        if not exists(clean_fasta) or force:
            clean_taxa, clean_fasta = clean_database(
              data['Reference tax path'], data['Reference file path'], db_dir)

        simulated_reads_fp = simulated_reads_filepath(
            clean_fasta, data['Fwd primer id'], data['Rev primer id'],
            min_read_length=min_read_length, trim_primers=trim_primers,
            truncate=truncate)
        if not exists(simulated_reads_fp) or force:
            with open(simulated_reads_fp, 'w') as simulated_reads:
                if trim_primers:
                    seqs = Artifact.import_data(
                        'FeatureData[Sequence]', clean_fasta)
                    trunc_len = read_length if truncate else 0
                    trimmed = feature_classifier.methods.extract_reads(
                        sequences=seqs, trunc_len=trunc_len,
                        f_primer=data['Fwd primer'],
                        r_primer=data['Rev primer']).reads
                    for seq in trimmed.view(DNAIterator):
                        if (min_read_length is None
                                or len(seq) >= min_read_length):
                            seq.write(simulated_reads, format='fasta')
                else:
                    for seq in io.read(clean_fasta, format='fasta'):
                        out_seq = seq[:read_length] if truncate else seq
                        if (min_read_length is None
                                or len(out_seq) >= min_read_length):
                            out_seq.write(simulated_reads, format='fasta')
        else:
            print('simulated reads and amplicons exist: skipping extraction')

        # Print sequence counts to see how many seqs were filtered
        print(index, 'Sequence Counts')
        print('Raw Fasta:           ', seq_count(data['Reference file path']))
        print('Clean Fasta:         ', seq_count(clean_fasta))
        print('Simulated Reads:     ', seq_count(simulated_reads_fp))

        # Generate simulated community query and reference seqs/taxa pairs.
        if make_cv_taxa or make_novel:
            cv_taxa_dir = cv_dir if make_cv_taxa else novel_cv_tmp_dir
            generate_cross_validated_sequences(
                clean_taxa, simulated_reads_fp, index, iterations, cv_taxa_dir)

        if make_cv_trad:
            generate_cross_validated_trad_sequences(
                clean_taxa, simulated_reads_fp, index, iterations, cv_trad_dir,
                force=force, query_size=trad_cv_query_size)

        # Generate novel query and reference seqs/taxa pairs
        if make_novel:
            # Scope to this database: the loop calls this once per database, so
            # an unscoped glob would rebuild the earlier databases' novel folds
            # again and would read any half-written fold left in cv_taxa_dir.
            generate_novel_sequence_sets(
                cv_taxa_dir, novel_dir, levelrange=levelrange, index=index)
            # If cv-taxa output wasn't requested, discard temporary folds.
            if not make_cv_taxa:
                for fold_dir in glob(
                        join(cv_taxa_dir, format_cv_fold_dirname(index, '*'))):
                    rmtree(fold_dir)

    if make_novel and not make_cv_taxa and exists(novel_cv_tmp_dir):
        rmtree(novel_cv_tmp_dir)


def generate_self_validated_datasets(dataframe, data_dir, read_length=None,
                                     force=False, min_read_length=80,
                                     trim_primers=True, truncate=True):
    '''Build self-validation datasets: classify every reference sequence
    against the full database.

    Unlike cross-validated methods, no sequences are removed from the
    reference and there are no CV folds — one dataset per reference database.
    '''
    if truncate and read_length is None:
        raise ValueError(
            'read_length is required when truncate=True; use truncate=False '
            'if you do not need a truncation length.')

    sv_dir = self_validated_root(data_dir)

    for index, data in dataframe.iterrows():
        db_dir = join(ref_dbs_root(data_dir), data['Reference id'])
        if not exists(db_dir):
            makedirs(db_dir)

        clean_fasta = join(db_dir, '{0}_clean.fasta'.format(
            basename(splitext(data['Reference file path'])[0])))
        clean_taxa = join(db_dir, '{0}_clean.tsv'.format(
            basename(splitext(data['Reference tax path'])[0])))
        if not exists(clean_fasta) or force:
            clean_taxa, clean_fasta = clean_database(
                data['Reference tax path'], data['Reference file path'], db_dir)

        simulated_reads_fp = simulated_reads_filepath(
            clean_fasta, data['Fwd primer id'], data['Rev primer id'],
            min_read_length=min_read_length, trim_primers=trim_primers,
            truncate=truncate)
        if not exists(simulated_reads_fp) or force:
            with open(simulated_reads_fp, 'w') as simulated_reads:
                if trim_primers:
                    seqs = Artifact.import_data(
                        'FeatureData[Sequence]', clean_fasta)
                    trunc_len = read_length if truncate else 0
                    trimmed = feature_classifier.methods.extract_reads(
                        sequences=seqs, trunc_len=trunc_len,
                        f_primer=data['Fwd primer'],
                        r_primer=data['Rev primer']).reads
                    for seq in trimmed.view(DNAIterator):
                        if (min_read_length is None
                                or len(seq) >= min_read_length):
                            seq.write(simulated_reads, format='fasta')
                else:
                    for seq in io.read(clean_fasta, format='fasta'):
                        out_seq = seq[:read_length] if truncate else seq
                        if (min_read_length is None
                                or len(out_seq) >= min_read_length):
                            out_seq.write(simulated_reads, format='fasta')
        else:
            print('simulated reads and amplicons exist: skipping extraction')

        print(index, 'Sequence Counts')
        print('Raw Fasta:           ', seq_count(data['Reference file path']))
        print('Clean Fasta:         ', seq_count(clean_fasta))
        print('Simulated Reads:     ', seq_count(simulated_reads_fp))

        db_sv_dir = join(sv_dir, index)
        if not exists(db_sv_dir):
            makedirs(db_sv_dir)

        query_taxa_fp = join(db_sv_dir, QUERY_TAXA_TSV)
        query_fp = join(db_sv_dir, QUERY_FASTA)
        ref_fp = join(db_sv_dir, REF_SEQS_FASTA)
        ref_taxa_fp = join(db_sv_dir, REF_TAXA_TSV)

        if force or not exists(query_taxa_fp):
            simulated_reads = list(io.read(simulated_reads_fp, format='fasta'))
            taxonomy = Artifact.import_data(
                'FeatureData[Taxonomy]', clean_taxa,
                view_type='HeaderlessTSVTaxonomyFormat')
            taxonomy_series = taxonomy.view(pd.Series)
            seq_ids = [seq.metadata['id'] for seq in simulated_reads]
            all_list = [
                '\t'.join([sid, str(taxonomy_series.loc[sid]).strip()])
                for sid in sorted(seq_ids)]
            export_list_to_file(all_list, query_taxa_fp)

            with open(query_fp, 'w') as query_fasta:
                for seq in simulated_reads:
                    seq.write(query_fasta, format='fasta')

            ref_reads_abs = abspath(simulated_reads_fp)
            ref_taxa_abs = abspath(clean_taxa)
            _replace_symlink(ref_fp, ref_reads_abs)
            _replace_symlink(ref_taxa_fp, ref_taxa_abs)

            shared_ref_seqs_qza = join(db_dir, '_trad_cv_shared_ref_seqs.qza')
            shared_ref_taxa_qza = join(db_dir, '_trad_cv_shared_ref_taxa.qza')
            if force or not exists(shared_ref_seqs_qza):
                if exists(shared_ref_seqs_qza):
                    remove(shared_ref_seqs_qza)
                artifact = Artifact.import_data(
                    'FeatureData[Sequence]', simulated_reads_fp)
                artifact.save(shared_ref_seqs_qza)
            if force or not exists(shared_ref_taxa_qza):
                if exists(shared_ref_taxa_qza):
                    remove(shared_ref_taxa_qza)
                artifact = Artifact.import_data(
                    'FeatureData[Taxonomy]', clean_taxa,
                    view_type='HeaderlessTSVTaxonomyFormat')
                artifact.save(shared_ref_taxa_qza)

            _replace_symlink(ref_fp[:-5] + 'qza', shared_ref_seqs_qza)
            _replace_symlink(ref_taxa_fp[:-3] + 'qza', shared_ref_taxa_qza)

            artifact = Artifact.import_data('FeatureData[Sequence]', query_fp)
            artifact.save(query_fp[:-5] + 'qza')


def generate_novel_sequence_sets(cv_dir, novel_dir,
                                 levelrange=range(6, 0, -1), index=None):
    '''Generate paired query/reference fastas and taxonomies for novel taxa
    analysis, given an input of simulated amplicon taxonomies (read_taxa)
    and fastas (simulated_reads_fp), the index (database) name, # of
    iterations to perform (cross-validated data subsets), the output dir
    (novel_dir).
    novel_dir: path
        base output directory to contain simulated datasets
    cv_dir: path
        directory containing cross-validation data sets for filtering
    index: str or None
        Reference database name. Only that database's fold directories
        (``<index>-iter*``) are read. ``None`` reads every fold in *cv_dir*,
        which is only correct when it holds a single database: this function is
        called once per database, so an unscoped glob reprocesses the earlier
        databases' folds and, worse, trips over a half-written fold left in
        ``cv_dir`` by a previous failed run (one with ``ref_taxa.tsv`` but no
        ``query_taxa.tsv`` yet).
    '''

    fold_pattern = '*' if index is None else format_cv_fold_dirname(index, '*')
    for cv_fold_dir in sorted(glob(join(cv_dir, fold_pattern))):
        query_taxa_in = join(cv_fold_dir, QUERY_TAXA_TSV)
        if not isfile(query_taxa_in):
            # A fold directory without its query taxonomy is left over from a
            # run that died partway through writing it. Skipping is safe: this
            # run rewrites the folds it owns before getting here.
            print('WARNING: skipping {0}: no {1} (left over from an '
                  'interrupted run?)'.format(cv_fold_dir, QUERY_TAXA_TSV))
            continue
        cv_parts = parse_cv_dataset_id(basename(cv_fold_dir))
        fold_db, iteration = cv_parts.database, cv_parts.iteration
        for level in levelrange:
            novel_fold_dir = join(
                novel_dir,
                format_novel_fold_dirname(fold_db, level, int(iteration)))
            if not exists(novel_fold_dir):
                makedirs(novel_fold_dir)

            # Trim taxonomy strings to level X and copy them into the directory
            query_taxa = trim_taxonomy_strings(
                join(cv_fold_dir, QUERY_TAXA_TSV), level)
            query_taxa_fp = join(novel_fold_dir, QUERY_TAXA_TSV)
            export_list_to_file(query_taxa, query_taxa_fp)

            # Create REF TAXONOMY from list of taxonomies that
            #    DO NOT match QUERY taxonomies
            ref_taxa_fp = join(novel_fold_dir, REF_TAXA_TSV)
            name_list = '^' + '$|^'.join(list(set(extract_taxa_names(
                query_taxa_fp, slice(1, level+1), stripchars='')))) + '$'
            ref_taxa = string_search(join(cv_fold_dir, REF_TAXA_TSV),
                                     name_list, discard=True,
                                     field=slice(1, level+1))
            export_list_to_file(ref_taxa, ref_taxa_fp)

            # Now trim the taxonomy strings to level X-1 for consistency with
            # CV and remove any that are unclassified at that level
            query_taxa = trim_taxonomy_strings(query_taxa_fp, level-1)
            separated = defaultdict(list)
            for s in query_taxa:
                separated[s.count(';')].append(s)
            query_taxa = separated[max(separated)]

            # Removing the query taxon can remove its parent as well (a
            # monotypic genus goes with its only species). A rank the reference
            # no longer holds cannot be returned by any classifier, so cut each
            # expected taxonomy back to its deepest rank still in this fold's
            # reference; queries with no rank left are dropped.
            truncated = truncate_taxa_to_reference(query_taxa, ref_taxa_fp)
            before = dict(
                line.split('\t', 1) for line in query_taxa if '\t' in line)
            after = dict(
                line.split('\t', 1) for line in truncated if '\t' in line)
            n_trimmed = sum(
                1 for sid, taxon in after.items() if before.get(sid) != taxon)
            n_dropped = len(before) - len(after)
            if n_trimmed or n_dropped:
                message = (
                    '{0}: expected taxonomy truncated to the reference for {1} '
                    'of {2} queries'.format(
                        basename(novel_fold_dir), n_trimmed, len(before)))
                if n_dropped:
                    message += (
                        '; {0} dropped (no rank left in reference)'.format(
                            n_dropped))
                print(message)
            export_list_to_file(truncated, query_taxa_fp)

            # Create REF: Filter ref database to contain only seqs that
            #    match non-matching taxonomy strings
            ref_fp = join(novel_fold_dir, REF_SEQS_FASTA)
            filter_sequences(join(cv_fold_dir, REF_SEQS_FASTA),
                             ref_fp, ref_taxa_fp, keep=True)

            # Create QUERY: Filter ref database to contain only seqs
            #    that match QUERY TAXONOMY
            query_fp = join(novel_fold_dir, QUERY_FASTA)
            filter_sequences(join(cv_fold_dir, QUERY_FASTA),
                             query_fp, query_taxa_fp, keep=True)

            # Encode as Artifacts for convenience
            artifact = Artifact.import_data('FeatureData[Sequence]', ref_fp)
            artifact.save(ref_fp[:-5] + 'qza')
            artifact = Artifact.import_data('FeatureData[Sequence]', query_fp)
            artifact.save(query_fp[:-5] + 'qza')
            artifact = Artifact.import_data(
                'FeatureData[Taxonomy]', ref_taxa_fp,
                view_type='HeaderlessTSVTaxonomyFormat')
            artifact.save(ref_taxa_fp[:-3] + 'qza')


def generate_cross_validated_sequences(read_taxa, simulated_reads_fp, index,
                                       iterations, cv_dir):
    '''Generates simulated community files (fasta and taxonomy) as subsets of
    simulated amplicons/taxa for cross-validated taxonomy assignment. Selects
    duplicated taxa names, evenly allocates these among subsets as query taxa
    (test set), generates ref taxa (training set) that do not match query fasta
    IDs, and creates fasta files to match each of these sets.

    Each query's expected taxonomy is truncated to its deepest prefix that the
    fold's training set still contains. A query with **no** prefix in training --
    not even its first rank -- is dropped from the fold, and the per-fold counts
    are printed. That happens when every sequence sharing a first rank lands in
    the same test fold, which is guaranteed when only one sequence holds it (the
    split runs on reads surviving amplicon extraction, so a clade can be
    plentiful in the database and a singleton here). Dropped queries are left
    out of ``query.fasta`` as well, keeping it aligned with ``query_taxa.tsv``.
    read_taxa: list or path
        list or file of taxonomies corresponding to simulated_reads_fp
    simulated_reads_fp: path
        simulated amplicon reads (fasta format file)
    index: str
        reference database name
    iterations: int >= 2
        number of subsets to create
    cv_dir: path
        base output directory to contain simulated datasets
    '''
    if iterations < 2:
        raise ValueError('Must perform two or more iterations for '
                         'construction of cross-validated datasets.')

    # Stratify the data and form the CV data sets
    simulated_reads = list(io.read(simulated_reads_fp, format='fasta'))
    taxonomy = Artifact.import_data(
        'FeatureData[Taxonomy]', read_taxa,
        view_type='HeaderlessTSVTaxonomyFormat')
    tree = build_tree(taxonomy, simulated_reads)
    strata = get_strata(tree, iterations)
    print(index + ': generating', iterations, 'folds on', len(strata),
          'strata')
    X, y = zip(*[(s, t) for t, ss in strata for s in ss])
    skf = StratifiedKFold(
        n_splits=iterations, shuffle=True, random_state=0)
    splits = []
    for train, test in skf.split(X, y):
        train_set = {X[i] for i in train}
        test_set = {X[i] for i in test}
        splits.append((train_set, test_set))

    # Output the CV data sets in the expected formats
    taxonomy_series = taxonomy.view(pd.Series)
    for iteration, (train, test) in enumerate(splits):
        db_iter_dir = join(cv_dir, format_cv_fold_dirname(index, iteration))
        if not exists(db_iter_dir):
            makedirs(db_iter_dir)
        query_taxa_fp = join(db_iter_dir, QUERY_TAXA_TSV)
        query_fp = join(db_iter_dir, QUERY_FASTA)
        ref_fp = join(db_iter_dir, REF_SEQS_FASTA)
        ref_taxa_fp = join(db_iter_dir, REF_TAXA_TSV)

        # Output the taxa files (pandas >= 2.2 disallows set indexers)
        train_ids = list(train)
        train_series = taxonomy_series.loc[train_ids]
        train_series.to_csv(ref_taxa_fp, sep='\t', header=False)
        # If a taxonomy in the test set doesn't exist in the training set, trim
        # it until it does
        train_taxonomies = set()
        for taxonomy in train_series.values:
            taxonomy = taxonomy.split(';')
            for level in range(1, len(taxonomy)+1):
                train_taxonomies.add(';'.join(taxonomy[:level]))
        test_list = []
        n_trimmed = 0
        dropped = []
        for sid in test:
            taxonomy = taxonomy_series[sid].split(';')
            for level in range(len(taxonomy), 0, -1):
                if ';'.join(taxonomy[:level]) in train_taxonomies:
                    if level < len(taxonomy):
                        n_trimmed += 1
                    test_list.append(
                        '\t'.join([sid, ';'.join(taxonomy[:level]).strip()]))
                    break
            else:
                # Not even the first rank survives in this fold's training set.
                # That happens when every sequence sharing a first rank landed
                # in this test fold -- guaranteed when only one sequence holds
                # it. No classifier can return a rank the reference does not
                # contain, so grading against it would score every method as
                # wrong whatever it did. Drop the query from the fold, the same
                # way generate_novel_sequence_sets does.
                dropped.append(sid)
        if n_trimmed or dropped:
            message = (
                '{0}: expected taxonomy truncated to the training set for {1} '
                'of {2} queries'.format(
                    format_cv_fold_dirname(index, iteration), n_trimmed,
                    len(test)))
            if dropped:
                missing = sorted({
                    taxonomy_series[s].split(';')[0] for s in dropped})
                message += (
                    '; {0} dropped (first rank absent from the training set: '
                    '{1}{2})'.format(
                        len(dropped), ', '.join(repr(m) for m in missing[:5]),
                        ', ...' if len(missing) > 5 else ''))
            print(message)
        if not test_list:
            raise RuntimeError(
                '{0}: every query was dropped, leaving an empty fold. The '
                'reference database has too little overlap between folds to '
                'cross-validate; reduce iterations or clean the '
                'taxonomy.'.format(format_cv_fold_dirname(index, iteration)))
        export_list_to_file(test_list, query_taxa_fp)
        # Output the reference files. Dropped queries are left out of
        # query.fasta as well, so it stays aligned with query_taxa.tsv -- a
        # sequence with no expected taxonomy would otherwise be classified and
        # then have nothing to score against.
        dropped_ids = set(dropped)
        with open(ref_fp, 'w') as ref_fasta:
            with open(query_fp, 'w') as query_fasta:
                for seq in simulated_reads:
                    sid = seq.metadata['id']
                    if sid in train:
                        seq.write(ref_fasta, format='fasta')
                    elif sid not in dropped_ids:
                        seq.write(query_fasta, format='fasta')

        # Encode as Artifacts for convenience
        artifact = Artifact.import_data('FeatureData[Sequence]', ref_fp)
        artifact.save(ref_fp[:-5] + 'qza')
        artifact = Artifact.import_data('FeatureData[Sequence]', query_fp)
        artifact.save(query_fp[:-5] + 'qza')
        artifact = Artifact.import_data(
            'FeatureData[Taxonomy]', ref_taxa_fp,
            view_type='HeaderlessTSVTaxonomyFormat')
        artifact.save(ref_taxa_fp[:-3] + 'qza')


_QUERY_SIZE_MEANING = (
    'query_size is the TOTAL pool of query sequences, divided evenly between '
    'the iterations. A float in (0, 1] means a FRACTION of the database (0.2 '
    'means 20% of sequences in total, so 20%/iterations per fold); an int '
    'means an ABSOLUTE NUMBER of sequences (10 means 10 sequences in total, '
    'not 10% and not 10 per fold).')


def _resolve_query_pool_size(query_size, n_sequences, iterations, index):
    '''Resolve a ``cross-validated-trad`` query pool size to a sequence count.

    *query_size* is the total number of sequences queried across all folds:
    a float in ``(0, 1]`` as a fraction of the database, or an int as an
    absolute count. Returns that count as an int, or raises ``ValueError``
    naming both readings, since ``0.2`` and ``20`` mean very different things.

    An int larger than the database is **not** an error: a warning is printed
    and the whole database is used, so one int can be shared across databases
    of different sizes. The pool must still hold at least one sequence per
    fold, so the resolved count is required to be >= *iterations*.
    '''
    # bool is a subclass of int; reject it before the int branch accepts True.
    if isinstance(query_size, bool):
        raise ValueError(
            '{0}: query_size must be a float or an int, got bool {1!r}. '
            '{2}'.format(index, query_size, _QUERY_SIZE_MEANING))
    if isinstance(query_size, int):
        n_pool = query_size
        if n_pool < 1:
            raise ValueError(
                '{0}: query_size={1} sequences is out of range; an int '
                'query_size must be at least 1. {2}'.format(
                    index, query_size, _QUERY_SIZE_MEANING))
        if n_pool > n_sequences:
            # Asking for more sequences than the database holds is expected
            # when one int is used to equalize query counts across databases
            # of different sizes; fall back to the whole database.
            print('WARNING: {0}: query_size={1} exceeds the {2} sequences in '
                  'this database; querying the entire database ({2} sequences) '
                  'instead. Query counts will not match databases large enough '
                  'to supply {1} sequences.'.format(
                      index, query_size, n_sequences))
            n_pool = n_sequences
    elif isinstance(query_size, float):
        if not 0 < query_size <= 1:
            hint = ''
            if query_size > 1:
                hint = (' To query {0:.0f} sequences in total, pass the int '
                        '{0:.0f} instead.'.format(query_size))
            raise ValueError(
                '{0}: query_size={1} is out of range; a float query_size must '
                'be greater than 0 and at most 1 (1.0 means the whole '
                'database, which is the default). {2}{3}'.format(
                    index, query_size, _QUERY_SIZE_MEANING, hint))
        n_pool = round(query_size * n_sequences)
    else:
        raise ValueError(
            '{0}: query_size must be a float, an int, or None, got {1!r} of '
            'type {2}. {3}'.format(
                index, query_size, type(query_size).__name__,
                _QUERY_SIZE_MEANING))
    if n_pool < iterations:
        # No point suggesting a bigger query_size once the pool is the whole
        # database: the database itself is the limit.
        fix = ('Reduce iterations to {0} or fewer.'.format(n_sequences)
               if n_pool >= n_sequences
               else 'Increase query_size or reduce iterations.')
        raise ValueError(
            '{0}: query_size={1} resolves to a total query pool of {2} '
            'sequence(s) out of {3}, which cannot be divided between '
            '{4} iterations (at least 1 sequence per fold is required). '
            '{5} {6}'.format(
                index, query_size, n_pool, n_sequences, iterations,
                fix, _QUERY_SIZE_MEANING))
    return n_pool


def _replace_symlink(link_path, target_path):
    '''Create or replace *link_path* with a symlink to *target_path* (absolute).'''
    target_abs = abspath(target_path)
    if lexists(link_path):
        remove(link_path)
    symlink(target_abs, link_path)


def generate_cross_validated_trad_sequences(read_taxa, simulated_reads_fp, index,
                                            iterations, cv_dir, force=False,
                                            query_size=None):
    '''Cross-validation folds by random split of sequence IDs (traditional CV).

    For each fold, test sequences are drawn at random. Query FASTA and query
    taxonomy hold only the test fold. The reference FASTA and taxonomy are
    symbolic links to the full simulated-reads FASTA and cleaned taxonomy TSV,
    so **the reference is the entire database in every fold, queries
    included** — each query sequence has an exact self-match in the reference
    it is classified against. Query taxonomy lines use the full expected
    string from the database — no trimming to match the training taxonomies
    and no check that test taxa appear in the reference.

    Because the reference is the whole database, every query's full lineage is
    present in it by construction. So unlike the taxonomy-aware builder, this
    one never truncates or drops a query: there is no equivalent here of a
    first rank going missing from the training set.

    ``ref_seqs.qza`` and ``ref_taxa.qza`` are also symlinked to shared artifacts
    under the reference database directory (created once per database) to avoid
    duplicating QIIME artifacts across folds.

    Parameters
    ----------
    query_size : float, int, or None, optional
        Size of the **total query pool**, which is then divided evenly between
        the ``iterations`` folds. A **float** in ``(0, 1]`` is a **fraction**
        of the database; an **int** is an **absolute number of sequences**.
        Both describe the total across all folds, not the size of one fold.

        For example ``query_size=0.2`` with ``iterations=8`` queries 20% of
        the database in total, so each fold holds 20%/8 = 2.5% of it.
        Likewise ``query_size=8000`` with ``iterations=8`` gives 8 folds of
        1000 sequences each.

        ``None`` (default) uses the whole database as the pool, which is
        equivalent to ``query_size=1.0``: each fold holds
        ``n_sequences / iterations`` sequences.

        An **int larger than this database** is not an error: a warning is
        printed and the entire database is used as the pool. This lets one
        int be shared across databases of different sizes, equalizing query
        counts wherever the database is big enough to supply them.

        In every case a random pool of the requested size is drawn (with a
        fixed seed), then split with
        ``KFold(n_splits=iterations, shuffle=True)``, so folds are disjoint
        and together cover the pool exactly once. Folds differ by at most one
        sequence when the pool does not divide evenly.

    Notes
    -----
    Every fold's ``query.fasta`` and ``query_taxa.tsv`` are rewritten on each
    call, so a changed *query_size* takes effect without any cache flag.
    ``force`` here only deletes the shared ``_trad_cv_shared_ref_*.qza``
    artifacts, which hold the full-database reference and do not depend on
    *query_size*. (Under Tourmaline the catch is one level up: snakemake skips
    ``tax_credit_prepare_datasets`` on a finished run, so delete
    ``.datasets.done`` or use ``--forcerun`` to make this function run at all.
    ``force_regenerate`` does not trigger it.)
    '''
    if iterations < 2:
        raise ValueError('Must perform two or more iterations for '
                         'construction of cross-validated datasets.')

    db_dir = dirname(simulated_reads_fp)
    shared_ref_seqs_qza = join(db_dir, '_trad_cv_shared_ref_seqs.qza')
    shared_ref_taxa_qza = join(db_dir, '_trad_cv_shared_ref_taxa.qza')
    if force:
        for p in (shared_ref_seqs_qza, shared_ref_taxa_qza):
            if exists(p):
                remove(p)

    simulated_reads = list(io.read(simulated_reads_fp, format='fasta'))
    seq_ids = [seq.metadata['id'] for seq in simulated_reads]
    taxonomy = Artifact.import_data(
        'FeatureData[Taxonomy]', read_taxa,
        view_type='HeaderlessTSVTaxonomyFormat')
    taxonomy_series = taxonomy.view(pd.Series)

    # The query pool is what gets divided between the folds. By default it is
    # the whole database; query_size shrinks it to a random subset first.
    if query_size is None:
        pool_ids = seq_ids
        print('{0}: dividing all {1} sequences between {2} folds '
              '(~{3} query sequences per fold)'.format(
                  index, len(seq_ids), iterations, len(seq_ids) // iterations))
    else:
        n_pool = _resolve_query_pool_size(
            query_size, len(seq_ids), iterations, index)
        # Fixed seed so the pool is reproducible across runs.
        pool_idx = random.RandomState(0).permutation(len(seq_ids))[:n_pool]
        pool_ids = [seq_ids[i] for i in sorted(pool_idx)]
        print('{0}: query_size={1!r} -> dividing a random pool of {2} of {3} '
              'sequences ({4:.2%} of the database) between {5} folds '
              '(~{6} query sequences per fold, ~{7:.2%} of the database '
              'each)'.format(
                  index, query_size, n_pool, len(seq_ids),
                  n_pool / len(seq_ids), iterations, n_pool // iterations,
                  (n_pool / iterations) / len(seq_ids)))

    ref_reads_abs = abspath(simulated_reads_fp)
    ref_taxa_abs = abspath(read_taxa)

    splitter = KFold(n_splits=iterations, shuffle=True, random_state=0)
    for iteration, (_train_idx, test_idx) in enumerate(splitter.split(pool_ids)):
        test = {pool_ids[i] for i in test_idx}
        db_iter_dir = join(cv_dir, format_cv_fold_dirname(index, iteration))
        if not exists(db_iter_dir):
            makedirs(db_iter_dir)
        query_taxa_fp = join(db_iter_dir, QUERY_TAXA_TSV)
        query_fp = join(db_iter_dir, QUERY_FASTA)
        ref_fp = join(db_iter_dir, REF_SEQS_FASTA)
        ref_taxa_fp = join(db_iter_dir, REF_TAXA_TSV)

        test_list = [
            '\t'.join([sid, str(taxonomy_series.loc[sid]).strip()])
            for sid in sorted(test)]
        export_list_to_file(test_list, query_taxa_fp)

        with open(query_fp, 'w') as query_fasta:
            for seq in simulated_reads:
                if seq.metadata['id'] in test:
                    seq.write(query_fasta, format='fasta')

        _replace_symlink(ref_fp, ref_reads_abs)
        _replace_symlink(ref_taxa_fp, ref_taxa_abs)

        if not exists(shared_ref_seqs_qza):
            artifact = Artifact.import_data(
                'FeatureData[Sequence]', simulated_reads_fp)
            artifact.save(shared_ref_seqs_qza)
        if not exists(shared_ref_taxa_qza):
            artifact = Artifact.import_data(
                'FeatureData[Taxonomy]', read_taxa,
                view_type='HeaderlessTSVTaxonomyFormat')
            artifact.save(shared_ref_taxa_qza)

        _replace_symlink(ref_fp[:-5] + 'qza', shared_ref_seqs_qza)
        _replace_symlink(ref_taxa_fp[:-3] + 'qza', shared_ref_taxa_qza)

        artifact = Artifact.import_data('FeatureData[Sequence]', query_fp)
        artifact.save(query_fp[:-5] + 'qza')


def test_cross_validated_sequences(data_dir):
    '''confirm that test (query) taxa IDs are not in training (ref) set, but
    that all taxonomy strings are.
    '''
    simulated_dir = cross_validated_root(data_dir)
    for db_iter_dir in glob(join(simulated_dir, GLOB_CV_FOLD_DIRS)):
        query_taxa = import_to_list(
            join(db_iter_dir, QUERY_TAXA_TSV), field=0)
        ref_taxa = import_to_list(
            join(db_iter_dir, REF_TAXA_TSV), field=0)
        common = set(query_taxa).intersection(set(ref_taxa))
        if common:
            print('sequences in training and test sets:')
            print(common)


def test_novel_taxa_datasets(data_dir):
    '''confirm that test (query) taxa IDs and taxonomies are not in training
    (ref) set, but sister branch taxa are.
    '''
    novel_dir = novel_taxa_simulations_root(data_dir)
    for db_iter_dir in glob(join(novel_dir, GLOB_NOVEL_FOLD_DIRS)):
        query_taxa = import_taxonomy_to_dict(join(db_iter_dir,
                                                  QUERY_TAXA_TSV))
        ref_taxa = import_taxonomy_to_dict(join(db_iter_dir,
                                                REF_TAXA_TSV))
        for key, value in query_taxa.items():
            if key in ref_taxa:
                print('key duplicate:', basename(db_iter_dir), key)
            if value in ref_taxa.values():
                print('value duplicate:', basename(db_iter_dir), value)


def recall_simulated_taxa_dirs(data_dir, databases, iterations,
                               ref_seqs=REF_SEQS_FASTA, ref_taxa=REF_TAXA_TSV,
                               max_level=6, min_level=0, multilevel=True):
    '''Build sweep inputs for simulated reference datasets (novel-taxa or CV).

    Returns a list of ``(dataset_id, reference_id)`` pairs (here both the same
    string) and a dict mapping each ``dataset_id`` to ``(ref_seqs_path,
    ref_taxa_path)`` under ``data_dir``.

    Parameters
    ----------
    data_dir : str
        Root containing fold directories (e.g. ``novel-taxa-simulations/`` or
        ``cross-validated/``), not the repo root.
    databases : iterable of str
        Reference database names used when generating simulated datasets.
    iterations : int
        Number of CV / novel folds (``iter0`` … ``iter{n-1}``).
    ref_seqs, ref_taxa : str
        Basenames of reference sequence and taxonomy files in each fold
        (defaults: ``ref_seqs.fasta``, ``ref_taxa.tsv``).
    max_level, min_level : int
        For ``multilevel=True``, level range for novel-taxa fold names
        ``<db>-L<level>-iter<i>`` (e.g. ``max_level=6``, ``min_level=0`` yields
        levels 6..1). Ignored when ``multilevel=False``.
    multilevel : bool
        If ``True``, use novel-taxa fold naming. If ``False``, use cross-validated
        fold naming ``<db>-iter<i>`` (still one level slot in the loop; set
        ``max_level`` / ``min_level`` to span a single level when needed).

    Returns
    -------
    tuple
        ``(dataset_reference_combinations, reference_dbs)`` for
        ``parameter_sweep`` / ``gen_param_sweep``.
    '''
    dataset_reference_combinations = list()
    reference_dbs = dict()
    for database in databases:
        for level in range(max_level, min_level, -1):
            for iteration in range(0, iterations):
                if multilevel is True:
                    dataset_name = format_novel_fold_dirname(database,
                                                             level,
                                                             iteration)
                else:
                    dataset_name = format_cv_fold_dirname(database, iteration)
                dataset_reference_combinations.append((dataset_name,
                                                       dataset_name))
                reference_dbs[dataset_name] = (join(data_dir, dataset_name,
                                                    ref_seqs),
                                               join(data_dir, dataset_name,
                                                    ref_taxa))
    return dataset_reference_combinations, reference_dbs


def recall_self_validated_dirs(data_dir, databases,
                               ref_seqs=REF_SEQS_FASTA, ref_taxa=REF_TAXA_TSV):
    '''Build sweep inputs for self-validated datasets (one per reference DB).

    Returns the same tuple shape as :func:`recall_simulated_taxa_dirs`.
    '''
    dataset_reference_combinations = list()
    reference_dbs = dict()
    for database in databases:
        dataset_reference_combinations.append((database, database))
        reference_dbs[database] = (join(data_dir, database, ref_seqs),
                                   join(data_dir, database, ref_taxa))
    return dataset_reference_combinations, reference_dbs


def trad_cv_shared_reference_qzas(data_dir, reference_id):
    '''Absolute paths to the shared QIIME artifacts for ``cross-validated-trad``.

    All folds for a reference database reuse the same ``FeatureData[Sequence]``
    and ``FeatureData[Taxonomy]`` artifacts under ``ref_dbs/<reference_id>/``.

    Parameters
    ----------
    data_dir : str
        Top-level data directory (parent of ``ref_dbs``), same as used in
        ``generate_simulated_datasets``.
    reference_id : str
        The ``Reference id`` / ``ref_dbs`` subdirectory name for that database.

    Returns
    -------
    tuple
        ``(ref_seqs_qza, ref_taxa_qza)`` paths.
    '''
    db_dir = join(ref_dbs_root(data_dir), reference_id)
    return (
        join(db_dir, '_trad_cv_shared_ref_seqs.qza'),
        join(db_dir, '_trad_cv_shared_ref_taxa.qza'),
    )


def _split_fit_classify_params(parameters):
    '''Split method entry into fit vs classify param dicts for trad CV helper.

    If *parameters* contains keys ``fit`` and ``classify`` (``fit`` must be a
    ``dict``), returns ``(parameters['fit'], parameters.get('classify') or {})``.
    Otherwise returns ``(parameters, {})`` so all keys are fit-only (backward
    compatible with :func:`parameter_sweep`).
    '''
    if isinstance(parameters, dict) and 'fit' in parameters and isinstance(
            parameters['fit'], dict):
        fit_p = parameters['fit']
        classify_p = parameters.get('classify') or {}
        if not isinstance(classify_p, dict):
            raise TypeError(
                '"classify" entry must be a dict; got {!r}'.format(classify_p))
        return fit_p, classify_p
    return parameters, {}


def _parameter_comb_id(parameter_ids, parameter_combination):
    return ':'.join(map(str, parameter_combination))


def _classify_sklearn_cli_flags(classify_params, confidence, classify_n_jobs):
    '''Build ``--p-* ...`` string for ``classify-sklearn`` from a param dict.'''
    flags = dict(classify_params)
    if 'p-confidence' not in flags:
        flags['p-confidence'] = confidence
    if 'p-n-jobs' not in flags:
        flags['p-n-jobs'] = classify_n_jobs
    parts = []
    for key in sorted(flags.keys()):
        parts.append('--{0} {1}'.format(key, flags[key]))
    return ' '.join(parts)


def trad_cv_naive_bayes_commands_single_classifier(
        trad_sim_data_dir,
        project_data_dir,
        results_dir,
        database_names,
        iterations,
        method_parameters_combinations,
        reference_id_by_database=None,
        confidence=0.7,
        classify_n_jobs=5,
        force=False):
    '''QIIME 2 naive Bayes commands for ``cross-validated-trad`` with one fit per DB.

    For traditional CV sims the reference is identical in every fold (full database).
    This returns two command lists: **fit** once per (database, **fit** parameter
    combo), then **classify** per (fold, fit combo × classify combo) reusing
    ``results_dir/<database>/<database>/<method>/<fit-params>/classifier.qza``.
    Assignment outputs use the same depth as :func:`parameter_sweep` with
    ``recall_simulated_taxa_dirs(..., multilevel=False)``:
    ``results_dir/<fold-id>/<fold-id>/<method>/<combined-params>/``.

    **method_parameters_combinations** supports two shapes per method:

    - **Fit only (legacy):** same as :func:`parameter_sweep` — a flat dict of
      parameter names to lists, all passed to ``fit-classifier-naive-bayes``.
      Classify uses ``confidence`` and ``classify_n_jobs`` only.

    - **Fit + classify:** ``{'fit': {...}, 'classify': {...}}`` — each is a dict
      of QIIME flag stems (e.g. ``p-classify--alpha``) to lists for the Cartesian
      product. Fit params train the classifier; classify params (e.g.
      ``p-confidence``, ``p-n-jobs``, ``p-reads-per-batch``) are passed to
      ``classify-sklearn``. If ``p-confidence`` or ``p-n-jobs`` are omitted from
      ``classify``, the function defaults them from ``confidence`` and
      ``classify_n_jobs``.

    Output folder names: ``<fit-id>`` when there are no classify sweeps; otherwise
    ``<fit-id>__cls__<classify-id>`` so each assignment path is unique.

    Parameters
    ----------
    trad_sim_data_dir : str
        Root containing fold dirs (e.g. ``.../data/cross-validated-trad``).
    project_data_dir : str
        Parent of ``ref_dbs`` (e.g. ``.../data``), used with
        :func:`trad_cv_shared_reference_qzas`.
    results_dir : str
        Scratch/output root for classifier and assignments (mirrors
        :func:`parameter_sweep` layout).
    database_names : iterable of str
        Keys matching fold prefixes (e.g. ``CRABS_ecoPCR`` for
        ``CRABS_ecoPCR-iter0``).
    iterations : int
        Number of folds per database.
    method_parameters_combinations : dict
        Per-method flat fit dict, or ``{'fit': {...}, 'classify': {...}}``.
    reference_id_by_database : dict or None
        Maps each database name to its ``ref_dbs/<id>`` folder name; default is
        identity (``db -> db``).
    confidence : float
        Default ``--p-confidence`` when not listed under ``classify``.
    classify_n_jobs : int
        Default ``--p-n-jobs`` when not listed under ``classify``.
    force : bool
        If True, always emit commands even when outputs exist.

    Returns
    -------
    tuple
        ``(fit_commands, classify_commands)`` — shell strings for QIIME 2.
    '''
    if reference_id_by_database is None:
        reference_id_by_database = {db: db for db in database_names}

    method_parameters_fit = {}
    method_parameters_classify = {}
    for method, parameters in method_parameters_combinations.items():
        fit_p, cls_p = _split_fit_classify_params(parameters)
        method_parameters_fit[method] = fit_p
        method_parameters_classify[method] = cls_p

    reference_dbs_fit = {
        db: trad_cv_shared_reference_qzas(
            project_data_dir, reference_id_by_database[db])
        for db in database_names}
    fit_combinations = [(db, db) for db in database_names]

    command_template_fit = (
        'mkdir -p "{0}"; '
        'qiime feature-classifier fit-classifier-naive-bayes --o-classifier '
        '"{0}/classifier.qza" --i-reference-reads {2} --i-reference-taxonomy {3} {5}'
    )
    fit_commands = parameter_sweep(
        trad_sim_data_dir, results_dir, reference_dbs_fit, fit_combinations,
        method_parameters_fit, command_template_fit,
        infile='query.qza', output_name='classifier.qza', force=force)

    classify_commands = []
    for db in database_names:
        for iteration in range(iterations):
            fold_id = format_cv_fold_dirname(db, iteration)
            query_reads = join(trad_sim_data_dir, fold_id, 'query.qza')
            for method, fit_params in method_parameters_fit.items():
                classify_params_template = method_parameters_classify[method]
                fit_ids = sorted(fit_params.keys())
                fit_product = list(product(*[fit_params[k] for k in fit_ids])
                                   ) if fit_ids else [()]
                cls_ids = sorted(classify_params_template.keys())
                cls_product = list(product(
                    *[classify_params_template[k] for k in cls_ids])
                ) if cls_ids else [()]

                for fit_combo in fit_product:
                    fit_id = (_parameter_comb_id(fit_ids, fit_combo)
                              if fit_ids else '')
                    for cls_combo in cls_product:
                        if cls_ids:
                            cls_id = _parameter_comb_id(cls_ids, cls_combo)
                            run_id = '{0}__cls__{1}'.format(fit_id, cls_id)
                            classify_arg_source = dict(
                                zip(cls_ids, cls_combo))
                        else:
                            run_id = fit_id
                            classify_arg_source = {}
                        param_output_dir = join(
                            results_dir, fold_id, fold_id, method, run_id)
                        classifier_fp = join(
                            results_dir, db, db, method, fit_id,
                            'classifier.qza')
                        if (not exists(join(
                                param_output_dir,
                                'rep_seqs_tax_assignments.qza'))
                                or force):
                            classify_flags = _classify_sklearn_cli_flags(
                                classify_arg_source, confidence,
                                classify_n_jobs)
                            cmd = (
                                'mkdir -p "{out}"; '
                                'qiime feature-classifier classify-sklearn '
                                '{flags} '
                                '--o-classification "{out}/rep_seqs_tax_assignments.qza" '
                                '--i-classifier "{clf}" --i-reads "{query}"; '
                                'qiime tools export --input-path '
                                '"{out}/rep_seqs_tax_assignments.qza" --output-path '
                                '"{out}/taxonomy.tsv" --output-format TSVTaxonomyFormat; '
                                'mv "{out}/taxonomy.tsv" '
                                '"{out}/query_tax_assignments.txt"; '.format(
                                    out=param_output_dir,
                                    clf=classifier_fp,
                                    query=query_reads,
                                    flags=classify_flags))
                            classify_commands.append(cmd)

    return fit_commands, classify_commands


# tag for removal — if match == recall, do we need to keep LCA?
def find_last_common_ancestor(taxon_a, taxon_b):
    '''Given two taxonomy strings, find the first level at which taxa mismatch.
    '''
    i = -1
    for i, (ta, tb) in enumerate(zip(taxon_a.split(';'), taxon_b.split(';'))):
        if ta != tb:
            return i
    return i + 1


def evaluate_classification(obs_taxon, exp_taxon):
    '''Given an observed and actual taxonomy string corresponding to a cross-
    validated simulated community, score as match, overclassification,
    underclassification, or misclassification.

    Trailing 'NA' / empty ranks count as unassigned and are ignored; internal
    'NA' ranks are kept so ranks are compared position by position.'''
    obs_taxon = normalize_taxon(obs_taxon)
    exp_taxon = normalize_taxon(exp_taxon)
    if obs_taxon == exp_taxon:
        return 'match'
    if is_unassigned_taxon(obs_taxon):
        return 'underclassification'
    obs_ranks = obs_taxon.split(';')
    exp_ranks = exp_taxon.split(';')
    # compare whole ranks, so 'A;Lut' is not a prefix of 'A;Lutjanus'
    if exp_ranks[:len(obs_ranks)] == obs_ranks:
        return 'underclassification'
    if obs_ranks[:len(exp_ranks)] == exp_ranks:
        return 'overclassification'
    return 'misclassification'


def load_taxa(obs_fp, level=slice(0, 7), field=1, sort=True):
    '''Mount observed/expected taxonomy observations.
    obs_fp: path
        Input file containing taxonomy strings and IDs.
    level: slice
        Taxonomic range of interest. 0 = kingdom, 1 = phylum, 2 = class,
        3 = order, 4 = family, 5 = genus, 6 = species.
        Must use slice notation. For species, use level=slice(6, 7)
    field: int
        ab-delimited field containing taxonomy strings.
    '''
    raw_obs = accept_list_or_file(obs_fp)
    if sort:
        raw_obs = sorted(raw_obs)
    obs = extract_taxa_names(raw_obs, field=field, level=level)
    obs = [';'.join([l.strip() for l in line.split(';')]) for line in obs
           if not line.startswith(('taxonomy', 'Taxon'))]
    return obs


def count_records(record_counter, record_name, line_count):
    '''Tally ratio of results in record counter.
    record_counter: counter
        Name of counter containing record_name and line_count counts.
    record_name: str
        Key name for record to tally.
    count: str
        Key name for record of total lines, used as denominator.
    '''
    count = record_counter[line_count]
    try:
        ratio = record_counter[record_name] / count
    except ZeroDivisionError:
        ratio = 0
    return ratio


def precision_recall_fscore(exp, obs, sample_weight=None, exclude=None):
    # precision, recall, fscore, calculated using microaveraging
    if exclude is None:
        exclude = []
    if sample_weight is None:
        sample_weight = [1]*len(exp)
    tp, fp, fn = 0, 0, 0
    for e, o, w in zip(exp, obs, sample_weight):
        if e in exclude:
            continue
        classification = evaluate_classification(o, e)
        if classification == 'match':
            # tp for the true class, the rest are tn
            tp += w
        elif classification == 'underclassification':
            # fn for the true class
            fn += w
            # no fp for the predicted class, because it was right to some level
            # the rest are tn
        else:
            # fp the the predicted class
            fp += w
            # fn for the true class, the rest are tn
            fn += w
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2.*p*r / (p + r)
    return p, r, f


def compute_prf(exp, obs, test_type='cross-validated',
                l_range=range(1, 7), sample_weight=None, exclude=None):
    '''Compute precision, recall, and F-measure using sklearn.
    exp_taxa: list
        Expected observations for each sample (sequence).
    obs_taxa: list
        Predicted observations for each sample (sequence).
    avg: str
        Score averaging method using in sklearn. 'micro', 'weighted', or
        'macro'.
    test_type: str
        'novel-taxa', 'cross-validated', 'cross-validated-trad', or
        'self-validated'.
    l_range: range
        Range of taxonomic levels for ``cross-validated`` and
        ``cross-validated-trad``.
    sample_weight: array-like of shape = [n_samples], optional
        Sample weights.
    '''

    if test_type in ('mock', 'novel-taxa'):
        p, r, f = precision_recall_fscore(
            exp, obs, sample_weight=sample_weight, exclude=exclude)
    elif test_type in ('cross-validated', 'cross-validated-trad',
                       'self-validated'):
        # initialize p/r/f as lists of 0s, representing each taxonomic level.
        p, r, f = [0] * 7, [0] * 7, [0] * 7
        # iterate over multiple taxonomic levels
        for lvl in l_range:
            _obs = extract_taxa_names(obs, level=slice(0, lvl+1))
            _exp = extract_taxa_names(exp, level=slice(0, lvl+1))
            p[lvl], r[lvl], f[lvl] = precision_recall_fscore(
                _exp, _obs, sample_weight=sample_weight, exclude=exclude)
    else:
        raise ValueError(
            'test_type must be "novel-taxa", "cross-validated", '
            '"cross-validated-trad", "self-validated", or "mock".')

    return p, r, f


def load_prf(obs_fp, exp_fp, level=slice(0, 7), sort=True):
    exp_taxa = load_taxa(exp_fp, level=level, sort=sort)
    obs_taxa = load_taxa(obs_fp, level=level, sort=sort)

    # raise error if obs_taxa and exp_taxa are not same length
    if len(obs_taxa) != len(exp_taxa):
        raise RuntimeError(
            'Lengths of expected and observed taxa do not match. '
            'Check inputs: {0}, {1}'.format(obs_fp, exp_fp))

    return exp_taxa, obs_taxa


def runtime_make_test_data(seqs_in, results_dir, sampling_depths):
    '''Repeatedly subsample a fasta sequence file at multiple sequence depths
    to generate query/test data for testing method runtimes.

    seqs_in: path
        fasta format reference sequences.
    results_dir: path
        Output directory.
    sampling_depths: list of integers
        Number of sequences to subsample from seqs.
    '''
    if not exists(results_dir):
        makedirs(results_dir)

    seqs = [seq for seq in io.read(seqs_in, format='fasta')]
    for depth in sampling_depths:
        subset = sample(seqs, depth)
        tmpfile = join(results_dir, str(depth)) + '.fna'
        with open(tmpfile, "w") as output_fasta:
            for s in subset:
                s.write(output_fasta, format='fasta')


def runtime_make_commands(input_dir, results_dir, methods,
                          ref_taxa, sampling_depths, num_iters=1,
                          subsample_ref=True):
    '''Generate list of commands to benchmark method runtimes.

    input_dir: path
        Input directory, containing query/ref sequences.
    results_dir: path
        Output directory.
    methods: dict
        Dictionary of method:parameters pairs in format:
            {'method' : (command-template, method-specific-parameters)}
    ref_taxa: path
        Taxonomy map for ref sequences in tab-separated format:
            seqID   ACGTGTAGTCGATGCTAGCTACG
    sampling_depths: list of integers
        Number of sequences to subsample from seqs.
    num_iters: int
        Number of iterations to perform.
    subsample_ref: bool
        If True (default), ref seqs are subsampled at depths defined in
        sampling_depths, and query seqs default to smallest depth. If false,
        query seqs are subsampled at these depths, and ref defaults to largest
        sampling depth.
    '''

    commands = []
    for iteration in range(num_iters):
        for method, template in methods.items():
            for depth in sampling_depths:
                # default: subsample ref seqs, query = smallest sample
                if subsample_ref is True:
                    q_depth = str(sampling_depths[0])
                    r_depth = str(depth)
                # or subsample query seqs, ref = largest sample
                else:
                    q_depth = str(depth)
                    r_depth = str(sampling_depths[-1])
                query = join(input_dir, q_depth) + '.fna'
                ref = join(input_dir, r_depth) + '.fna'
                command = (template[0].format(results_dir, query, ref,
                                              ref_taxa, method, template[1]),
                           method, q_depth, r_depth, iteration)
                commands.append(command)
    return commands


def clock_runtime(command, results_fp, force=False):
    '''Execute a command and record the runtime.

    command: str
        Command to be executed.
    results_fp: path
        Output file
    force: bool
        Overwrite results? If false, will append to any existing results
    '''
    if force is True:
        remove(results_fp)

    _command, method, q_frac, r_frac, iteration = command
    start = time()
    system(_command)
    end = time()
    results = [method, q_frac, r_frac, iteration, end - start]
    with open(results_fp, 'a') as timeout:
        timeout.write('\t'.join(map(str, results)) + '\n')


from tax_credit.novel_evaluation import (  # noqa: E402
    extract_per_level_accuracy,
    novel_taxa_classification_evaluation,
)
