from cogent3.phylo import distance
from cogent3.evolve import models
from cogent3 import LoadSeqs

import multiprocessing as mp
import pandas as pd
import numpy as np
import itertools
import argparse
import os

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Don't know what the ssGN and GN models are
# Full list of models: ['JC69', 'K80', 'F81', 'HKY85', 'TN93', 'GTR', 'ssGN', 'GN']
MODELS_NAMES = models.nucleotide_models[:-2]
selected_models = []

# Muscle and clustal alignments don't keep order
# alignments = ['muscle', 'mafft', 'clustal']
alignments = ['mafft']

DIM = 1066
DATA_FOLDER = '1066_data'


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_model(model_name):
    """
    Getting appropriate model object from cogent3.evolve.models by name
    :param model_name: model name
    :return: model object
    """
    return models.__dict__[model_name]()


def csv_to_fasta():
    """
    Format all_genes.csv to all_genes.fasta file
    """
    all_genes_raw = pd.read_csv('{}/all_genes.csv'.format(DATA_FOLDER), sep='\t', header=None, index_col=None)
    records = []

    for index, row in all_genes_raw.iterrows():
        row_data = row.tolist()[0].split(',')
        seq_simple = Seq(row_data[2])
        info = '_'.join(row_data[0:2])
        record = SeqRecord(seq_simple, name=info, description=info, id=info)
        records.append(record)

    SeqIO.write(records, '{}/all_genes.fasta'.format(DATA_FOLDER), 'fasta')
    return records


def split_alignments():
    for align in alignments:
        records = list(SeqIO.parse('{}/all_genes_{}.fasta'.format(DATA_FOLDER, align), 'fasta'))
        records_chunks = list(divide_chunks(records, 82))

        create_folder('{}/chunks'.format(DATA_FOLDER))

        for idx_r in range(13):
            for idx_c in range(idx_r, 13):
                file_name = '{}/chunks/all_genes_{}_{}_{}.fasta'.format(DATA_FOLDER, align, idx_r, idx_c)
                if idx_r != idx_c:
                    SeqIO.write(records_chunks[idx_r] + records_chunks[idx_c], file_name, 'fasta')
                else:
                    SeqIO.write(records_chunks[idx_r], file_name, 'fasta')


def cosine_similarities(vectors):
    similarities = np.zeros([DIM, DIM])
    for idx_r, row_a in vectors.iterrows():
        for idx_c, row_b in vectors.iterrows():
            if idx_r == idx_c:
                continue
            similarities[idx_r][idx_c] = np.dot(row_a, row_b) / (np.linalg.norm(row_a) * np.linalg.norm(row_b))

    np.savetxt('{}/cosine_similarities.tsv'.format(DATA_FOLDER), similarities, delimiter='\t')


def euclidean_distances(vectors):
    distances_euc = np.zeros([DIM, DIM])
    for idx_r, row_a in vectors.iterrows():
        for idx_c, row_b in vectors.iterrows():
            if idx_r == idx_c:
                continue
            distances_euc[idx_r][idx_c] = np.sqrt(np.sum((row_a - row_b) ** 2))
    np.savetxt('{}/distances_euc.tsv'.format(DATA_FOLDER), distances_euc, delimiter='\t')


def cosine_and_euclidean_vectors_calculation():
    """
    Calculate cosine similarities and Euclidean distances using 2 processes
    """
    vectors = pd.read_csv('{}/genes_vec_8grams_1066_genes.tsv'.format(DATA_FOLDER), sep='\t', header=None,
                          index_col=None)

    cos_proc = mp.Process(target=cosine_similarities, args=(vectors,))
    euc_proc = mp.Process(target=euclidean_distances, args=(vectors,))
    cos_proc.start()
    euc_proc.start()
    cos_proc.join()
    euc_proc.join()


def distances_chunks(align, model_name, idx_r, idx_c):
    al = LoadSeqs('{}/chunks/all_genes_{}_{}_{}.fasta'.format(DATA_FOLDER, align, idx_r, idx_c))

    d = distance.EstimateDistances(al, submodel=get_model(model_name))
    d.run()

    create_folder('{}/distances/{}'.format(DATA_FOLDER, align))

    # Format output from EstimateDistances() using pandas with temporary file
    temp_file_name = '{}/distances/{}/{}_{}_{}_tmp.txt'.format(DATA_FOLDER, align, model_name, idx_r, idx_c)
    with open(temp_file_name, 'w') as f:
        f.write(d.__str__())
    tmp = pd.read_csv(temp_file_name, engine='python',
                      delim_whitespace=True, skiprows=3, header=None,
                      index_col=None, skipfooter=1).iloc[:, 1:]
    os.remove(temp_file_name)

    tmp.to_csv('{}/distances/{}/{}_{}_{}.tsv'.format(DATA_FOLDER, align, model_name, idx_r, idx_c),
               sep='\t', header=None, index=False)


def distances_chunks_concatenate():
    pass


def dist_mp():
    create_folder('{}/distances'.format(DATA_FOLDER))

    pool = mp.Pool(mp.cpu_count())

    # combs = list(itertools.product(alignments, selected_models, range(3), range(3)))
    combs = []
    for align in alignments:
        for model in selected_models:
            for idx_r in range(2):
                for idx_c in range(idx_r, 2):
                    combs.append((align, model, idx_r, idx_c, ))

    pool.starmap_async(distances_chunks,
                       [(align, model_name, idx_r, idx_c) for align, model_name, idx_r, idx_c in combs]).get(99999)
    distances_chunks_concatenate()


def correlation():
    for align in alignments:
        for model_name in selected_models:
            with open('correlation_{}.txt'.format(model_name), 'w') as out_file:
                # Get rid of * symbol after pycogent3's EstimateDistances() method call on main diagonal
                # using list comprehensions and calculate correlation
                evolution_distances = [0 if x == '*' else float(x) for x in pd.read_csv('{}/distances/{}/{}.tsv'
                                                                  .format(DATA_FOLDER,
                                                                          align,
                                                                          model_name),
                                                                  sep='\t', header=None,
                                                                  index_col=None).values.flatten()]

                data_frame_distances_euc = pd.read_csv('{}/distances_euc.tsv'.format(DATA_FOLDER),
                                                       sep='\t', header=None, index_col=None)
                similarities             = pd.read_csv('{}/similarities.tsv'.format(DATA_FOLDER),
                                                       sep='\t', header=None, index_col=None)

                correlations_euc = np.corrcoef(data_frame_distances_euc.values.flatten(),   evolution_distances)
                correlations_sim = np.corrcoef(similarities.values.flatten(),               evolution_distances)

                out_file.write(str(correlations_euc))
                out_file.write(str(correlations_sim))


def main(args):
    create_folder(DATA_FOLDER)

    if args.vectors:
        cosine_and_euclidean_vectors_calculation()

    if args.format:
        csv_to_fasta()

    if args.split:
        split_alignments()

    if args.distances:
        dist_mp()

    if args.correlation:
        correlation()


def parse_args():
    global MODELS_NAMES
    global selected_models

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--format', action='store_true', help='convert all_genes.csv -> all_genes.fasta')
    parser.add_argument('-s', '--split', action='store_true', help='split alignments in to chunks')
    parser.add_argument('-d', '--distances', action='store_true', help='calculate distances using pycogent3')
    parser.add_argument('-v', '--vectors', action='store_true',
                        help='calculate cosine similarity, and euclidean distances')
    parser.add_argument('-c', '--correlation', action='store_true',
                        help='calculate correlation between:\n[cosine similarity, euclidean distances] and\n'
                             '[models of DNA evolution metrics]')
    parser.add_argument('-m', '--models', type=str, nargs='+',
                        help='list your models;\n'
                             'JC69:     Jukes and Cantor 1969;\n'
                             'K80:      Kimura 1980;\n'
                             'F81:      Felsenstein 1981;\n'
                             'HKY85:    Hasegawa, Kishino and Yano 1985;\n'
                             'TN93:     Tamura and Nei 1993,\n'
                             'GTR:      Tavaré 1986')

    args = parser.parse_args()

    if args.models:
        for model in args.models[0].split(','):
            if model.upper() in MODELS_NAMES:
                selected_models.append(model)
    else:
        selected_models = MODELS_NAMES.copy()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
