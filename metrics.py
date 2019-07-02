from cogent3.phylo import distance
from cogent3.evolve import models
from cogent3 import LoadSeqs

import matplotlib.pyplot as plt
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


def get_model(model_name):
    """
    Getting appropriate model object from cogent3.evolve.models by name
    :param model_name: model name
    :return: model object
    """
    return models.__dict__[model_name]()


def cos(vectors):
    similarities, distances_cos, = np.zeros([DIM, DIM]), np.zeros([DIM, DIM])
    for idx_r, row_a in vectors.iterrows():
        for idx_c, row_b in vectors.iterrows():
            if idx_r == idx_c:
                continue
            similarities[idx_r][idx_c] = np.dot(row_a, row_b) / (np.linalg.norm(row_a) * np.linalg.norm(row_b))
            distances_cos[idx_r][idx_c] = 1 - similarities[idx_r][idx_c]

    np.savetxt('{}/similarities.tsv'.format(DATA_FOLDER), similarities, delimiter='\t')
    np.savetxt('{}/distances_cos.tsv'.format(DATA_FOLDER), distances_cos, delimiter='\t')


def euc(vectors):
    distances_euc = np.zeros([DIM, DIM])
    for idx_r, row_a in vectors.iterrows():
        for idx_c, row_b in vectors.iterrows():
            if idx_r == idx_c:
                continue
            distances_euc[idx_r][idx_c] = np.sqrt(np.sum((row_a - row_b) ** 2))
    np.savetxt('{}/distances_euc.tsv'.format(DATA_FOLDER), distances_euc, delimiter='\t')


def vectors():
    """
    Calculate cosine similarities, cosine and Euclidean distances using 2 processes
    """
    vectors = pd.read_csv('{}/genes_vec_8grams_1066_genes.tsv'.format(DATA_FOLDER), sep='\t', header=None,
                          index_col=None)

    cos_proc = mp.Process(target=cos, args=(vectors,))
    euc_proc = mp.Process(target=euc, args=(vectors,))
    cos_proc.start()
    euc_proc.start()
    cos_proc.join()
    euc_proc.join()


def format():
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


def calc_dist(seq_1, seq_2, model):
    al = LoadSeqs(data=[['x', seq_1], ['y', seq_2]], array_align=False)
    d = distance.EstimateDistances(al, submodel=model)
    d.run()
    _, value = d._param_ests.popitem()
    return value['length']


def calc_dist_sparse(model_name, records):
    arr = np.full([DIM, DIM], 5, dtype=np.float64)

    # 10 if lengths are different
    # 100 on main diagonal
    # any other positive value stands for distance
    for row_n, record in enumerate(records):
        for col_n in range(row_n + 1, len(records)):
            if len(record.seq) == len(records[col_n].seq):
                if row_n != col_n:
                    arr[row_n][col_n] = calc_dist(record, records[col_n], get_model(model_name))
                else:
                    arr[row_n][col_n] = 100

    # Symmetric matrix
    arr += arr.T

    plt.imshow(arr, cmap='plasma')
    plt.savefig('{}/sparse/{}.png'.format(DATA_FOLDER, model_name), dpi=2000)

    np.savetxt('{}/sparse/{}.tsv'.format(DATA_FOLDER, model_name), arr, delimiter='\t')


def dist_sparse_mp(records):
    if not os.path.exists('{}/sparse'.format(DATA_FOLDER)):
        os.mkdir('{}/sparse'.format(DATA_FOLDER))

    pool = mp.Pool(mp.cpu_count())
    pool.starmap_async(calc_dist_sparse,
                       [(model_name, records) for model_name in selected_models]).get(99999)


def dist_dense(align, model_name):
    al = LoadSeqs('{}/all_genes_{}.fasta'.format(DATA_FOLDER, align))

    d = distance.EstimateDistances(al, submodel=get_model(model_name))
    d.run()

    if not os.path.exists('{}/dense/{}'.format(DATA_FOLDER, align)):
        os.mkdir('{}/dense/{}'.format(DATA_FOLDER, align))

    with open('{}/dense/{}/{}_tmp.txt'.format(DATA_FOLDER, align, model_name), 'w') as f:
        f.write(d.__str__())
    tmp = pd.read_csv('{}/dense/{}/{}_tmp.txt'.format(DATA_FOLDER, align, model_name), engine='python',
                      delim_whitespace=True, skiprows=3, header=None,
                      index_col=None, skipfooter=1).iloc[:, 1:]
    os.remove('{}/dense/{}/{}_tmp.txt'.format(DATA_FOLDER, align, model_name))

    tmp.to_csv('{}/dense/{}/{}.tsv'.format(DATA_FOLDER, align, model_name), sep='\t', header=None, index=False)


def dist_dense_mp():
    if not os.path.exists('{}/dense'.format(DATA_FOLDER)):
        os.mkdir('{}/dense'.format(DATA_FOLDER))

    pool = mp.Pool(mp.cpu_count())
    combs = list(itertools.product(alignments, selected_models))
    pool.starmap_async(dist_dense, [(align, model_name) for align, model_name in combs]).get(99999)


def correlation():
    for align in alignments:
        for model_name in selected_models:
            with open('correlation_{}.txt'.format(model_name), 'w') as out_file:
                # Get rid of * symbol after pycogent3's EstimateDistances() method call on main diagonal
                # using list comprehensions and calculate correlation
                evolution_distances = [0 if x == '*' else float(x) for x in pd.read_csv('{}/dense/{}/{}.tsv'
                                                                  .format(DATA_FOLDER,
                                                                          align,
                                                                          model_name),
                                                                  sep='\t', header=None,
                                                                  index_col=None).values.flatten()]

                data_frame_distances_cos = pd.read_csv('{}/distances_cos.tsv'.format(DATA_FOLDER), sep='\t', header=None, index_col=None)
                data_frame_distances_euc = pd.read_csv('{}/distances_euc.tsv'.format(DATA_FOLDER), sep='\t', header=None, index_col=None)
                similarities             = pd.read_csv('{}/similarities.tsv'.format(DATA_FOLDER), sep='\t', header=None, index_col=None)

                correlations_cos = np.corrcoef(data_frame_distances_cos.values.flatten(),   evolution_distances)
                correlations_euc = np.corrcoef(data_frame_distances_euc.values.flatten(),   evolution_distances)
                correlations_sim = np.corrcoef(similarities.values.flatten(),               evolution_distances)

                out_file.write(str(correlations_cos))
                out_file.write(str(correlations_euc))
                out_file.write(str(correlations_sim))


def main(args):
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    if args.vectors:
        vectors()

    if args.format:
        records = format()
    else:
        # Load all_genes.fasta
        records = list(SeqIO.parse('{}/all_genes.fasta'.format(DATA_FOLDER), 'fasta'))

    if args.sparse:
        dist_sparse_mp(records)

    if args.dense:
        dist_dense_mp()

    if args.correlation:
        correlation()


def parse_args():
    global MODELS_NAMES
    global selected_models

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--format', action='store_true', help='convert all_genes.csv -> all_genes.fasta')
    parser.add_argument('-s', '--sparse', action='store_true', help='calculate metrics for sparse matrices')
    parser.add_argument('-d', '--dense', action='store_true', help='calculate metrics for dense matrices')
    parser.add_argument('-v', '--vectors', action='store_true',
                        help='calculate cosine similarity, euclidean and cosine distances')
    parser.add_argument('-c', '--correlation', action='store_true',
                        help='calculate correlation between: cosine [similarity, euclidean and cosine distances] '
                             'and [models of DNA evolution metrics]')
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
    main(parse_args())
