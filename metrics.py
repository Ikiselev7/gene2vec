from cogent3.phylo import distance
from cogent3.evolve import models
from cogent3 import LoadSeqs

import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import numpy as np
import itertools
import argparse
import pickle
import os

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Don't know what the ssGN and GN models are
# Full list of models: ['JC69', 'K80', 'F81', 'HKY85', 'TN93', 'GTR', 'ssGN', 'GN']
models_names = models.nucleotide_models[:-2]

DATA_FOLDER = '1066_data'

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)


def get_model(model_name):
    """
    Getting appropriate model object from cogent3.evolve.models by name
    :param model_name: model name
    :return: model object
    """
    return models.__dict__[model_name]()


def cos(vectors):
    similarities, distances_cos, = np.zeros([1066, 1066]), np.zeros([1066, 1066])
    for idx_r, row_a in vectors.iterrows():
        for idx_c, row_b in vectors.iterrows():
            if idx_r == idx_c:
                continue
            similarities[idx_r][idx_c] = np.dot(row_a, row_b) / (np.linalg.norm(row_a) * np.linalg.norm(row_b))
            distances_cos[idx_r][idx_c] = 1 - similarities[idx_r][idx_c]

    np.savetxt('{}/similarities.tsv'.format(DATA_FOLDER), similarities, delimiter='\t')
    np.savetxt('{}/distances_cos.tsv'.format(DATA_FOLDER), distances_cos, delimiter='\t')


def euc(vectors):
    distances_euc = np.zeros([1066, 1066])
    for idx_r, row_a in vectors.iterrows():
        for idx_c, row_b in vectors.iterrows():
            if idx_r == idx_c:
                continue
            distances_euc[idx_r][idx_c] = np.sqrt(np.sum((row_a - row_b) ** 2))
    np.savetxt('{}/distances_euc.tsv'.format(DATA_FOLDER), distances_euc, delimiter='\t')


def vectors():
    vectors = pd.read_csv('{}/genes_vec_8grams_1066_genes.tsv'.format(DATA_FOLDER), sep='\t', header=None,
                          index_col=None)

    cos_proc = mp.Process(target=cos, args=(vectors,))
    euc_proc = mp.Process(target=euc, args=(vectors,))
    cos_proc.start()
    euc_proc.start()
    cos_proc.join()
    euc_proc.join()


def convert():
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
    arr = np.full([1066, 1066], 5, dtype=np.float64)

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

    # 10 if lengths are different
    # 100 on main diagonal
    # any other positive value stands for distance
    np.savetxt('{}/sparse/{}.tsv'.format(DATA_FOLDER, model_name), arr, delimiter='\t')


def calc_dist_dense(align, model_name):
    al = LoadSeqs('{}/all_genes_{}_test.fasta'.format(DATA_FOLDER, align))

    d = distance.EstimateDistances(al, submodel=get_model(model_name))
    d.run()

    if not os.path.exists('{}/dense/{}'.format(DATA_FOLDER, align)):
        os.mkdir('{}/dense/{}'.format(DATA_FOLDER, align))

    with open('{}/dense/{}/{}_tmp.txt'.format(DATA_FOLDER, align, model_name), 'w') as f:
        f.write(d.__str__())

    tmp = pd.read_csv('{}/dense/{}/{}_tmp.txt'.format(DATA_FOLDER, align, model_name), engine='python',
                      delim_whitespace=True, skiprows=3, header=None,
                      index_col=None, skipfooter=1).iloc[:, 1:]
    tmp.to_csv('{}/dense/{}/{}.tsv'.format(DATA_FOLDER, align, model_name), sep='\t', header=None, index=False)


def main(args):
    # Calculate cosine similarity, cosine and Euclidean distances
    if args.vectors:
        vectors()

    # Convert all_genes.csv to all_genes.fasta file
    if args.convert:
        records = convert()

    # Load all_genes.fasta
    else:
        records = list(SeqIO.parse('{}/all_genes.fasta'.format(DATA_FOLDER), 'fasta'))

    # Calculate distances spare
    if args.sparse:
        pool = mp.Pool(mp.cpu_count())
        pool.starmap_async(calc_dist_sparse,
                           [(model_name, records) for model_name in models_names]).get(99999)

    # Calculate distances dense
    if args.dense:
        pool = mp.Pool(mp.cpu_count())

        # alignments = ['muscle', 'mafft', 'clustal']
        alignments = ['mafft']
        combs = list(itertools.product(alignments, models_names))
        pool.starmap_async(calc_dist_dense, [(align, model_name) for align, model_name in combs]).get(99999)

    # Calculate correlation
    if args.correlation:
        for model_name in models_names:
            print(np.corrcoef(pd.read_csv('{}/distances_cos.tsv'
                                          .format(DATA_FOLDER), sep='\t', header=None, index_col=None).values.flatten(),
                              [0 if x == '*' else float(x) for x in pd.read_csv('{}/dense/mafft/{}.tsv'
                                                                                .format(DATA_FOLDER, model_name),
                                                                                sep='\t', header=None,
                                                                                index_col=None).values.flatten()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--convert', action='store_true', help='convert all_genes.csv -> all_genes.fasta')
    parser.add_argument('-s', '--sparse', action='store_true', help='calculate metrics for sparse matrices')
    parser.add_argument('-d', '--dense', action='store_true', help='calculate metrics for dense matrices')
    parser.add_argument('-v', '--vectors', action='store_true', help='calculate cosine similarity, '
                                                                     'euclidean and cosine distances')
    parser.add_argument('-r', '--correlation', action='store_true', help='calculate correlation between: cosine '
                                                                         'similarity, euclidean and cosine distances '
                                                                         'and models of DNA evolution metrics')

    args = parser.parse_args()
    main(args)
