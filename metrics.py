from cogent3.phylo import distance
from cogent3.evolve import models
from cogent3 import LoadSeqs

import multiprocessing as mp
import pandas as pd
import numpy as np
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
CHUNK_SIZE = 82
CHUNK_NUM = DIM // CHUNK_SIZE


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def read_tsv(path):
    return pd.read_csv(path, sep='\t', header=None, index_col=None)


def save_tsv(path, data):
    np.savetxt(path, data, delimiter='\t')


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
    all_genes_raw = read_tsv('{}/all_genes.csv'.format(DATA_FOLDER))
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
        records_chunks = list(divide_chunks(records, CHUNK_SIZE))

        create_folder('{}/chunks'.format(DATA_FOLDER))

        for idx_r in range(CHUNK_NUM):
            for idx_c in range(idx_r, CHUNK_NUM):
                file_name = '{}/chunks/all_genes_{}_{}_{}.fasta'.format(DATA_FOLDER, align, idx_r, idx_c)
                if idx_r != idx_c:
                    SeqIO.write(records_chunks[idx_r] + records_chunks[idx_c], file_name, 'fasta')
                else:
                    SeqIO.write(records_chunks[idx_r], file_name, 'fasta')


def cosine_similarities(vectors):
    similarities = np.full([DIM, DIM], fill_value=1)

    for idx_r, row_a in vectors.iterrows():
        for idx_c, row_b in vectors.iterrows():
            if idx_r == idx_c:
                continue

            similarities[idx_r][idx_c] = np.dot(row_a, row_b) / (np.linalg.norm(row_a) * np.linalg.norm(row_b))

    save_tsv('{}/cosine_similarities.tsv'.format(DATA_FOLDER), similarities)


def euclidean_distances(vectors):
    distances_euc = np.zeros([DIM, DIM])

    for idx_r, row_a in vectors.iterrows():
        for idx_c, row_b in vectors.iterrows():
            if idx_r == idx_c:
                continue

            distances_euc[idx_r][idx_c] = np.sqrt(np.sum((row_a - row_b) ** 2))

    save_tsv('{}/distances_euc.tsv'.format(DATA_FOLDER), distances_euc)


def cosine_and_euclidean_vectors_calculation():
    """
    Calculate cosine similarities and Euclidean distances using 2 processes
    """
    vectors = read_tsv('{}/genes_vec_8grams_1066_genes.tsv'.format(DATA_FOLDER))

    cos_proc = mp.Process(target=cosine_similarities, args=(vectors,))
    euc_proc = mp.Process(target=euclidean_distances, args=(vectors,))
    cos_proc.start()
    euc_proc.start()
    cos_proc.join()
    euc_proc.join()


def format_array_after_dist_calc(path):
    """
    Get rid of * symbol on main diagonal after distance calculation and cast to float
    """
    tmp_array = read_tsv(path)
    final_array = np.zeros(shape=tmp_array.shape)

    for r in range(tmp_array.shape[0]):
        for c in range(tmp_array.shape[0]):
            if tmp_array[r][c] == '*':
                final_array[r][c] = 0
            else:
                final_array[r][c] = tmp_array[r][c]

    return final_array


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
    for align in alignments:
        for model in selected_models:
            path = '{}/distances/{}/'.format(DATA_FOLDER, align)
            files = os.listdir(path)

            def get_r_index(file_name):
                return int(file_name.split('_')[2].split('.')[0])

            def get_c_index(file_name):
                return int(file_name.split('_')[1])

            files = sorted(files, key=get_r_index)
            files = sorted(files, key=get_c_index)

            final_array = np.ndarray((DIM, DIM))

            # Don't know how to write it simple using python\numpy methods
            # Here we gonna C style
            for file in files:

                r = get_r_index(file)
                c = get_c_index(file)

                tmp_array = format_array_after_dist_calc('{}/distances/{}/{}'
                                                         .format(DATA_FOLDER, align, file))

                # If chunk on main diagonal just copy values
                if r == c:

                    for r_idx in range(CHUNK_SIZE):
                        for c_idx in range(CHUNK_SIZE):
                            final_array[r_idx + CHUNK_SIZE * r][c_idx + CHUNK_SIZE * c] = tmp_array[r_idx][c_idx]

                # 1) Extract values from 1 quarter
                # 2) Place them using files idx
                # 3) Transpose matrix\ndarray | swap r and c indexes
                # 4) Place them using reverse idx
                else:
                    # 1st step
                    values = np.ndarray([CHUNK_SIZE, CHUNK_SIZE])

                    for r_idx in range(CHUNK_SIZE):
                        for c_idx in range(CHUNK_SIZE):
                            values[r_idx][c_idx] = tmp_array[r_idx][c_idx + CHUNK_SIZE]

                    # 2nd step
                    for r_idx in range(CHUNK_SIZE):
                        for c_idx in range(CHUNK_SIZE):
                            final_array[r_idx + CHUNK_SIZE * r][c_idx + CHUNK_SIZE * c] = values[r_idx][c_idx]

                    # 3d step
                    values = values.T
                    r, c = c, r

                    # 4th step
                    for r_idx in range(CHUNK_SIZE):
                        for c_idx in range(CHUNK_SIZE):
                            final_array[r_idx + CHUNK_SIZE * r][c_idx + CHUNK_SIZE * c] = values[r_idx][c_idx]

            save_tsv('{}/distances/{}_{}_concatenated.tsv'.format(DATA_FOLDER, align, model), final_array)


def dist_mp():
    create_folder('{}/distances'.format(DATA_FOLDER))

    pool = mp.Pool(mp.cpu_count())

    # Itertools?
    combs = []
    for align in alignments:
        for model in selected_models:
            for idx_r in range(CHUNK_NUM):
                for idx_c in range(idx_r, CHUNK_NUM):
                    combs.append((align, model, idx_r, idx_c,))

    pool.starmap_async(distances_chunks,
                       [(align, model_name, idx_r, idx_c) for align, model_name, idx_r, idx_c in combs]).get(99999)

    distances_chunks_concatenate()


def correlation():
    """
    Have to transform to 1-dim array/list to calculate correlation using numpy corrcoef method
    """
    for align in alignments:
        for model_name in selected_models:
            with open('{}/correlation_{}.txt'.format(DATA_FOLDER, model_name), 'w') as out_file:
                evolution_distances = read_tsv('{}/distances/{}_{}_concatenated.tsv'
                                               .format(DATA_FOLDER, align, model_name)).values.flatten()

                cosine_sim = read_tsv('{}/cosine_similarities.tsv'.format(DATA_FOLDER)).values.flatten()
                dist_euc = read_tsv('{}/distances_euc.tsv'.format(DATA_FOLDER)).values.flatten()

                correlations_sim = np.corrcoef(cosine_sim, evolution_distances)
                correlations_euc = np.corrcoef(dist_euc, evolution_distances)

                out_file.write('cosine_similarities/evolution_distances\n' +
                               str(correlations_sim) + '\n\n\n' +
                               'distances_euc/evolution_distances\n' +
                               str(correlations_euc))


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

    return args


if __name__ == '__main__':
    main(parse_args())
