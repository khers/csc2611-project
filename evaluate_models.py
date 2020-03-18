#!/usr/bin/python3

# pylint: disable=method-hidden

from gensim.models import KeyedVectors
import pathlib
import argparse
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import json
import numpy
import sys
from multiprocessing.pool import Pool
import pandas as pd


def measure_windowed_differences(args, start_index, end_index):
    start = KeyedVectors.load_word2vec_format('{}/vectors_{}_{}.word2vec'.format(args.models, args.windowing, start_index))
    end = KeyedVectors.load_word2vec_format('{}/vectors_{}_{}.word2vec'.format(args.models, args.windowing, end_index))

    # Collect new and retired words by comparing the two vocabularies
    start_set = set(start.vocab)
    end_set = set(end.vocab)
    overlap = list(start_set.intersection(end_set))
    new_words = end_set - start_set
    retired_words = start_set - end_set

    # Collect the vectors for words that exists in both so we can build
    # truncated pair of models
    s_v = list()
    e_v = list()
    for word in overlap:
        s_v.append(start.get_vector(word))
        e_v.append(end.get_vector(word))

    filtered_start = pd.DataFrame(s_v, index=overlap)
    filtered_end = pd.DataFrame(e_v, index=overlap)

    # To measure change in words that appear in both models we need to rotate the starting model into the
    # with the orthogonal Procustes solution
    R,_ = orthogonal_procrustes(filtered_start, filtered_end, check_finite=False)
    filtered_start = filtered_start @ R

    cos_diff = []
    euc_diff = []
    for entry in overlap:
        in1 = filtered_start.loc[entry].to_numpy().reshape(1,300)
        in2 = filtered_end.loc[entry].to_numpy().reshape(1,300)
        cos_diff.append((entry, cosine_similarity(in1, in2)))
        euc_diff.append((entry, euclidean(filtered_start.loc[entry], filtered_end.loc[entry])))
    cos_diff.sort(key=lambda t : t[1])
    euc_diff.sort(key=lambda t : t[1], reverse=True)
    ret = {}
    ret['cosine_differences'] = cos_diff
    ret['euclidean_differences'] = euc_diff
    ret['new_words'] = new_words
    ret['retired_words'] = retired_words
    return ret


def evaluate_windowed_models(args):
    inputs = [(args, index, index + 1) for index in range(args.start, args.end)]
    with Pool() as p:
        key = args.start + 1
        results = {}
        for entry in p.starmap(measure_windowed_differences, inputs):
            results[key] = entry
            key += 1
    return results


def load_vocab(path, windowing, index):
    vocab = list()
    with open('{}/vocab_{}_{}.txt'.format(path, windowing, index), 'r') as fd:
        for line in fd:
            vocab.append(line.split()[0])
    return vocab

single_model = None


def measure_differences(args, index):
    old = load_vocab(args.models, args.windowing, index)
    new = load_vocab(args.models, args.windowing, index + 1)

    retired = set(old) - set(new)
    new_words = set(new) - set(old)
    overlap = [ w for w in old if w in new ]
    cos_diff = []
    euc_diff = []
    for word in overlap:
        v1 = single_model['{}_{}'.format(word, index)]
        v2 = single_model['{}_{}'.format(word, index + 1)]
        cos_diff.append((word, cosine_similarity(v1.reshape(1,300), v2.reshape(1,300))))
        euc_diff.append((word, euclidean(v1, v2)))
    cos_diff.sort(key=lambda t : t[1])
    euc_diff.sort(key=lambda t : t[1], reverse=True)
    ret = {}
    ret['cosine_differences'] = cos_diff
    ret['euclidean_differences'] = euc_diff
    ret['new_words'] = new_words
    ret['retired_words'] = retired
    return ret


def evaluate_single_model(args):
    global single_model
    single_model = KeyedVectors.load_word2vec_format('{}/vectors_none_0.word2vec'.format(args.models))
    input = [(args, index) for index in range(args.start, args.end)]
    results = {}
    with Pool() as p:
        key = args.start + 1
        for entry in p.starmap(measure_differences, input):
            results[key] = entry
            key += 1
    return results


def dump_results(results, filename, windowing):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, numpy.integer):
                return int(obj)
            elif isinstance(obj, numpy.floating):
                return float(obj)
            elif isinstance(obj, numpy.ndarray):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            else:
                return super(MyEncoder, self).default(obj)

    with open('./{}_{}.json'.format(filename, windowing), 'w') as fd:
        json.dump(results, fd, cls=MyEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models for semantic change')
    parser.add_argument('-m', '--models', default='./models', help='Path to directory containing models')
    parser.add_argument('-w', '--windowing', choices=['middle', 'leading', 'none'], default='middle', help='The windowing method to test')
    parser.add_argument('-s', '--start', type=int, default=55, help='Congress to start on')
    parser.add_argument('-e', '--end', type=int, default=110, help='Congress to end on')
    parser.add_argument('-o', '--output', default='results', help='Where to store the output JSON')

    args = parser.parse_args(sys.argv[1:])

    if args.windowing == 'none':
        results = evaluate_single_model(args)
    else:
        results = evaluate_windowed_models(args)

    dump_results(results, args.output, args.windowing)