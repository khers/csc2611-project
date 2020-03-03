#!/usr/bin/python3

from gensim.models import KeyedVectors
import pathlib
import argparse
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy
import sys
from multiprocessing.pool import Pool

def load_vocab(models, index, windowing):
    vocab = set()
    with open('{}/vocab_{}_{}.txt'.format(models, windowing, index), 'r') as fd:
        for line in fd:
            vocab.add(line.split()[0])
    return vocab


def measure_differences(args, start_index, end_index):
    start = KeyedVectors.load_word2vec_format('{}/vectors_{}_{}.word2vec'.format(args.models, args.windowing, start_index))
    end = KeyedVectors.load_word2vec_format('{}/vectors_{}_{}.word2vec'.format(args.models, args.windowing, end_index))

    # Collect new and retired words by comparing the two vocabularies
    start_set = load_vocab(args.models, start_index, args.windowing)
    end_set = load_vocab(args.models, end_index, args.windowing)
    new_words = end_set - start_set
    retired_words = start_set - end_set

    # To measure change in words that appear in both models we need to rotate the starting model into the
    # with the orthogonal Procustes solution
    R,_ = orthogonal_procrustes(start.vectors, end.vectors, check_finite=False)
    start.vectors = start.vectors @ R

    diff = []
    for entry in start_set.intersection(end_set):
        if entry not in start.vocab:
            print('{} missing from {} {} model, skipping entry'.format(entry, start_index, args.windowing))
            continue
        if entry not in end.vocab:
            print('{} missing from {} {} model, skipping entry'.format(entry, end_index, args.windowing))
            continue
        diff.append((entry, end.distances(start.get_vector(entry), other_words=[entry])[0]))
    diff.sort(key=lambda t : t[1])
    return diff,new_words,retired_words


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models for semantic change')
    parser.add_argument('-m', '--models', default='./models', help='Path to directory containing models')
    parser.add_argument('-w', '--windowing', choices=['middle', 'leading'], default='middle', help='The windowing method to test')
    parser.add_argument('-s', '--start', type=int, default=55, help='Congress to start on')
    parser.add_argument('-e', '--end', type=int, default=110, help='Congress to end on')
    parser.add_argument('-o', '--output', default='results', help='Where to store the output JSON')

    args = parser.parse_args(sys.argv[1:])

    inputs = [(args, index, index + 1) for index in range(args.start, args.end)]

    with Pool(processes=12) as p:
        key = args.start + 1
        results = {}
        for entry in p.starmap(measure_differences, inputs):
            results[key] = entry
            key += 1

    with open('./{}_{}.json'.format(args.output, args.windowing), 'w') as fd:
        json.dump(results, fd, cls=MyEncoder)
