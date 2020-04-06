#!/usr/bin/python3

# Copyright 2020 Eric Munson

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import multiprocessing
import argparse
from subprocess import Popen, PIPE
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
from nltk.collocations import BigramAssocMeasures,BigramCollocationFinder
import pandas as pd
from nltk.lm import Vocabulary
from utilities import load_target_words


def read_windowed_input_data(windowing, target, training):
    sources = []
    if windowing == 'middle':
        sources.append('{}/prepared_{:03d}.txt'.format(training, target - 1))
        sources.append('{}/prepared_{:03d}.txt'.format(training, target))
        sources.append('{}/prepared_{:03d}.txt'.format(training, target + 1))
    elif windowing == 'leading':
        sources.append('{}/prepared_{:03d}.txt'.format(training, target - 2))
        sources.append('{}/prepared_{:03d}.txt'.format(training, target - 1))
        sources.append('{}/prepared_{:03d}.txt'.format(training, target))
    elif windowing == 'single':
        sources.append('{}/prepared_{:03d}.txt'.format(training, target))
    input = []
    for entry in sources:
        with open(entry, 'r') as fd:
            for line in fd:
                input.append(line)
    return ' '.join(input)


def read_and_tag_file(arg):
    file_path,target_session,target_words = arg
    input = []
    with open(file_path, 'r') as fd:
        for line in fd:
            input_words = [entry for entry in line.split() if entry and not entry.isspace()]
            input_words = [entry if entry not in target_words else "{}_{}".format(entry, target_session) for entry in input_words]
            input.append((' '.join(input_words)).strip())
    return ' '.join(input)


def read_tagged_input(training, start, end, targets_path):
    target_words = load_target_words(targets_path)
    input = []
    for target in range(start, end + 1):
        input.append(('{}/prepared_{:03d}.txt'.format(training, target), target, target_words))
    with multiprocessing.Pool() as p:
        output = p.map(read_and_tag_file, input)
    return ' '.join(output)


def run_vocab(input, args, target):
    exe = "{}/vocab_count".format(args.glove)
    cmd = [exe, '-min-count', str(args.min_count), '-verbose', '2']
    p = Popen(cmd, bufsize=268435456, stdin=PIPE, stdout=PIPE, universal_newlines=True)
    out,_ = p.communicate(input=input)
    vocab_file = '{}/vocab_{}_{}.txt'.format(args.output, args.windowing, target)
    with open(vocab_file, 'w') as fd:
        fd.write(out)
    return vocab_file


def run_cooccur_and_shuffle(input, args, vocab_file, target):
    exe = '{}/cooccur'.format(args.glove)
    cmd = [exe, '-memory', str(args.memory), '-verbose', '2', '-vocab-file', vocab_file, '-window-size', '15']
    p = Popen(cmd, bufsize=268435456, stdin=PIPE, stdout=PIPE)
    coocur,_ = p.communicate(input=bytes(input, 'utf-8'))

    exe = '{}/shuffle'.format(args.glove)
    cmd = [exe, '-memory', str(args.memory), '-verbose', '2']
    p = Popen(cmd, bufsize=268435456, stdin=PIPE, stdout=PIPE)
    out,_ = p.communicate(input=coocur)
    cooccur_shuff_file = '{}/cooccurrence_{}_{}.shuf.bin'.format(args.output, args.windowing, target)
    with open(cooccur_shuff_file, 'wb') as fd:
        fd.write(out)
    return cooccur_shuff_file


def run_glove(args, vocab_file, cooccur_shuff_file, target):
    numCpus = multiprocessing.cpu_count()
    exe = '{}/glove'.format(args.glove)
    cmd = [exe, '-save-file', '{}/vectors_{}_{:03d}'.format(args.output, args.windowing, target), '-threads',
    str(numCpus), '-input-file', cooccur_shuff_file, '-x-max', '10', '-iter', str(args.max_iterations),
    '-vector-size', str(args.vector_size), '-binary', '2', '-vocab-file', vocab_file, '-verbose', '2']
    p = Popen(cmd)
    p.communicate()

    os.remove(cooccur_shuff_file)
    os.remove(vocab_file)
    os.remove('{}/vectors_{}_{:03d}.bin'.format(args.output, args.windowing, target))

    glove2word2vec('{}/vectors_{}_{:03d}.txt'.format(args.output, args.windowing, target),
                '{}/vectors_{}_{}.word2vec'.format(args.output, args.windowing, target))
    os.remove('{}/vectors_{}_{:03d}.txt'.format(args.output, args.windowing, target))


def glove_training(args):
    if args.windowing == 'none':
        # First run vocab extraction per congress to be used in model evaluation
        for target in range(args.start, args.end + 1):
            with open('{}/prepared_{:03d}.txt'.format(args.training, target), 'r') as fd:
                run_vocab(fd.read(), args, target)
        input = read_tagged_input(args.training, args.start, args.end, args.target_words)
        vocab_file = run_vocab(input, args, 0)

        cooccur_shuff_file = run_cooccur_and_shuffle(input, args, vocab_file, 0)

        run_glove(args, vocab_file, cooccur_shuff_file, 0)

    else:
        for target in range(args.start, args.end + 1):
            input = read_windowed_input_data(args.windowing, target, args.training)

            vocab_file = run_vocab(input, args, target)

            cooccur_shuff_file = run_cooccur_and_shuffle(input, args, vocab_file, target)

            run_glove(args, vocab_file, cooccur_shuff_file, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GloVe training steps  with specified configuration.")
    parser.add_argument('-t', '--training', default='prepared_txt', help='path to prepared data')
    parser.add_argument('-w', '--windowing', default='middle', choices=['single', 'middle', 'leading', 'none'], help='How to window the training data around the target, middle is wtw and leading is wwt, none is a single model with words tagged by congress')
    parser.add_argument('-o', '--output', default='models', help='path where resulting models and intermediate files should be placed')
    parser.add_argument('-g', '--glove', default='./GloVe/build', help='Path where the GloVe executables can be found')
    parser.add_argument('-s', '--start',  default=55, type=int, help='Congress to start with')
    parser.add_argument('-e', '--end', default=110, type=int, help='Congress to end with')
    parser.add_argument('-m', '--memory', default=6.0, type=float, help='The amount of memory in Gigabyte to allow the training process to use')
    parser.add_argument('-c', '--min-count', default=5, type=int, help='Minimum number of occurences to consider a word')
    parser.add_argument('-v', '--vector-size', default=300, type=int, help='Dimensionality of resulting word vectors')
    parser.add_argument('-i', '--max-iterations', default=20, type=int, help='Maximum iterations allowed for the GloVe method')
    parser.add_argument('-T', '--target-words', default='', help='Path to file with list of target words separated by whitespace, for none type windowing (Temporally Referenced)')

    args = parser.parse_args(sys.argv[1:])

    glove_training(args)