#!/usr/bin/python3

import os
import sys
import multiprocessing
import argparse
from subprocess import Popen, PIPE
from gensim.scripts.glove2word2vec import glove2word2vec

def read_input_data(windowing, target, training):
    if windowing == 'middle':
        p1 = '{}/prepared_{:03d}.txt'.format(training, target - 1)
        p2 = '{}/prepared_{:03d}.txt'.format(training, target)
        p3 = '{}/prepared_{:03d}.txt'.format(training, target + 1)
    elif windowing == 'leading':
        p1 = '{}/prepared_{:03d}.txt'.format(training, target - 2)
        p2 = '{}/prepared_{:03d}.txt'.format(training, target - 1)
        p3 = '{}/prepared_{:03d}.txt'.format(training, target)
    input = []
    for entry in [p1, p2, p3]:
        with open(entry, 'r') as fd:
            for line in fd:
                input.append(line)
    return ' '.join(input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GloVe training steps  with specified configuration.")
    parser.add_argument('-t', '--training', default='prepared_txt', help='path to prepared data')
    parser.add_argument('-w', '--windowing', default='middle', choices=['middle', 'leading'], help='How to window the training data')
    parser.add_argument('-o', '--output', default='models', help='path where resulting models and intermediate files should be placed')
    parser.add_argument('-g', '--glove', default='./GloVe/build', help='Path where the GloVe executables can be found')
    parser.add_argument('-s', '--start',  default=55, type=int, help='Congress to start with')
    parser.add_argument('-e', '--end', default=111, type=int, help='Congress to end with')
    parser.add_argument('-m', '--memory', default=6.0, type=float, help='The amount of memory in Gigabyte to allow the training process to use')
    parser.add_argument('-c', '--min-count', default=5, type=int, help='Minimum number of occurences to consider a word')
    parser.add_argument('-v', '--vector-size', default=300, type=int, help='Dimensionality of resulting word vectors')
    parser.add_argument('-i', '--max-iterations', default=20, type=int, help='Maximum iterations allowed for the GloVe method')

    args = parser.parse_args(sys.argv[1:])

    numCpus = multiprocessing.cpu_count()

    for target in range(args.start, args.end + 1):
        input = read_input_data(args.windowing, target, args.training)
        exe = "{}/vocab_count".format(args.glove)
        cmd = [exe, '-min-count', str(args.min_count), '-verbose', '2']
        p = Popen(cmd, bufsize=268435456, stdin=PIPE, stdout=PIPE, universal_newlines=True)
        out,_ = p.communicate(input=input)
        vocab_file = '{}/vocab_{}_{}.txt'.format(args.output, args.windowing, target)
        with open(vocab_file, 'w') as fd:
            fd.write(out)

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

        exe = '{}/glove'.format(args.glove)
        cmd = [exe, '-save-file', '{}/vectors_{}_{}'.format(args.output, args.windowing, target), '-threads',
        str(numCpus), '-input-file', cooccur_shuff_file, '-x-max', '10', '-iter', str(args.max_iterations),
        '-vector-size', str(args.vector_size), '-binary', '2', '-vocab-file', vocab_file, '-verbose', '2']
        p = Popen(cmd)
        _,_ = p.communicate()

        os.remove(vocab_file)
        os.remove(cooccur_shuff_file)
        os.remove('{}/vectors_{}_{}.bin'.format(args.output, args.windowing, target))

        glove2word2vec('{}/vectors_{}_{}.txt'.format(args.output, args.windowing, target),
                       '{}/vectors_{}_{}.word2vec'.format(args.output, args.windowing, target))
        os.remove('{}/vectors_{}_{}.txt'.format(args.output, args.windowing, target))
