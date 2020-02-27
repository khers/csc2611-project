#!/usr/bin/python3


import csv
import glob
import json
import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool
import codecs

from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_numeric, strip_punctuation
from gensim.parsing.preprocessing import strip_short, remove_stopwords, preprocess_string

from utilities import detect_encoding

stop_words = set()


# Parse the specified descr_*.txt file and extract the names of listed speakers.
def extract_names_from_file(path):
    names = set()
    try:
        encoding = detect_encoding(path)['encoding']
        with codecs.open(path, 'r', encoding=encoding, errors='ignore') as fd:
            reader = csv.DictReader(fd, delimiter='|')
            for line in reader:
                first = line['first_name'].lower()
                last = line['last_name'].lower()
                if first.isalpha() and first != 'unknown':
                    names.add(first)
                if last.isalpha() and last != 'unknown':
                    names.add(last)
    except UnicodeDecodeError as err:
        print("Failed to parse {} with error {}".format(path, err))
    return names


# Collect any words we want to filter out of the training data.  This is
# currently limited to the names of speakers listed in the description files.
def collect_stop_words(dataset_base_path):
    names = set()
    for entry in glob.glob("{}/descr_*.txt".format(dataset_base_path), recursive=False):
        names = names.union(extract_names_from_file(entry))
    return names


# Load the stop words set from JSON file, if present.  Otherwise collect the stop
# words and save to JSON file.
def load_stop_words(dataset_base_path):
    names = set()
    try:
        with open("{}/stop_words.json".format(dataset_base_path), 'r') as p:
            names = set(json.load(p))
    except (json.decoder.JSONDecodeError,FileNotFoundError):
        with open("{}/stop_words.json".format(dataset_base_path), 'w') as p:
            names = collect_stop_words(dataset_base_path)
            json.dump(list(names), p)
    return names


def filter_string(input):
    global stop_words
    if len(stop_words) is 0:
        return input
    out = [item for item in input.split() if item not in stop_words]
    return " ".join(out)


# Read a speeches_*.txt file, apply filters to clean the data, and return
# a single string of the cleaned text in this file.
def clean_input(file_path):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_numeric, strip_punctuation, filter_string,
                      remove_stopwords, strip_short, strip_multiple_whitespaces]
    out = []
    try:
        encoding = detect_encoding(file_path)['encoding']
        with codecs.open(file_path, 'r', encoding=encoding, errors='ignore') as fd:
            reader = csv.DictReader(fd, delimiter='|')
            for line in reader:
                out.append(" ".join(preprocess_string(line['speech'], filters=CUSTOM_FILTERS)))
    except UnicodeDecodeError as e:
        print("Failed to parse {} with discovered encoding {} and error {}".format(file_path, encoding, e))
        print("Last read line started with {}".format(line['speech_id']))
    return " ".join(out)


def prep_file(base, out, index):
    infile = "{}/speeches_{:03d}.txt".format(base, index)
    outfile = "{}/prepared_{:03d}.txt".format(out, index)
    if os.path.isfile(outfile):
        return
    with open(outfile, 'w') as fd:
        fd.write(clean_input(infile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare speech files for use in training GloVe models")
    parser.add_argument('-d', '--directory', default='./hein-bound', help='The top level directory where the data set is stored')
    parser.add_argument('-s', '--start', default=55, type=int, help='The Congress to use as a starting point')
    parser.add_argument('-e', '--end', default=106, type=int, help='The Congress to use as an ending point')
    parser.add_argument('-o', '--out', default='prepared_txt', help='Directory where the prepared text files should be placed')
    args = parser.parse_args(sys.argv[1:])
    Path(args.out).mkdir(parents=True, exist_ok=True)

    csv.field_size_limit(sys.maxsize)

    stop_words = set(load_stop_words(args.directory))
    inputs = []
    for i in range(args.start, args.end + 1):
        inputs.append((args.directory, args.out, i))
    p = Pool()
    p.starmap(prep_file, inputs)
