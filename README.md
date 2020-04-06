# csc2611-project

All the code in this project is distributed under the GPLv3, see the LICENSE file for the full license text.

The scripts in the repo all do what they are named for and have help text that will describe the available options.

## Prerequisites
You will need to download and unpack the hein-bound.zip from the [Stanford Congress Corpus](https://data.stanford.edu/congress_text)

You will need to clone and build the [GloVe project](https://github.com/stanfordnlp/GloVe/tree/master/src)

You will need to install Gensim, nltk, scikit-learn, scipy, and numpy.

## Assembling training data

The assemble_training_data.py script will parse the files from the hein-bound package and emit pre-processed files separated by congressional session.

For example:
```
$ ./assemble_training_data.py -d ~/data/hein-bound -s 55 -e 110 -o ~/data/prepared_data
```

## Training Models

The train_models.py script will build models using any of the three supported windowing methods.

For example:
```
$ ./train_models.py -t ~/data/prepared_data -w leading -o ~/models -g ~/GloVe/build -s 55 -e 110 -m 8 -c 10 -v 300 -i 20
```

## Evaluating Models

The evaluate_models.py script calculates the distances (both cosine and Euclidean) between overlapping words in all adjacent congressional sessions.  It will emit these scores and the sets of novel and retired words between them.

For example:
```
$ ./evaluate_models.py -m ~/models -w leading -s 55 -e 110 -o ~/results
```
