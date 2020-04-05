# csc2611-project

The scripts in the repo all do what they are named for and have help text that will describe the available options.

## Prerequisites
You will need to download and unpack the hein-bound.zip from the [Stanford Congress Corpus](https://data.stanford.edu/congress_text)

You will need to clone and build the [GloVe project](https://github.com/stanfordnlp/GloVe/tree/master/src)

You will need to install Gensim, nltk, scikit-learn, scipy, and numpy.

## Assembling training data

The assemble_training_data.py script will parse the files from the hein-bound package and emit pre-processed files separated by congressional session.

## Training Models

The train_models.py script will build models using any of the three supported windowing methods.

## Evaluating Models

The evaluate_models.py script calculates the distances (both cosine and Euclidean) between overlapping words in all adjacent congressional sessions.  It will emit these scores and the sets of novel and retired words between them.
