#!/usr/bin/env python
"""
   !!! Not certified fit for any purpose, use at your own risk !!!

   Copyright (c) Rex Sutton 2016.

   Demonstrate machine learning for binary classification,
       using a Gaussian Process, on USPS hand written digits, 3's vs 5's task.

"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np

from Tools import logstring
from Tools import print_log

import Usps
import LaplaceBinaryGpClassifier as gpc

GOLD_LOGSIGMAF = 2.35
GOLD_LOGL = 2.85
GOLD_LOG_MARGINAL_LIKELIHOOD = -99.0

__pos_label__ = 3
__neg_label__ = 5
__max_patterns__ = 1000


def benchmark(path):
    """Benchmark the solution of Rasmussen and Williams, published for the USPS
        data 3 versus 5 classifier using the squared exponential kernel.
    Args:
        path (str): The path to save the classifier to.
    """
    # load data from disk
    print_log("starting...")
    training_patterns, training_classifications \
        = Usps.load_training_data(pos_label=__pos_label__,
                                  neg_label=__neg_label__,
                                  max_patterns=__max_patterns__)
    print_log("loaded data...")
    # observation_vector is D rows, n cols
    training_patterns = np.transpose(training_patterns)
    # use gold params
    params = np.empty([2], gpc.__precision__)
    params[0] = GOLD_LOGSIGMAF
    params[1] = GOLD_LOGL
    # compute the derived data
    kernel = gpc.SquaredExponentialKernel()
    kernel.set_params(params)
    data = gpc.DerivedData(training_patterns, training_classifications, kernel)
    print_log("calculated derived data...")
    # build the classifier
    pred = gpc.Classifier(data)
    print_log("initialized classifier...")
    # save the classifier
    pred.save(path)
    print_log("saved classifier...")
    # run test
    test(path)


def test(path):
    """ Test the performance of the classifier saved to disc.
    Args:
        path (str): The path to load the classifier from.
    """
    # load patterns
    test_patterns, test_classifications \
        = Usps.load_test_data(pos_label=__pos_label__,
                              neg_label=__neg_label__,
                              max_patterns=__max_patterns__)
    # load classifier
    pred = gpc.Classifier.load(path)
    # print parameters
    print logstring(), "using logsigmaf:", pred.data.kernel.__kernel__.logsigmaf
    print logstring(), "using logl:", pred.data.kernel.__kernel__.logl
    # print log marginal likelihood and derivatives
    derivatives = pred.log_marginal_likelihood_deriv()
    print logstring(), "log marginal likelihood:", pred.log_marginal_likelihood()
    print logstring(), "dlogsigmaf:", derivatives[0]
    print logstring(), "dlogl:", derivatives[1]
    # print information and errors
    predicted_probabilities \
        = [pred.predict(pattern) for pattern in test_patterns]
    predicted_classifications \
        = [pred.threshold(probability) for probability in predicted_probabilities]
    results \
        = np.subtract(predicted_classifications, test_classifications)  # an error if opposite signs
    num_errors = np.count_nonzero(results)
    information = -1.0 * np.average(np.log2(predicted_probabilities))
    print logstring(), "average Information (bytes):", information
    # print performance summary
    num_test_patterns = len(test_classifications)
    num_correct_classifications = num_test_patterns - num_errors
    percent = 100.0 * float(num_correct_classifications) / float(num_test_patterns)
    print logstring(), "correctly classified:", num_correct_classifications,\
        "of", num_test_patterns, "digits"
    print logstring(), "correctly classified:", percent, "%"
    # print mis-classifications
    print logstring(), "mis-classification indices:", list(np.nonzero(results)[0])


def peek(path, pattern_idx):
    """ Classify a pattern from the USPS test patterns using the classifier saved to disk.
    Args:
        path (str): The path to save the classifier to.
        pattern_idx (int): The index of the selected pattern.
    """
    # load patterns
    test_patterns, dummy \
        = Usps.load_test_data(pos_label=__pos_label__,
                              neg_label=__neg_label__,
                              max_patterns=__max_patterns__)
    # print the predicted classification
    pred = gpc.Classifier.load(path)
    prob = pred.predict(test_patterns[pattern_idx])
    if gpc.Classifier.threshold(prob) > 0.0:
        print "*** machine predicted digit:", __pos_label__, "with probability:", prob
    else:
        print "*** machine predicted digit:", __neg_label__, "with probability:", 1.0 - prob
    # call Usps to peek at the pattern
    Usps.peek(pattern_idx,
              pos_label=__pos_label__,
              neg_label=__neg_label__,
              max_patterns=__max_patterns__)


def train(path):
    """ Train the classifer, saving the results to disk.
    Args:
        path (str): The path to save the model to.
    """
    # load patterns
    training_patterns, training_classifications \
        = Usps.load_training_data(pos_label=__pos_label__,
                                  neg_label=__neg_label__,
                                  max_patterns=__max_patterns__)
    # observation_vector is D rows, n cols
    training_patterns = np.transpose(training_patterns)
    # train the model
    print logstring(), "training..."
    kernel = gpc.SquaredExponentialKernel()
    params = gpc.Classifier.train(kernel, training_patterns, training_classifications)
    # print results
    print logstring(), "optimal logsigmaf:", params[0]
    print logstring(), "optimal logl:", params[1]
    # save the predictor
    kernel.set_params(params)
    data = gpc.DerivedData(training_patterns, training_classifications, kernel)
    pred = gpc.Classifier(data)
    pred.save(path)
    print_log("saved classifier...")
    # run test
    test(path)


def main():
    """ The main entry point function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command",
                        help="The command to invoke.",
                        choices=["bench", "test", "peek", "train"],
                        default="peek")
    parser.add_argument("-p", "--path",
                        help="The path to load/save the classifier from/to.",
                        default="classifier")
    parser.add_argument("-i", "--idx",
                        help="The index of the pattern to peek at.",
                        type=int,
                        default=0)

    args = parser.parse_args()

    if args.command == "bench":
        benchmark(args.path)
    elif args.command == "test":
        test(args.path)
    elif args.command == "train":
        train(args.path)
    elif args.command == "peek":
        peek(args.path, args.idx)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
