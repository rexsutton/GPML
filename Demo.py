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

from Tools import log_string
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


def train(path):
    """ Train the classifier, saving the results to disk.
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
    print log_string(), "training..."
    kernel = gpc.SquaredExponentialKernel()
    params = gpc.Classifier.train(kernel, training_patterns, training_classifications)
    # print results
    print log_string(), "optimal log_sigma_f:", params[0]
    print log_string(), "optimal log_l:", params[1]
    # save the predictor
    kernel.set_params(params)
    data = gpc.DerivedData(training_patterns, training_classifications, kernel)
    pred = gpc.Classifier(data)
    pred.save(path)
    print_log("saved classifier...")


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


def show(prompt, path, indices):
    """ Show the user randomly selected patterns from the subset of indices.
    Args:
        prompt (str): The indices of the in-correctly classified patterns.
        path (str): The path to load the classifier from.
        indices (int): The indices of the patterns
    """
    while True:
        if raw_input(prompt) != "y":
            break
        index = np.random.choice(indices)
        peek(path, index)


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
    print "*** using log_sigma_f:", pred.data.kernel.__kernel__.log_sigma_f
    print "*** using log_l:", pred.data.kernel.__kernel__.log_l
    # print log marginal likelihood and derivatives
    derivatives = pred.log_marginal_likelihood_deriv()
    print "*** log marginal likelihood:", pred.log_marginal_likelihood()
    print "*** derivative log_sigma_f:", derivatives[0]
    print "*** derivative log_l:", derivatives[1]
    # print information and errors
    predicted_probabilities \
        = [pred.predict(pattern) for pattern in test_patterns]
    predicted_classifications \
        = [pred.threshold(probability) for probability in predicted_probabilities]
    results \
        = np.subtract(predicted_classifications, test_classifications)  # an error if opposite signs
    num_errors = np.count_nonzero(results)
    information = -1.0 * np.average(np.log2(predicted_probabilities))
    print "*** average Information (bytes):", information
    # print performance summary
    num_test_patterns = len(test_classifications)
    num_correct_classifications = num_test_patterns - num_errors
    percent = 100.0 * float(num_correct_classifications) / float(num_test_patterns)
    print "*** correctly classified:", num_correct_classifications,\
        "of", num_test_patterns, "digits"
    print "*** correctly classified:", percent, "%"
    show("*** View a randomly selected, correctly classified digit (y)? ",
         path,
         list(set(range(0, num_test_patterns)) - set(list(np.nonzero(results)[0]))))
    show("*** View a randomly selected, __in-correctly__ classified digit (y)? ",
         path,
         list(np.nonzero(results)[0]))


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
