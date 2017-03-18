#!/usr/bin/env python
"""
   !!! Not certified fit for any purpose, use at your own risk !!!

   Copyright (c) Rex Sutton 2016.

   Python interdace to USPS data.

   The *_patterns variables contain a raster scan
    of the 16 by 16 grey level pixel intensities,
    which have scaled such that the range is [-1; 1].

   The *_labels variables contain a one-of-k encoding,
    with values -1 and +1 of the classification,
    one +1 per column.

   The USPS data is at
       http://www.gaussianprocess.org/gpml/data/.

"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io

__pattern_data_type__ = np.float64
__label_data_type__ = np.int16
__class_data_type__ = np.float64


def load_data():
    """Load the USPS training and test patterns from disk.
    Returns:
        The data matrix.
    """
    return scipy.io.loadmat('usps_resampled/usps_resampled.mat')


def extract_pattern_vector(all_patterns, pattern_idx):
    """Copy pattern from array format to a vector.
    Args:
        all_patterns(matrix): The patterns.
        pattern_idx (int): The index of the selected pattern.
    Returns:
        vector: The pattern
    """
    vec = np.empty([256], dtype=__pattern_data_type__)
    for i in range(0, 256):
        vec[i] = all_patterns[i][pattern_idx]
    return vec


def extract_pattern_label(all_labels, pattern_idx):
    """Return the label.
    Args:
        all_labels(matrix): The patterns.
        pattern_idx (int): The index of the selected pattern.
    Returns:
        int: The pattern label
    """
    for i in range(0, 10):
        if all_labels[i][pattern_idx] == 1:
            return i


def pattern_vector_to_image_matrix(vec):
    """Format the vector as a matrix.
    Args:
        vec (vec): A vector.
    Returns:
        A 2d matrix view.
    """
    return vec.view().reshape([16, 16])


def show(pattern_vec):
    """Show the pattern as a greyscale image.
    Args:
        pattern_vec (vector): A pattern vector.
    """
    color_norm = matplotlib.colors.Normalize(
        vmin=-1.0, vmax=1.0, clip=False)

    plt.imshow(pattern_vector_to_image_matrix(pattern_vec), cmap='Greys',
               norm=color_norm, interpolation='nearest')
    plt.show()


def count_patterns(pos_label, neg_label, max_patterns, all_labels):
    """Count patterns from the input array of labels.
    Args:
        pos_label (int): Patterns with this label have +ve classification.
        neg_label (int): Patterns with this label have -ve classification.
        max_patterns (int): The maximum number of patterns to return, applies to both
            training and test patterns.
        all_labels: A vector of labels to select from.
    Returns:
        int: The number of patterns.
    """
    max_patterns = np.minimum(len(all_labels[0]), max_patterns)
    num_patterns = 0
    for pat in range(0, len(all_labels[0])):
        if num_patterns == max_patterns:
            break
        is_pos = all_labels[pos_label][pat] == 1
        is_neg = all_labels[neg_label][pat] == 1
        if is_pos or is_neg:
            num_patterns += 1
    return num_patterns


def select_patterns(pos_label, neg_label, max_patterns, all_patterns, all_labels):
    """Select patterns from the input array of patterns and labels.
    Args:
        pos_label (int): Patterns with this label have +ve classification.
        neg_label (int): Patterns with this label have -ve classification.
        max_patterns (int): The maximum number of patterns to return, applies to both
            training and test vectors.
        all_patterns: A vector of patterns to select from.
        all_labels: A vector of labels to select from.
    Returns:
        tuple: Elements are, the matrix of patterns,
                their classification vector.
    """
    num_patterns = count_patterns(pos_label, neg_label, max_patterns, all_labels)
    patterns, classifications = np.empty([num_patterns, 256], dtype=__pattern_data_type__),\
                                np.empty([num_patterns], dtype=__class_data_type__)

    idx_write_pattern = 0
    for idx_read_pattern in range(0, len(all_labels[0])):

        if idx_write_pattern == num_patterns:
            break

        is_pos = all_labels[pos_label][idx_read_pattern] == 1
        is_neg = all_labels[neg_label][idx_read_pattern] == 1

        if is_pos:
            classifications[idx_write_pattern] = 1.0
        if is_neg:
            classifications[idx_write_pattern] = -1.0

        if is_pos or is_neg:
            for i in range(0, 256):
                patterns[idx_write_pattern][i] = all_patterns[i][idx_read_pattern]

            idx_write_pattern += 1

    return patterns, classifications


def load_test_data(pos_label, neg_label, max_patterns):
    """Load selected USPS test patterns from disk.
    Args:
        pos_label (int): Patterns with this label are recorded with + classification.
        neg_label (int): Patterns with this label are recorded with - classification.
        max_patterns (int): The maximum number of patterns to return,
            applies to both training and test vectors.
    Returns:
        tuple: Elements are,
            the test patterns, their classification vector.
    """
    mat = load_data()
    return select_patterns(pos_label, neg_label, max_patterns,
                           mat['test_patterns'], mat['test_labels'])


def load_training_data(pos_label, neg_label, max_patterns):
    """Load selected USPS training patterns from disk.
    Args:
        pos_label (int): Patterns with this label are recorded with + classification.
        neg_label (int): Patterns with this label are recorded with - classification.
        max_patterns (int): The maximum number of patterns to return,
            applies to both training and test vectors.
    Returns:
        tuple: Elements are,
            the training patterns, their classification vector,
    """
    mat = load_data()
    return select_patterns(pos_label, neg_label, max_patterns,
                           mat['train_patterns'], mat['train_labels'])


def load_pattern_data(pos_label, neg_label, max_patterns):
    """Load selected USPS training and test patterns from disk.
    Args:
        pos_label (int): Patterns with this label are recorded with + classification.
        neg_label (int): Patterns with this label are recorded with - classification.
        max_patterns (int): The maximum number of patterns to return,
            applies to both training and test vectors.
    Returns:
        tuple: Elements are,
            the training patterns, their classification vector,
            the test patterns, their classification vector.
    """
    mat = load_data()
    training_patterns, training_classifications \
        = select_patterns(pos_label, neg_label, max_patterns,
                          mat['train_patterns'], mat['train_labels'])
    test_patterns, test_classifications \
        = select_patterns(pos_label, neg_label, max_patterns,
                          mat['test_patterns'], mat['test_labels'])
    return training_patterns, training_classifications, test_patterns, test_classifications


def load_test_pattern(pattern_idx):
    """Load selected USPS test pattern and label from disk.

    Args:
        pattern_idx (int): The index of the selected pattern.

    Returns:
        tuple: Elements are,
            the pattern_vector, the pattern classification.
    """
    mat = load_data()
    return extract_pattern_vector(mat['test_patterns'], pattern_idx),\
           extract_pattern_label(mat['test_labels'], pattern_idx)


def peek(pattern_idx, pos_label, neg_label, max_patterns):
    """ Display a pattern from the USPS test patterns using the index w.r.t a pair of digits.
    Args:
        pattern_idx (int): The index of the selected pattern.
        pos_label (int): Patterns with this label are recorded with + classification.
        neg_label (int): Patterns with this label are recorded with - classification.
        max_patterns (int): The maximum number of patterns to return,
            applies to both training and test vectors.
    """
    # load patterns
    test_patterns, test_classifications \
        = load_test_data(pos_label, neg_label, max_patterns)
    # print the actual digit a human has classified the selected pattern as
    if test_classifications[pattern_idx] > 0:
        print "*** labelled digit:", pos_label
    else:
        print "*** labelled digit:", neg_label
    # show the image
    print "*** (close the plot window to continue)"
    show(test_patterns[pattern_idx])


def peep(pattern_idx):
    """ Display a pattern from the USPS test patterns.
    Args:
        pattern_idx (int): The index of the selected pattern.
    """
    # load patterns
    pattern, label = load_test_pattern(pattern_idx)
    print "*** labelled digit:", label
    # show the image
    print "*** (close the plot window to continue)"
    show(pattern)


def main():
    """ The main entry point function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx",
                        help="The index of the pattern to peep at.", type=int, default=2)
    args = parser.parse_args()
    if args.idx >= 0:
        peep(args.idx)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
