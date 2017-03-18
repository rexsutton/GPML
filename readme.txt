Rex Sutton 2016

This is the `readme' for my python implementation of binary classification, using a Gaussian Process with Laplace's approximation.

Included is a demonstration taken from `Rasmussen, Williams 2006 Gaussian Processes For Machine Learning', see `http://www.gaussianprocess.org/gpml/'.

The demo uses machine learning to classify images of hand-written digits (collected by United States Postal Service), in particular images of `3' and `5's.

The USPS data is taken from `http://www.gaussianprocess.org/gpml/data/'.

setup.sh
--------

A bash script for downloading and de-compressing the training and test data.

demo.sh
--------

A bash script for running the demonstration, first training the model then displaying a digit and classification.

Demo.py
-------

python Demo.py -ctrain

Training refers to learning the optimal hyper-parameters.

Training should take less than two minutes to complete on a modern laptop.

At the end if training, the classifier is tested on the test data.

The indices of digits that are mis-classified are printed out.

It is interesting to view (see below) the digits both successfully and un-successfully classified, by passing the (zero-based) index of the image on the command line.

python Demo.py -cpeek -iINDEX
