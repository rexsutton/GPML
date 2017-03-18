#!/bin/bash
#
# setup.sh
#
# Rex Sutton 2016
#
#
# test to see if the data is already there
if [ -d usps_resampled ]
then
    echo "The demo data has already been downloaded."
    echo "Remove the data (y)?"
    read confirm
    if [ $confirm = 'y' ] ; then
        rm -rf usps_resampled
    else
        exit
    fi
fi
# download the training data
wget 'http://www.gaussianprocess.org/gpml/data/usps_resampled.tar.bz2'
if [[ $? != 0 ]]
then
    echo "Failed to download USPS data."
    exit
fi
# de-compress the data
tar -xjf usps_resampled.tar.bz2
if [[ $? != 0 ]]
then
    echo "Failed to de-compress USPS data."
    exit
fi
echo "*** Training data downloaded and de-compressed successfully. ***"
