#!/bin/bash
#
# demo.sh
#
# Rex Sutton 2016.
#
# test for data
if [ ! -d usps_resampled ]
then
    echo "*** The demo data has not been downloaded. ***"
    exit
fi
# train the gp
if [ -e classifier ]
then
    echo "*** The classifier has already been trained. ***"
    echo "*** Remove the classifier to repeat training (y)? ***"
    read confirm
    if [ $confirm = 'y' ] ; then
        rm -rf classifier
    fi
fi
# train the classifier
if [ ! -e classifier ]
then
    echo "*** Begin training (should take less than two minutes). ***"
    python Demo.py -ctrain
    if [[ $? != 0 ]]
    then
        echo "*** Failed to run Demo.py, use python 2.7 ***"
        exit
    fi
    echo "*** Training complete. ***"
fi
# test the gp
echo "*** Begin testing. ***"
python Demo.py -ctest
if [[ $? != 0 ]]
then
    echo "*** Failed to run Demo.py, use python 2.7 ***"
    exit
fi
echo "*** Testing complete. ***"
