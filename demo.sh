#!/bin/bash
#
# demo.sh
#
# Rex Sutton 2016.
#
# test for data
if [ -d usps_resampled ]
then
    echo "The demo data has not been downloaded."
fi
# train the gp
echo "*** Begin training ***"
python Demo.py -ctrain
if [[ $? != 0 ]]
then
    echo "Failed to run Demo.py, use python 2.7"
    exit
fi
echo "*** Training complete  ***"
python Demo.py -cpeek
