#!/bin/bash

if [ $# -ne 1 ] || [ ! -d "$1" ]
then
    echo "Usage: `basename $0` directory" && exit 85
fi

PYTHONPATH="." python ./resilient/result_analysis.py ./"$1"/*.dat
