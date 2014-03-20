#!/bin/bash

USAGE="Usage: `basename $0` python_experiment"
E_WRONGARGS=85

if [ $# -ne 1 ]
then
    echo ${USAGE} && exit ${E_WRONGARGS}
fi

if [ ! -e "./$1" ]
then
    echo ${USAGE} && exit ${E_WRONGARGS}
fi

RESULTS_DIR="$(cd ../; python ./next_results_dir.py)"
for i in `seq 1 10`
do
    PYTHONPATH=".." python ./$1 ${i} ${RESULTS_DIR} || exit 1
done

echo "Results directory: ${RESULTS_DIR}"
paplay /usr/share/sounds/gnome/default/alerts/bark.ogg & exit 0
