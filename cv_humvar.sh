#!/bin/bash

USAGE="Usage: `basename $0` experiment_module"
E_WRONGARGS=85

if [ $# -ne 1 ]
then
    echo ${USAGE} && exit ${E_WRONGARGS}
fi

for i in `seq 1 10`
do
    dataset_name=`printf humvar_%02d ${i}`
    python ./experimenter.py $1 $2 ${dataset_name} || exit 1
done

paplay /usr/share/sounds/gnome/default/alerts/bark.ogg & exit 0
