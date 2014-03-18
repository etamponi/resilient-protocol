#!/bin/bash

USAGE="Usage: `basename $0` python_experiment"
E_WRONGARGS=85

if [ $# -ne 1 ]
then
    echo $USAGE && exit $E_WRONGARGS
fi

if [ ! -e "./$1" ]
then
    echo $USAGE && exit $E_WRONGARGS
fi

for i in `seq 1 10`
do
    PYTHONPATH=".." python ./$1 $i || exit 1
done

paplay /usr/share/sounds/gnome/default/alerts/bark.ogg & exit 0
