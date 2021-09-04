#!/bin/bash
ulimit -n 65536;

machine=$1;
batch=$2;
length=$3;
kcmin=$4;
kcmax=$5;
label=$6;
# kc=$5;

./MLKM_Discriminator.py ${machine} ${batch} ${length} ${kcmin} ${kcmax} ${label};# ${kc};
