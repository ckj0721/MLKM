#!/bin/bash
ulimit -n 65536;

machine=$1;
batch=$2;
length=$3;
label=$4;
kc=$5;
kcmin=$6;
kcmax=$7;

./MLKM_Discriminator.py ${machine} ${batch} ${length} ${label} ${kc} ${kcmin} ${kcmax};
