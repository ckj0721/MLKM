#!/bin/bash
ulimit -n 65536;

machine=$1;
devices=$2;
batch=$3;
length=$4;
model=$5;
label=$6;
epoch=$7;
trial=$8;

./MLKM_Estimator.py ${machine} ${devices} ${batch} ${length} ${model} ${label} ${epoch} ${trial};

paste Label_test3.dat saves/L${length}/${model}_${label}/outputs_${trial}.dat > saves/L${length}/${model}_${label}/outputs_${trial}_marking.dat