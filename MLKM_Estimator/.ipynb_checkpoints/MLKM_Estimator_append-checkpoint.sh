#!/bin/bash
ulimit -n 65536;

machine=$1;
devices=$2;
batch=$3;
length=$4;
model=$5;
label=$6;
epoch=$7;
prev_trial=$8;
trial=$9;


./MLKM_Estimator_append.py ${machine} ${devices} ${batch} ${length} ${model} ${label} ${epoch} ${prev_trial} ${trial};

paste Label_test3.dat saves/L${length}/${model}_${label}/outputs_${trial}.dat > saves/L${length}/${model}_${label}/outputs_${trial}_marking.dat