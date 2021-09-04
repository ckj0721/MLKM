#!/bin/bash
ulimit -n 65536;

machine=$1;
devices=$2;
batch=$3;
lr=$4;
length=$5;
model=$6;
label=$7;
epoch=$8;
prev_trial=$9;
trial=${10};


./MLKM_Estimator_append.py ${machine} ${devices} ${batch} ${lr} ${length} ${model} ${label} ${epoch} ${prev_trial} ${trial};

paste Label_test3.dat saves/L${length}/${model}_${label}/outputs_${trial}.dat > saves/L${length}/${model}_${label}/outputs_${trial}_marking.dat