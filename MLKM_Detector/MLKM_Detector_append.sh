#!/bin/bash
ulimit -n 65536;

machine=$1;
devices=$2;
batch=$3;
lr=$4;
N=$5;
K=$6;
w=$7;
ft=$8;
dt=$9;
model=${10};
label=${11};
data=${12};
max_epoch=${13};
prev_trial=${14};
trial=${15};
train_ini=${16};
train_end=${17};

./MLKM_Detector_append.py ${machine} ${devices} ${batch} ${lr} ${N} ${K} ${w} ${ft} ${dt} ${model} ${label} ${data} ${max_epoch} ${prev_trial} ${trial} ${train_ini} ${train_end};
