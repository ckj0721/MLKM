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
trial=${14};
train_ini=${15};
train_end=${16};

./MLKM_Detector.py ${machine} ${devices} ${batch} ${lr} ${N} ${K} ${w} ${ft} ${dt} ${model} ${label} ${data} ${max_epoch} ${trial} ${train_ini} ${train_end};

 