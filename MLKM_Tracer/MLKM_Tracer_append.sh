#!/bin/bash
ulimit -n 65536;

MACHINE=$1;
DEVICES=$2;
BATCHING=$3;
LEARNING_RATE=$4;
SYSTEM_SIZE=$5;
COUPLING=$6;
LENGTH_IN=$7;
MODEL=$8;
LABEL=$9;
DATA=${10};
MAX_EPOCH=${11};
PREV_TRIAL=${12};
TRIAL=${13};

./MLKM_Tracer_append.py ${MACHINE} ${DEVICES} ${BATCHING} ${LEARNING_RATE} ${SYSTEM_SIZE} ${COUPLING} ${LENGTH_IN} ${MODEL} ${LABEL} ${DATA} ${MAX_EPOCH} ${PREV_TRIAL} ${TRIAL};
