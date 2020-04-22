#!/bin/bash
set -ex
# bash inference.sh &

GPUID=0

# test data
DATASETS_DIR=./datasets/
TARGET_ACTOR=room1

# model
MODEL=TexRepNet
TEX_N_LAYERS=8
TEX_N_FREQ=10
#TEX_N_FREQ=12
TEX_NGF=256

###############################################################################
###############################    TESTING     ################################
###############################################################################

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
#DATE_WITH_TIME=20200421-123717
DATE_WITH_TIME=20200421-224344
NAME=$MODEL/$TARGET_ACTOR-TEX2048--$DATE_WITH_TIME-VGG

# output directory
RESULT_DIR=./results/$NAME/
mkdir -p $RESULT_DIR

# training
python test.py --tex_n_layers $TEX_N_LAYERS --tex_n_freq $TEX_N_FREQ --tex_ngf $TEX_NGF  --results_dir $RESULT_DIR --name $NAME --dataroot $DATASETS_DIR/$TARGET_ACTOR --model $MODEL --gpu_ids $GPUID
