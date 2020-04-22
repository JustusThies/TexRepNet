#!/bin/bash
set -ex
# bash texture_optimization.sh &

GPUID=0

# training data
DATASETS_DIR=./datasets/
TARGET_ACTOR=room1
DATASET_MODE=uvpos


# texture
TEX_N_LAYERS=8
#TEX_N_FREQ=10
TEX_N_FREQ=12
TEX_NGF=256

# models
MODEL=TexRepNet

# optimizer parameters
BATCH_SIZE=1
LR=0.0001
N_ITER=130
N_ITER_LR_DECAY=10000


# save frequency
SAVE_FREQ=100


# loss weights
LAMBDA_L1=10.0
#LAMBDA_L1_DIFF=20.0
LAMBDA_L1_DIFF=0.0
LAMBDA_VGG=10.0

# regularizer
LAMBDA_REG_TEX=0.0


################################################################################
###############################    TRAINING     ################################
################################################################################

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
NAME=$MODEL/$TARGET_ACTOR-TEX_N_FREQ$TEX_N_FREQ-$LOSS-$DATE_WITH_TIME-VGG
DISPLAY_NAME=${MODEL}/${TARGET_ACTOR}-TEX_N_FREQ$TEX_N_FREQ-VGG

# output directory
RESULT_DIR=./results/$NAME/
mkdir -p $RESULT_DIR

# training
# --continue_train 
python train.py --results_dir $RESULT_DIR --name $NAME --save_epoch_freq $SAVE_FREQ --tex_n_layers $TEX_N_LAYERS --tex_n_freq $TEX_N_FREQ --tex_ngf $TEX_NGF --lambda_L1 $LAMBDA_L1 --lambda_L1_Diff $LAMBDA_L1_DIFF --lambda_Reg_Tex $LAMBDA_REG_TEX --lambda_VGG $LAMBDA_VGG --display_env $DISPLAY_NAME --niter $N_ITER --niter_decay $N_ITER_LR_DECAY --dataroot $DATASETS_DIR/$TARGET_ACTOR --model $MODEL --dataset_mode $DATASET_MODE --gpu_ids $GPUID --lr $LR --batch_size $BATCH_SIZE

