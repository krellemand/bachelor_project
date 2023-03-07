#!/bin/bash
PYTHON='/usr/local/bin/python'
export CUDA_VISIBLE_DEVICES=0

hostname
nvidia-smi

# Get unique log file
SAVE_DIR=logs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.tests.openset_test_imagenet
#> ${SAVE_DIR}logfile_${EXP_NUM}.out