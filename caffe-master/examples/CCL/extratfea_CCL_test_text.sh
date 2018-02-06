#!/usr/bin/env sh

RUN_BIN=./build/tools

MODEL_NAME=./examples/CCL/CCL_model/CCL_seg_full_iter_20000.caffemodel.h5
FEA_DIR=./examples/CCL/CCL_test_text_res/
PROTO=./examples/CCL/CCL_trip_test.prototxt

echo "Begin Extract fea"
echo $MODEL_NAME
echo $FEA_DIR
echo $PROTO

#TOTAL_NUM=2274
TOTAL_NUM=100
BATCH_NUM=80
FEA_NUM=`expr $TOTAL_NUM / $BATCH_NUM + 1`

echo "Total Feature num: ${FEA_NUM}"

GLOG_logtostderr=0
${RUN_BIN}/extract_features.bin ${MODEL_NAME} ${PROTO} inter_prob_txt ${FEA_DIR} ${FEA_NUM} leveldb GPU 0
