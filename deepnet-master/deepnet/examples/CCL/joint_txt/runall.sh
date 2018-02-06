#!/bin/bash
deepnet=../..
prefix=${deepnet}/examples/CCL/joint_txt

gpu_mem=4G
main_mem=25G

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=${prefix}/joint_models
data_output_dir=${prefix}/joint_reps

echo "Training joint layer"
python ${trainer} ${prefix}/joint_txt.pbtxt ${prefix}/train.pbtxt ${prefix}/eval.pbtxt || exit 1
echo "Train over"
python ${extract_rep} ${model_output_dir}/joint_txt_LAST ${prefix}/train.pbtxt joint_txt_hidden ${data_output_dir}/joint_txt_LAST ${gpu_mem} ${main_mem} || exit 1
echo "over"
