#!/bin/bash
# Trains a feed forward net on MNIST.
deepnet=../..
train_deepnet=${deepnet}/trainer.py
ff_txt_dir=${deepnet}/examples/CCL/ff_txt
python ${train_deepnet} ${ff_txt_dir}/model.pbtxt ${ff_txt_dir}/train.pbtxt ${ff_txt_dir}/eval.pbtxt

sh ${ff_txt_dir}/extract_reps.sh
