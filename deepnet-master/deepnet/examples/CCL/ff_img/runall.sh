#!/bin/bash
# Trains a feed forward net on MNIST.
deepnet=../..
train_deepnet=${deepnet}/trainer.py
ff_img_dir=${deepnet}/examples/CCL/ff_img
python ${train_deepnet} ${ff_img_dir}/model.pbtxt ${ff_img_dir}/train.pbtxt ${ff_img_dir}/eval.pbtxt

sh ${ff_img_dir}/extract_reps.sh
