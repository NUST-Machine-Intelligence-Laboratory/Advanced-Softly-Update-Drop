#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore"

# resnet18, resnet50, bcnn
export NET='bcnn'
export path='bcnn'
export data_base='fg-web-data/web-bird'
export N_CLASSES=200
export label_weight=0.6
export drop_rate=0.25
export epochs=80
export tk=5
export lr=1e-2
export w_decay=1e-8
export batchsize=64
export step=1

python train.py --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data_base} --lr ${lr} --w_decay ${w_decay} --batch_size ${batchsize} --epochs ${epochs} --label_weight ${label_weight} --drop_rate ${drop_rate} --tk ${tk} --step ${step} --denoise --smooth --warm 5 --cos

sleep 100

export epochs=80
export lr=1e-2
export w_decay=1e-5
export batchsize=32
export step=2

python train.py --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data_base} --lr ${lr} --w_decay ${w_decay} --batch_size ${batchsize} --epochs ${epochs} --label_weight ${label_weight} --drop_rate ${drop_rate} --tk ${tk} --step ${step} --denoise --smooth --warm 5 --cos
