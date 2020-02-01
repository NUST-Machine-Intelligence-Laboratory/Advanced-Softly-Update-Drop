#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore"

# resnet18, resnet50
export NET='resnet18'
export path='resnet18'
export data_base='fg-web-data/web-bird'
export N_CLASSES=200
export lr=0.01
export w_decay=1e-5
export label_weight=0.5

python train.py --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data_base} --lr ${lr} --w_decay ${w_decay} --label_weight ${label_weight} --denoise --smooth --warm 5 --cos
