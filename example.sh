#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0   python  cifarmain.py  --dataset ciafr100nc  --noise_type symmetric --noise_rate 0.5 --epochs 200 