#!/bin/bash
mkdir experiment_cep
python ../train.py --savedir ./experiment_cep --save-interval 800 --train-size 40000 --test-size 256 --valid-size 64 --fitspath ./paper --save-stats --buffer-size 32 --epochs 40 --batch-size 32 --predict-sigma --normalise-basic --mask-thresh 0.0001 --log-interval 100 --num-points 250 --lr 0.0004 --sigma-file sigma_cep.csv
