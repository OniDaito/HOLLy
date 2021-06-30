#!/bin/bash
mkdir experiment_noise
python ../train.py --savedir ./experiment_noise --save-interval 800 --train-size 2000 --test-size 256 --valid-size 64 --objpath ../objs/bunny_large.obj --save-stats --aug --num-aug 20 --buffer-size 32 --epochs 40 --batch-size 32 --cont --predict-sigma --normalise-basic --mask-thresh 0.0001 --log-interval 100 --wobble 0.06 --max-spawn 4 --spawn-rate 0.3 --num-points 350 --lr 0.0004 --sigma-file sigma_bunny.csv
