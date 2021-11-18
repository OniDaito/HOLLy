#!/bin/bash
# usage sigma_exp.sh <path to experiment>
X=10
Y=20
Z=30

for i in 10 9.0 8.1 7.29 6.56 5.9 5.31 4.78 4.3 3.87 3.65 3.28 2.95 2.66 2.39 2.15 1.94 1.743 1.57 1.41
do
    python ../render.py --obj ../objs/bunny_large.obj --sigma $i --rot $X,$Y,$Z --norm
    mv renderer.fits $i.fits
    mv renderer.jpg $i.jpg
    python ../run.py --load $1 --image $i.fits --points $1/last.ply --sigma $i --no-cuda
    mv guess.jpg $i_guess.jpg
    mv guess.fits $i_guess.fits
done


for i in 10 9.0 8.1 7.29 6.56 5.9 5.31 4.78 4.3 3.87 3.65 3.28 2.95 2.66 2.39 2.15 1.94 1.743 1.57 1.41
do
    python ../render.py --obj ../objs/bunny_large.obj --sigma $i --rot 10,20,30 --norm
    mv renderer.fits i$i.fits
    python ../render.py --obj ../objs/bunny_large.obj --sigma $i --rot 20,30,40 --norm
    mv renderer.fits j$i.fits
    python ../loss.py --i i$i.fits --j j$i.fits --norm
done