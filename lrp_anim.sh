#!/bin/bash

count=$(find lrp_anim/heat_1* -maxdepth 1 -type f|wc -l)
for i in $( eval echo {0..$count} )
do
    idx=$(printf "%03d" $i)
    montage lrp_anim/heat_4_${idx}.jpg lrp_anim/heat_3_${idx}.jpg lrp_anim/heat_2_${idx}.jpg lrp_anim/heat_1_${idx}.jpg lrp_anim/heat_0_${idx}.jpg lrp_anim/in_${idx}.jpg lrp_anim/out_${idx}.jpg lrp_anim/montage_${idx}.jpg
done

ffmpeg -r 24 -i lrp_anim/montage_%03d.jpg montage.mp4