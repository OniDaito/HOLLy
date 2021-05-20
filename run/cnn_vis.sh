#!/bin/bash
python3 ../cnn_vis.py --savedir $1 --no-cuda
counter=1
while [ $counter -le 6 ] 
do
montage $1/layer_vis_l${counter}* $1/conv${counter}_vis.png
convert $1/conv${counter}_vis.png -resize 25% $1/conv${counter}_vis_small.png
((counter++))
done
rm $1/layer_vis_l*
echo "Completed Conv Vis"