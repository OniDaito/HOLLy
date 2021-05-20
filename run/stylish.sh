#!/bin/bash
montage $1/in_e0* -background 'black' in.jpg
montage $1/out_e0* -background 'black' out.jpg
convert in.jpg -flatten +level-colors black,HotPink pink.jpg
convert out.jpg -flatten +level-colors black,cyan blue.jpg
convert pink.jpg blue.jpg -compose lighten -composite  pink_and_blue.jpg
rm pink.jpg
rm blue.jpg
rm in.jpg
rm out.jpg
