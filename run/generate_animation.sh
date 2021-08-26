#!/bin/bash
if [ $# != 4 ]
then
  echo "Please pass in the <save folder> <object_in> <object_out> and <sigma>."
  exit 1
fi

rm -rf ./animation_in
mkdir ./animation_in
rm -rf ./animation_out
mkdir ./animation_out

X=0;
Y=0;
Z=0;
TX=-1;
TY=-1;
TT=$((1/3600))
IDX=0;

for i in {1..3600}
do
	python ../render.py --rot $X,$Y,$Z --obj $2 --sigma $4
	a="renderer.fits"
	mv $a animation_in/`printf %04d.%s ${IDX%.*} ${a##*.}`
	a="renderer.jpg"
	mv $a animation_in/`printf %04d.%s ${IDX%.*} ${a##*.}`
	let IDX="$IDX+1"
	let X="$X+1"
	let Y="$Y+1"
	let Z="$Z+1"
	let TX=$(($TX + $TT))
	let TY=$(($TY + $TT))
done

let IDX=0;

for i in {1..3600}
do
	a="renderer.fits"
	python ../run.py --load $1 --image animation_in/`printf %04d.%s ${IDX%.*} ${a##*.}` --points $3 --sigma $4 --no-cuda
	a="guess.jpg"
	mv $a animation_out/`printf %04d.%s ${IDX%.*} ${a##*.}`
	let IDX="$IDX+1"
done

cd animation_in
ffmpeg -i %4d.jpg -pix_fmt rgb24 input.gif
cd ../animation_out
ffmpeg -i %4d.jpg -pix_fmt rgb24 output.gif
cd ..
rm anim_output.gif
rm anim_output.mp4
ffmpeg -i animation_in/input.gif  -i animation_out/output.gif -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map [vid] -c:v libx264 -crf 23 -preset veryfast anim_output.mp4
ffmpeg -i anim_output.mp4 anim_output.gif