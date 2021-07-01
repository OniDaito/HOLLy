#!/bin/bash
#    ___           __________________  ___________
#   / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
#  / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
# /_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
#
# Author : Benjamin Blundell - k1803390@kcl.ac.uk
#
# Run our various scripts and package everything up in the data dir
# then upload to the server for viewing.
#
# Example usage: ./generate_stats.sh  -i /tmp/runs/test_run -g ../objs/teapot.obj -a -z
# 
# -i : input path of the experiment
# -g : the groundtruth obj file, if there was one
# -a : evaluate angles and create an animation
# -z : no translation was used
# -n : Use basic normalisation (needs to match the experiment)
#
# https://misc.flogisoft.com/bash/tip_colors_and_formatting

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Colour

echo -e "\U1F6A7 " ${YELLOW}GENERATING STATS${NC} "\U1F6A7"

#if [ $# < 2 ]
#then
#  echo "Please pass in the directory of results and the groundtruth"
#  exit 1
#fi

#VARS=`getopt -o ni:g: --long normalise,input:,ground: -- "$@"`
#eval set -- "$vars"

extras=''
animate=false
base=''
groundtruth=''
while test $# -gt 0; do
    case "$1" in
      -n|--normalise)
        extras+=' --normalise-basic'
        shift
        ;;
      -a|--animate)
        animate=true
        shift
        ;;
      -z|--notrans)
        extras+=' --no-translate'
        shift
        ;;
      -i|--input)
        shift
	base=$1
        shift
        ;;
      -g|--ground)
	shift
        groundtruth=$1
        shift
        ;;
    esac
done
echo $base
echo $groundtruth
echo $extras

# Generate a single animated JSON 3D points model from our OBJ files.
python ../stats/objs_to_json.py --path $base/objs --limit -1

# Find out which sigma file we used.
sigmafile=`grep "sigma" $base/run.conf | sed -En "s/.*sigma-file ([^ ]+).*/\1/p"`
echo "Sigma File $sigmafile"

# Because I messed up copying the sigma I've put in copy here
if [ -f $sigmafile ]; then
  cp $sigmafile $base/$sigmafile
fi

lastply=`ls -rt $base/plys/shape*ply | tail -1`
cp $lastply $base/last.ply

# Create a subdir we can rsync to the server.
subdir="${1##*/}"

# Create our various montages for viewing. Relies on ImageMagick
rm $base/montage_in.jpg $base/montage_out.jpg
montage $base/jpgs/in_*.jpg $base/montage_in.jpg
montage $base/jpgs/out_*.jpg $base/montage_out.jpg

files_in=($base/jpgs/in_*.jpg)
files_out=($base/jpgs/out_*.jpg)
counter=1
while [ $counter -le ${#files_in[@]} ] 
do
out_file=`printf pair_%05d.jpg ${counter}`;
montage -tile 2x1 ${files_in[$counter]} ${files_out[$counter]} /tmp/$out_file
((counter++))
done

montage -frame 3 -geometry 128x64 /tmp/pair_* $base/pair_montage.jpg
rm /tmp/pair_*


# Copy the stats jupyter notebook
cp stats.ipynb $base

# Make our diff image with a nice shift.
convert '(' $base/montage_out.jpg -flatten -grayscale Rec709Luminance ')'\
	'(' $base/montage_in.jpg -flatten -grayscale Rec709Luminance ')'\
	'(' -clone 0-1 -compose luminize -composite ')'\
       	-channel RGB -combine $base/diff.png

# Evaluate the network and use the output file with the meshscore program if we have it installed
if [ "$animate" = true ]; then
  python ../eval.py --no-cuda --savedir $base --obj $groundtruth --animate $extras

  # Create an animated gif
  ffmpeg -i $base/eval_in_%3d.jpg -r 5 -pix_fmt rgb24 $base/input.gif
  ffmpeg -i $base/eval_out_%3d.jpg -r 5 -pix_fmt rgb24 $base/output.gif
  rm $base/anim_output.gif
  rm $base/anim_output.mp4
  ffmpeg -i $base/input.gif  -i $base/output.gif -r 5 -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map [vid] -c:v libx264 -crf 23 -preset veryfast $base/anim_output.mp4
  ffmpeg -i $base/anim_output.mp4 $base/anim_output.gif
  rm $base/eval_in*.jpg
  rm $base/eval_out*.jpg
else
  python ../eval.py --no-cuda --savedir $base --obj $groundtruth $extras
fi

# Mesh score won't be installed on other systems so lets comment out for now
#mesh_score -b $groundtruth -t $base/eval_out.ply > $base/mesh_score.txt 

# Export the redis data to a set of CSVs which we will then add to a zip
# but only if they don't already exist
#if [ ! -f $base/redis_csv_data.zip ]
#then	
#  echo -e "\U1F69A " ${GREEN}EXPORTING REDIS DATA${NC} "\U1F69A"
#  mkdir parts
#  for i in `redis-cli  KEYS "$subdir:*" | tr -d '"' `
#  do
#    redis-cli --csv ZRANGE $i 0 -1 > parts/$i.csv
#  done

#  cd parts
#  zip redis_csv_data.zip *
#  mv redis_csv_data.zip ../$base/.
#  cd ..
#  rm -rf parts
#fi
