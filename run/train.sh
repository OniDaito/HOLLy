#!/bin/bash
RED='\033[0;31m'
NC='\033[0m' # No Colour

echo -e "\U1F3CB " ${RED}TRAINING${NC} "\U1F3CB"

if ! git diff-files --quiet --ignore-submodules -- ; then
  echo "You have changes that have not been committed. Please commit before continuing."
  exit 1
fi

if [ $# < 2 ]
  then
    echo "Please pass the directory to save and an overall comment as to what this network is doing."
    exit 1
fi

baseops=""
date=$(date +'%Y_%m_%d')
#basedir="../runs/$date"
basedir=$1
echo $2 > $basedir/notes.txt

gitbranch=$(git rev-parse --abbrev-ref HEAD)

if [ -f $1/run.conf ]; then
  baseops=`tail -1 $1/run.conf`
  tdir=`head -3 $1/run.conf | tail -1` # Get the basedir at runtime
  vdir=`head -4 $1/run.conf | tail -1` # Get the datadir at runtime
  basedir=${tdir:2}
  datadir=${vdir:2}
  echo "Running with ops from conf file: " $baseops 
elif [ -f run.conf ]; then
  baseops=`tail -1 run.conf`
  tdir=`head -3 run.conf | tail -1` # Get the basedir at runtime
  vdir=`head -4 run.conf | tail -1` # Get the datadir at runtime
  basedir=${tdir:2}
  datadir=${vdir:2}
  echo "Running with ops from conf file: " $baseops
  cp run.conf $basedir/run.conf
else
  if [ $# != 3 ]
    then
      echo "Please pass in the directory to save results to, e.g train.sh <savedir> <comment> <datadir>"
      exit 1
  fi
  basedir=$1
  datadir=$2
  baseops="python ../train.py --savedir $basedir --save-interval 100 --train-size 99500 --valid-size 250 --test-size 250 --ceppath $2 --save-stats --epochs 20 --batch-size 20 --log-interval 100 --num-points 302"
  echo "Running with default ops: " $baseops
fi

#sigmafile=`grep "sigma" $basedir/run.conf | sed -En "s/.* --sigma-file (.*)/\1/p"`
#groundtruth=`grep "sigma" $basedir/run.conf | sed -En "s/.* --gt (.*) --save.*/\1/p"`

#if [ -f $sigmafile ]; then
#  cp $sigmafile $basedir/$sigmafile
#  cat $sigmafile >> $basedir/notes.txt
#fi
git log --format=%B -n 1 HEAD >> $basedir/notes.txt

echo "Time to train."

echo "#!/bin/bash" > $basedir/run.conf
echo "# Version of code tag: " `git describe --abbrev=0 --tags` >> $basedir/run.conf
echo "# " $basedir >> $basedir/run.conf
echo "# " $datadir >> $basedir/run.conf
echo "# branch " $gitbranch >> $basedir/run.conf
echo "# Execute the following commands to get the code associated with this run:" >> $basedir/run.conf
echo "#git clone gogs@git.benjamin.computer:oni/shaper.git" >> $basedir/run.conf
echo "#git reset --hard " `git rev-parse HEAD` >> $basedir/run.conf
echo "$baseops" >> $basedir/run.conf

time $baseops

echo -e "\U1F37B"
