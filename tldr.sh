#!/bin/bash
#    ___           __________________  ___________
#   / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
#  / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
# /_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
#
# Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk
#
# tldr.sh - create a miniconda instance called holly,
# download all the required packages, train a
# network and generate some outputs.

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Colour
echo -e "\U1F69A " ${GREEN}CREATING MINICONDA ENVIRONMENT${NC} "\U1F69A"

# Create miniconda environment
conda create -n holly python
conda activate holly
pip install -r requirements.txt # -y doesnt always work sadly

mkdir experiment

echo -e "\U1F3CB " ${RED}TRAINING${NC} "\U1F3CB"

# Start the training
python3 train.py --obj ./objs/teapot_large.obj \
--train-size 40000 --lr 0.0004 --savedir experiment \
--num-points 230 --epochs 40 \
--sigma-file ./run/sigma.csv

# Generate the final stats
cd run
./generate_stats.sh -i ../experiment -g ../objs/teapot_large.obj
cd ..

echo "${YELLOW}Take a look in the experiment directory for the new outputs.${NC}"
