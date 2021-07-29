# HOLLy - Hypothesised Object from Light Localisations
## A neural network for 3D Reconstruction from 2D Microscopy images

![Evolving structure](https://shutr.benjamin.computer/static/shutr_files/original/num2021_05_14_cep_0.gif)

A [PyTorch](https://pytorch.org/) based neural network program designed to recontruct single molecules in 3D from a series of 2D storm images.

The goal of this network is to take a set of 2D representations of a molecule (or other simulated object) and create a 3D pointcloud of the molecule. We predict the pose and structure using a Convolutional Neural Network.

## Overview

HOLLy takes a number of images, or generates images from a ground-truth point-cloud, trains on these attempting to improve it's own internal representation of what it thinks the ground-truth object is. At the end of training, an obj or ply file will be produced, representing the object the network converged upon.

HOLLy is split into *train.py* and *run.py* with the actual net stored in in *net/net.py*. *eval.py* will evaluate a trained network, creating statistics and visualistions. 

*net/renderer.py* contains the code for the differentiable renderer. *data/loader.py*, along with *data/buffer.py* and *data/batcher.py* create our simulated data for all the tests, including adding noise. *data/imageload.py* is similar, but for pre-rendered images.

## Installation

### Requirements

Requirements include:

* A Linux system setup to perform deeplearning, with appropriate nvidia drivers installed.
* Python version 3
* CUDA setup and running
* A GPU with at least 4G of memory (more if you are fitting more points).

And one of the following

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
* [pyenv/virtualenv](https://github.com/pyenv/pyenv).
* [Docker](https://www.docker.com/).

If you want to generate lots of statistics for use later you'll also need all of the following installed and running. Initially, you shouldn't need these but you might see a few warnings pop up:

* [PostgreSQL](https://www.postgresql.org/)
* [Redis](https://redis.io/)

If you want some fancy formatted results or if you are running the tl;dr script, install the following:

* [Imagemagick](https://imagemagick.org/index.php)
* [ffmpeg](https://ffmpeg.org/)

Chances are, if you are running a Linux distribution, you will have these already and if not, they'll be in your repository management system (like apt or pacman).

All of the python requirements are listed in the requirements.txt file (there aren't too many).

### tl;dr version

Run the script *tldr.sh* from the top directory. You will need [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed and a computer running Linux, capable of trainin a large neural network with Pytorch. As a guide, a machine with several cores, >=16G memory and an nVida GPU such as a GTX 2080Ti will work reasonably well.

The script creates a miniconda environment called "holly", downloads packages and starts running an experiment, followed by a generation of results and a *final.ply* file representing the learned model. Once this script has completed, take a look at the *Outputs* section of this readme to understand what the network has produced.

### via miniconda

Assuming you have [installed miniconda](https://docs.conda.io/en/latest/miniconda.html), start with the following:

    conda create -n holly python
    conda activate holly
    pip install -r requirements.txt

### via docker

There is a dockerfile included in *docker/holly* that will create a container that you can use to train our network.

Assuming you have docker installed, you can create the docker container like so:

    docker build -t holly docker/holly

## Training

### Simulated data

We can simulate data from a [wavefront obj file](https://en.wikipedia.org/wiki/Wavefront_.obj_file), several of which are included with this repository. Alternatively, you can provide your own. 

To train a new network on simulated data, run the following command in the top level of the project, assuming you are inside the miniconda environment:

    python train.py --obj objs/teapot.obj --train-size 80000 --lr 0.0004 --savedir /tmp/runs/test_run --num-points 230 --no-translation --no-data-translate --epochs 20 --sigma-file run/sigma_quick.csv

It is also possible to use the provided bash script to execute training in a given directory, saving the various options and code versions to various files. This is more useful if you want to run many tests, saving the various settings.

    cd run
    ./train.sh <path of output file>

The bash script *train.sh* looks for a file called *run.conf* that contains the required data for training a network. The file *run.conf.example* can be copied to *run.conf*. Make sure the directory referred to for saving the results exists.

Using docker, one can run the same command as follows:

    mkdir experiment
    docker run --gpus=all --volume="$PWD:/app" holly python train.py --obj objs/teapot.obj --train-size 80000 --lr 0.0004 --savedir /app/experiment --num-points 230 --no-translate --no-data-translate --epochs 20 --sigma-file run/sigma_quick.csv

Confirm that docker can see the gpu:
    docker run --gpus all holly nvidia-smi

Your GPU should be listed.

Docker needs the [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) package in order to access gpus. On Ubuntu 20.04 LTS one needs to run the following commands:

    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list > /etc/apt/sources.list.d/nvidia-docker.list
    apt update
    apt -y install nvidia-docker2
    systemctl restart docker 

There are a large number of options you can pass to the network. These are listed in the *train.py* file. The most important are listed below:

* --save-stats - save on-going train and test statistics.
* --obj - when simulating data, this is the path to the object file we want to use.
* --train-size - the size of the training set
* --lr - the learning rate
* --num-points - how many points do we want to try and fit to the structure.
* --epochs - how many epochs should we run for?
* --sigma-file - the path to the file containing the per-epoch sigmas.

Training usually takes around 4 hours on a nVidia 2080Ti system when running for 20 epochs on an 80,000 size training set.

The final structure discerned by the network is saved as both an obj and a ply file. Both can be viewed in a program like [Meshlab](https://www.meshlab.net/) or [Blender](https://www.blender.org/). The files can be found in:

    <save_directory>/plys/shape_e<epoch>_s<step>.ply
    <save_directory>/objs/shape_e<epoch>_s<step>.obj

### Real Image data

To use real, experimental data one must have a number of images, rendered using a particular sigma. These images need to be held within a directory stucture that matches the sigma file passed in. For example:

    images
    |___10.0
        |____image0.fits
        |____image1.fits
    |___8
        |____image0.fits
        |____image1.fits
    |___1.2
        |____image0.fits
        |____image1.fits

The sigma file (sigma_images.csv) would be:
    10,8,1.2

Images should be in [FITS Format](https://en.wikipedia.org/wiki/FITS) (they support floating point values), and be 128x128 in size. Future versions of the software will support variable image sizes.

We would then run the following command:
    python train.py --fitspath images --train-size 80000 --lr 0.0004 --savedir /tmp/runs/test_run --num-points 230 --no-translation --epochs 20 --sigma-file sigma_images.csv

A test data set is available for download from [Zenodo](https://zenodo.org/record/4751057). This is a big download - just under 50G! However it does have all the data ready for use, pre-rendered.  

Once you've donwloaded and unzipped the data, place all the images into a directory structure like this:

    holly
    |___run
           |___paper
                    |____10.0
                    |____8
                    etc
                    ...

Use the script *run_cp.sh* in the *run* directory. This assumes you've installed the environment with either miniconda or docker. This script assumes you have downloaded the dataset and placed it the correct directory.

If you want to see the results of a real CEP152 run - the ones from our paper - you can download them from [Zenodo](https://zenodo.org/record/4836173) too!

#### Generating your own images from STORM localisations.

The data used in the paper comes from [Suliana Manley's research group](https://www.epfl.ch/labs/leb/) and is available as a MATLAB/HDF5 file. In order to use it with this network, you will need to render the data to a series of FITs images. We have written a program to do this called [CEPrender](https://github.com/OniDaito/CEPrender). You can download and install this program, then generate the data as follows:

    cargo run --release --bin render /data/Cep152_all.mat /paper/10 24 10 10 cep_all_accepted.txt
    cargo run --release --bin render /data/Cep152_all.mat /paper/9 24 9 10 cep_all_accepted.txt
    ...
    ..
    .
    cargo run --release --bin render /data/Cep152_all.mat /paper/1.41 24 1.41 10 cep_all_accepted.txt

... filling the missing steps with the other sigma levels. This takes quite a while, even with only 10 augmentations. The pre-rendered data can be found on [Zenodo](https://zenodo.org/record/4751057) complete the with the full instructions for generating it.

## Outputs

Once you have a trained network, you can generate the final outputs using the *generate_stats.sh* script found in the *run* directory.

    ./generate_stats.sh  -i <path to your results directory> -g <path to object file> -a -z

So with the test example given above:
    
    ./generate_stats.sh  -i /tmp/runs/test_run -g ../objs/teapot.obj -a -z

This relies on having imagemagick 'montage' command and the correct python libs installed.

This script loads the network and tests it against the ground_truth (if we are using simulated data)

## Training and Test statistics

Due to the nature of this work, we used [Jupyter Notebooks](https://jupyter.org/), Redis and PostgreSQL to save our various statistics, using the code found in *stats/stats.py*. By default, these statistics include:

* Train and test errors.
* Input translations and rotations that make up the dataset.
* Output translations and rotations the network decided upon.
* Input and output sigma.

The training machine must be running an instance of either Redis or PostgreSQL. If both are running, the program will use both. Imagemagick and ffmpeg are used to create montage images and animations.

At some point in the future, we might move to [Tensorboard](https://www.tensorflow.org/tensorboard/) and / or [Weights and Biases](wandb.ai/).

## Tests

Our code comes with a considerable number of tests. These can be run from the top level directory as follows:

    python test.py

Individual tests can be run as follows:

    python -m unittest test.data.Data.test_wobble

## Our published paper and data

Our paper "3D Structure from SMLM images using Deep Learning" is under review at the moment and hopefully will be available soon.

The CEP152 data we used in the paper with HOLLy can be found on Zenodo at [https://zenodo.org/record/4751057](https://zenodo.org/record/4751057) and the results we mention in the paper can also be downloaded from Zenodo at [https://zenodo.org/record/4836173](https://zenodo.org/record/4836173).


## Command line and configuration options

When running train.py, there are a number of options one can choose.

    --batch-size
    Input batch size for training (default: 20).

    --epochs
    Number of epochs to train (default: 10).

    --lr
    Learning rate (default: 0.0004).

    --mask-thresh
    Threshold for what we consider in the loss (default: 0.05)

    --plr
    Learning rate for points (default: same as the learning rate).

    --spawn-rate
    Probabilty of spawning a point (default: 1.0).

    --max-trans
    The scalar on the translation we generate and predict (default: 0.1).

    --max-spawn
    How many flurophores are spawned total (default: 1).

    --save-stats
    Save the stats of the training for later graphing.

    --predict-sigma
    Predict the sigma (default: False).

    --no-cuda
    Disables CUDA training.

    --deterministic
    Run deterministically.

    --no-translate",
    Turn off translation prediction in the network (default: false).

    --no-data-translate
    Turn off translation in the data loader(default: false).

    --normalise-basic
    Normalise with torch basic intensity divide.

    --scheduler
    Use a scheduler on the loss.

    --seed
    Random seed (default: 1).

    --cont
    Continuous sigma values

    --log-interval
    How many batches to wait before logging training status (default 100)

    --num-points
    How many points to optimise (default 200).

    --aug
    Do we augment the data with XY rotation (default False)?

    --num-aug
    How many augmentations to perform per datum (default 10).

    --save-interval
    How many batches to wait before saving (default 1000).

    --load
    A checkpoint file to load in order to continue training.

    "--savename
    The name for checkpoint save file.

    --savedir
    The name for checkpoint save directory.

    --allocfile
    An optional data order allocation file.

    --sigma-file
    Optional file for the sigma blur dropoff.

    --dropout
    When coupled with objpath, what is the chance of a point being dropped? (default 0.0)

    --wobble
    Distance to wobble our fluorophores (default 0.0)

    --fitspath
    Path to a directory of FITS files.

    --objpath
    Path to the obj for generating data
    
    --train-size
    The size of the training set (default: 50000)

    --image-size
    The size of the images involved, assuming square (default: 128).

    --test-size
    The size of the training set (default: 200)

    --valid-size
    The size of the training set (default: 200)

    --buffer-size
    How big is the buffer in images?  (default: 40000)
    

## Contributing

You can contribute to this project by submitting a pull request through github. Suggestions and feedback are welcomed through Github's Issue tracker. If you want to contact the authors please email Benjamin Blundell at benjamin.blundell@kcl.ac.uk. 

### Useful links

* Docker image based on - [https://github.com/anibali/docker-pytorch](https://github.com/anibali/docker-pytorch)

*"And the moral of the story; appreciate what you've got, because basically, I'm fantastic!" - Holly. Red Dwarf*
