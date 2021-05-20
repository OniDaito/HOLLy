# Holly - a neural network for 3D Reconstruction from 2D Microscopy images

A [PyTorch](https://pytorch.org/) based neural network program designed to recontruct single molecules in 3D from a series of 2D storm images.

The goal of this network is to take a set of 2D representations of a molecule (or other simulated object) and create a 3D pointcloud of the molecule. We predict the pose and structure using a Convolutional Neural Network.

## Overview

It is split into *train.py* and *run.py* with the actual net stored in in *net/net.py*. *eval.py* will evaluate a trained network, creating statistics and visualistions. 

*net/renderer.py* contains the code for the differentiable renderer. *data/loader.py*, along with *data/buffer.py* and *data/batcher.py* create our simulated data for all the tests, including adding noise. *data/imageload.py* is similar, but for pre-rendered images.

## tl;dr version

Run the script *tldr.sh* from this directory. You will need miniconda and a computer capable of running a large neural network with Pytorch.

It will creata miniconda environment called "holly", download, install and start running an experiment, followed by a generation of results and a *final.ply* file representing the learned model. Once this script has completed, take a look at the *Outputs* section of this readme to understand what the network has produced.

## Installation

Taking it a little slower, there are various options for running our network using either miniconda or docker.

### Requirements

Requirements include:

* A Linux system setup to perform deeplearning, with appropriate drivers installed.
* Python version 3
* CUDA setup and running
* A GPU with at least 4G of memory (more if you are fitting more points).

And one of the following

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or a [pyenv/virtualenv](https://github.com/pyenv/pyenv) setup.
* [Docker](https://www.docker.com/) 

If you want to generate lots of statistics for use later you'll also need the following installed and running:

* [PostgreSQL](https://www.postgresql.org/)
* [Redis](https://redis.io/)
* [Imagemagick](https://imagemagick.org/index.php)

All of the python requirements are listed in the requirements.txt file (there aren't too many).

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

A test data set is available for download. Use the script *cep152_experiment.sh* in the *run* directory. This assumes you've installed the environment with either miniconda or docker. This script will download the data and once downloaded, start training a network.

The data used in the paper comes from [Suliana Manley's research group](https://www.epfl.ch/labs/leb/) and is available as a MATLAB/HDF5 file. In order to use it with this network, you will need to render the data to a series of FITs images. We have written a program to do this called [CEPrender](https://github.com/OniDaito/CEPrender). You can download and install this program, then generate the data as follows:

    cargo run --release --bin render /data/Cep152_all.mat /paper/10 24 10 10 cep_all_accepted.txt
    cargo run --release --bin render /data/Cep152_all.mat /paper/9 24 9 10 cep_all_accepted.txt
    ...
    ..
    .
    cargo run --release --bin render /data/Cep152_all.mat /paper/1.41 24 1.41 10 cep_all_accepted.txt

... filling the missing steps with the other sigma levels. This takes quite a while, even with only 10 augmentations. The pre-rendered data can be found [here]() complete the with the full instructions for generating it.

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

## Contributing

You can contribute to this project by submitting a pull request through github. Suggestions and feedback are welcomed through Github's Issue tracker. If you want to contact the authors please email Benjamin Blundell at benjamin.blundell@kcl.ac.uk. 

### Useful links

* Docker image based on - [https://github.com/anibali/docker-pytorch](https://github.com/anibali/docker-pytorch)
