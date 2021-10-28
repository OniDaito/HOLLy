Usage
=====
.. _usage:

To use HOLLy, you need to train the network against a particular set of images. These images should be in `FITS format <https://fits.gsfc.nasa.gov/>`_. For testing purposes though, HOLLy can also accept a path to a `Wavefront OBJ file <https://en.wikipedia.org/wiki/Wavefront_.obj_file>`_ or a `PLY file <https://en.wikipedia.org/wiki/PLY_(file_format)>`_, generating images from these objects on the fly.


Using a virtual environment
###########################

We can simulate data from a Wavefront OBJ file, several of which are included with this repository. Alternatively, you can provide your own. 

To train a new network on simulated data, run the following command in the top level of the project, assuming you are inside the miniconda environment
::
    python train.py --obj objs/teapot.obj --train-size 80000 --lr 0.0004 --savedir /tmp/runs/test_run --num-points 230 --no-translation --no-data-translate --epochs 20 --sigma-file run/sigma_quick.csv

It is also possible to use the provided bash script to execute training in a given directory, saving the various options and code versions to various files. This is more useful if you want to run many tests, saving the various settings.
::
    cd run
    ./train.sh <path of output file>

The bash script *train.sh* looks for a file called *run.conf* that contains the required data for training a network. The file *run.conf.example* can be copied to *run.conf*. Make sure the directory referred to for saving the results exists.

Using Docker
############

Using docker, one can run the same command as follows
::
    mkdir experiment
    docker run --gpus=all --volume="$PWD:/app" holly python train.py --obj objs/teapot.obj --train-size 80000 --lr 0.0004 --savedir /app/experiment --num-points 230 --no-translate --no-data-translate --epochs 20 --sigma-file run/sigma_quick.csv

Confirm that docker can see the gpu
::
    docker run --gpus all holly nvidia-smi

Your GPU should be listed.


Working with images
###################

To use real, experimental data one must have a number of images, rendered using a particular sigma. These images need to be held within a directory stucture that matches the sigma file passed in. For example
::
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

The sigma file (sigma_images.csv) would be
::
    10,8,1.2

Images should be in FITS Format(they support floating point values), and be 128x128 in size. Future versions of the software will support variable image sizes.

We would then run the following command:
    python train.py --fitspath images --train-size 80000 --lr 0.0004 --savedir /tmp/runs/test_run --num-points 230 --no-translation --epochs 20 --sigma-file sigma_images.csv

A test data set is available for download from [Zenodo](https://zenodo.org/record/4751057). This is a big download - just under 50G! However it does have all the data ready for use, pre-rendered.  

Once you've donwloaded and unzipped the data, place all the images into a directory structure like this:
::
    holly
    |___run
           |___paper
                    |____10.0
                    |____8
                    etc
                    ...

Use the script *run_cp.sh* in the *run* directory. This assumes you've installed the environment with either miniconda or docker. This script assumes you have downloaded the dataset and placed it the correct directory.

Assuming you have 4000 images per sigma level, and a sigma file called **sigma.csv** that matches the directory structure of the image directory, the command to run HOLLy would look like this
::
    python train.py --fitspath ./images --train-size 4000 --lr 0.0004 --savedir /tmp/runs/test_run --num-points 230 --epochs 20 --sigma-file run/sigma.csv


Further options
###############

There are a number of advanced options one can play with. For example, one can set the level of noise in the simulated data using **--wobble N** where N is a number between 0 and 1. More options can be found in the **train.py** file.


Testing a trained net
#####################

Once a network has been trained, we can test how well it predicts the pose of an object from a single input image, using the **run.py** file. Assuming we have a trained net in the **trained_net** directory, an input image called **test.fits** and a final points set called **./trained_net/last.ply** the command would look like this
::
    python run.py --load ./trained_net --image test.fits --points ./trained_net/last.ply


