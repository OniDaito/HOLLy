Install
=======
.. _install:
There are two ways of installing HOLLy that we have included in the repository. The first is to use a virtual environment such as `Conda <https://docs.conda.io/en/latest/>`_, `PyEnv <https://github.com/pyenv/pyenv>`_ or `Virtual Env <https://pythonbasics.org/virtualenv/>`_. The second is to use a GPU enabled container with docker.

CUDA and GPU acceleration
#########################

HOLLy relies on having some form of GPU acceleration. Please refer to the `Pytorch Getting Started docs <https://pytorch.org/get-started/locally/>`_. Typically, if you can run the program **nvidia-smi** successfully, you are off to a good start.

via a virtual environment install
#################################

Once you've picked your choice of virtual environment, you can run **pip** to install the required packages. In the example below, I'm using `Miniconda <https://docs.conda.io/en/latest/miniconda.html/>`_:
::
    conda create -n holly python
    conda activate holly
    pip install -r requirements.txt


via the included docker
#######################

There is a dockerfile included in *docker/holly* that will create a container that you can use to train our network.

Assuming you have docker installed, you can create the docker container like so:
::
    docker build -t holly docker/holly


With this setup, you can run a test example : :ref:`usage`