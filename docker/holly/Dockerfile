FROM nvidia/cuda:11.0-base-ubuntu20.04
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    imagemagick \
    libpq-dev \
    postgresql \
    build-essential \
    python3-pip \
    graphviz \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Holly specific steps
RUN pip3 install torch astropy numpy pillow psycopg2 redis pyquaternion pywavefront tqdm scipy rich graphviz
# Set the default command to python3
CMD ["python3"]