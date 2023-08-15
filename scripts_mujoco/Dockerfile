# Using the python as the base image which is based on ubuntu
FROM python:3.10
#can pass argument when buliding
ARG user1=rootUser
ENV CODE_DIR /root/code
COPY ./../src $CODE_DIR/baselines/src
COPY /scripts/setup.py $CODE_DIR/baselines
COPY ./scripts/requirements.txt $CODE_DIR/baselines
WORKDIR $CODE_DIR/baselines

COPY ./scripts/cuda-keyring_1.0-1_all.deb $CODE_DIR/baselines

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update \
    && apt-get install -y \
       apt-utils \
       build-essential \
       curl \
       vim \
       libglvnd0 \
       libgl1 \
       libglx0 \
       libegl1 \
       libxext6 \
       libx11-6 \
       libgl1-mesa-dev \
       libgl1-mesa-glx \
       libglew-dev \
       libosmesa6-dev \
       swig \
       software-properties-common \
       net-tools \
       vim \
       virtualenv \
       wget \
       xpra \
       xserver-xorg-dev \
       ffmpeg

ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}


COPY ./scripts/vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy
COPY ./scripts/vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Installing the mujoco binaries
RUN cd ~ && \
    mkdir .mujoco && \
    cd .mujoco && \
    mkdir mjpro150 && \
    curl -L -o mujoco.tar.gz https://github.com/deepmind/mujoco/releases/download/2.3.6/mujoco-2.3.6-linux-aarch64.tar.gz && \
    tar -xzf mujoco.tar.gz -C mjpro150 --strip-components=1 && \
    export PATH=$PATH:~/.mujoco/mjpro150/bin/
#Copy the license over
COPY ./scripts/mjkey.txt ~/.mujoco/mjpro150/
# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .[test]  && \
    python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

RUN echo "export PATH=$PATH:~/.mujoco/mujoco210/bin/ in your .bashrc file"
#have to somehow install mujuco in the image

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
CMD /bin/bash

#Not tested yet
