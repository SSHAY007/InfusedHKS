# Using the python as the base image
FROM python:3.10
#can pass argument when buliding
ARG user1=rootUser

RUN apt-get update \
    && apt-get install -y \
       apt-utils \
       build-essential \
       curl \
       vim \
       ffmpeg


ENV CODE_DIR /root/code

COPY . $CODE_DIR/baselines
WORKDIR $CODE_DIR/baselines

# Installing the mujoco binaries
RUN cd ~ && \
    mkdir .mujoco && \
    cd .mujoco && \
    mkdir mujoco210 && \
    curl -L -o mujoco.tar.gz https://github.com/deepmind/mujoco/releases/download/2.3.6/mujoco-2.3.6-linux-aarch64.tar.gz && \
    tar -xzf mujoco.tar.gz -C mujoco210 --strip-components=1 && \
    export PATH=$PATH:~/.mujoco/mujoco210/bin/
# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .[test]  && \
    python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

RUN echo "export PATH=$PATH:~/.mujoco/mujoco210/bin/ in your .bashrc file"
#have to somehow install mujuco in the image
CMD /bin/bash

#Not tested yet
