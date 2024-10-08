#!/bin/bash

ENV_NAME=pytorch_lac
CONDA_HOME=$(which conda)
source ${CONDA_HOME::-10}/etc/profile.d/conda.sh
conda create -n $ENV_NAME -y python=3.9
conda activate $ENV_NAME
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

CUR_DIR=$PWD
# cd#  $CUR_DIR/applications
python setup.py install
# cd $CUR_DIR