#!/bin/bash

ENVNAME='./fmnist_env'
if ! [ -x "$(command -v conda)" ]; then
  echo "Could not find 'conda'. Please make sure it is installed for generating the virtual environment."
  exit
fi

if [ -d "$ENVNAME" ]; then
  echo "Found existing conda environment in local directory './fmnist_env'"
else
  echo "Creating conda environment in local directory './fmnist_env'"
  conda env create --prefix $ENVNAME -f fmnist_env.yml
fi

source activate $ENVNAME

# standard gauss prior
python fmnist_experiment.py --mb-size 32 --latent-dim 35 --train-model