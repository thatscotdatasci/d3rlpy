#! /bin/bash

#################################################
# Steps to create suitable environment on the HPC
#################################################

# Load the miniconda module
module load miniconda/3

# Create environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install HPC requirements
pip intall -r hpc_requirements.txt
