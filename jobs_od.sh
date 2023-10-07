#!/bin/bash
#SBATCH --output=/home/mila/m/marawan.gamal/projects/tensor-net/outputs/slurm-%j.out
#SBATCH --error=/home/mila/m/marawan.gamal/projects/tensor-net/outputs/slurm-error-%j.out
#SBATCH --mem=10G                                         # Ask for 10 GB of RAM
#SBATCH --time=8:00:00                                    # The job will run for 8 hours
#SBATCH -x cn-g[005-012,017-026]

# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source ~/scratch/.venv/rosa/bin/activate


# 3. Train
python train_object_detection.py pmodel.method=none
python train_object_detection.py pmodel.method=loraconv2d pmodel.rank=4
python train_object_detection.py pmodel.method=lora pmodel.rank=4