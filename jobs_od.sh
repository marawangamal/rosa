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

#python eval_od.py -e /home/mila/m/marawan.gamal/scratch/lora-tensor/cppe-5/e10_l0.0001_b16_w0.0001_fTrue_mnone_r4_t0

# 3. Train
#python train_od.py pmodel.method=none train.lr=1e-3
#python train_od.py pmodel.method=none train.lr=1e-4
#python train_od.py pmodel.method=none train.lr=1e-5
#python train_od.py pmodel.method=none train.lr=1e-6

#python train_od.py pmodel.method=none train.lr=1e-4 train.batch_size=64
python train_od.py pmodel.method=none train.lr=2e-4 train.batch_size=64
python train_od.py pmodel.method=none train.lr=5e-4 train.batch_size=64
#python train_od.py pmodel.method=none train.lr=1e-4 train.batch_size=128
#python train_od.py pmodel.method=none train.lr=1e-4 train.batch_size=256
#python train_od.py pmodel.method=none train.lr=1e-4 train.batch_size=16

#python train_od.py pmodel.method=loraconv2d pmodel.rank=4
#python train_od.py pmodel.method=none
#python train_od.py pmodel.method=lora pmodel.rank=4