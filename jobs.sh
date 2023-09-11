#!/bin/bash
#SBATCH --output=/home/mila/m/marawan.gamal/projects/tensor-net/outputs/slurm-%j.out
#SBATCH --error=/home/mila/m/marawan.gamal/projects/tensor-net/outputs/slurm-error-%j.out
#SBATCH --mem=10G                                         # Ask for 10 GB of RAM
#SBATCH --time=8:00:00                                    # The job will run for 8 hours
#SBATCH -x cn-g[005-012,017-026]

# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source /home/mila/m/marawan.gamal/.venv/rosa/bin/activate

# 3. Copy your dataset on the compute node
cp -r /home/mila/m/marawan.gamal/.cache/huggingface $SLURM_TMPDIR/huggingface
python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=5e-5 train.optimizer.name=adamw fnmodel.params.rank=1



#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=lora train.lr=5e-7 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=lora train.lr=5e-6 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=lora train.lr=5e-5 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=lora train.lr=5e-4 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=lora train.lr=5e-3 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=lora train.lr=5e-2 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=lora train.lr=5e-1 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=5e-7 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=5e-6 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=5e-5 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=5e-4 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=5e-3 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=5e-2 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=5e-1 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=none train.lr=5e-7 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=none train.lr=5e-6 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=none train.lr=5e-5 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=none train.lr=5e-4 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=none train.lr=5e-3 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=none train.lr=5e-2 train.optimizer.name=sgd
#python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=none train.lr=5e-1 train.optimizer.name=sgd