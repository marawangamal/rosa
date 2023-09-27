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
#cp -r /home/mila/m/marawan.gamal/.cache/huggingface $SLURM_TMPDIR/huggingface

# sst2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-5 fnmodel.factorize_freq=2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-4 fnmodel.factorize_freq=2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-3 fnmodel.factorize_freq=2
#e10_l0.0002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-08_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r2_leepoch_factrandom_factoequal_uFalse_t0

## axb
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=axb train.epochs=10 train.lr=2e-5

# rte
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=axb train.epochs=10  train.batch_size=16 fnmodel.name=lora fnmodel.params.bias_requires_grad=False fnmodel.params.rank=2 train.lr=2e-5
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=lora fnmodel.params.bias_requires_grad=False fnmodel.params.rank=2 train.lr=2e-4
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=lora fnmodel.params.bias_requires_grad=False fnmodel.params.rank=2 train.lr=2e-3
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=2 train.lr=2e-5
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=2 train.lr=2e-4
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=2 train.lr=2e-3
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=2 train.lr=2e-5 fnmodel.factorize_freq=2
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=2 train.lr=2e-4 fnmodel.factorize_freq=2
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=2 train.lr=2e-3 fnmodel.factorize_freq=2


#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=lora fnmodel.params.bias_requires_grad=False fnmodel.params.rank=8 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=lora fnmodel.params.bias_requires_grad=False fnmodel.params.rank=8 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=lora fnmodel.params.bias_requires_grad=False fnmodel.params.rank=8 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=8 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=8 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=8 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=8 train.lr=2e-5 fnmodel.factorize_freq=2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=8 train.lr=2e-4 fnmodel.factorize_freq=2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=rte train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.bias_requires_grad=False fnmodel.params.rank=8 train.lr=2e-3 fnmodel.factorize_freq=2

# Wandb sweep commands
# wandb sweep --project rosa-mlm-sweep sweep_mlm.yaml
# wandb agent tensor-lab/rosa-mlm-sweep/6isjw8pm --count 5
