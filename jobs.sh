#!/bin/bash
#SBATCH --output=/home/mila/m/marawan.gamal/projects/tensor-net/outputs/slurm-%j.out
#SBATCH --error=/home/mila/m/marawan.gamal/projects/tensor-net/outputs/slurm-error-%j.out
#SBATCH --mem=10G                                         # Ask for 10 GB of RAM
#SBATCH --time=8:00:00                                    # The job will run for 8 hours
#SBATCH -x cn-g[005-012,017-026]

# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source /home/mila/m/marawan.gamal/.venv/tn/bin/activate

# 3. Copy your dataset on the compute node
cp -r /home/mila/m/marawan.gamal/.cache/huggingface $SLURM_TMPDIR/huggingface


# 4. Run your code
python train.py dataset.cache=$SLURM_TMPDIR/huggingface train.epochs=5 train.lr=5e-5 train.scheduler.name=linear train.batch_size=10 train.optimizer.name=adamw model.name=gpt2-medium fnmodel.name=none


# 5. Evaluate your results
python eval.py eval.experiment=/home/mila/m/marawan.gamal/scratch/rosa/runs/e2e_nlg/e5_l5e-05_b5_f1.0_s512_mTrue_nadamw_mo0.9_w0.01_nalinear_nu500_namgpt2_namenone_r0.5_leepoch_sarandom_cTrue_t0