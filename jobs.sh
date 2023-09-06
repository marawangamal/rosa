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

python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=none train.lr=5e-2 train.optimizer.name=sgd


python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=lora train.lr=5e-2 train.optimizer.name=sgd
python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=5e-2 train.optimizer.name=sgd


# Best models
# ROSA (2.298 PPL valid)
#python eval.py -e ~/scratch/rosa/runs/e2e_nlg/e5_l0.05_b10_f1.0_s512_mTrue_nsgd_mo0.9_w0.01_nalinear_nu500_namgpt2_namerosa_r0.01_leepoch_sarandom_cTrue_t0
#
## LoRA (2.238 PPL valid)
#python eval.py -e ~/scratch/rosa/runs/e2e_nlg/e5_l0.5_b10_f1.0_s512_mTrue_nsgd_mo0.9_w0.01_nalinear_nu500_namgpt2_namelora_r0.01_leepoch_sarandom_cTrue_t0
#
## FT (2.162 PPL valid)
#python eval.py -e ~/scratch/rosa/runs/e2e_nlg/e5_l0.05_b10_f1.0_s512_mTrue_nsgd_mo0.9_w0.01_nalinear_nu500_namgpt2_namenone_r0.01_leepoch_sarandom_cTrue_t0

#cd ~/scratch/rosa/runs/e2e_nlg/e5_l0.05_b10_f1.0_s512_mTrue_nsgd_mo0.9_w0.01_nalinear_nu500_namgpt2_namerosa_r0.01_leepoch_sarandom_cTrue_t0
#~/projects/e2e-metrics/measure_scores.py -p test_references.txt test_predictions_latest.txt >> metrics.txt
#
#cd ~/scratch/rosa/runs/e2e_nlg/e5_l0.5_b10_f1.0_s512_mTrue_nsgd_mo0.9_w0.01_nalinear_nu500_namgpt2_namelora_r0.01_leepoch_sarandom_cTrue_t0
#~/projects/e2e-metrics/measure_scores.py -p test_references.txt test_predictions_latest.txt >> metrics.txt
#
#cd ~/scratch/rosa/runs/e2e_nlg/e5_l0.05_b10_f1.0_s512_mTrue_nsgd_mo0.9_w0.01_nalinear_nu500_namgpt2_namenone_r0.01_leepoch_sarandom_cTrue_t0
#~/projects/e2e-metrics/measure_scores.py -p test_references.txt test_predictions_latest.txt >> metrics.txt


python train.py dataset.cache=$SLURM_TMPDIR/huggingface train.epochs=2 fnmodel.name=lora train.lr=0.05 train.optimizer.name=sgd