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

# MRPC
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=none train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=none train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 fnmodel.factorize_freq=2 train.lr=2e-3

#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=8 train.lr=2e-6 **
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=8 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=8 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=8 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=lora fnmodel.params.rank=8 train.lr=2e-2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=mrpc train.epochs=10 train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 fnmodel.factorize_freq=2  train.lr=2e-3


# BOOLQ
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=none train.lr=2e-6
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=none train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=none train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=none train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=none train.lr=2e-2


#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=1 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=1 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=1 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=1 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=1 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=1 train.lr=2e-3
#
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=2 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-3
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=2 train.lr=2e-3 fnmodel.factorize_freq=2

#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=4 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=4 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=4 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=4 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=4 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=4 train.lr=2e-3

#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=8 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=8 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=lora fnmodel.params.rank=8 train.lr=2e-3
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-5
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-3
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-2
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=boolq train.epochs=10 train.batch_size=32 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-3 fnmodel.factorize_freq=2




# Wandb sweep commands
# wandb sweep --project rosa-mlm-sweep sweep_mlm.yaml
# wandb agent tensor-lab/rosa-mlm-sweep/6isjw8pm --count 5

## Eval commands
#python evaltools.py -e ~/scratch/rosa/runs/e2e_nlg/e5_l0.0002_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namelora_r0.01_leepoch_sarandom_cTrue_t0
#python evaltools.py -e ~/scratch/rosa/runs/e2e_nlg/e5_l0.0002_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namerosa_r0.01_leepoch_sarandom_cTrue_t0
##python evaltools.py -e ~/scratch/rosa/runs/e2e_nlg/e5_l5e-05_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namenone_r0.01_leepoch_sarandom_cTrue_t0
#
#export E2E_METRICS_EXEC=/home/mila/m/marawan.gamal/projects/e2e-metrics/measure_scores.py
#$E2E_METRICS_EXEC -p refs_5.txt preds_5.txt
#cd ~/scratch/rosa/runs/e2e_nlg/e5_l0.002_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namelora_r0.01_leepoch_sarandom_cTrue_t0
#$E2E_METRICS_EXEC -p test_references.txt test_predictions_best.txt >> metrics.txt
#
#cd ~/scratch/rosa/runs/e2e_nlg/e5_l0.002_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namerosa_r0.01_leepoch_sarandom_cTrue_t0
#$E2E_METRICS_EXEC -p test_references.txt test_predictions_best.txt >> metrics.txt
#
#cd ~/scratch/rosa/runs/e2e_nlg/e5_l5e-05_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namenone_r0.01_leepoch_sarandom_cTrue_t0
#$E2E_METRICS_EXEC -p test_references.txt test_predictions_best.txt >> metrics.txt