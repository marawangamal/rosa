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
python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=2e-3 fnmodel.params.factorize_mode=bottom fnmodel.params.rank=2 train.epochs=5
python train.py dataset.cache=$SLURM_TMPDIR/huggingface fnmodel.name=rosa train.lr=2e-3 fnmodel.params.factorize_mode=top fnmodel.params.rank=2 train.epochs=5


## Eval
#python eval.py -e ~/scratch/rosa/runs/e2e_nlg/e5_l0.0002_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namelora_r0.01_leepoch_sarandom_cTrue_t0
#python eval.py -e ~/scratch/rosa/runs/e2e_nlg/e5_l0.0002_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namerosa_r0.01_leepoch_sarandom_cTrue_t0
##python eval.py -e ~/scratch/rosa/runs/e2e_nlg/e5_l5e-05_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namenone_r0.01_leepoch_sarandom_cTrue_t0
#
export E2E_METRICS_EXEC=/home/mila/m/marawan.gamal/projects/e2e-metrics/measure_scores.py
$E2E_METRICS_EXEC -p refs_5.txt preds_5.txt
#cd ~/scratch/rosa/runs/e2e_nlg/e5_l0.002_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namelora_r0.01_leepoch_sarandom_cTrue_t0
#$E2E_METRICS_EXEC -p test_references.txt test_predictions_best.txt >> metrics.txt
#
#cd ~/scratch/rosa/runs/e2e_nlg/e5_l0.002_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namerosa_r0.01_leepoch_sarandom_cTrue_t0
#$E2E_METRICS_EXEC -p test_references.txt test_predictions_best.txt >> metrics.txt
#
#cd ~/scratch/rosa/runs/e2e_nlg/e5_l5e-05_b10_f1.0_s512_iTrue_nadamw_m0.9_w0.01_nalinear_nu500_namgpt2_namenone_r0.01_leepoch_sarandom_cTrue_t0
#$E2E_METRICS_EXEC -p test_references.txt test_predictions_best.txt >> metrics.txt





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