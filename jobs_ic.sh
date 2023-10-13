#SBATCH --error=/home/mila/m/marawan.gamal/projects/tensor-net/outputs/slurm-error-%j.out
#SBATCH --mem=10G                                         # Ask for 10 GB of RAM
#SBATCH --time=8:00:00                                    # The job will run for 8 hours
#SBATCH -x cn-g[005-012,017-026]

# 1. Load the required modules
#module load python/3.8

# 2. Load your environment
#source /home/mila/m/marawan.gamal/.venv/rosa/bin/activate

# 3. Copy your dataset on the compute node

# Last FT
#python train_ic.py pmodel.method=last train.lr=5e-5
#python train_ic.py pmodel.method=last train.lr=5e-4
#python train_ic.py pmodel.method=last train.lr=5e-3
#
## Full FT
#python train_ic.py train.lr=5e-5
#python train_ic.py train.lr=5e-4
#python train_ic.py train.lr=5e-3
#
## rank 2
#python train_ic.py pmodel.method=loraconv2d pmodel.rank=2 train.lr=5e-3
#python train_ic.py pmodel.method=loraconv2d pmodel.rank=2 train.lr=5e-4
#python train_ic.py pmodel.method=loraconv2d pmodel.rank=2 train.lr=5e-5
#
## rank 4
#python train_ic.py pmodel.method=loraconv2d pmodel.rank=2 train.lr=5e-3
#python train_ic.py pmodel.method=loraconv2d pmodel.rank=2 train.lr=5e-4
#python train_ic.py pmodel.method=loraconv2d pmodel.rank=2 train.lr=5e-5

# rank 2
python train_ic.py pmodel.method=loraconv2d pmodel.ignore=only1d pmodel.rank=2 train.lr=5e-3
python train_ic.py pmodel.method=loraconv2d pmodel.ignore=only1d pmodel.rank=2 train.lr=5e-4
python train_ic.py pmodel.method=loraconv2d pmodel.ignore=only1d pmodel.rank=2 train.lr=5e-5

# rank 4
python train_ic.py pmodel.method=loraconv2d pmodel.ignore=only1d pmodel.rank=2 train.lr=5e-3
python train_ic.py pmodel.method=loraconv2d pmodel.ignore=only1d pmodel.rank=2 train.lr=5e-4
python train_ic.py pmodel.method=loraconv2d pmodel.ignore=only1d pmodel.rank=2 train.lr=5e-5