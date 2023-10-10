# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source ~/scratch/.venv/rosa/bin/activate

# 3. Train

# LoRA
python train_od.py pmodel.method=lora pmodel.rank=8 train.lr=5e-5 train.batch_size=8 train.epochs=10
python train_od.py pmodel.method=lora pmodel.rank=8 train.lr=5e-4 train.batch_size=8 train.epochs=10
python train_od.py pmodel.method=lora pmodel.rank=8 train.lr=5e-6 train.batch_size=8 train.epochs=10

# LoRA with 1D conv
#python train_od.py pmodel.method=lorafull pmodel.rank=8 pmodel.ignore_regex=.*conv2.* train.lr=5e-5 train.batch_size=8 train.epochs=5
#python train_od.py pmodel.method=lorafull pmodel.rank=8 pmodel.ignore_regex=.*conv2.* train.lr=5e-6 train.batch_size=8 train.epochs=5

# LoRA with 1D conv + 2D conv
#python train_od.py pmodel.method=lorafull pmodel.rank=4 train.lr=5e-5 train.batch_size=8 train.epochs=5
#python train_od.py pmodel.method=lorafull pmodel.rank=16 train.lr=5e-5 train.batch_size=8 train.epochs=5
#python train_od.py pmodel.method=lorafull pmodel.rank=32 train.lr=5e-5 train.batch_size=8 train.epochs=5
#python train_od.py pmodel.method=lorafull pmodel.rank=8 train.lr=5e-6 train.batch_size=8 train.epochs=5

