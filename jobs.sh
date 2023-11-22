#source /home/mila/m/marawan.gamal/scratch/.venv/rosa/bin/activate
#cp -r /home/mila/m/marawan.gamal/.cache/huggingface $SLURM_TMPDIR/huggingface
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=stsb train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-3 fnmodel.factorize_freq=20 fnmodel.params.factorize_method=add
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=stsb train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-5 fnmodel.factorize_freq=20 fnmodel.params.factorize_method=add
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=stsb train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-4 fnmodel.factorize_freq=20
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=stsb train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-5 fnmodel.factorize_freq=20
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=sst2 train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-3 fnmodel.factorize_freq=2
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=stsb train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-4 fnmodel.factorize_freq=2
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=stsb train.epochs=10  train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=8 train.lr=2e-5 fnmodel.factorize_freq=2
