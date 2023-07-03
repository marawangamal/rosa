# ROSA: Random Orthogonal Subspace Adaptation
This repository is the official implementation of [ROSA: Random Orthogonal Subspace Adaptation](). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train ROSA model(s) in the paper, run this command:

```commandline
python train.py 
    dataset.cache=/path/to/save/downloaded/dataset
    output.path=/path/to/save/model/weights
    dataset.name=e2e 
    train.epochs=5 
    train.batch_size=128 
    train.lr=1e-3 
    fnmodel.name=rosa
```

To train LoRA model(s) in the paper, run this command:

```commandline
python train.py 
    dataset.cache=/output/path/to/save/downloaded/dataset
    output.path=/output/path/to/save/model/weights
    dataset.name=e2e 
    train.epochs=5 
    train.batch_size=128 
    train.lr=1e-3 
    fnmodel.name=lora
```

The E2E or ELI5 dataset will be downloaded and cached in the path specified 
by `dataset.cache`. The model will be saved in the path specified by `output.path`.

## Evaluation

To visualize train/validation curves of model(s) in the paper, run:

```commandline
tensorboard --logdir=/output/path/to/saved/model/weights
```

## Recreating the results
To run all experiments in the paper simultaneously on SLURM, run:

```commandline
python jobrunner.py --fn jobs.yaml
```

Monitor the status of the jobs with:

```commandline
python jobrunner.py -s
```

## Citation
If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{hameed2023rosa,
  title={ROSA: Random Orthogonal Subspace Adaptation},
  author={Marawan Gamal Abdel Hameed and Guillaume Rabusseau}
  maintitle = {International Conference on Machine Learning},
  booktitle = {Efficient Systems for Foundation Models},
  year={2023}
}
```
