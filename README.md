# ROSA: Random Orthogonal Subspace Adaptation
This repository is the official implementation of [ROSA: Random Orthogonal Subspace Adaptation](https://openreview.net/forum?id=4P9vOFpb63). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train ROSA/LoRA model(s) in the paper, run this command:

```commandline
python train.py 
    dataset.cache=/path/to/save/downloaded/dataset
    output.path=/path/to/saved/runs
    dataset.name=e2e 
    train.epochs=5 
    train.batch_size=128 
    train.lr=1e-3 
    fnmodel.name=<rosa or lora>
```

The E2E or ELI5 dataset will be downloaded and cached in the path specified 
by `dataset.cache`. The model will be saved in the path specified by `output.path`.

## Evaluation

To evaluate test metrics for all models, run the following commands:

```commandline
python eval.py path/to/saved/runs // generate predictions 
bash eval.sh path/to/saved/runs // compute metrics
python compile_results.py path/to/saved/runs // compile results of all models
```

with `path/to/saved/runs` being the root containing the saved runs of the model(s) in the paper.


## Visualize train/validation curves of model(s)
Run the following command to visualize the train/validation curves of model(s) in the paper:

```commandline
tensorboard --logdir=/path/to/saved/runs
```

To create figures of model(s) in the paper for e2e or eli5 dataset, run:

```commandline
python figures.py --fn /path/to/saved/runs/for/dataset/<dataset_name>
```
with `<dataset_name>` being `e2e_nlg` or `eli5`.


### Evaluate metrics of model(s) on test set

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
