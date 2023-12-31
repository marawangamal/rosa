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
    dataset.cache=/path/to/save/downloaded/dataset # path to save downloaded dataset
    output.path=/path/to/saved/runs  # path to save model checkpoints
    dataset.name=e2e # e2e or eli5
    model.name=gpt2  # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    fnmodel.name=rosa # rosa or lora
    train.epochs=5 
    train.batch_size=10 
    train.lr=5e-5
```

The E2E or ELI5 dataset will be downloaded and cached in the path specified 
by `dataset.cache`. The model will be saved in the path specified by `output.path`.

## Evaluation

To evaluate test metrics for all models, first clone the 
[e2e-metrics](https://github.com/tuetschek/e2e-metrics/tree/master) repository. 
Then, run the following commands:

```commandline
export E2E_METRICS_EXEC=/path/to/e2e-metrics/measure_scores.py
python eval.py -r path/to/saved/runs // generate predictions 
bash eval.sh $E2E_METRICS_EXEC path/to/saved/runs // evaluate predictions
python compile.py path/to/saved/runs // compile results of all models
```

with `path/to/saved/runs` being the root containing the saved runs of the model(s) in the paper.


To evaluate test metrics for a single model, first clone run:

```commandline
export E2E_METRICS_EXEC=/path/to/e2e-metrics/measure_scores.py
python eval.py -e /path/to/saved/experiment
$E2E_METRICS_EXEC -p /path/to/references.txt path/to/predictions.txt
```

[//]: # (export E2E_METRICS_EXEC=/home/mila/m/marawan.gamal/projects/e2e-metrics/measure_scores.py)

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
