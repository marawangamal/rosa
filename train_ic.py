import os
import os.path as osp

import wandb
from typing import Optional, List, Dict

import hydra
import evaluate
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
from transformers import ResNetForImageClassification

from transformers import TrainingArguments
from transformers import Trainer
import numpy as np

import peftnet as pn
from utils.utils_cv import get_dataloaders_ic, get_dataloaders_ic_hf
from utils.utils import get_experiment_name, set_seeds, save_object, transform_dict
accuracy = evaluate.load("accuracy")

# https://huggingface.co/microsoft/resnet-50
# https://huggingface.co/docs/transformers/tasks/image_classification


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def get_ignore_list(model, mode='none'):
    """Ignore Conv2d layers with kernel size > 1."""
    ignore_list = []
    if mode.lower() == 'none':
        return ignore_list
    elif mode.lower() == 'only1d':
        for name, module in model.named_modules():
            # If primitive layer
            if isinstance(module, nn.Conv2d) and any([k > 1 for k in module.kernel_size]):
                ignore_list.append(name)
        return ignore_list
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of ['none', 'only1d'].")


@hydra.main(version_base=None, config_path="configs", config_name="conf_ic")
def main(cfg: DictConfig):

    # Experiment tracking and logging
    args = OmegaConf.to_container(cfg, resolve=True)
    if args['override']:
        args = transform_dict(args)
        print("=> Overriding arguments")
    print(OmegaConf.to_yaml(cfg))

    # Load dataset
    model_checkpoint = "microsoft/resnet-50"
    train_dataset, test_dataset, data_collator, image_processor, id2label, label2id, labels = get_dataloaders_ic(
        name=args['dataset']['name'],
        root=args['dataset']['root'],
        image_processor_checkpoint=model_checkpoint,
    )

    for t in range(max(1, args["runs"])):

        # Set diff seeds for each run
        if args['seed'] > 0:
            set_seeds(int(args['seed'] + t))

        # Experiment path
        experiment_name = get_experiment_name(
            {"train": args["train"], "pmodel": args["pmodel"], "trial": t}, mode='str'
        )
        experiment_path = osp.join(args['output']['path'], args['dataset']['name'])
        experiment_path = osp.join(experiment_path, "hp_search") if args['hp_search'] else experiment_path
        output_path = osp.join(experiment_path, experiment_name)

        wandb.init(
            name=experiment_name, config=args, project='lora-tensor-classification-{}'.format(args['dataset']['name'])
        )

        if not osp.exists(output_path):
            os.makedirs(output_path)

        save_object(args, osp.join(output_path, 'args.pkl'))

        model = ResNetForImageClassification.from_pretrained(
            model_checkpoint,
            # id2label=id2label,
            # label2id=label2id,
            num_labels=len(labels),
            ignore_mismatched_sizes=True,
        )

        # PEFT Model
        print("-" * 100)
        if args['pmodel']['method'].lower() == 'last':
            for param in model.parameters():
                param.requires_grad = False
        elif args['pmodel']['method'].lower() != 'none':
            print(f"=> Training PEFTNet model: {args['pmodel']['method']}")
            # cmodel = pn.PEFTNet(model, ignore_regex=".*conv(1|3).*", **args['pmodel'])
            ignore_list = get_ignore_list(model, mode=args['pmodel']['ignore'])
            cmodel = pn.PEFTNet(model, ignore_list=ignore_list, **args['pmodel'])
            model = cmodel.peft_model
        else:
            print("=> Training baseline model")

        # Make last layer trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Print model
        print(model)
        print(f"\n{pn.PEFTNet.get_report(model)}\n")

        # Print ratio of different layer types to total number of layers
        print(f"=> Ratio of different layer types to total number of layers:")
        n_conv_params = 0
        n_trainable_conv_params = 0
        n_linear_params = 0
        n_trainable_linear_params = 0
        n_total_params = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # in millions

        for name, module in model.named_modules():
            # If primitive layer
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if isinstance(module, nn.Conv2d) and any([k > 1 for k in module.kernel_size]):
                    n_conv_params += module.weight.numel()
                    n_trainable_conv_params += module.weight.numel() if module.weight.requires_grad else 0
                else:
                    n_linear_params += module.weight.numel()
                    n_trainable_linear_params += module.weight.numel() if module.weight.requires_grad else 0

        n_conv_params /= 1e6
        n_linear_params /= 1e6
        print(f"=> Total number of parameters: {n_total_params}")
        print(f"=> # Conv2d params: {n_conv_params}M (ratio: {n_conv_params / n_total_params:.3f})")
        print(f"=> # Linear params: {n_linear_params}M (ratio: {n_linear_params / n_total_params:.3f})")
        print(f"=> # Trainable params: {n_trainable_params:.3f}M (ratio: {n_trainable_params / n_total_params:.3f})")

        wandb.log({
            "n_trainable_params": n_trainable_params,
            "n_total_params": n_total_params,
        })

        training_args = TrainingArguments(
            output_dir=output_path,
            run_name=experiment_name,
            report_to='wandb',
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            eval_steps=5,
            save_strategy="epoch",
            learning_rate=args['train']['lr'],
            per_device_train_batch_size=16,
            # gradient_accumulation_steps=4,
            # per_device_eval_batch_size=16,
            num_train_epochs=args['train']['epochs'],
            warmup_ratio=args['train']['warmup_ratio'],
            logging_strategy="steps",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            # push_to_hub=True,
        )

        print("=> Running training")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_path)


if __name__ == "__main__":
    main()