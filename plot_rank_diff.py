import copy
import os
import os.path as osp
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
import evaluate as eval_lib
from transformers import default_data_collator
from transformers import AutoTokenizer, get_scheduler
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

from utils.utils import preprocess_function_mlm, get_ignore_list_glue, set_seeds

import peftnet as pn
from peftnet.peft_module.rosalinear import RosaLinear
from peftnet.peft_module.loralinear import LoraLinear
import pandas as pd


base = {
    "train": "train",
    "validation": "validation",
    "test": "validation"
}

task_to_split = {
    "mnli": {
        "train": "train",
        "validation": "validation_matched",
        "test": "validation_matched"
    },
    "qnli": base.copy(),
    "stsb": base.copy(),
    "cola": base.copy(),
    "rte": base.copy(),
    "mrpc": base.copy(),
    "sst2": base.copy(),
    "qqp": base.copy(),
    "wnli": base.copy(),
    "axb": base.copy(),
    "axg": base.copy(),
    "boolq": base.copy(),
    "cb": base.copy(),
    "copa": base.copy(),
    "multirc": base.copy(),
    "record": base.copy(),
    "wic": base.copy(),
    "wsc.fixed": base.copy(),
}


def get_dataloaders(args, tokenizer):
    # Load dataset
    assert args['dataset']['name'] in ["glue", "super_glue"], "Dataset not supported"
    assert args['dataset']['task_name'] in task_to_split.keys(), "Task not supported"

    train_dataset = load_dataset(
        args['dataset']['name'], args['dataset']['task_name'],
        split=task_to_split[args['dataset']['task_name']]['train'],
        cache_dir=args['dataset']['cache']
    )
    test_dataset = load_dataset(
        args['dataset']['name'], args['dataset']['task_name'],
        split=task_to_split[args['dataset']['task_name']]['test'],
        cache_dir=args['dataset']['cache']
    )
    valid_dataset = load_dataset(
        args['dataset']['name'], args['dataset']['task_name'],
        split=task_to_split[args['dataset']['task_name']]['validation'],
        cache_dir=args['dataset']['cache']
    )

    # Filter for faster training (debug)
    num_train_pts, _ = train_dataset.shape
    train_dataset = train_dataset.select(range(int(num_train_pts * args['train']['fraction'])))

    # Apply tokenizer to dataset
    train_tokenized = train_dataset.map(
        lambda examples: preprocess_function_mlm(
            examples, tokenizer, task_name=args['dataset']['task_name'], max_length=args['train']['seq_len']
        ),
        # batched=True,
        # num_proc=4,
    )

    valid_tokenized = valid_dataset.map(
        lambda examples: preprocess_function_mlm(
            examples, tokenizer, task_name=args['dataset']['task_name'], max_length=args['train']['seq_len']),
        # batched=True
    )

    test_tokenized = test_dataset.map(
        lambda examples: preprocess_function_mlm(
            examples, tokenizer, task_name=args['dataset']['task_name'], max_length=args['train']['seq_len']),
        # batched=True
    )

    # Only include tokenized ids
    train_tokenized_reduced = train_tokenized.remove_columns(train_dataset.column_names)
    valid_tokenized_reduced = valid_tokenized.remove_columns(valid_dataset.column_names)
    test_tokenized_reduced = test_tokenized.remove_columns(test_dataset.column_names)

    # use default data collator
    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_tokenized_reduced, shuffle=True, batch_size=args['train']['batch_size'], collate_fn=data_collator,
        pin_memory=True, num_workers=1
    )
    valid_dataloader = DataLoader(
        valid_tokenized_reduced, batch_size=args['train']['batch_size'], collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        test_tokenized_reduced, batch_size=args['train']['batch_size'], collate_fn=data_collator
    )

    # tweak for now since we can't evaluate on test (it's private there's no labels)
    # return train_dataloader, valid_dataloader, test_dataloader, test_dataset
    return train_dataloader, valid_dataloader, None, None


def main():

    out_path = "figures/singular_values_diff_lora.pdf"
    arr_path = "figures/singular_values_diff_lora.npy"
    # Load model from
    ckpt_root = "/home/mila/m/marawan.gamal/scratch/rosa/runs/glue/cola"
    # exp = "e10_l0.002_b32_f1.0_s512_nadamw_be0.9_0.98_ep1e-08_w0.1_nalinear_wa0.06_namroberta-base_namerosa_fa1_facepoch_iTrue_r8_leepoch_factrandom_factoequal_uFalse_t0"
    exp = "e10_l0.0002_b32_f1.0_s512_nadamw_be0.9_0.98_ep1e-08_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r8_leepoch_factrandom_factoequal_uFalse_t1"
    # exp = "e10_l0.0002_b32_f1.0_s512_nadamw_be0.9_0.98_ep1e-08_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r8_leepoch_factrandom_factoequal_uFalse_t0"
    ckpt = os.path.join(ckpt_root, exp)
    # Load args.pkl
    args = pickle.load(open(os.path.join(ckpt, 'args.pkl'), 'rb'))
    dct_best = torch.load(os.path.join(ckpt, 'model_best.pth'), map_location='cpu')

    tokenizer = AutoTokenizer.from_pretrained(
        args['model']['name'],
        cache_dir=args['dataset']['cache'],
        use_fast=True
    )

    train_dataloader, valid_dataloader, test_dataloader, test_dataset = get_dataloaders(args, tokenizer)
    is_regression = args['dataset']['task_name'] == 'stsb'
    if is_regression:
        num_labels = 1
    else:
        num_labels = len(train_dataloader.dataset.unique('labels'))

    config = AutoConfig.from_pretrained(
        args['model']['name'],
        cache_dir=args['dataset']['cache'],
        num_labels=num_labels,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args['model']['name'],
        config=config,
        cache_dir=args['dataset']['cache']
    )

    ignore_list = get_ignore_list_glue(model) if args['fnmodel']['ignore_list'] else None
    cmodel = {
        "rosa": pn.RosaNet,
        "lora": pn.LoraNet,
        "ia3": pn.IA3Net,
        "none": lambda x, **kwargs: x
    }[args['fnmodel']['name'].lower()](copy.deepcopy(model), ignore_list=ignore_list, **args['fnmodel']['params'])

    cmodel.load_state_dict(dct_best['model_state_dict'])
    cmodel_merged = cmodel.merge()
    # cmodel_merged = cmodel

    with torch.no_grad():

        cumsum_singular_vals = {}
        for name, module in cmodel_merged.named_modules():
            if isinstance(module, RosaLinear) or isinstance(module, LoraLinear):
                corresponding_layer_name = name.replace("peft_model.", "") + ".weight"
                w_t = module.w
                w_0 = model.state_dict()[corresponding_layer_name]
                delta_t = w_t - w_0.T

                _, s, _ = torch.svd(delta_t)
                s_norm = s / torch.sum(s)
                cumsum_singular_vals[name] = torch.cumsum(s_norm, dim=0).cpu().numpy()

        # Check if any layers matched
        if not cumsum_singular_vals:
            print("No layers matched the regular expression.")
            return

        # Convert the dictionary to a 2D NumPy array
        max_length = max(len(vals) for vals in cumsum_singular_vals.values())
        array_data = np.zeros((len(cumsum_singular_vals), max_length))
        row_labels = []

        for i, (name, vals) in enumerate(cumsum_singular_vals.items()):
            array_data[i, :len(vals)] = vals
            row_labels.append(name)

        # Plot
        # plt.figure(figsize=(15, 8))
        # plt.imshow(array_data, interpolation='none', aspect='auto', cmap='viridis')

        # save array to arr.npy
        np.save(arr_path, array_data)

        plt.imshow(array_data[:80], interpolation='none')
        plt.colorbar()
        plt.yticks(np.arange(len(row_labels)), labels=[])
        plt.xlabel('Singular Value Index')
        plt.ylabel('Layer')
        plt.title('Cumulative Sum of Singular Values in Model Layers')
        plt.savefig(out_path)


if __name__ == '__main__':
    main()