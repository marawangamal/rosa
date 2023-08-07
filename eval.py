""" Evaluate BLEU for FactorizedNet or LoraNet model

Example:
python eval_lorank.py eval.experiment=runs/e2e_nlg/e64_l1e-05_b32_f1.0_nsgd_m0.9_w0.01_nanone_nu100_namdistillgpt2_namelora_r0.1_leepoch_srandom_t0

"""

import os.path as osp
import math
import logging
import torch.nn as nn

import csv
import argparse
from tqdm import tqdm
import string

import torch

import factorizednet as fn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import load_object


def get_data(experiment_args):
    # Load dataset
    assert experiment_args['dataset']['name'] in ["eli5", "e2e_nlg"], "Dataset not supported"

    test_split = {"eli5": "validation_asks", "e2e_nlg": "test"}[experiment_args['dataset']['name']]

    test_dataset = load_dataset(
        experiment_args['dataset']['name'], split=test_split, cache_dir=experiment_args['dataset']['cache']
    ).flatten()
    return test_dataset


# @hydra.main(version_base=None, config_path="./", config_name="configs")
def evaluate_experiment(experiment_root):
    # Convert config to dict
    experiment_args = load_object(osp.join(experiment_root, "args.pkl"))

    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = AutoModelForCausalLM.from_pretrained(experiment_args['model']['name'])
    tokenizer = AutoTokenizer.from_pretrained(experiment_args['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    test_dataset = get_data(experiment_args)
    logging.info("=> Using {} model ...".format(experiment_args['fnmodel']['name'].lower()))

    # Factorize model
    cmodel = {
        "factorized": fn.RosaNet, "lora": fn.LoraNet, "none": lambda x, **kwargs: x
    }[experiment_args['fnmodel']['name'].lower()](model, **experiment_args['fnmodel']['params'])
    dct_best = torch.load(osp.join(experiment_root, "model_best.pth"))
    cmodel.load_state_dict(dct_best['model_state_dict'])
    cmodel.to(device)

    # Evaluate model
    logging.info("=> Evaluating model ...")
    model_fn = cmodel.module if isinstance(cmodel, nn.DataParallel) else cmodel
    model_fn = model_fn.factorized_model if (isinstance(model_fn, fn.RosaNet) or isinstance(model_fn, fn.LoraNet)) \
        else model_fn

    predictor = pipeline(
        'text-generation',
        model=model_fn,
        tokenizer=tokenizer,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if experiment_args['dataset']['name'] == "e2e_nlg":

        output_path_refs = osp.join(experiment_root, "e2e_test_references.txt")
        output_path_preds = osp.join(experiment_root, "e2e_test_predictions.txt")

        with open(output_path_refs, "w", newline="") as f:
            current_mr = ""
            for i, datapoint in enumerate(tqdm(test_dataset, total=len(test_dataset))):
                if datapoint['meaning_representation'] != current_mr:
                    if i != 0:
                        f.write("\n")
                    current_mr = datapoint['meaning_representation']
                hr_with_spaces = datapoint['human_reference'].translate(
                    str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
                f.write(hr_with_spaces + "\n")

        with open(output_path_preds, "w", newline="") as f:
            writer = csv.writer(f)
            current_mr = ""
            for datapoint in tqdm(test_dataset, total=len(test_dataset)):
                if datapoint['meaning_representation'] != current_mr:
                    current_mr = datapoint['meaning_representation'].replace('"', "")
                    input_str = "Input: {} Output: ".format(current_mr)

                    output_str = predictor(
                        input_str,
                        return_full_text=False,
                        length_penalty=0.8,
                        no_repeat_ngram_size=4,
                        repetition_penalty=1.0,
                        num_beams=10,
                        num_return_sequences=1,
                        max_length=512,
                    )[0]['generated_text'].strip().replace("\xa0", " ")

                    if output_str == "":
                        output_str = "NONE"

                    # Write to CSV
                    writer.writerow([output_str])

    else:
        raise NotImplementedError("Dataset {} not supported".format(experiment_args['dataset']['name']))


def main(args):
    evaluate_experiment(args.experiment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, required=True, help='Experiment directory')
    args = parser.parse_args()
    main(args)
