""" Evaluate BLEU for FactorizedNet or LoraNet model

Example:
python eval_lorank.py eval.experiment=runs/e2e_nlg/e64_l1e-05_b32_f1.0_nsgd_m0.9_w0.01_nanone_nu100_namdistillgpt2_namelora_r0.1_leepoch_srandom_t0

"""
import os
import os.path as osp
import uuid
import logging
import torch.nn as nn

import csv
import argparse
from tqdm import tqdm
import string

import torch

import factorizednet as fn
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate
from utils import load_object, AverageMeter
from itertools import groupby


def get_data(dataset_name, dataset_cache):
    # Load dataset
    assert dataset_name in ["eli5", "e2e_nlg"], "Dataset not supported"

    test_split = {"eli5": "validation_asks", "e2e_nlg": "test"}[dataset_name]

    test_dataset = load_dataset(
        dataset_name, split=test_split, cache_dir=dataset_cache
    ).flatten()
    return test_dataset


def evaluate_model(cmodel, output_path_preds, output_path_refs, test_dataset, tokenizer, device=None):
    # Evaluate model
    model_fn = cmodel.module if isinstance(cmodel, nn.DataParallel) else cmodel
    model_fn = model_fn.factorized_model if (isinstance(model_fn, fn.RosaNet) or isinstance(model_fn, fn.LoraNet)) \
        else model_fn
    predictor = pipeline(
        'text-generation',
        model=model_fn,
        tokenizer=tokenizer,
        device=device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    logging.info("=> Testing model bleu scores (Device={}) ...".format(predictor.device))
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
                current_mr = datapoint['meaning_representation']
                input_str = "Input: {} Output: ".format(current_mr)

                output_str = predictor(
                    input_str,
                    return_full_text=False,
                    # length_penalty=0.8,
                    no_repeat_ngram_size=4,
                    num_beams=5,
                    max_length=512,
                    # early_stopping=True,
                )[0]['generated_text'].strip().replace("\xa0", " ")

                if output_str == "":
                    output_str = "NONE"

                # Write to CSV
                writer.writerow([output_str])


def evaluate_model_bleu(cmodel, test_dataset, tokenizer, device=None, batch_size=32):
    with torch.no_grad():
        logging.info("=> Testing model bleu scores (Device={}) ...".format(device))
        uuid_str = str(uuid.uuid1())
        BLEU = evaluate.load("bleu", experiment_id=uuid_str, cache_dir="~/.cache/huggingface/evaluate/{}".format(uuid_str))
        bleu_average_meter = AverageMeter()

        # Initialize model
        cmodel.to(device)
        model_fn = cmodel.module if isinstance(cmodel, nn.DataParallel) else cmodel
        model_fn = model_fn.factorized_model if (
                    isinstance(model_fn, fn.RosaNet) or isinstance(model_fn, fn.LoraNet)) else model_fn
        model_fn.eval()

        # Sort dataset by 'meaning_representation' to ensure all similar items are together
        sorted_dataset = sorted(test_dataset, key=lambda x: x['meaning_representation'])

        # Group the sorted dataset by 'meaning_representation'
        grouped_data = [list(group) for key, group in groupby(sorted_dataset, key=lambda x: x['meaning_representation'])]

        # Combine all references for each group
        grouped_data = [
            {
                "meaning_representation": group[0]['meaning_representation'],
                "human_reference": [item['human_reference'] for item in group]
            }
            for group in grouped_data
        ]

        num_pts = len(grouped_data)
        for i in tqdm(range(0, num_pts, batch_size)):
            batch = grouped_data[i:i + batch_size]

            input_strs = ["Input: {} Output: ".format(dp['meaning_representation']) for dp in batch]
            references = [item['human_reference'] for item in batch]

            inputs = tokenizer(
                input_strs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            output_ids = model_fn.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                no_repeat_ngram_size=4,
                num_beams=5,
                max_length=512,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            output_strs = [
                out.replace("Input: {} Output: ".format(dp['meaning_representation']), "").strip() for out, dp in
                zip(output_strs, batch)
            ]

            # Compute BLEU
            for output_str, reference in zip(output_strs, references):
                if len(" ".strip()) == 0:
                    bleu_score = 0
                else:
                    results = BLEU.compute(predictions=[output_str], references=[reference])
                    bleu_score = results["bleu"]
                bleu_average_meter.add(bleu_score)

        return {
            "bleu": bleu_average_meter.value,
        }


def evaluate_model_bleu_datapoint(cmodel, test_dataset, tokenizer, device=None, batch_size=32):
    BLEU = evaluate.load("bleu")
    bleu_average_meter = AverageMeter()

    # Initialize model
    model_fn = cmodel.module if isinstance(cmodel, nn.DataParallel) else cmodel
    model_fn = model_fn.factorized_model if (
                isinstance(model_fn, fn.RosaNet) or isinstance(model_fn, fn.LoraNet)) else model_fn

    # Set the device
    device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_fn.to(device)

    # Create predictor
    predictor = pipeline(
        'text-generation',
        model=model_fn,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    num_data_points = len(test_dataset)

    for i in tqdm(range(0, num_data_points, batch_size)):
        batch = test_dataset[i:i + batch_size]

        input_strs = ["Input: {} Output: ".format(dp['meaning_representation']) for dp in batch]

        # Generate batch outputs
        output_strs = predictor(
            input_strs,
            return_full_text=False,
            no_repeat_ngram_size=4,
            num_beams=5,
            max_length=512,
        )

        for idx, output in enumerate(output_strs):
            references = [batch[idx]['human_reference']]
            output_text = output['generated_text'].strip().replace("\xa0", " ")

            # Compute BLEU
            results = BLEU.compute(predictions=[output_text], references=[references])
            bleu_score = results["bleu"]
            bleu_average_meter.add(bleu_score)

    return {
        "bleu": bleu_average_meter.value,
    }


def evaluate_model_bleu_old(cmodel, test_dataset, tokenizer, device=None, batch_size=32):
    with torch.no_grad():
        BLEU = evaluate.load("bleu")
        bleu_average_meter = AverageMeter()

        # Initialize model
        model_fn = cmodel.module if isinstance(cmodel, nn.DataParallel) else cmodel
        model_fn = model_fn.factorized_model if (
                    isinstance(model_fn, fn.RosaNet) or isinstance(model_fn, fn.LoraNet)) else model_fn
        model_fn.eval()

        predictor = pipeline(
            'text-generation',
            model=model_fn,
            tokenizer=tokenizer,
            device=device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


        # Sort dataset by 'meaning_representation' to ensure all similar items are together
        sorted_dataset = sorted(test_dataset, key=lambda x: x['meaning_representation'])

        # Group the sorted dataset by 'meaning_representation'
        grouped_data = [list(group) for key, group in groupby(sorted_dataset, key=lambda x: x['meaning_representation'])]

        # Combine all references for each group
        grouped_data = [
            {
                "meaning_representation": group[0]['meaning_representation'],
                "human_reference": [item['human_reference'] for item in group]
            }
            for group in grouped_data
        ]

        num_pts = len(grouped_data)
        for i in tqdm(range(0, num_pts, batch_size)):
            batch = grouped_data[i:i + batch_size]

            input_strs = ["Input: {} Output: ".format(dp['meaning_representation']) for dp in batch]
            references = [item['human_reference'] for item in batch]

            output_strs = predictor(
                input_strs,
                return_full_text=False,
                no_repeat_ngram_size=4,
                num_beams=5,
                max_length=512,
            )

            output_strs = [
                output[0]['generated_text'].strip().replace("\xa0", " ") for output in output_strs
            ]

            # Compute BLEU
            bleu_score_sum = sum([BLEU.compute(predictions=[output_str], references=[reference])["bleu"] for output_str, reference in zip(output_strs, references)])
            bleu_average_meter.add(bleu_score_sum, n=len(batch))

        return {
            "bleu": bleu_average_meter.value,
        }


# https://stackoverflow.com/questions/76465343/huggingface-transformers-model-config-reported-this-is-a-deprecated-strategy-to
# @hydra.main(version_base=None, config_path="./", config_name="configs")
def evaluate_experiment(experiment_root, test_dataset, overwrite=False, min_records=620, all=False):
    experiment_args = load_object(osp.join(experiment_root, "args.pkl"))
    model_names = [name for name in os.listdir(experiment_root) if name.startswith("model_") and name.endswith(".pth")]

    if all:
        print("\t=> Evaluating all {} models".format(len(model_names)))
    else:
        model_names = ["model_latest.pth"]
        print("\t=> Evaluating latest model only".format(len(model_names)))

    for model_name in model_names:
        output_path_refs = osp.join(experiment_root, "test_references.txt")
        output_filename = model_name.replace("model_", "test_predictions_").replace(".pth", ".txt")
        output_path_preds = osp.join(experiment_root, output_filename)

        # Check number of lines in output_path_preds
        if osp.exists(output_path_preds) and not overwrite:
            # ~/scratch/rosa/runs/e2e_nlg/e5_l0.0005_b10_f1.0_s512_mTrue_nadamw_mo0.9_w0.01_nalinear_nu500_namgpt2_namenone_r0.5_leepoch_sarandom_cTrue_t0
            with open(output_path_preds, "r") as f:
                num_lines = sum(1 for _ in f)
            if num_lines >= min_records:
                print("\t=> Skipping evaluation. {} already exists and has {} records".format(output_path_preds,
                                                                                              num_lines))
                continue

        # Define model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForCausalLM.from_pretrained(experiment_args['model']['name'])
        tokenizer = AutoTokenizer.from_pretrained(experiment_args['model']['name'])
        tokenizer.pad_token = tokenizer.eos_token

        # Factorize & Load pretrained model
        cmodel = {
            "rosa": fn.RosaNet, "lora": fn.LoraNet, "none": lambda x, **kwargs: x
        }[experiment_args['fnmodel']['name'].lower()](model, **experiment_args['fnmodel']['params'])

        print("\t=> Loading model {} ...".format(model_name))
        print("\t=> Using device {}".format(device))
        dct_best = torch.load(osp.join(experiment_root, model_name))
        cmodel.load_state_dict(dct_best['model_state_dict'])
        cmodel.to(device)

        if experiment_args['dataset']['name'] != "e2e_nlg":
            raise NotImplementedError("Dataset {} not supported".format(experiment_args['dataset']['name']))

        evaluate_model(cmodel, output_path_preds, output_path_refs, test_dataset, tokenizer, device)


def main(args):
    test_dataset = get_data(args.dataset, args.cache)
    if args.experiment == '':
        experiments = [osp.join(args.root, d) for d in os.listdir(args.root) if osp.isdir(osp.join(args.root, d))]
        for i, experiment in enumerate(experiments):
            print("\n=> [{}/{}] Evaluating experiment {}".format(i + 1, len(experiments), experiment))
            evaluate_experiment(experiment, test_dataset=test_dataset, all=args.all)
    else:
        print("=> Generating predictions for experiment {}".format(args.experiment))
        evaluate_experiment(args.experiment, test_dataset=test_dataset, all=args.all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, default='', required=False, help='Experiment directory')
    parser.add_argument('-a', '--all', action='store_true', help='Evaluate all model.pth weights')
    parser.add_argument('-r', '--root', type=str, default='', help='Root directory of many experiments')
    parser.add_argument('-d', '--dataset', type=str, default='e2e_nlg', help='Dataset name')
    parser.add_argument('-c', '--cache', type=str, default='/home/mila/m/marawan.gamal/.cache/huggingface',
                        help='Dataset cache directory')
    args = parser.parse_args()

    assert args.experiment != '' or args.root != '', "Either experiment or root must be specified"
    main(args)
