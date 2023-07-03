""" Evaluate BLEU for FactorizedNet or LoraNet model

Example:
python eval_lorank.py eval.experiment=runs/e2e_nlg/e64_l1e-05_b32_f1.0_nsgd_m0.9_w0.01_nanone_nu100_namdistillgpt2_namelora_r0.1_leepoch_srandom_t0

"""

import os.path as osp
import math
import logging
import torch.nn as nn

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from utils import AverageMeter, load_object, BLEU

import factorizednet as fn


def get_dataloaders(args):
    # Load dataset
    assert args['dataset']['name'] in ["eli5", "e2e_nlg"], "Dataset not supported"

    valid_split = {"eli5": "validation_asks", "e2e_nlg": "test"}[args['dataset']['name']]

    valid_dataset = load_dataset(
        args['dataset']['name'], split=valid_split, cache_dir=args['dataset']['cache']
    ).flatten()
    # valid_dataset = valid_dataset.remove_columns(valid_dataset.column_names)
    return valid_dataset


def preprocess_function(examples, tokenizer, dataset_name="eli5", max_length=512):
    """Concat all questions/answers into one text and tokenize them afterwards."""

    if dataset_name == "eli5":
        return tokenizer([" ".join(x) for x in examples["answers.text"]])
    elif dataset_name == "e2e_nlg":
        output = tokenizer(
            [" ".join([x, y]) for x, y in zip(examples['meaning_representation'], examples['human_reference'])],
            max_length=max_length,
            truncation=True,
        )
        return output
    else:
        raise NotImplementedError


def group_texts(examples, block_size=128):
    # Concatenate all texts across batches. {ids: [List_1, .., List_N]} => [*List_1, ..., *List_N]
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.
    result = {
        column_name: [column_vals[i: i + block_size] for i in range(0, total_length, block_size)]
        for column_name, column_vals in concatenated_examples.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result


def bpc_fn(inputs, outputs, targets):
    # Compute bpc
    loss = outputs.loss.mean()
    bpc = (loss / math.log(2))
    return bpc


def ppl_fn(inputs, outputs, targets):
    # Compute ppl
    loss = outputs.loss.mean()
    ppl = torch.exp(loss)
    return ppl


def evaluate_fn(model, device, eval_dataloader, metric_fns):
    with torch.no_grad():

        average_meters = {k: AverageMeter() for k in metric_fns.keys()}

        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], labels=batch['labels'])

            for k in metric_fns.keys():
                average_meters[k].add(metric_fns[k](batch['input_ids'], batch['labels'], outputs))

    return {k: v.value for k, v in average_meters.items()}


@hydra.main(version_base=None, config_path="./", config_name="lorank_configs")
def main(cfg: DictConfig):
    # Convert config to dict
    eval_args = OmegaConf.to_container(cfg, resolve=True)
    args = load_object(osp.join(eval_args['eval']['experiment'], "args.pkl"))

    # Load dataset
    # tokenizer, test_dataloader, test_dataset = get_dataloaders(args)

    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(args['model']['name'])
    tokenizer = AutoTokenizer.from_pretrained(args['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    test_dataset = get_dataloaders(args)

    # Factorize model
    logging.info("=> Using {} model ...".format(args['tnmodel']['name'].lower()))

    # Regular model
    cmodel = {
        "factorized": fn.RosaNet, "lora": fn.LoraNet, "none": lambda x, **kwargs: x
    }[args['tnmodel']['name'].lower()](model, **args['tnmodel']['params'])
    dct_best = torch.load(osp.join(eval_args['eval']['experiment'], "model_best.pth"))
    cmodel.load_state_dict(dct_best['model_state_dict'])
    cmodel.to(device)

    # Evaluate model
    logging.info("=> Evaluating model ...")

    # Bleu score
    model_fn = cmodel.module if isinstance(cmodel, nn.DataParallel) else cmodel
    model_fn = model_fn.factorized_model if isinstance(model_fn, fn.RosaNet) else model_fn
    model_fn = model_fn.factorized_model if isinstance(model_fn, fn.LoraNet) else model_fn
    bleu_fn = BLEU(osp.join(eval_args['eval']['experiment'], "eval.csv"))

    bleu_avg_meter = AverageMeter()

    for i, x in enumerate(test_dataset):
        prompt = x['meaning_representation']
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        output_ids = model_fn.generate(
            input_ids,
            max_length=512,
            num_beams=10,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        # output_ids = model_fn.generate(input_ids, max_length=512, num_beams=10,no_repeat_ngram_size=2, early_stopping=True, pad_token_id=tokenizer.eos_token_id)

        output_str = tokenizer.decode(output_ids[0])[len(tokenizer.decode(input_ids[0])):].strip()
        results = bleu_fn(predictions=[output_str], references=[x['human_reference']], outpath=None)
        bleu_avg_meter.add(results['bleu'])
        print("[{}/{}] BLEU: {}\n\tPrompt: {}\n\tReference: {}\n\tOutput: {}".format(
            i + 1, len(test_dataset), results['bleu'], prompt, x['human_reference'], output_str
        ))

    print("Evaluation finished! ({})".format(eval_args['eval']['experiment']))
    print("=> (AVG) BLEU: {}".format(bleu_avg_meter.value))


if __name__ == '__main__':
    main()
