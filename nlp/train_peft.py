import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

batch_size = 32
model_name_or_path = "roberta-base"
task = "boolq"
peft_type = PeftType.LORA
device = "cuda"
num_epochs = 20

CACHE_DIR = "/home/mila/a/aristides.milios/scratch/hf_cache"

r = 2
lr = 3e-4
#lr = 3e-3 <== causes training to collapse

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=r, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "key", "value", "output.dense"])

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("super_glue", task, cache_dir=CACHE_DIR)
metric = evaluate.load("super_glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["question"], examples["passage"], truncation=True, max_length=None)
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question", "passage"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, cache_dir=CACHE_DIR)
model = get_peft_model(model, peft_config)

print(model)

#import pdb; pdb.set_trace()

for param in model.classifier.parameters():
    param.requires_grad = False

model.print_trainable_parameters()

optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model.to(device)
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)
