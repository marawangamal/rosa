import os
import os.path as osp
import math
import subprocess
import time
import logging
import torch.nn as nn

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, load_metric
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, get_scheduler
from transformers import AutoModelForCausalLM

from utils import get_num_params, get_experiment_name, get_latency, AverageMeter, save_object, LatencyReport, \
    CudaMemoryTracker, preprocess_function_mlm
from eval import evaluate_model_bleu

import factorizednet as fn
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:.3f}'.format)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_dataloaders(args, tokenizer):
    # Load dataset
    assert args['dataset']['name'] == 'glue', "Dataset not supported"
    assert args['dataset']['task_name'] in task_to_keys.keys(), "Task not supported"

    train_dataset = load_dataset(
        args['dataset']['name'], args['dataset']['task_name'], split="train",
        cache_dir=args['dataset']['cache']
    )
    test_dataset = load_dataset(
        args['dataset']['name'], args['dataset']['task_name'], split="test",
        cache_dir=args['dataset']['cache']
    )
    valid_dataset = load_dataset(
        args['dataset']['name'], args['dataset']['task_name'],
        split="validation", cache_dir=args['dataset']['cache']
    )

    # Filter for faster training (debug)
    num_train_pts, _ = train_dataset.shape
    train_dataset = train_dataset.select(range(int(num_train_pts * args['train']['fraction'])))

    # Apply tokenizer to dataset
    train_tokenized = train_dataset.map(
        lambda examples: preprocess_function_mlm(
            examples, tokenizer, task_name=args['dataset']['task_name'], max_length=args['train']['seq_len']
        ),
        batched=True,
        # num_proc=4,
    )

    valid_tokenized = valid_dataset.map(
        lambda examples: preprocess_function_mlm(
            examples, tokenizer, task_name=args['dataset']['task_name'], max_length=args['train']['seq_len']),
        batched=True
    )

    test_tokenized = test_dataset.map(
        lambda examples: preprocess_function_mlm(
            examples, tokenizer, task_name=args['dataset']['task_name'], max_length=args['train']['seq_len']),
        batched=True
    )

    # Only include tokenized ids
    train_tokenized_reduced = train_tokenized.remove_columns(train_dataset.column_names)
    valid_tokenized_reduced = valid_tokenized.remove_columns(valid_dataset.column_names)
    test_tokenized_reduced = test_tokenized.remove_columns(test_dataset.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, 
                                            return_tensors="pt", padding=True, pad_to_multiple_of=8)

    train_dataloader = DataLoader(
        train_tokenized_reduced, shuffle=True, batch_size=args['train']['batch_size'], collate_fn=data_collator,
        pin_memory=True, num_workers=2
    )
    valid_dataloader = DataLoader(
        valid_tokenized_reduced, batch_size=args['train']['batch_size'], collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        test_tokenized_reduced, batch_size=args['train']['batch_size'], collate_fn=data_collator
    )

    return train_dataloader, valid_dataloader, test_dataloader, test_dataset


def evaluate(model, device, eval_dataloader, task="cola"):
    glue_metric = load_metric('glue', task)

    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions.extend(torch.argmax(outputs.logits, dim=-1).tolist())
            references.extend(batch['labels'].tolist())
    
    return glue_metric.compute(predictions=predictions, references=references)

def sample_trainable(args, model, lr_scheduler, optimizer, steps_counter, num_training_steps):
    # Mask gradients
    model = model.module.sample_trainable() if isinstance(model, nn.DataParallel) else model.sample_trainable()

    # New optimizer
    opt_cls = optimizer.__class__
    del optimizer
    optimizer = opt_cls(
        model.parameters(),
        lr=args["train"]["lr"]
    )

    # New scheduler
    if lr_scheduler is not None:
        del lr_scheduler
        lr_scheduler = get_scheduler(
            name=args['train']['scheduler']['name'],
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            **args['train']['scheduler']['params']
        )
        for i in range(steps_counter):
            lr_scheduler.step()

    return model, optimizer, lr_scheduler


def mark_only_rosa_or_lora_as_trainable(model, verbose=False):
    for name, param in model.named_parameters():
        if ("rosa" in name and "fixed" not in name) or "lora" in name:
            param.requires_grad = True
            if verbose:
                print("+ Marked {} as trainable".format(name))
        else:
            if verbose:
                print("- Marked {} as not trainable".format(name))
            param.requires_grad = False


def get_num_trainable_params(model):
    n_trainable_params = 0
    for name, param in model.named_parameters():
        n_trainable_params += param.numel() if param.requires_grad else 0
    return n_trainable_params


def printmodel(model):
    for name, param in model.named_parameters():
        print("Name: {} | Shape: {} | Requires grad: {}".format(name, param.shape, param.requires_grad))


def train_epoch(args, model, device, train_dataloader, optimizer, lr_scheduler, epoch, print_freq=10,
                report_latency=True, steps_counter=0, writer=None):
    loss_average_meter = AverageMeter()
    ppl_average_meter = AverageMeter()

    latency_report = LatencyReport()

    cuda_memory_tracker = CudaMemoryTracker()

    cuda_memory_tracker.track("[train_epoch] Initial")
    model.train()
    model.to(device)
    cuda_memory_tracker.track("[train_epoch] After model to device")

    curr_step = lambda epoch, i_step: epoch * len(train_dataloader) + i_step

    # Get trainable parameters
    n_trainable_params = get_num_trainable_params(model)

    if args['train']['mark_only_rosa_or_lora_as_trainable'] and not args['fnmodel']['name'] == 'none':
        mark_only_rosa_or_lora_as_trainable(model, verbose=False)

    for i_step, batch in enumerate(train_dataloader):

        batch = {k: v.to(device) for k, v in batch.items()}
        cuda_memory_tracker.track("[train_epoch] After batch to device")

        # Masking FactorizedNet gradients
        latency_report.start()
        if args['fnmodel']['name'] == 'rosa' and args['fnmodel']['params']['level'] == "batch":
            num_training_steps = args['train']['epochs'] * len(train_dataloader)

            model, optimizer, lr_scheduler = sample_trainable(
                args, model, lr_scheduler, optimizer, steps_counter, num_training_steps
            )
            cuda_memory_tracker.track("[train_epoch] After sample_trainable")

            latency_report.stop(name="sample_trainable")

            n_trainable_params = get_num_trainable_params(model)

        # Forward pass
        # import pdb; pdb.set_trace()
        outputs = model(batch["input_ids"], labels=batch['labels'])
        cuda_memory_tracker.track("[train_epoch] After forward")
        latency_report.stop(name="forward")

        if len(outputs.loss.shape) > 0:
            loss = outputs.loss.mean()
        else:
            loss = outputs.loss
        latency_report.stop(name="loss.mean()")

        loss.backward()
        cuda_memory_tracker.track("[train_epoch] After loss.backward()")
        latency_report.stop(name="loss.backward()")

        optimizer.step()
        cuda_memory_tracker.track("[train_epoch] After optimizer.step()")
        latency_report.stop(name="optimizer.step()")

        if lr_scheduler is not None:
            lr_scheduler.step()

        optimizer.zero_grad()
        cuda_memory_tracker.track("[train_epoch] After optimizer.zero_grad()")

        if i_step % print_freq == 0:
            logging.info(
                "\tEpoch {:4d} | step {:4d}/{:4d} | trainable: {:,} | lr: {:.6f} | loss {:5.2f} | ppl {:8.2f} | "
                "bpc {:8.2f}".format(
                    epoch, i_step, len(train_dataloader), n_trainable_params, optimizer.param_groups[0]['lr'],
                    loss.item(), torch.exp(loss).item(),
                    loss.item() / math.log(2)
                ) + ("" if not report_latency else " | " + latency_report.report())
            )
            logging.info("{}\n".format(cuda_memory_tracker.report()))

        loss_average_meter.add(loss.item())
        ppl_average_meter.add(torch.exp(loss).item())
        steps_counter += 1

    model_fn = model.module if isinstance(model, nn.DataParallel) else model
    if isinstance(model_fn, fn.RosaNet) or isinstance(model_fn, fn.LoraNet):
        df = model_fn.get_report()
        logging.info(df)
        logging.info(model_fn)

    if writer is not None:
        for i, (k, v) in enumerate(cuda_memory_tracker.memory_allocated.items()):
            writer.add_scalar("train_epoch/memory_allocated", curr_step(epoch, i))

        for i, (k, v) in enumerate(cuda_memory_tracker.memory_reserved.items()):
            writer.add_scalar("train_epoch/memory_reserved", curr_step(epoch, i))

    return {"loss": loss_average_meter.value,
            "ppl": ppl_average_meter.value,
            "bpc": loss_average_meter.value / math.log(2)}, optimizer


def train(args, cmodel, optimizer, lr_scheduler, train_dataloader, valid_dataloader, device, output_path,
          tokenizer, writer, curr_epoch=1, best_valid_metrics=None, baseline_runtime_metrics=None,
          cuda_memory_tracker=None, test_dataloader=None, test_dataset=None):

    # Get runtime metrics
    cuda_memory_tracker = CudaMemoryTracker() if cuda_memory_tracker is None else cuda_memory_tracker
    factorized_mean_mean, factorized_mean_std = get_latency(
        cmodel, device=device, inp=torch.ones(args['logging']['input_size'], dtype=torch.long)
    )
    factorized_mean_params = get_num_params(cmodel)

    factorized_runtime_metrics = {
        "mean": factorized_mean_mean, "std": factorized_mean_std, "params": factorized_mean_params
    }
    cuda_memory_tracker.track("[train] After computing latency of factorized model")

    # Train loop
    steps_counter = 0
    for i_epoch in range(curr_epoch, args["train"]["epochs"] + 1):

        cuda_memory_tracker.track("[train] Loop start")
        _ = writer.add_scalar("train/memory_allocated", torch.cuda.memory_allocated(), i_epoch)

        epoch_start_time = time.time()

        # Mask gradients of RosaNet
        if args['fnmodel']['name'] == "rosa" and args['fnmodel']['params']['level'] == "epoch":
            num_training_steps = args['train']['epochs'] * len(train_dataloader)
            cmodel, optimizer, lr_scheduler = sample_trainable(
                args, cmodel, lr_scheduler, optimizer, steps_counter, num_training_steps
            )

            cuda_memory_tracker.track("[train] After epoch level sample trainable")

        # Train
        cuda_memory_tracker.track("[train] Before train epoch")
        _ = writer.add_scalar("train/memory_allocated", torch.cuda.memory_allocated(), i_epoch)

        train_metrics, optimizer = train_epoch(
            args, cmodel, device, train_dataloader, optimizer, lr_scheduler, i_epoch,
            print_freq=args["logging"]["print_freq"], writer=writer
        )
        train_end_time = time.time()

        cuda_memory_tracker.track("[train] After train epoch")
        _ = writer.add_scalar("train/memory_allocated", torch.cuda.memory_allocated(), i_epoch)

        # Evaluate
        valid_metrics = evaluate(cmodel, device, valid_dataloader, \
                                 task=args['dataset']['task_name'])
        valid_end_time = time.time()

        # Test
        logging.info("=> Computing test metrics...")
        test_metrics_advanced = evaluate_model_bleu(cmodel, test_dataset, tokenizer, device=device) \
            if test_dataset is not None else None
        test_metrics = evaluate(cmodel, device, test_dataloader, \
            task=args['dataset']['task_name']) if test_dataloader is not None else None

        # Log metrics
        logging.info(
            "=> Epoch {:4d}/{:4d} | Elapsed: tr={:5.2f}s tot={:5.2f}s | ".format(
                i_epoch, args["train"]["epochs"], (train_end_time - epoch_start_time),
                (valid_end_time - epoch_start_time)
            )
            + " | ".join([f"Train {k}: {v:.2f}" for k, v in train_metrics.items()]) + " | "
            + " | ".join([f"Valid {k}: {v:.2f}" for k, v in valid_metrics.items()]) + " | "
            + " | ".join([f"Test {k}: {v:.2f}" for k, v in test_metrics.items()]) + " | "
            + " | ".join([f"Test {k}: {v:.2f}" for k, v in test_metrics_advanced.items()])
        )
        logging.info(cuda_memory_tracker.report())

        # Ckpt object
        try:
            model_state_dict = cmodel.module.state_dict()
        except AttributeError:
            model_state_dict = cmodel.state_dict()
        ckpt = {
            'epoch': i_epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model_state_dict,
            'torchrandom_state': torch.get_rng_state(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics,
            'baseline_runtime_metrics': baseline_runtime_metrics,
            'factorized_runtime_metrics': factorized_runtime_metrics,
            'config': args
        }

        # Save model checkpoint
        if best_valid_metrics is None or valid_metrics['loss'] < best_valid_metrics['loss']:
            best_valid_metrics = valid_metrics
            torch.save(ckpt, osp.join(output_path, "model_best.pth"))
            torch.save(ckpt, osp.join(output_path, "model_latest.pth"))
            torch.save(ckpt, osp.join(output_path, "model_{}.pth".format(i_epoch)))

        elif i_epoch % 1 == 0 or i_epoch == args["train"]["epochs"]:
            torch.save(ckpt, osp.join(output_path, "model_latest.pth"))
            torch.save(ckpt, osp.join(output_path, "model_{}.pth".format(i_epoch)))

        # Log to tensorboard
        for m in valid_metrics.keys():
            if m is not None:
                writer.add_scalar("valid/{}".format(m), valid_metrics[m], i_epoch)

        for m in train_metrics.keys():
            if m is not None:
                writer.add_scalar("train/{}".format(m), train_metrics[m], i_epoch)

        if test_metrics is not None:
            for m in test_metrics.keys():
                if m is not None:
                    writer.add_scalar("test/{}".format(m), test_metrics[m], i_epoch)

        if test_metrics_advanced is not None:
            for m in test_metrics_advanced.keys():
                if m is not None:
                    writer.add_scalar("test/{}".format(m), test_metrics_advanced[m], i_epoch)

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], i_epoch)

        # Get trainable parameters
        n_trainable_params = 0
        for name, param in cmodel.named_parameters():
            n_trainable_params += param.numel() if param.requires_grad else 0
        writer.add_scalar("train/trainable_params", n_trainable_params, i_epoch)

        # Log best valid metrics
        logging.info("=> Best valid metrics: " + " | ".join(
            [f"{k}: {v:.2f}" for k, v in best_valid_metrics.items()]
        ))

        # Sample
        prompt = {
            "eli5": "Somatic hypermutation allows the immune system to",
            "e2e_nlg": "name[Blue Spice], eatType[coffee shop], area[city centre]",
        }[args["dataset"]["name"]]

        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        model_fn = cmodel.module if isinstance(cmodel, nn.DataParallel) else cmodel
        model_fn = model_fn.factorized_model if isinstance(model_fn, fn.RosaNet) else model_fn
        model_fn = model_fn.factorized_model if isinstance(model_fn, fn.LoraNet) else model_fn
        outputs = model_fn.generate(
            inputs,
            max_length=args['train']['seq_len'],
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        sample_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        writer.add_text('Sample', sample_str, i_epoch)
        logging.info("Sample: \n{}\nEND of Epoch\n=========\n".format(sample_str))


@hydra.main(version_base=None, config_path="./", config_name="configs_mlm")
def main(cfg: DictConfig):
    # Experiment tracking and logging
    args = OmegaConf.to_container(cfg, resolve=True)

    for t in range(max(1, args["runs"])):

        exp_name = get_experiment_name(
            {"train": args["train"], "model": args["model"], "fnmodel": args["fnmodel"], "trial": t}
        )
        folder_name = "_".join(["{}{}".format(k, v) for k, v in exp_name.items()])

        dct_latest, dct_best = None, None

        output_path = osp.join(args['output']['path'], args['dataset']['name'], folder_name)
        if not osp.exists(output_path):
            os.makedirs(output_path)
            save_object(args, osp.join(output_path, 'args.pkl'))
            print("=> Running Experiment: `{}`".format(folder_name))

        elif not osp.exists(osp.join(output_path, 'model_latest.pth')):
            print("=> Running Experiment: `{}`".format(folder_name))

        else:  # Experiment already exists
            dct_latest = torch.load(osp.join(output_path, 'model_latest.pth'))
            dct_best = torch.load(osp.join(output_path, 'model_best.pth'))
            if dct_latest['epoch'] >= args['train']['epochs']:
                print("=> Experiment `{}` already exists. (Latest @ epoch {})".format(
                    folder_name, dct_latest['epoch']
                ))
                continue

            else:
                print("=> Running Experiment: `{}`".format(folder_name))

        writer = SummaryWriter(log_dir=output_path)

        # Logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s',
            datefmt='%H:%M:%S',
            filemode='a'
        )
        logging.getLogger().addHandler(logging.FileHandler(osp.join(output_path, "logging.txt")))

    cuda_memory_tracker = CudaMemoryTracker()
    cuda_memory_tracker.track('[main] Initial')

    model = AutoModelForCausalLM.from_pretrained(args['model']['name'])
    tokenizer = AutoTokenizer.from_pretrained(args['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    # train_dataloader, valid_dataloader, valid_dataset, test_dataset = get_dataloaders(args, tokenizer)
    train_dataloader, valid_dataloader, test_dataloader, test_dataset = get_dataloaders(args, tokenizer)
    # import pdb; pdb.set_trace()
    logging.info("Model:\n{}".format(model))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda_memory_tracker.track('[main] Created base model loaded to cpu')

    logging.info("=> Computing baseline latency...")
    model.to(device)
    torch.cuda.empty_cache()
    cuda_memory_tracker.track('[main] Moved baseline model loaded to gpu')

    baseline_mean, baseline_std = get_latency(
        model, device=device, inp=torch.ones(args['logging']['input_size'], dtype=torch.long)
    ) if not args['debug'] else (1, 1)
    baseline_params = get_num_params(model)

    baseline_runtime_metrics = {
        "mean": baseline_mean, "std": baseline_std, "params": baseline_params
    }

    # Factorize model either using ROSA or LORA
    logging.info("=> Using {} model ...".format(args['fnmodel']['name'].lower()))
    cmodel = {
        "rosa": fn.RosaNet, "lora": fn.LoraNet, "none": lambda x, **kwargs: x
    }[args['fnmodel']['name'].lower()](model, **args['fnmodel']['params'])
    logging.info("Factorized Model:\n{}".format(cmodel))

    cuda_memory_tracker.track('[main] Created factorized model loaded to cpu')

    opt = {
        "adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD
    }[args['train']['optimizer']['name']]

    cuda_memory_tracker.track('[main] Created optimizer on cpu')

    # Resume training
    if dct_latest is not None:
        cmodel.load_state_dict(dct_latest['model_state_dict'])
        cmodel.to(device)

        torch.cuda.empty_cache()
        cuda_memory_tracker.track('[main] Moved factorized model loaded to gpu')

        if args['train']['optimizer']['name'] == "sgd":
            optimizer = opt(
                cmodel.parameters(),
                lr=args["train"]["lr"],
                **args['train']['optimizer']['params']
            )
        # Catch all exceptions
        else:
            optimizer = opt(
                cmodel.parameters(),
                lr=args["train"]["lr"]
            )

        cuda_memory_tracker.track('[main] Optimizer passed network parameters')

        optimizer.load_state_dict(dct_latest['optimizer_state_dict'])
        curr_epoch = dct_latest['epoch']
        curr_best_valid_metrics = dct_best['valid_metrics']
        logging.info("=> Resuming training from from epoch {}".format(dct_latest['epoch']))

    else:
        curr_epoch = 0
        curr_best_valid_metrics = None
        cmodel.to(device)
        torch.cuda.empty_cache()
        cuda_memory_tracker.track('[main] Moved factorized model loaded to gpu')

        optimizer = opt(
            cmodel.parameters(),
            lr=args["train"]["lr"]
        )
        cuda_memory_tracker.track('[main] Optimizer passed network parameters')
        logging.info("=> Starting training from scratch ...")

    # Scheduler
    n_training_steps = args['train']['epochs'] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=args['train']['scheduler']['name'],
        optimizer=optimizer,
        num_training_steps=n_training_steps,
        **args['train']['scheduler']['params']
    ) if args['train']['scheduler']['name'] != "none" else None

    # Compute tn model latency
    logging.info("=> Computing factorized latency...")
    factorized_mean, factorized_std = get_latency(
        cmodel, device=device, inp=torch.ones(args['logging']['input_size'], dtype=torch.long)
    ) if not args['debug'] else (1, 1)
    factorized_params = get_num_params(cmodel)

    logging.info("=> Baseline {:.4f} ms| Factorized {:.4f} ms | Speedup {:.4f} | Compression {:.4f}".format(
        baseline_mean, factorized_mean, baseline_mean / factorized_mean, factorized_params / baseline_params
    ))

    # Parallelize the model
    if torch.cuda.device_count() >= 1:
        logging.info("=> Using {} GPU(s)".format(torch.cuda.device_count()))
        cmodel = nn.DataParallel(cmodel)

    # Training
    logging.info(cuda_memory_tracker.report())
    train(
        args, cmodel, optimizer, lr_scheduler, train_dataloader, valid_dataloader, device,
        output_path, tokenizer, writer, curr_epoch=curr_epoch + 1, best_valid_metrics=curr_best_valid_metrics,
        baseline_runtime_metrics=baseline_runtime_metrics, cuda_memory_tracker=cuda_memory_tracker,
        test_dataloader=test_dataloader, test_dataset=test_dataset
    )

    print("=> Experiment: `{}` Succeeded".format(folder_name))


if __name__ == '__main__':
    main()