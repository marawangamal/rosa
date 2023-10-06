import os
import os.path as osp
import math
import time
import logging
import torch.nn as nn

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, get_scheduler
from transformers import AutoModelForCausalLM

from utils.utils import get_num_params, get_experiment_name, get_latency, AverageMeter, save_object, LatencyReport, \
    CudaMemoryTracker, preprocess_function, get_ignore_list_e2e, set_seeds
from eval import evaluate_model

import peftnet as pn
import pandas as pd

wandb.login()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:.3f}'.format)

# todo: model(**batch) to include attn mask [done]
# todo: divide by rank in model [done]
# todo: add ia3 model [done]
# todo: add NIST score
# todo: add in_out for ia3
# todo: change model api to do a `factorize` if train mode and a `merge` if evaltools mode


def get_dataloaders(args, tokenizer):
    # Load dataset
    assert args['dataset']['name'] in ["eli5", "e2e_nlg"], "Dataset not supported"

    train_split = {"eli5": "train_asks", "e2e_nlg": "train"}[args['dataset']['name']]
    valid_split = {"eli5": "validation_asks", "e2e_nlg": "validation"}[args['dataset']['name']]
    test_split = {"eli5": "validation_asks", "e2e_nlg": "test"}[args['dataset']['name']]

    train_dataset = load_dataset(
        args['dataset']['name'], split=train_split, cache_dir=args['dataset']['cache']
    )
    test_dataset = load_dataset(
        args['dataset']['name'], split=test_split, cache_dir=args['dataset']['cache']
    )
    valid_dataset = load_dataset(
        args['dataset']['name'], split=valid_split, cache_dir=args['dataset']['cache']
    )

    # Filter for faster training (debug)
    num_train_pts, _ = train_dataset.shape
    train_dataset = train_dataset.select(range(int(num_train_pts * args['train']['fraction'])))

    # Apply tokenizer to dataset
    train_tokenized = train_dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, dataset_name=args['dataset']['name'], max_length=args['train']['seq_len']
        ),
        batched=True,
        # num_proc=4,
    )

    valid_tokenized = valid_dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, dataset_name=args['dataset']['name'], max_length=args['train']['seq_len']),
        batched=True
    )

    test_tokenized = test_dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, dataset_name=args['dataset']['name'], max_length=args['train']['seq_len']),
        batched=True
    )

    # Only include tokenized ids
    train_tokenized_reduced = train_tokenized.remove_columns(train_dataset.column_names)
    valid_tokenized_reduced = valid_tokenized.remove_columns(valid_dataset.column_names)
    test_tokenized_reduced = test_tokenized.remove_columns(test_dataset.column_names)

    if args['dataset']['name'] == "eli5":
        train_tokenized_reduced_grouped = train_tokenized_reduced.map(group_texts, batched=True)
        valid_tokenized_reduced_grouped = valid_tokenized_reduced.map(group_texts, batched=True)
    elif args['dataset']['name'] == "e2e_nlg":
        train_tokenized_reduced_grouped = train_tokenized_reduced
        valid_tokenized_reduced_grouped = valid_tokenized_reduced
        test_tokenized_reduced_grouped = test_tokenized_reduced
    else:
        raise NotImplementedError

    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors="pt", padding=True)
    train_dataloader = DataLoader(
        train_tokenized_reduced_grouped, shuffle=True, batch_size=args['train']['batch_size'], collate_fn=data_collator,
        pin_memory=True, num_workers=2
    )
    valid_dataloader = DataLoader(
        valid_tokenized_reduced_grouped, batch_size=args['train']['batch_size'], collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        test_tokenized_reduced_grouped, batch_size=args['train']['batch_size'], collate_fn=data_collator
    )

    return train_dataloader, valid_dataloader, test_dataloader, test_dataset


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


def evaluate(model, device, eval_dataloader):
    model.eval()
    with torch.no_grad():
        loss_average_meter = AverageMeter()
        ppl_average_meter = AverageMeter()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            if len(outputs.loss.shape) > 0:
                loss = outputs.loss.mean()
            else:
                loss = outputs.loss

            loss_average_meter.add(loss.item())
            ppl_average_meter.add(torch.exp(loss).item())

    return {"loss": loss_average_meter.value,
            "ppl": ppl_average_meter.value,
            "bpc": loss_average_meter.value / math.log(2)
            }


def refactorize(args, model, lr_scheduler, optimizer, steps_counter, num_training_steps):
    """Refactorize model and update optimizer and scheduler accordingly"""

    # (Re)factorize model
    model = model.module.factorize() if isinstance(model, nn.DataParallel) else model.factorize()

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


def get_num_trainable_params(model):
    n_trainable_params = 0
    for name, param in model.named_parameters():
        n_trainable_params += param.numel() if param.requires_grad else 0
    return n_trainable_params


def train_epoch(args, model, device, train_dataloader, optimizer, lr_scheduler, epoch, print_freq=10,
                report_latency=True, steps_counter=0, writer=None,
                valid_dataloader=None, test_dataloader=None, tokenizer=None, test_dataset=None):
    loss_average_meter = AverageMeter()
    ppl_average_meter = AverageMeter()

    latency_report = LatencyReport()

    cuda_memory_tracker = CudaMemoryTracker()

    cuda_memory_tracker.track("[train_epoch] Initial")
    model.train()
    model.to(device)
    cuda_memory_tracker.track("[train_epoch] After model to device")

    # Get trainable parameters
    n_trainable_params = get_num_trainable_params(model)
    step = epoch * len(train_dataloader)

    for i_step, batch in enumerate(train_dataloader):

        step = epoch * len(train_dataloader) + i_step

        if args['logging']['eval_level'] == "batch" and i_step % args['logging']['eval_freq'] == 0:


            # Evaluate
            valid_metrics = evaluate(model, device, valid_dataloader)

            # Test
            test_metrics_advanced = evaluate_model(
                model,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                device=device
            ) if test_dataset is not None else None
            test_metrics = evaluate(model, device, test_dataloader) if test_dataloader is not None else None

            # Combine test metrics
            test_metrics = {**test_metrics, **test_metrics_advanced}

            # Log to tensorboard
            if valid_metrics is not None:
                for m in valid_metrics.keys():
                    if m is not None:
                        writer.add_scalar("valid/{}".format(m), valid_metrics[m], step)
                        wandb.log({"valid/{}".format(m): valid_metrics[m]}, step=step)

            if test_metrics is not None:
                for m in test_metrics.keys():
                    if m is not None:
                        writer.add_scalar("test/{}".format(m), test_metrics[m], step)
                        wandb.log({"test/{}".format(m): test_metrics[m]}, step=step)

            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)
            wandb.log({"train/lr": optimizer.param_groups[0]['lr']}, step=step)

            # Get trainable parameters
            n_trainable_params = 0
            for name, param in model.named_parameters():
                n_trainable_params += param.numel() if param.requires_grad else 0
            writer.add_scalar("train/trainable_params", n_trainable_params, step)

        batch = {k: v.to(device) for k, v in batch.items()}
        cuda_memory_tracker.track("[train_epoch] After batch to device")

        # Masking FactorizedNet gradients
        latency_report.start()
        if args['fnmodel']['name'] == 'rosa' and args['fnmodel']['params']['level'] == "batch":
            num_training_steps = args['train']['epochs'] * len(train_dataloader)

            model, optimizer, lr_scheduler = refactorize(
                args, model, lr_scheduler, optimizer, steps_counter, num_training_steps
            )
            cuda_memory_tracker.track("[train_epoch] After sample_trainable")

            latency_report.stop(name="sample_trainable")

            n_trainable_params = get_num_trainable_params(model)

        # Forward pass
        # import pdb; pdb.set_trace()
        outputs = model(**batch)
        cuda_memory_tracker.track("[train_epoch] After forward")
        latency_report.stop(name="forward")

        if len(outputs.loss.shape) > 0:
            loss = outputs.loss.mean()
        else:
            loss = outputs.loss

        if torch.isnan(loss):
            raise ValueError("Loss is NAN. Discontinuing training.")

        else:
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
    if isinstance(model_fn, pn.RosaNet) or isinstance(model_fn, pn.LoraNet):
        df = model_fn.get_report()
        logging.info("\n{}".format(df))
        logging.info(model_fn)

    if writer is not None:
        for i, (k, v) in enumerate(cuda_memory_tracker.memory_allocated.items()):
            writer.add_scalar("train_epoch/memory_allocated/{}".format(k), v,  step)

        for i, (k, v) in enumerate(cuda_memory_tracker.memory_reserved.items()):
            writer.add_scalar("train_epoch/memory_reserved/{}".format(k), v,  step)

    return {"loss": loss_average_meter.value,
            "ppl": ppl_average_meter.value,
            "bpc": loss_average_meter.value / math.log(2)}, optimizer


def train(args, pmodel, optimizer, lr_scheduler, train_dataloader, valid_dataloader, device, output_path,
          tokenizer, writer, curr_epoch=0, best_valid_metrics=None, baseline_runtime_metrics=None,
          cuda_memory_tracker=None, test_dataloader=None, test_dataset=None):

    # Get runtime metrics
    cuda_memory_tracker = CudaMemoryTracker() if cuda_memory_tracker is None else cuda_memory_tracker
    factorized_mean_mean, factorized_mean_std = get_latency(
        pmodel, device=device, inp=torch.ones(args['logging']['input_size'], dtype=torch.long)
    )
    factorized_mean_params = get_num_params(pmodel)

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

        # Refactorize if using ROSA model
        if (args['fnmodel']['name'] == "rosa" and args['fnmodel']['params']['level'] == "epoch"
                and i_epoch > 0 and args['fnmodel']['factorize_freq'] % i_epoch == 0):
            logging.info("=> Re-sampling trainable parameters...")
            num_training_steps = args['train']['epochs'] * len(train_dataloader)
            pmodel, optimizer, lr_scheduler = refactorize(
                args, pmodel, lr_scheduler, optimizer, steps_counter, num_training_steps
            )

            cuda_memory_tracker.track("[train] After epoch level sample trainable")

        # Train
        if i_epoch > 0:
            cuda_memory_tracker.track("[train] Before train epoch")
            _ = writer.add_scalar("train/memory_allocated", torch.cuda.memory_allocated(), i_epoch)
            train_metrics, optimizer = train_epoch(
                args, pmodel, device, train_dataloader, optimizer, lr_scheduler, i_epoch,
                print_freq=args["logging"]["print_freq"], writer=writer
            )
            train_end_time = time.time()
        else:
            train_metrics = None
            train_end_time = epoch_start_time

        cuda_memory_tracker.track("[train] After train epoch")
        _ = writer.add_scalar("train/memory_allocated", torch.cuda.memory_allocated(), i_epoch)

        # Evaluate
        valid_metrics = evaluate(pmodel, device, valid_dataloader)
        valid_end_time = time.time()

        # Test
        logging.info("=> Computing test metrics...")
        test_metrics_advanced = evaluate_model(
            pmodel,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            device=device,
            output_path_refs=osp.join(output_path, "refs_{}.txt".format(i_epoch)),
            output_path_preds=osp.join(output_path, "preds_{}.txt".format(i_epoch))
        ) if test_dataset is not None else None
        test_metrics = evaluate(pmodel, device, test_dataloader) if test_dataloader is not None else None

        # Combine test metrics
        test_metrics = {**test_metrics, **test_metrics_advanced}

        # Log metrics
        logging.info(
            "=> Epoch {:4d}/{:4d} | Elapsed: tr={:5.2f}s tot={:5.2f}s | ".format(
                i_epoch, args["train"]["epochs"], (train_end_time - epoch_start_time),
                (valid_end_time - epoch_start_time)
            )
            + " | ".join([f"Train {k}: {v:.2f}" for k, v in train_metrics.items()]) + " | " if i_epoch > 0 else ""
            + " | ".join([f"Valid {k}: {v:.2f}" for k, v in valid_metrics.items()]) + " | "
            + " | ".join([f"Test {k}: {v:.2f}" for k, v in test_metrics.items()])
        )
        logging.info(cuda_memory_tracker.report())

        # Ckpt object
        try:
            model_state_dict = pmodel.module.state_dict()
        except AttributeError:
            model_state_dict = pmodel.state_dict()
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
        if best_valid_metrics is None or test_metrics['bleu'] > best_valid_metrics['bleu']:
            best_valid_metrics = test_metrics
            torch.save(ckpt, osp.join(output_path, "model_best.pth"))
            torch.save(ckpt, osp.join(output_path, "model_latest.pth"))
            torch.save(ckpt, osp.join(output_path, "model_{}.pth".format(i_epoch)))

        elif i_epoch % 1 == 0 or i_epoch == args["train"]["epochs"]:
            torch.save(ckpt, osp.join(output_path, "model_latest.pth"))
            torch.save(ckpt, osp.join(output_path, "model_{}.pth".format(i_epoch)))

        # Log to tensorboard
        if valid_metrics is not None:
            for m in valid_metrics.keys():
                if m is not None:
                    writer.add_scalar("valid/{}".format(m), valid_metrics[m], i_epoch)
                    wandb.log({"valid/{}".format(m): valid_metrics[m]}, step=i_epoch)

        if train_metrics is not None:
            for m in train_metrics.keys():
                if m is not None:
                    writer.add_scalar("train/{}".format(m), train_metrics[m], i_epoch)
                    wandb.log({"train/{}".format(m): train_metrics[m]}, step=i_epoch)

        if test_metrics is not None:
            for m in test_metrics.keys():
                if m is not None:
                    writer.add_scalar("test/{}".format(m), test_metrics[m], i_epoch)
                    wandb.log({"test/{}".format(m): test_metrics[m]}, step=i_epoch)

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], i_epoch)
        wandb.log({"train/lr": optimizer.param_groups[0]['lr']}, step=i_epoch)

        # Get trainable parameters
        n_trainable_params = 0
        for name, param in pmodel.named_parameters():
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
        model_fn = pmodel.module if isinstance(pmodel, nn.DataParallel) else pmodel

        if any([isinstance(model_fn, k) for k in [pn.RosaNet, pn.LoraNet, pn.IA3Net]]):
            model_fn = model_fn.peft_model
        else:
            model_fn = model_fn

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


@hydra.main(version_base=None, config_path="./configs", config_name="conf_clm")
def main(cfg: DictConfig):
    # Experiment tracking and logging
    args = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))

    for t in range(max(1, args["runs"])):

        # Set diff seeds for each run
        if args['seed'] > 0:
            set_seeds(int(args['seed'] + t))

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

        run = wandb.init(
            mode="disabled",
            # Set the project where this run will be logged
            project="rosa",
            name=folder_name,
            # Track hyperparameters and run metadata
            config=args,
        )

        cuda_memory_tracker = CudaMemoryTracker()
        cuda_memory_tracker.track('[main] Initial')

        model = AutoModelForCausalLM.from_pretrained(args['model']['name'])
        tokenizer = AutoTokenizer.from_pretrained(args['model']['name'])
        tokenizer.pad_token = tokenizer.eos_token
        # train_dataloader, valid_dataloader, valid_dataset, test_dataset = get_dataloaders(args, tokenizer)
        train_dataloader, valid_dataloader, test_dataloader, test_dataset = get_dataloaders(args, tokenizer)
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
        ignore_list = get_ignore_list_e2e(model) if args['fnmodel']['ignore_list'] else None

        # Create PEFT model
        pmodel = pn.PEFTNet(
            model,
            peft_method=args['fnmodel']['name'],
            factorize_list=['Linear', 'Conv1D'],
            ignore_list=ignore_list,
            **args['fnmodel']['params']
        ) if args['fnmodel']['name'].lower() != "none" else model

        logging.info("Factorized Model:\n{}".format(pmodel))
        model_fn = model.module if isinstance(model, nn.DataParallel) else model
        if isinstance(model_fn, pn.RosaNet) or isinstance(model_fn, pn.LoraNet):
            df = model_fn.get_report()
            logging.info("\n{}".format(df))
            logging.info(model_fn)

        cuda_memory_tracker.track('[main] Created factorized model loaded to cpu')

        opt = {
            "adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD
        }[args['train']['optimizer']['name']]

        cuda_memory_tracker.track('[main] Created optimizer on cpu')

        # Resume training
        if dct_latest is not None:
            pmodel.load_state_dict(dct_latest['model_state_dict'])
            pmodel.to(device)

            torch.cuda.empty_cache()
            cuda_memory_tracker.track('[main] Moved factorized model loaded to gpu')

            if args['train']['optimizer']['name'] == "sgd":
                optimizer = opt(
                    pmodel.parameters(),
                    lr=args["train"]["lr"],
                    **args['train']['optimizer']['params']
                )
            # Catch all exceptions
            else:
                optimizer = opt(
                    pmodel.parameters(),
                    lr=args["train"]["lr"]
                )

            cuda_memory_tracker.track('[main] Optimizer passed network parameters')

            optimizer.load_state_dict(dct_latest['optimizer_state_dict'])
            curr_epoch = dct_latest['epoch'] + 1
            curr_best_valid_metrics = dct_best['test_metrics']
            logging.info("=> Resuming training from from epoch {}".format(dct_latest['epoch']))

        else:
            curr_epoch = 0
            curr_best_valid_metrics = None
            pmodel.to(device)
            torch.cuda.empty_cache()
            cuda_memory_tracker.track('[main] Moved factorized model loaded to gpu')

            optimizer = opt(
                pmodel.parameters(),
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
            pmodel, device=device, inp=torch.ones(args['logging']['input_size'], dtype=torch.long)
        ) if not args['debug'] else (1, 1)
        factorized_params = get_num_params(pmodel)

        logging.info("=> Baseline {:.4f} ms| Factorized {:.4f} ms | Speedup {:.4f} | Compression {:.4f}".format(
            baseline_mean, factorized_mean, baseline_mean / factorized_mean, factorized_params / baseline_params
        ))

        # Parallelize the model
        if torch.cuda.device_count() >= 1:
            logging.info("=> Using {} GPU(s)".format(torch.cuda.device_count()))
            pmodel = nn.DataParallel(pmodel)

        # Training
        logging.info(cuda_memory_tracker.report())
        train(
            args, pmodel, optimizer, lr_scheduler, train_dataloader, valid_dataloader, device,
            output_path, tokenizer, writer, curr_epoch=curr_epoch, best_valid_metrics=curr_best_valid_metrics,
            baseline_runtime_metrics=baseline_runtime_metrics, cuda_memory_tracker=cuda_memory_tracker,
            test_dataloader=test_dataloader, test_dataset=test_dataset
        )

        print("=> Experiment: `{}` Succeeded".format(folder_name))


if __name__ == '__main__':
    main()
