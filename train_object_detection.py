import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer

import peftnet as pn
from utils_cv import collate_fn, get_dataloaders
from utils.utils import  get_experiment_name, set_seeds


@hydra.main(version_base=None, config_path="configs", config_name="conf_od")
def main(cfg: DictConfig):
    # Experiment tracking and logging
    args = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))

    for t in range(max(1, args["runs"])):

        # Set diff seeds for each run
        if args['seed'] > 0:
            set_seeds(int(args['seed'] + t))

        # Experiment path
        experiment_name = get_experiment_name(
            {"train": args["train"], "pmodel": args["pmodel"], "trial": t}, mode='str'
        )
        experiment_path = osp.join(
            args['output']['path'], args['dataset']['name'], experiment_name
        )
        output_path = osp.join(experiment_path, "checkpoint")

        # PEFT Model
        checkpoint = "facebook/detr-resnet-50"
        train_dataset, test_dataset, test_dataloader, image_processor, id2label, label2id = get_dataloaders(
            image_processor_checkpoint=checkpoint,
            dataset=args['dataset']['name']
        )

        model = AutoModelForObjectDetection.from_pretrained(
            checkpoint,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        print(model)

        print("-" * 100)
        if args['pmodel']['method'].lower() != 'none':
            print(f"=> Training PEFTNet model: {args['pmodel']['method']}")
            cmodel = pn.PEFTNet(model, ignore_regex=".*conv(1|3).*", **args['pmodel'])
            print(f"\n{cmodel.get_report()}\n")
            model = cmodel.peft_model
        else:
            print("=> Training baseline model")

        # Number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # in millions
        print(f"=> Number of trainable parameters: {trainable_params:.3f}M")
        print(f"=> Total number of parameters: {total_params:.3f}M")
        print(f"=> Ratio: {trainable_params / total_params:.3f}")
        print("-" * 100)

        training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy="steps",
            eval_steps=50,
            # logging_strategy="epoch",
            report_to='tensorboard',
            prediction_loss_only=True,
            # fp16=True,
            # save_steps=200,
            # logging_steps=50,
            # save_total_limit=2,
            remove_unused_columns=False,
            per_device_train_batch_size=args['train']['batch_size'],
            num_train_epochs=args['train']['epochs'],
            learning_rate=args['train']['lr'],
            weight_decay=1e-4

            # output_dir = "/Content/mod",
            # evaluation_strategy = "epoch",  # Can be epoch or steps
            # learning_rate = 2e-5,  # According to original bert paper
            # per_device_train_batch_size = 32,  # According to original bert paper
            # per_device_eval_batch_size = 32,
            # num_train_epochs = 3,  # should be inbetween 2-4 according to the paper
            # weight_decay = 0.01,
            # prediction_loss_only = True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=lambda x: collate_fn(x, image_processor),
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=image_processor,
        )
        trainer.train()


if __name__ == "__main__":
    main()
