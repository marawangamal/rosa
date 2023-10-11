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
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer

import peftnet as pn
from utils.utils_cv import collate_fn, get_dataloaders
from utils.utils import  get_experiment_name, set_seeds, save_object

# todo: vary bs

# Load dataset
checkpoint = "facebook/detr-resnet-50"
(train_dataset, test_dataset, test_dataloader, test_ds_coco_format, test_dl_coco_format,
 image_processor, id2label, label2id) = get_dataloaders(
    image_processor_checkpoint=checkpoint,
    dataset='cppe-5',
    create_coco=True
)


def compute_objective(metrics):
    # import pdb; pdb.set_trace()
    return metrics["eval_AP-IoU=0.50-area=all-maxDets=100"]


class CustomTrainer(Trainer):

    def eval_internal(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        return output.metrics

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):

        # Load evaluator
        module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)

        with torch.no_grad():
            # Call the original evaluate method
            results_total = self.eval_internal(eval_dataset, ignore_keys, metric_key_prefix)

            eval_dataloader = test_dl_coco_format
            model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)

            for i, batch in enumerate(eval_dataloader):
                inputs = self._prepare_inputs(batch)
                pixel_values = inputs["pixel_values"]
                pixel_mask = inputs["pixel_mask"]

                labels = [
                    {k: v for k, v in t.items()} for t in inputs["labels"]
                ]  # these are in DETR format, resized + normalized

                # forward pass
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
                # convert outputs of model to COCO api
                results = image_processor.post_process(outputs, orig_target_sizes)

                module.add(prediction=results, reference=labels)
                del inputs
                del batch

        results = module.compute()
        results_total.update(
            {f"{metric_key_prefix}_{key}": value for key, value in results['iou_bbox'].items()}
        )
        self.log(results_total)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, results_total)
        self._memory_tracker.stop_and_update_metrics(results_total)
        return results_total


def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "objective", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-3},
            "per_device_train_batch_size": {"values": [8, 16]},
        },
    }


@hydra.main(version_base=None, config_path="configs", config_name="conf_od")
def main(cfg: DictConfig):
    # Experiment tracking and logging
    args = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
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

        run = wandb.init(config=wandb_config, project="lora-tensor", name=experiment_name)

        if not osp.exists(output_path):
            os.makedirs(output_path)

        save_object(args, osp.join(output_path, 'args.pkl'))

        model = AutoModelForObjectDetection.from_pretrained(
            checkpoint,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        # PEFT Model
        print("-" * 100)
        if args['pmodel']['method'].lower() != 'none':
            print(f"=> Training PEFTNet model: {args['pmodel']['method']}")
            # cmodel = pn.PEFTNet(model, ignore_regex=".*conv(1|3).*", **args['pmodel'])
            cmodel = pn.PEFTNet(model, **args['pmodel'])
            model = cmodel.peft_model
        else:
            print("=> Training baseline model")

        def model_init(trial):
            return model

        # Make last layer trainable
        for param in model.class_labels_classifier.parameters():
            param.requires_grad = True
        for param in model.bbox_predictor.parameters():
            param.requires_grad = True

        # Print model
        print(model)
        print(f"\n{pn.PEFTNet.get_report(model)}\n")

        # Print ratio of different layer types to total number of layers
        print(f"=> Ratio of different layer types to total number of layers:")
        n_conv_params = 0
        n_linear_params = 0
        n_total_params = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # in millions

        for name, module in model.named_modules():
            # If primitive layer
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if isinstance(module, nn.Conv2d) and any([k > 1 for k in module.kernel_size]):
                    n_conv_params += module.weight.numel()
                else:
                    n_linear_params += module.weight.numel()

        n_conv_params /= 1e6
        n_linear_params /= 1e6
        print(f"=> Total number of parameters: {n_total_params}")
        print(f"=> # Conv2d params: {n_conv_params}M (ratio: {n_conv_params / n_total_params:.3f})")
        print(f"=> # Linear params: {n_linear_params}M (ratio: {n_linear_params / n_total_params:.3f})")
        print(f"=> # Trainable params: {n_trainable_params:.3f}M (ratio: {n_trainable_params / n_total_params:.3f})")
        print("-" * 100)

        training_args = TrainingArguments(
            output_dir=output_path,
            run_name=experiment_name,
            evaluation_strategy="steps",
            eval_steps=100,
            logging_strategy="steps",
            logging_steps=50,
            report_to='wandb',
            save_total_limit=4,
            prediction_loss_only=True,
            load_best_model_at_end=True,
            remove_unused_columns=False,
            per_device_train_batch_size=args['train']['batch_size'],
            num_train_epochs=args['train']['epochs'],
            learning_rate=args['train']['lr'],
            weight_decay=args['train']['weight_decay'],
            fp16=args['train']['fp16']
        )

        if args['hp_search']:
            print("=> Running hyperparameter search")
            trainer = CustomTrainer(
                model=None,
                args=training_args,
                data_collator=lambda x: collate_fn(x, image_processor),
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=image_processor,
                model_init=model_init
            )

            best_trial = trainer.hyperparameter_search(
                direction="maximize",
                backend="wandb",
                hp_space=wandb_hp_space,
                n_trials=20,
                compute_objective=compute_objective,
            )
            print(best_trial)

        else:
            print("=> Running training (no search)")
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                data_collator=lambda x: collate_fn(x, image_processor),
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=image_processor
            )
            trainer.train()
            trainer.save_model(output_path)


if __name__ == "__main__":
    main()
