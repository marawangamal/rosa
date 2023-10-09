import os
import os.path as osp
from tqdm import tqdm
from typing import Optional, List

import hydra
import evaluate
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

import torch
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer

import peftnet as pn
from utils_cv import collate_fn, get_dataloaders, coco_bbox_to_pascal_bbox
from utils.utils import  get_experiment_name, set_seeds, save_object

# todo: vary bs
# todo: stop removing invalid images / check num images

import numpy as np
from mean_average_precision import MetricBuilder

# Load dataset
checkpoint = "facebook/detr-resnet-50"
(train_dataset, test_dataset, test_dataloader, test_ds_coco_format, test_dl_coco_format,
 image_processor, id2label, label2id) = get_dataloaders(
    image_processor_checkpoint=checkpoint,
    dataset='cppe-5',
    create_coco=True
)

# Load evaluator
module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


class CustomTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        with torch.no_grad():
            # Call the original evaluate method
            results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

            eval_dataloader = test_dl_coco_format
            model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
            import pdb;
            pdb.set_trace()
            for i, batch in enumerate(eval_dataloader):
                inputs = self._prepare_inputs(batch)
                pixel_values = inputs["pixel_values"]
                pixel_mask = inputs["pixel_mask"]

                labels = [
                    {k: v for k, v in t.items()} for t in batch["labels"]
                ]  # these are in DETR format, resized + normalized

                # forward pass
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)

                # convert outputs of model to COCO api
                results = image_processor.post_process_object_detection(outputs, orig_target_sizes)

                module.add(prediction=results, reference=labels)
                del batch

        results = module.compute()
        print(results)

    def evaluate_old(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        with torch.no_grad():

            metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=5)

            # Call the original evaluate method
            results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)

            # Type: {'boxes': torch.Tensor, 'labels': torch.Tensor, 'scores': torch.Tensor, 'pred_boxes': torch.Tensor}
            for i, batch in enumerate(eval_dataloader):
                inputs = self._prepare_inputs(batch)
                outputs = model(**inputs)

                gt_bbox = [batch['labels'][i]['boxes'] for i in range(len(batch['labels']))]
                gt_labels = [batch['labels'][i]['class_labels'] for i in range(len(batch['labels']))]
                gt_iscrowd = [batch['labels'][i]['iscrowd'] for i in range(len(batch['labels']))]
                gt_difficult = [torch.zeros(
                    batch['labels'][i]['iscrowd'].shape, device=batch['labels'][i]['iscrowd'].device
                ) for i in range(len(batch['labels']))]
                gt_bbox_flat = torch.cat(gt_bbox)  # [n, 4]
                gt_labels_flat = torch.cat(gt_labels)
                gt_iscrowd_flat = torch.cat(gt_iscrowd)
                gt_difficult_flat = torch.cat(gt_difficult)

                img_dims = batch['pixel_values'].shape[-2:]
                gt_bbox_pascal_ = [coco_bbox_to_pascal_bbox(box, img_dims) for box in gt_bbox_flat]
                gt_bbox_pascal = torch.cat([torch.tensor(r).reshape(1, -1) for r in gt_bbox_pascal_])

                # [xmin, ymin, xmax, ymax, class_id, difficult, crowd] gt
                gt_pascal = torch.stack(
                    [*[gt_bbox_pascal[:, i] for i in range(gt_bbox_pascal.shape[1])],
                     gt_labels_flat, gt_difficult_flat, gt_iscrowd_flat], dim=1
                )

                # [xmin, ymin, xmax, ymax, class_id, confidence] pred
                pred_bbox_flat = outputs.pred_boxes.reshape(-1, 4)  # [n, 4]
                pred_bbox_pascal_ = [coco_bbox_to_pascal_bbox(box, img_dims) for box in pred_bbox_flat]
                pred_bbox_pascal = torch.cat([torch.tensor(r).reshape(1, -1) for r in pred_bbox_pascal_])

                pred_conf_tot = outputs.logits.softmax(dim=-1).reshape(-1, outputs.logits.shape[-1])  # [n, 4]
                pred_class_flat, pred_conf_flat = pred_conf_tot.max(dim=-1)  # [n, 1]
                pred_class_flat = pred_class_flat.to(pred_bbox_pascal.device)
                pred_conf_flat = pred_conf_flat.to(pred_bbox_pascal.device)

                pred_pascal = torch.stack(
                    [*[pred_bbox_pascal[:, i] for i in range(pred_bbox_pascal.shape[1])],
                     pred_class_flat, pred_conf_flat], dim=1
                )

                # Compute metrics
                metric_fn.add(pred_pascal.numpy(), gt_pascal.numpy())

            map_score = metric_fn.value(
                iou_thresholds=np.arange(0.5, 1.0, 0.05),
                recall_thresholds=np.arange(0., 1.01, 0.01),
                mpolicy='soft'
            )['mAP']

            results.update({'map_score': map_score})
            print(results)
            return results


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
    }


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
        experiment_path = osp.join(args['output']['path'], args['dataset']['name'])
        output_path = osp.join(experiment_path, experiment_name)

        if not osp.exists(output_path):
            os.makedirs(output_path)

        save_object(args, osp.join(output_path, 'args.pkl'))

        model = AutoModelForObjectDetection.from_pretrained(
            checkpoint,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        print(model)

        # PEFT Model
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
            eval_steps=1,
            logging_strategy="steps",
            logging_steps=50,
            report_to='tensorboard',
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

        # best_trial = trainer.hyperparameter_search(
        #     direction="maximize",
        #     backend="optuna",
        #     hp_space=optuna_hp_space,
        #     n_trials=20,
        #     compute_objective=compute_objective,
        # )



if __name__ == "__main__":
    main()
