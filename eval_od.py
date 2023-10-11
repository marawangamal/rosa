import os
import os.path as osp
from tqdm import tqdm

import torch
import evaluate
from transformers import AutoModelForObjectDetection

import peftnet as pn
from utils.utils_cv import get_dataloaders
from utils.utils import load_object

import argparse


def main(args):

    # experiment_path = "/home/mila/m/marawan.gamal/scratch/lora-tensor/cppe-5/e10_l1e-05_b16_w0.0001_fTrue_mnone_r4_t0"
    experiment_path = args.experiment_path
    # experiment_path="devonho/detr-resnet-50_finetuned_cppe5"
    # Load dataset
    # checkpoint = "facebook/detr-resnet-50"
    # checkpoint = osp.join(experiment_path, "checkpoint-1000")

    coco_path = "/home/mila/m/marawan.gamal/scratch/lora-tensor/eval"
    if osp.exists(coco_path):
        # rm -rf /home/mila/m/marawan.gamal/scratch/lora-tensor/eval
        print("=> Removing coco eval dir")
        os.system(f"rm -rf {coco_path}")

    train_dataset, _, _, test_dataset, test_dataloader, image_processor, id2label, label2id = (
        get_dataloaders(image_processor_checkpoint=experiment_path, dataset="cppe-5", create_coco=True,
                        coco_path=coco_path)
    )

    # Load evaluator
    module = evaluate.load("ybelkada/cocoevaluate", coco=test_dataset.coco)

    # Load model
    model = AutoModelForObjectDetection.from_pretrained(
        experiment_path,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.eval()

    args = load_object(osp.join(experiment_path, 'args.pkl'))

    # PEFT Model
    print("-" * 100)
    if args['pmodel']['method'].lower() != 'none':
        print(f"=> Evaluating PEFTNet model: {args['pmodel']['method']}")
        cmodel = pn.PEFTNet(model, ignore_regex=".*conv(1|3).*", **args['pmodel'])
        print(f"\n{cmodel.get_report()}\n")
        model = cmodel.peft_model
    else:
        print("=> Evaluating baseline model")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader)):
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]

            labels = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  # these are in DETR format, resized + normalized

            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = image_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api

            module.add(prediction=results, reference=labels)
            del batch

    results = module.compute()
    print(results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment_path', type=str, default=None)
    args = parser.parse_args()

    main(args)

