import os
import json
from tqdm import tqdm
import torch
import torchvision
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from utils_cv import get_dataloaders

# def get_dataloaders():
#
#     image_processor_checkpoint = "facebook/detr-resnet-50"
#     image_processor = AutoImageProcessor.from_pretrained(image_processor_checkpoint)
#
#     train_dataset = load_dataset("cppe-5", split="train")
#     train_dataset = remove_invalid_images(train_dataset)
#     train_dataset = train_dataset.with_transform(
#         lambda x: transform_aug_ann(x, image_processor)
#     )
#
#     categories = train_dataset.features["objects"].feature["category"].names
#     id2label = {index: x for index, x in enumerate(categories, start=0)}
#     label2id = {v: k for k, v in id2label.items()}
#
#     return train_dataset, image_processor, id2label, label2id
#
#
# def collate_fn(batch, image_processor):
#     pixel_values = [item["pixel_values"] for item in batch]
#     encoding = image_processor.pad(pixel_values, return_tensors="pt")
#     labels = [item["labels"] for item in batch]
#     return {
#         "pixel_values": encoding["pixel_values"],
#         "pixel_mask": encoding["pixel_mask"],
#         "labels": labels
#     }
#
#
# def val_formatted_anns(image_id, objects):
#     return [
#         {
#             "id": objects["id"][i],
#             "category_id": objects["category"][i],
#             "iscrowd": 0,
#             "image_id": image_id,
#             "area": objects["area"][i],
#             "bbox": objects["bbox"][i],
#         }
#         for i in range(len(objects["id"]))
#     ]
#
#
# def save_cppe5_annotation_file_images(dataset, id2label):
#     output_json = {}
#     path_output_cppe5 = os.path.join(os.getcwd(), "cppe5")
#
#     if not os.path.exists(path_output_cppe5):
#         os.makedirs(path_output_cppe5)
#
#     path_anno = os.path.join(path_output_cppe5, "cppe5_ann.json")
#     categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
#
#     output_json["images"] = [{
#         "image_id": example["image_id"],
#         "width": example["image"].width,
#         "height": example["image"].height,
#         "file_name": f"{example['image_id']}.png",
#     } for example in dataset]
#
#     output_json["annotations"] = [val_formatted_anns(example["image_id"], example["objects"]) for example in dataset]
#     output_json["categories"] = categories_json
#
#     with open(path_anno, "w") as file:
#         json.dump(output_json, file, ensure_ascii=False, indent=4)
#
#     for im, img_id in zip(dataset["image"], dataset["image_id"]):
#         path_img = os.path.join(path_output_cppe5, f"{img_id}.png")
#         im.save(path_img)
#
#     return path_output_cppe5, path_anno
#
#
# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, image_processor_instance, ann_file):
#         super().__init__(img_folder, ann_file)
#         self.image_processor = image_processor_instance
#
#     def __getitem__(self, idx):
#         img, target = super().__getitem__(idx)
#         image_id = self.ids[idx]
#         target = {"image_id": image_id, "annotations": target}
#         encoding = self.image_processor(images=img, annotations=target, return_tensors="pt")
#         return {
#             "pixel_values": encoding["pixel_values"].squeeze(),
#             "labels": encoding["labels"][0]
#         }


def main():
    # checkpoint = "/home/mila/m/marawan.gamal/scratch/detr-resnet-50_finetuned_cppe5/checkpoint-1200"
    # image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    #
    # cppe5 = load_dataset("cppe-5")
    # categories = cppe5["test"].features["objects"].feature["category"].names
    # id2label = {index: name for index, name in enumerate(categories)}
    # label2id = {name: index for index, name in id2label.items()}
    #
    # path_output_cppe5, path_anno = save_cppe5_annotation_file_images(cppe5["test"], id2label)
    # test_ds_coco_format = CocoDetection(path_output_cppe5, image_processor, path_anno)
    #
    # model = AutoModelForObjectDetection.from_pretrained(checkpoint)
    # evaluator_module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    # val_dataloader = torch.utils.data.DataLoader(
    #     test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4,
    #     collate_fn=lambda x: collate_fn(x, image_processor)
    # )

    checkpoint = "/home/mila/m/marawan.gamal/scratch/detr-resnet-50_finetuned_cppe5/checkpoint-1200"
    model = AutoModelForObjectDetection.from_pretrained(checkpoint)
    train_dataset, test_dataloader, image_processor, id2label, label2id = get_dataloaders()

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
            orig_target_sizes = torch.stack([target["orig_size"] for target in batch["labels"]], dim=0)
            results = image_processor.post_process(outputs, orig_target_sizes)
            # evaluator_module.add(prediction=results, reference=batch["labels"])

    # results = evaluator_module.compute()
    # print(results)


if __name__ == '__main__':
    main()
