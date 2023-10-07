import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor

import albumentations


def collate_fn(batch, image_processor):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


def remove_invalid_images(dataset):
    invalid_indices = []
    for idx, data in enumerate(dataset):
        width, height = data["width"], data["height"]
        for bbox in data["objects"]["bbox"]:
            x, y, w, h = bbox
            # Checking if any bounding box coordinate is outside image dimensions
            if x < 0 or y < 0 or (x + w) > width or (y + h) > height:
                invalid_indices.append(idx)
                print(f"Invalid image found at index {idx}")
                break
    # Return a new dataset with only the valid images
    valid_indices = [i for i in range(len(dataset)) if i not in invalid_indices]
    return dataset.select(valid_indices)


def formatted_anns(image_id, category, area, bbox):
    """Formats the annotations to the format expected by DETR"""
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


def transform_aug_ann(examples, image_processor, apply_aug=True):
    """Transforms the dataset to the format expected by DETR"""
    transform = albumentations.Compose(
        [
            albumentations.Resize(480, 480),
            albumentations.HorizontalFlip(p=1.0),
            albumentations.RandomBrightnessContrast(p=1.0),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
    )

    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        if apply_aug:
            out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        else:
            out = {"image": image, "bboxes": objects["bbox"], "category": objects["category"]}

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def get_dataloaders(image_processor_checkpoint="facebook/detr-resnet-50", dataset="cppe-5"):

    image_processor = AutoImageProcessor.from_pretrained(image_processor_checkpoint)
    train_dataset = load_dataset(dataset, split="train")

    # Remove invalid images
    remove_idx = [590, 821, 822, 875, 876, 878, 879]
    keep = [i for i in range(len(train_dataset)) if i not in remove_idx]
    train_dataset = train_dataset.select(keep)
    train_dataset = remove_invalid_images(train_dataset)

    train_dataset = train_dataset.with_transform(
        lambda x: transform_aug_ann(x, image_processor)
    )

    categories = train_dataset.features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    test_dataset = load_dataset(dataset, split="test")
    test_dataset = remove_invalid_images(test_dataset)
    test_dataset = test_dataset.with_transform(
        lambda x: transform_aug_ann(x, image_processor)
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: collate_fn(x, image_processor)
    )

    return train_dataset, test_dataset, test_dataloader, image_processor, id2label, label2id
