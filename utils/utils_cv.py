import json
import os

import numpy as np
import torch
import torchvision
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoImageProcessor

from transformers import DefaultDataCollator

import albumentations
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def coco_bbox_to_pascal_bbox(bbox, img_dims=(1, 1)):
    """Convert normalized [x-top-left, y-top-left, width, height] to normalized [xmin, ymin, xmax, ymax].

    Args:
    - bbox (list): List containing normalized [x_top_left, y_top_left, width, height]
    - img_dims (list): List containing [image_height, image_width]

    Returns:
    - List containing normalized [xmin, ymin, xmax, ymax]
    """

    # Extracting normalized values from the bbox list
    x_top_left, y_top_left, width, height = bbox
    img_height, img_width = img_dims

    # Calculating normalized xmin, ymin, xmax, ymax
    xmin = x_top_left * img_width
    ymin = y_top_left * img_height
    xmax = xmin + (width * img_width)
    ymax = ymin + (height * img_height)

    normalized_bbox = [xmin / img_width, ymin / img_height, xmax / img_width, ymax / img_height]

    return normalized_bbox


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


def get_dataloaders_od(image_processor_checkpoint="facebook/detr-resnet-50", dataset="cppe-5", create_coco=False,
                       coco_path=os.getcwd()):
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

    test_dataset_ = test_dataset.with_transform(
        lambda x: transform_aug_ann(x, image_processor)
    )
    test_dataloader_ = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: collate_fn(x, image_processor)
    )

    if create_coco:
        path_output_cppe5, path_anno = save_cppe5_annotation_file_images(test_dataset, id2label, coco_path)
        test_ds_coco_format = CocoDetection(path_output_cppe5, image_processor, path_anno)
        test_dl_coco_format = torch.utils.data.DataLoader(
            test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4,
            collate_fn=lambda x: collate_fn(x, image_processor)
        )

        return (train_dataset, test_dataset_, test_dataloader_, test_ds_coco_format, test_dl_coco_format,
                image_processor, id2label, label2id)

    return train_dataset, test_dataset_, test_dataloader_, image_processor, id2label, label2id


def val_formatted_anns(image_id, objects):
    annotations = []
    for i in range(0, len(objects["id"])):
        new_ann = {
            "id": objects["id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": objects["bbox"][i],
        }
        annotations.append(new_ann)

    return annotations


def save_cppe5_annotation_file_images(cppe5, id2label, path_root):
    # Save images and annotations into the files torchvision.datasets.CocoDetection expects
    output_json = {}
    # path_output_cppe5 = f"{os.getcwd()}/cppe5/"
    path_output_cppe5 = os.path.join(path_root, "cppe5")

    if not os.path.exists(path_output_cppe5):
        os.makedirs(path_output_cppe5)

    path_anno = os.path.join(path_output_cppe5, "cppe5_ann.json")
    categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
    output_json["images"] = []
    output_json["annotations"] = []

    for example in cppe5:
        ann = val_formatted_anns(example["image_id"], example["objects"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": example["image"].width,
                "height": example["image"].height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    for im, img_id in zip(cppe5["image"], cppe5["image_id"]):
        path_img = os.path.join(path_output_cppe5, f"{img_id}.png")
        im.save(path_img)

    return path_output_cppe5, path_anno


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, image_processor, ann_file):
        super().__init__(img_folder, ann_file)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target: converting target to DETR format,
        # resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.image_processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return {"pixel_values": pixel_values, "labels": target}


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


# Image Classification

dataset_to_split = {
    "Flowers102": {
        "train": "train",
        "test": "val",
    },
    "StanfordCars": {
        "train": "train",
        "test": "test",
    },
    "FGVCAircraft": {
        "train": "train",
        "test": "val",
    }
}


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, name="Flowers102", transform=torchvision.transforms.ToTensor()):

        assert name in dataset_to_split.keys(), \
            f"Dataset {name} not supported. Only {dataset_to_split.keys()} are supported."

        dataset_obj = getattr(torchvision.datasets, name)
        self.dataset = dataset_obj(
            root=root, split=split, download=True, transform=transform
        )
        self.labels = set([item[1] for item in self.dataset])

    def __getitem__(self, idx):
        return {"pixel_values": self.dataset[idx][0], "label": self.dataset[idx][1]}

    def __len__(self):
        return len(self.dataset)


def get_dataloaders_ic(
        root=os.getcwd(),
        name="Flowers102",
        image_processor_checkpoint="microsoft/resnet-50",
):
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    #     transforms.RandomCrop(32, padding=4),  # Randomly crop the image and pad it
    #     transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])

    image_processor = AutoImageProcessor.from_pretrained(image_processor_checkpoint)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    train_dataset = DictDataset(root=root, split=dataset_to_split[name]["train"], name=name, transform=_transforms)
    test_dataset = DictDataset(root=root, split=dataset_to_split[name]["test"], name=name, transform=_transforms)

    labels = train_dataset.labels

    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    data_collator = DefaultDataCollator()

    return train_dataset, test_dataset, data_collator, image_processor, id2label, label2id, labels


def get_dataloaders_ic_hf(image_processor_checkpoint="microsoft/resnet-50", dataset_name="food101", split="train[:5000]"):

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.train_test_split(test_size=0.2)
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    image_processor = AutoImageProcessor.from_pretrained(image_processor_checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    dataset = dataset.with_transform(transforms)
    data_collator = DefaultDataCollator()

    return dataset['train'], dataset['test'], data_collator, image_processor, id2label, label2id, labels
