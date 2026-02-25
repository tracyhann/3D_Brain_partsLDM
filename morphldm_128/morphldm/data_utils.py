import json
import os
from typing import Sequence

import pandas as pd
from monai import transforms
from monai.data import DataLoader, Dataset


def parse_n_samples(n_samples):
    if n_samples is None:
        return None
    if isinstance(n_samples, str):
        value = n_samples.strip()
        if value == "" or value.upper() == "ALL":
            return None
        return int(value)
    return int(n_samples)


def build_train_transforms(spacing=(1.0, 1.0, 1.0), channel=0, pad_k=32):
    keys = ["image"]
    spacing = tuple(float(v) for v in spacing)
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
            transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            transforms.EnsureTyped(keys=keys),
            transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear")),
            transforms.DivisiblePadd(
                keys=keys, k=pad_k, mode="constant", constant_values=-1.0
            ),
        ]
    )


def make_dataloaders_from_csv(
    csv_path,
    data_split_json_path,
    conditions=("age", "sex", "vol", "group"),
    train_transforms=None,
    n_samples=None,
    batch_size=1,
    num_workers=8,
    data_root=None,
):
    csv_path = os.path.expanduser(str(csv_path))
    split_path = os.path.expanduser(str(data_split_json_path))
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(csv_path)
    if not os.path.isabs(split_path):
        split_path = os.path.abspath(split_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    csv_size = os.path.getsize(csv_path)
    if csv_size == 0:
        raise ValueError(
            f"CSV file is empty: {csv_path}. "
            "Set `csv_path` to a non-empty table (e.g., ../data/whole_brain_0120.csv)."
        )
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split JSON file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    image_ids = {}
    for split_name in ("train", "val", "test"):
        for image_id in splits[split_name]:
            image_ids[image_id] = split_name

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"CSV has no parsable columns: {csv_path} (size={csv_size} bytes)."
        ) from exc
    train_data, val_data, test_data = [], [], []
    train_count = 0
    n_samples = parse_n_samples(n_samples)
    conditions = list(conditions)

    required_columns = {"image", "imageID", "age", "sex"}
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {missing_columns}")

    def resolve_image_path(path_value):
        raw = os.path.expanduser(str(path_value))
        candidates = []
        if os.path.isabs(raw):
            candidates.append(raw)
        else:
            candidates.append(os.path.abspath(raw))
            if data_root is not None:
                data_root_abs = os.path.abspath(os.path.expanduser(str(data_root)))
                candidates.append(os.path.abspath(os.path.join(data_root_abs, raw)))
            csv_dir = os.path.dirname(csv_path)
            candidates.append(os.path.abspath(os.path.join(csv_dir, raw)))
            candidates.append(os.path.abspath(os.path.join(os.path.dirname(csv_dir), raw)))
            candidates.append(
                os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(csv_dir)), raw))
            )

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                unique_candidates.append(candidate)
                seen.add(candidate)

        for candidate in unique_candidates:
            if os.path.exists(candidate):
                return candidate, unique_candidates
        return None, unique_candidates

    missing_path_records = []

    for _, row in df.iterrows():
        image_id = row["imageID"]
        split_name = image_ids.get(image_id)
        if split_name is None:
            continue

        resolved_image_path, tried_candidates = resolve_image_path(row["image"])
        if resolved_image_path is None:
            missing_path_records.append(
                {
                    "imageID": image_id,
                    "image": row["image"],
                    "tried": tried_candidates[:4],
                }
            )
            continue

        sample = {"image": resolved_image_path}
        for condition in conditions:
            if condition in row:
                sample[condition] = row[condition]

        if split_name == "train":
            train_data.append(sample)
            train_count += 1
            if n_samples is not None and train_count >= n_samples:
                break
        elif split_name == "val":
            val_data.append(sample)
        elif split_name == "test":
            test_data.append(sample)

    if not train_data:
        raise ValueError(
            "No training samples were found. Check csv_path/data_split_json_path and imageID values."
        )
    if not val_data:
        raise ValueError(
            "No validation samples were found. Check csv_path/data_split_json_path and imageID values."
        )
    if missing_path_records:
        preview = missing_path_records[:3]
        raise FileNotFoundError(
            "Some image files from CSV could not be resolved. "
            f"missing_count={len(missing_path_records)}. "
            f"Examples: {preview}. "
            "Set config `data_root` so relative image paths in CSV can be resolved."
        )

    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=train_transforms)
    test_ds = Dataset(data=test_data, transform=train_transforms)

    print(f"CSV: {csv_path}")
    print(f"Split JSON: {split_path}")
    if data_root is not None:
        print(f"Data root: {os.path.abspath(os.path.expanduser(str(data_root)))}")
    print(f"Transformed data shape: {train_ds[0]['image'].shape}")
    print(f"Number of training samples: {len(train_ds)}")
    print(f"Number of validation samples: {len(val_ds)}")
    print(f"Number of test samples: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
