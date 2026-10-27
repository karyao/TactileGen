"""Create a YOLO dataset without exact image duplicates across splits.

Keep test images first, then validation images, then training images. Images are
hard-linked to avoid duplicating PNG storage; labels are copied independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


# Earlier splits win when the same PNG appears in more than one split.
SPLITS = ("test", "val", "train")


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_data_yaml(output: Path, class_names: list[str]) -> None:
    """Write Ultralytics split paths and class IDs for this local dataset."""
    (output / "data.yaml").write_text(
        "# YOLO detection dataset; generated from classes.txt\n"
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(class_names)),
        encoding="utf-8",
    )


def read_manifest(
    path: Path, image_ids: set[str], split: str
) -> tuple[list[str], list[dict[str, str]]] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or "image_id" not in fields:
            raise ValueError(f"Invalid manifest: {path}")
        rows = list(reader)
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != image_ids:
        raise ValueError(f"Manifest/image IDs differ in {split}")
    return fields, rows


def clean_dataset(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if not (source / "classes.txt").is_file():
        raise FileNotFoundError(f"Missing classes.txt in {source}")
    class_names = (source / "classes.txt").read_text(encoding="utf-8").splitlines()
    if not class_names or any(not name or not name.replace("_", "").isalnum() for name in class_names):
        raise ValueError("classes.txt must contain simple, nonempty class names")

    # Validate every split and decide what to keep before creating output.
    kept: dict[str, list[tuple[Path, Path]]] = {}
    removed: dict[str, list[dict[str, str]]] = {}
    seen: dict[str, str] = {}
    manifests: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for split in SPLITS:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images or labels for {split}")
        images = {path.stem: path for path in image_dir.glob("*.png")}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        if images.keys() != labels.keys():
            raise ValueError(f"Image/label IDs differ in {split}")
        manifest = read_manifest(source / f"chart_types_{split}.csv", set(images), split)
        if manifest is not None:
            manifests[split] = manifest

        kept[split] = []
        removed[split] = []
        for image_id, image in sorted(images.items(), key=lambda item: int(item[0])):
            digest = image_hash(image)
            previous_split = seen.get(digest)
            if previous_split is not None and previous_split != split:
                removed[split].append({
                    "image_id": image_id,
                    "sha256": digest,
                    "matches_split": previous_split,
                })
            else:
                kept[split].append((image, labels[image_id]))
                # Repeated images within one split do not leak across splits.
                seen.setdefault(digest, split)

    # Only the selected images and their labels go into the new dataset.
    output.mkdir(parents=True)
    shutil.copy2(source / "classes.txt", output / "classes.txt")
    write_data_yaml(output, class_names)
    for split in SPLITS:
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image, label in kept[split]:
            (image_dir / image.name).hardlink_to(image)
            shutil.copy2(label, label_dir / label.name)
        if split in manifests:
            fields, rows = manifests[split]
            kept_ids = {image.stem for image, _ in kept[split]}
            with (output / f"chart_types_{split}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(row for row in rows if row["image_id"] in kept_ids)

    report = {
        "policy": "Keep test, then val, then train; remove exact SHA-256 matches across splits",
        "source": str(source),
        "counts": {
            split: {"kept": len(kept[split]), "removed": len(removed[split])}
            for split in SPLITS
        },
        "removed": removed,
    }
    (output / "leakage_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset_yolo_vbars"))
    parser.add_argument("--output", type=Path, default=Path("dataset_yolo_vbars_clean"))
    args = parser.parse_args()
    report = clean_dataset(args.source, args.output)
    for split in SPLITS:
        counts = report["counts"][split]
        print(f"{split}: kept {counts['kept']}, removed {counts['removed']}")
    print(f"Clean dataset: {args.output}")


if __name__ == "__main__":
    main()
