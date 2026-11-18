"""Download and safely unpack official PlotQA or ChartQA assets.

    python -m src.dataset.download plotqa --splits val
    python -m src.dataset.download chartqa

Install requirements-dataset.txt first. Only managed cache files are replaced;
existing split directories are never overwritten. Run one downloader per output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import zipfile

SPLITS = ("train", "val", "test")
SOURCES = json.loads(Path(__file__).with_name("sources.json").read_text())


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def transfer(asset: dict, destination: Path) -> None:
    """Dependencies are lazy so offline checks and tests use only stdlib."""
    if asset["provider"] == "drive":
        import gdown
        if not gdown.download(id=asset["id"], output=str(destination), quiet=False):
            raise OSError("Google Drive returned no file (check access or download quota)")
    else:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(
            repo_id=asset["repo_id"], filename=asset["filename"],
            revision=asset["revision"], repo_type="dataset", token=False,
            local_dir=destination.parent / "hf-cache",
        )
        shutil.copyfile(cached, destination)


def _valid_receipt(record: object, asset: dict) -> bool:
    """Reject malformed cache metadata before inspecting the cached file."""
    if not isinstance(record, dict):
        return False
    size = record.get("size")
    checksum = record.get("sha256")
    return (
        isinstance(record.get("asset"), dict) and record["asset"] == asset
        and type(size) is int and size > 0
        and isinstance(checksum, str) and len(checksum) == 64
        and all(character in "0123456789abcdef" for character in checksum)
    )


def fetch(asset: dict, cache: Path, name: str) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    receipt = target.with_suffix(target.suffix + ".json")
    if target.is_file() and receipt.is_file():
        try:
            record = json.loads(receipt.read_text())
        except (ValueError, OSError):
            record = {}
        if (_valid_receipt(record, asset) and target.stat().st_size == record["size"]
                and digest(target) == record["sha256"]):
            print(f"Verified cached download: {name}", flush=True)
            return target
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {name}", flush=True)
    try:
        transfer(asset, temporary)
        size = temporary.stat().st_size
        checksum = digest(temporary)
        if not size or ("size" in asset and size != asset["size"]):
            raise ValueError(f"Unexpected download size for {name}: {size}")
        if "sha256" in asset and checksum != asset["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")
        # Drive sometimes returns an HTML access/quota page instead of data.
        with temporary.open("rb") as stream:
            prefix = stream.read(256).lstrip().lower()
        if prefix.startswith((b"<!doctype html", b"<html")):
            raise ValueError(f"{name}: received HTML; check Drive access/download quota")
        temporary.replace(target)
        write_json(receipt, {"asset": asset, "size": size, "sha256": checksum})
    finally:
        temporary.unlink(missing_ok=True)
    return target


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return root.joinpath(*relative.parts)


def unpack(archive: Path, output: Path, split: str | None = None) -> None:
    """Extract regular files/directories only, never archive symlinks or devices."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                target = safe_path(output, item.filename)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError(f"Unsupported archive member: {item.filename}")
                if split is not None and split not in PurePosixPath(item.filename).parts:
                    continue
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(item) as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
    else:
        with tarfile.open(archive) as source:
            for item in source:
                target = safe_path(output, item.name)
                if item.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif item.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(item) as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
                else:
                    raise ValueError(f"Unsupported archive member: {item.name}")


def inventory(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != ".setup.json"}


def completed(output: Path, identity: dict) -> bool:
    if not output.exists():
        return False
    marker = output / ".setup.json"
    if marker.is_file():
        try:
            record = json.loads(marker.read_text())
        except (ValueError, OSError):
            record = {}
        if record.get("identity") == identity and record.get("files") == inventory(output):
            print(f"Verified prepared split: {output}", flush=True)
            return True
    raise FileExistsError(f"Unmanaged, changed, or incomplete split: {output}. Choose a new --output path.")


def validate_plotqa(root: Path) -> None:
    from src.dataset.prepare_plotqa import iter_annotations, png_size
    count = 0
    for annotation in iter_annotations(root / "annotations.json"):
        png_size(root / "png" / f"{annotation['image_index']}.png")
        count += 1
    if not count:
        raise ValueError("PlotQA annotations contain no images")


def validate_chartqa(root: Path, split: str) -> None:
    for directory, suffix in (("png", ".png"), ("annotations", ".json"), ("tables", ".csv")):
        if not any((root / directory).glob(f"*{suffix}")):
            raise ValueError(f"ChartQA {split}: missing or empty {directory}")
    for kind in ("human", "augmented"):
        path = root / f"{split}_{kind}.json"
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Invalid ChartQA QA file: {path}")
        for row in rows:
            name = row["imgname"]
            if Path(name).name != name or not (root / "png" / name).is_file():
                raise ValueError(f"ChartQA QA references missing/invalid image: {name}")


def setup(dataset: str, output: Path, splits: list[str], download_only: bool = False) -> None:
    config = SOURCES[dataset]
    cache = output / "downloads"
    chart_archive = None
    for split in dict.fromkeys(splits):
        assets = ([{"provider": "drive", "id": config["splits"][split][kind]}
                   for kind in ("images", "annotations")] if dataset == "plotqa" else
                  [{"provider": "hf", **{key: config[key] for key in
                    ("repo_id", "revision", "filename", "size", "sha256")}}])
        if dataset == "plotqa":
            archives = [fetch(asset, cache, f"{split}-{kind}{suffix}")
                        for asset, kind, suffix in zip(assets, ("images", "annotations"), (".tar.gz", ".json"))]
        else:
            if chart_archive is None:
                chart_archive = fetch(assets[0], cache, "chartqa.zip")
            archives = [chart_archive]
        if download_only:
            continue
        identity = {"dataset": dataset, "split": split, "assets": assets,
                    "sha256": [json.loads(path.with_suffix(path.suffix + ".json").read_text())["sha256"]
                               for path in archives]}
        target = output / split
        if completed(target, identity):
            continue
        with tempfile.TemporaryDirectory(prefix=f".{split}-", dir=output) as temporary:
            staging = Path(temporary)
            raw = staging / "raw"
            raw.mkdir()
            unpack(archives[0], raw, split if dataset == "chartqa" else None)
            if dataset == "plotqa":
                candidates = [p for p in raw.rglob("*") if p.is_dir() and any(p.glob("*.png"))]
                if len(candidates) != 1:
                    raise ValueError("Expected exactly one PlotQA PNG directory")
                prepared = staging / "prepared"
                prepared.mkdir()
                candidates[0].rename(prepared / "png")
                shutil.copyfile(archives[1], prepared / "annotations.json")
                validate_plotqa(prepared)
            else:
                candidates = [p for p in raw.rglob(split) if p.is_dir() and (p / "png").is_dir()]
                if len(candidates) != 1:
                    raise ValueError(f"Expected exactly one ChartQA {split} directory")
                prepared = candidates[0]
                validate_chartqa(prepared, split)
            write_json(prepared / ".setup.json", {"identity": identity, "files": inventory(prepared)})
            prepared.rename(target)
            print(f"Prepared {target}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(SOURCES))
    parser.add_argument("--output", type=Path, help="default: data/<dataset>")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        setup(args.dataset, args.output or Path("data") / args.dataset, args.splits, args.download_only)
    except KeyboardInterrupt:
        parser.exit(130, "Download interrupted; rerun the command to retry.\n")
    except ImportError as exc:
        parser.exit(2, f"error: {exc}. Install requirements-dataset.txt.\n")
    except Exception as exc:
        parser.exit(2, f"error: {exc}\nCheck source access, quota, network, and free disk space; rerun to retry.\n")


if __name__ == "__main__":
    main()
