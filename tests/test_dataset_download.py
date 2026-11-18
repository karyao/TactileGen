"""Downloader behavior using tiny archives; no network or optional dependencies."""
import io
import json
from pathlib import Path
import shutil
import struct
import sys
from types import SimpleNamespace
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from src.dataset import download as dl


PNG = b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 10, 10)


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "data"

    def plotqa(self):
        archive = self.root / "images.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            entry = tarfile.TarInfo("png/0.png")
            entry.size = len(PNG)
            stream.addfile(entry, io.BytesIO(PNG))
        annotations = self.root / "annotations.json"
        annotations.write_text(json.dumps([{"image_index": 0}]))

        def transfer(asset, destination):
            source = archive if asset["id"] == dl.SOURCES["plotqa"]["splits"]["val"]["images"] else annotations
            shutil.copyfile(source, destination)
        return transfer

    def test_provider_arguments(self):
        from unittest.mock import Mock
        destination = self.root / "download"
        drive = SimpleNamespace(download=Mock(return_value=str(destination)))
        with patch.dict(sys.modules, {"gdown": drive}):
            dl.transfer({"provider": "drive", "id": "official-id"}, destination)
        drive.download.assert_called_once_with(id="official-id", output=str(destination), quiet=False)
        cached = self.root / "cached.zip"
        cached.write_bytes(b"archive")
        hub = SimpleNamespace(hf_hub_download=Mock(return_value=str(cached)))
        config = dl.SOURCES["chartqa"]
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            dl.transfer({"provider": "hf", **config}, destination)
        self.assertEqual(destination.read_bytes(), b"archive")
        self.assertEqual(hub.hf_hub_download.call_args.kwargs["revision"], config["revision"])
        self.assertIs(hub.hf_hub_download.call_args.kwargs["token"], False)

    def test_broken_download_receipt_is_replaced(self):
        with patch.object(dl, "transfer", side_effect=self.plotqa()) as transfer:
            dl.setup("plotqa", self.output, ["val"], True)
            (self.output / "downloads/val-annotations.json.json").write_text("{")
            dl.setup("plotqa", self.output, ["val"], True)
            self.assertEqual(transfer.call_count, 3)

    def test_plotqa_setup_rerun_and_split_isolation(self):
        with patch.object(dl, "transfer", side_effect=self.plotqa()) as transfer:
            dl.setup("plotqa", self.output, ["val"])
            self.assertEqual(transfer.call_count, 2)
            dl.setup("plotqa", self.output, ["val"])
            self.assertEqual(transfer.call_count, 2)
        self.assertEqual((self.output / "val/png/0.png").read_bytes(), PNG)
        self.assertFalse((self.output / "train").exists())

    def test_corrupt_cache_is_downloaded_again(self):
        with patch.object(dl, "transfer", side_effect=self.plotqa()) as transfer:
            dl.setup("plotqa", self.output, ["val"], True)
            (self.output / "downloads/val-images.tar.gz").write_bytes(b"bad")
            dl.setup("plotqa", self.output, ["val"], True)
            self.assertEqual(transfer.call_count, 3)
        self.assertFalse((self.output / "val").exists())

    def test_interruption_does_not_publish_download(self):
        def fail(asset, destination):
            destination.write_bytes(b"partial")
            raise OSError("offline")
        with patch.object(dl, "transfer", side_effect=fail):
            with self.assertRaisesRegex(OSError, "offline"):
                dl.setup("plotqa", self.output, ["val"])
        self.assertEqual(list((self.output / "downloads").iterdir()), [])

    def test_preserves_existing_split(self):
        (self.output / "val").mkdir(parents=True)
        sentinel = self.output / "val/user.txt"
        sentinel.write_text("keep")
        with patch.object(dl, "transfer", side_effect=self.plotqa()):
            with self.assertRaises(FileExistsError):
                dl.setup("plotqa", self.output, ["val"])
        self.assertEqual(sentinel.read_text(), "keep")

    def test_missing_prepared_image_rejected(self):
        with patch.object(dl, "transfer", side_effect=self.plotqa()):
            dl.setup("plotqa", self.output, ["val"])
            (self.output / "val/png/0.png").unlink()
            with self.assertRaises(FileExistsError):
                dl.setup("plotqa", self.output, ["val"])

    def test_invalid_annotations_leave_no_partial_split(self):
        transfer = self.plotqa()
        (self.root / "annotations.json").write_text("[]")
        with patch.object(dl, "transfer", side_effect=transfer):
            with self.assertRaisesRegex(ValueError, "no images"):
                dl.setup("plotqa", self.output, ["val"])
        self.assertFalse((self.output / "val").exists())
        self.assertEqual(list(self.output.glob(".val-*")), [])

    def test_chartqa_shared_archive_and_structure(self):
        archive = self.root / "chartqa.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            for split in ("train", "val", "test"):
                prefix = f"ChartQA Dataset/{split}"
                for name, data in (("png/example.png", PNG), ("annotations/example.json", b"{}"),
                                   ("tables/example.csv", b"x,y\na,1\n")):
                    stream.writestr(f"{prefix}/{name}", data)
                for kind in ("human", "augmented"):
                    stream.writestr(f"{prefix}/{split}_{kind}.json", json.dumps([{"imgname": "example.png"}]))
        sources = {**dl.SOURCES, "chartqa": {**dl.SOURCES["chartqa"],
                   "sha256": dl.digest(archive), "size": archive.stat().st_size}}
        with patch.object(dl, "SOURCES", sources), patch.object(dl, "transfer", side_effect=lambda a, d: shutil.copyfile(archive, d)) as transfer:
            dl.setup("chartqa", self.output, ["train", "val"])
            self.assertEqual(transfer.call_count, 1)
            dl.setup("chartqa", self.output, ["train", "val"])
            self.assertEqual(transfer.call_count, 1)
        self.assertTrue((self.output / "val/annotations/example.json").is_file())
        self.assertFalse((self.output / "test").exists())

    def test_wrong_published_checksum_and_html_are_rejected(self):
        for content, extra in ((b"payload", {"sha256": "wrong"}), (b"<!DOCTYPE html>quota", {})):
            with self.subTest(content=content):
                with patch.object(dl, "transfer", side_effect=lambda a, d: d.write_bytes(content)):
                    with self.assertRaises(ValueError):
                        dl.fetch({"provider": "drive", "id": "example", **extra}, self.output, "file")
                self.assertFalse((self.output / "file").exists())

    def test_rejects_archive_traversal_and_links(self):
        for name in ("../outside", "/absolute", "dir\\evil"):
            archive = self.root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr(name, "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                dl.unpack(archive, self.root / "extracted")
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            archive = self.root / "unsafe.tar"
            with tarfile.open(archive, "w") as stream:
                member = tarfile.TarInfo("link")
                member.type = kind
                member.linkname = "outside"
                stream.addfile(member)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                dl.unpack(archive, self.root / "extracted")
        archive = self.root / "symlink.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            member = zipfile.ZipInfo("link")
            member.external_attr = 0o120777 << 16
            stream.writestr(member, "outside")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            dl.unpack(archive, self.root / "extracted")


if __name__ == "__main__":
    unittest.main()
