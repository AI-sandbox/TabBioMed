import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tabbiomed.models import checkpoints


class ModelCheckpoints(unittest.TestCase):
    def test_explicit_local_weights_do_not_download(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.ckpt"
            path.write_bytes(b"local weights")
            with patch("huggingface_hub.hf_hub_download") as download:
                self.assertEqual(
                    checkpoints.resolve_checkpoint("tabpfn_2_5", path.name, folder),
                    path,
                )
                download.assert_not_called()
            with self.assertRaises(FileNotFoundError):
                checkpoints.resolve_checkpoint("tabpfn_2_5", "missing.ckpt", folder)

    def test_new_weights_are_verified_and_reused(self):
        payload = b"published weights"
        source = {
            "tabpfn_2_5": {
                "repo_id": "author/model",
                "revision": "pinned",
                "sha256": {"model.ckpt": hashlib.sha256(payload).hexdigest()},
            }
        }
        with tempfile.TemporaryDirectory() as folder:

            def fetch(repo_id, filename, **kwargs):
                self.assertEqual(repo_id, "author/model")
                self.assertEqual(kwargs["revision"], "pinned")
                (kwargs["local_dir"] / filename).write_bytes(payload)

            with (
                patch.dict(os.environ, TABBIOMED_CHECKPOINTS=folder),
                patch.object(
                    checkpoints, "json", Mock(loads=Mock(return_value=source))
                ),
                patch("huggingface_hub.hf_hub_download", side_effect=fetch) as download,
            ):
                first = checkpoints.resolve_checkpoint("tabpfn_2_5", "model.ckpt")
                second = checkpoints.resolve_checkpoint("tabpfn_2_5", "model.ckpt")
                self.assertEqual(first, second)
                self.assertEqual(first.read_bytes(), payload)
                download.assert_called_once()

    def test_incorrect_cached_weights_are_rejected(self):
        payload = b"published weights"
        source = {
            "tabpfn_2_5": {
                "sha256": {"model.ckpt": hashlib.sha256(payload).hexdigest()}
            }
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tabpfn_2_5/model.ckpt"
            path.parent.mkdir()
            with (
                patch.dict(os.environ, TABBIOMED_CHECKPOINTS=folder),
                patch.object(
                    checkpoints, "json", Mock(loads=Mock(return_value=source))
                ),
                patch("huggingface_hub.hf_hub_download") as download,
            ):
                for verified_before in (False, True):
                    with self.subTest(verified_before=verified_before):
                        if verified_before:
                            path.write_bytes(payload)
                            checkpoints.resolve_checkpoint("tabpfn_2_5", path.name)
                        path.write_bytes(b"incorrect cached weights")
                        with self.assertRaisesRegex(
                            ValueError, "Checkpoint identity differs"
                        ):
                            checkpoints.resolve_checkpoint("tabpfn_2_5", path.name)
                        self.assertFalse(path.exists())
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
