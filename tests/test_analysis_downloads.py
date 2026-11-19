import json
from pathlib import Path
import tempfile
import unittest

from huggingface_hub.errors import HfHubHTTPError
from requests import Response
from unittest.mock import patch

from tabbiomed.analysis import downloads


class ResultDownloads(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.target = self.directory / "new/results"
        self.files = ["A/task/report.json", "A/task/predictions_best.pt"]
        self.source = {"repo_id": "TabBioMed/results", "revision": "pinned"}
        config = self.directory / "config"
        config.mkdir()
        (config / "results_source.json").write_text(json.dumps(self.source))
        for attribute, value in (
            ("CONFIG", config),
            ("DATA_CACHE", self.directory / "cache"),
        ):
            patcher = patch.object(downloads, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(downloads, "HfApi")
        self.api = patcher.start()
        self.addCleanup(patcher.stop)
        self.api.return_value.list_repo_files.return_value = self.files

    def fetch(self, **kwargs):
        self.assertEqual(kwargs["repo_id"], self.source["repo_id"])
        self.assertEqual(kwargs["revision"], "pinned")
        self.assertEqual(kwargs["repo_type"], "dataset")
        self.assertNotIn("token", kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")

    def test_creates_directory_and_downloads_missing_files(self):
        with patch.object(
            downloads, "hf_hub_download", side_effect=self.fetch
        ) as fetch:
            self.assertEqual(
                downloads.download_results(self.target), self.target.resolve()
            )
        self.assertCountEqual(
            [call.kwargs["filename"] for call in fetch.call_args_list], self.files
        )
        self.api.assert_called_once_with()

    def test_partial_directory_preserves_existing_files(self):
        existing = self.target / self.files[0]
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"local result")
        with patch.object(
            downloads, "hf_hub_download", side_effect=self.fetch
        ) as fetch:
            downloads.download_results(self.target)
        self.assertEqual(existing.read_bytes(), b"local result")
        self.assertEqual(fetch.call_args.kwargs["filename"], self.files[1])

    def test_complete_directory_reuses_cached_inventory_without_network(self):
        with patch.object(downloads, "hf_hub_download", side_effect=self.fetch):
            downloads.download_results(self.target)
        self.api.reset_mock()
        with patch.object(downloads, "hf_hub_download") as fetch:
            downloads.download_results(self.target)
            fetch.assert_not_called()
        self.api.assert_not_called()

    def test_interrupted_download_retries_only_remaining_files(self):
        def interrupt(**kwargs):
            if kwargs["filename"] == self.files[1]:
                raise ConnectionError("interrupted")
            self.fetch(**kwargs)

        with patch.object(downloads, "hf_hub_download", side_effect=interrupt):
            with self.assertRaises(ConnectionError):
                downloads.download_results(self.target)
        with patch.object(
            downloads, "hf_hub_download", side_effect=self.fetch
        ) as fetch:
            downloads.download_results(self.target)
        fetch.assert_called_once()
        self.assertEqual(fetch.call_args.kwargs["filename"], self.files[1])

    def test_rate_limit_retries_only_the_affected_file(self):
        counts = dict.fromkeys(self.files, 0)

        def attempt(**kwargs):
            name = kwargs["filename"]
            counts[name] += 1
            if name == self.files[1] and counts[name] == 1:
                response = Response()
                response.status_code = 429
                response.headers["RateLimit"] = '"api";r=0;t=9'
                raise HfHubHTTPError("quota exceeded", response=response)
            return self.fetch(**kwargs)

        with (
            patch("tabbiomed.data.hub.time.sleep") as sleep,
            patch.object(downloads, "hf_hub_download", side_effect=attempt),
        ):
            downloads.download_results(self.target)
        sleep.assert_called_once_with(10)
        self.assertEqual(counts, {self.files[0]: 1, self.files[1]: 2})

    def test_incomplete_download_is_reported(self):
        with patch.object(downloads, "hf_hub_download"):
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                downloads.download_results(self.target)

    def test_remote_path_cannot_escape_destination(self):
        self.api.return_value.list_repo_files.return_value = ["../outside.json"]
        with patch.object(downloads, "hf_hub_download") as fetch:
            with self.assertRaisesRegex(ValueError, "Invalid remote"):
                downloads.download_results(self.target)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
