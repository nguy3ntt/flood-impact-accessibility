"""Safety checks for the bounded, immutable source acquisition path."""

from contextlib import closing
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import acquire_sen1floods11 as acquisition


class PilotAcquisitionTests(unittest.TestCase):
    def test_rejects_unsafe_or_non_versioned_keys(self):
        for key in ("../secrets", "v1.1/../secrets", "v1.1\\evil", "v1.1/"):
            with self.subTest(key=key):
                self.assertFalse(acquisition.valid_key(key))

    def test_exact_bytes_are_saved_once_with_hash(self):
        payload = b"small pilot object"
        metadata = {"size": len(payload), "generation": "12"}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(acquisition, "RAW_ROOT", Path(directory)), \
                 patch.object(acquisition, "object_info", return_value=metadata), \
                 patch.object(acquisition, "urlopen", return_value=closing(io.BytesIO(payload))):
                record = acquisition.acquire("v1.1/pilot.bin", len(payload))
                self.assertEqual(record["bytes"], len(payload))
                self.assertEqual((Path(directory) / "v1.1/pilot.bin").read_bytes(), payload)
                self.assertEqual(len(record["sha256"]), 64)
                with self.assertRaises(FileExistsError):
                    acquisition.acquire("v1.1/pilot.bin", len(payload))

    def test_excess_bytes_leave_no_source_or_partial_file(self):
        payload = b"too many bytes"
        metadata = {"size": 3, "generation": "12"}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(acquisition, "RAW_ROOT", Path(directory)), \
                 patch.object(acquisition, "object_info", return_value=metadata), \
                 patch.object(acquisition, "urlopen", return_value=closing(io.BytesIO(payload))):
                with self.assertRaises(ValueError):
                    acquisition.acquire("v1.1/pilot.bin", 3)
            self.assertFalse((Path(directory) / "v1.1/pilot.bin").exists())
            self.assertFalse((Path(directory) / "v1.1/pilot.bin.part").exists())


if __name__ == "__main__":
    unittest.main()
