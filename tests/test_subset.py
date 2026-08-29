from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from offline.cli import _select_samples
from utils.subset import subset_ids


class SubsetTest(unittest.TestCase):
    def test_等间距取样且小子集嵌套在大子集里(self) -> None:
        offline = [f"{i:03d}" for i in range(1, 501)]
        ten, hundred = subset_ids(offline, 10), subset_ids(offline, 100)
        self.assertEqual(ten, ["001", "051", "101", "151", "201", "251", "301", "351", "401", "451"])
        self.assertEqual(len(hundred), 100)
        self.assertTrue(set(ten) <= set(hundred))
        self.assertEqual(subset_ids(offline, 500), offline)

        online = [f"{i:03d}" for i in range(1, 201)]
        self.assertTrue(set(subset_ids(online, 10)) <= set(subset_ids(online, 100)))
        self.assertEqual(subset_ids(online, 10)[:2], ["001", "021"])

    def test_子集大小越界报错(self) -> None:
        with self.assertRaises(ValueError):
            subset_ids(["001"], 2)
        with self.assertRaises(ValueError):
            subset_ids(["001"], 0)

    def test_offline评测的subset是partial(self) -> None:
        all_samples = [f"{i:03d}" for i in range(1, 501)]
        with patch("offline.cli.load_sample_ids", return_value=all_samples):
            selected, is_partial = _select_samples(SimpleNamespace(sample_ids=None, subset=10))
        self.assertEqual(selected[:2], ["001", "051"])
        self.assertTrue(is_partial)


if __name__ == "__main__":
    unittest.main()
