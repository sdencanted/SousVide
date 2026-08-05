import time
import unittest

from rich.progress import TextColumn

from sousvide.instruct.train_policy import _estimate_epoch_completion
from sousvide.visualize.rich_utilities import get_training_progress


class TrainingProgressTests(unittest.TestCase):
    def test_epoch_eta_uses_mean_completed_epoch_duration(self):
        now = 1_700_000_000.0
        eta,remaining = _estimate_epoch_completion([10.0,20.0],3,now=now)

        self.assertEqual(remaining,"45s")
        self.assertEqual(
            eta,time.strftime(
                "%Y-%m-%d %H:%M:%S",time.localtime(now+45)))

    def test_epoch_eta_formats_long_remaining_duration(self):
        _,remaining = _estimate_epoch_completion([3600.0],26,now=0.0)
        self.assertEqual(remaining,"1d 02h 00m")
        self.assertEqual(
            _estimate_epoch_completion([],5,now=0.0),
            ("calculating...","--"))
        self.assertEqual(
            _estimate_epoch_completion([10.0],0,now=0.0),
            ("complete","0s"))

    def test_training_progress_contains_labeled_eta_fields(self):
        progress = get_training_progress()
        text_formats = [
            column.text_format for column in progress.columns
            if isinstance(column,TextColumn)
        ]
        self.assertTrue(any("task.fields[eta]" in value for value in text_formats))
        self.assertTrue(any(
            "task.fields[remaining]" in value for value in text_formats))


if __name__ == "__main__":
    unittest.main()
