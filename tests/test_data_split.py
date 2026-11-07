import unittest
from pathlib import Path

try:
    from posenerf.data import FrameRecord, split_train_validation_frames
except ModuleNotFoundError:
    FrameRecord = None


@unittest.skipUnless(FrameRecord is not None, "Project dependencies are required for data split tests")
class DataSplitTests(unittest.TestCase):
    def test_train_validation_frames_are_disjoint_and_bounded(self):
        frames = [FrameRecord("sequence01", f"frame-{index:06d}", Path(f"{index}.color.png"), Path(f"{index}.depth.png"), Path(f"{index}.pose.txt")) for index in range(20)]
        train, validation = split_train_validation_frames(frames, 0.2, 12, 4)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(validation), 4)
        self.assertFalse({frame.stem for frame in train} & {frame.stem for frame in validation})


if __name__ == "__main__":
    unittest.main()
