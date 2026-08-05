import os
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import h5py
import numpy as np

from sousvide.synthesize.parallel_event_generator import (
    EventFrameBuffer,
    process_buffered_rollout,
)


class ParallelEventGenerationTests(unittest.TestCase):
    def test_frame_buffer_is_grayscale_and_disk_backed(self):
        buffer = EventFrameBuffer()
        rgb = np.zeros((4, 6, 3),dtype=np.uint8)
        rgb[..., 0] = 255
        buffer.process_frame(rgb,0.0,False)
        buffer.process_frame(rgb,0.05,True)

        with tempfile.TemporaryDirectory() as folder:
            frame_path = os.path.join(folder,"frames.npy")
            timestamps,close_windows = buffer.save(frame_path)
            frames = np.load(frame_path,mmap_mode="r",allow_pickle=False)
            self.assertEqual(frames.shape,(2,4,6))
            self.assertEqual(timestamps,(0.0,0.05))
            self.assertEqual(close_windows,(False,True))
            self.assertEqual(buffer.frames,[])

    def test_two_spawned_workers_write_independent_h5_rollouts(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs = []
            for index in range(2):
                frame_path = os.path.join(folder,f"{index}.frames.npy")
                h5_path = os.path.join(folder,f"{index}.h5")
                kronecker_path = os.path.join(folder,f"{index}.kronecker.npy")
                frames = np.stack((
                    np.zeros((8,8),dtype=np.uint8),
                    np.full((8,8),255-index,dtype=np.uint8),
                ))
                np.save(frame_path,frames,allow_pickle=False)
                jobs.append((frame_path,h5_path,kronecker_path))

            with ProcessPoolExecutor(
                max_workers=2,mp_context=get_context("spawn")
            ) as executor:
                futures = [
                    executor.submit(
                        process_buffered_rollout,
                        frame_path,(0.0,0.05),(False,True),
                        h5_path,kronecker_path,1,
                    )
                    for frame_path,h5_path,kronecker_path in jobs
                ]
                results = [future.result() for future in futures]

            for (kronecker_path,h5_path),(_,expected_h5,expected_kronecker) in zip(
                results,jobs
            ):
                self.assertEqual(kronecker_path,expected_kronecker)
                self.assertEqual(h5_path,expected_h5)
                kronecker = np.load(kronecker_path,allow_pickle=False)
                self.assertEqual(kronecker.shape,(1,8,8))
                self.assertEqual(kronecker.dtype,np.uint8)
                with h5py.File(h5_path,"r") as event_file:
                    self.assertEqual(event_file["events"].shape[1],4)


if __name__ == "__main__":
    unittest.main()
