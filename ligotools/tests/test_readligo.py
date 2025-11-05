import numpy as np
import os
from ligotools import readligo as rl

def test_empty_file_load_data(tmp_path):
    # loaddata() on empty file is (None, None, None)
    empty_file = tmp_path / "empty.hdf5"
    empty_file.write_text("")
    strain, time, chan_dict = rl.loaddata(str(empty_file))
    assert strain is None
    assert time is None
    assert chan_dict is None


def test_boolean_array_dq_channel_to_seglist():
    channel = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    segments = rl.dq_channel_to_seglist(channel, fs=1)
    assert isinstance(segments, list)
    assert all(isinstance(s, slice) for s in segments)
    assert len(segments) == 2
    assert segments[0].start == 2 and segments[0].stop == 5
    assert segments[1].start == 7 and segments[1].stop == 9
