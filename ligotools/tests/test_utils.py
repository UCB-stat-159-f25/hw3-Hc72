import numpy as np
from ligotools import utils


def test_whiten_returns_array():
    data = np.random.randn(1024)
    interp_psd = lambda f: np.ones_like(f)
    dt = 1 / 4096
    white = utils.whiten(data, interp_psd, dt)
    assert isinstance(white, np.ndarray)
    assert len(white) == len(data)
    assert np.isfinite(white).all()


def test_reqshift_changes_signal():
    data = np.random.randn(2048)
    shifted = utils.reqshift(data, fshift=200, sample_rate=4096)
    assert isinstance(shifted, np.ndarray)
    assert len(shifted) == len(data)
    assert not np.allclose(shifted, data)
