import numpy as np, pytest, rustymath as rm
from numpy.lib.stride_tricks import sliding_window_view as windows
from scipy import signal

def test_correlate_and_convolve():
    rng = np.random.default_rng(0)
    x, k = rng.standard_normal((7, 6)), rng.standard_normal((3, 2))
    for mode in ('valid', 'full', 'same'):
        assert np.allclose(rm.correlate(x, k, mode), signal.correlate(x, k, mode))
        assert np.allclose(rm.convolve(x, k, mode), signal.convolve(x, k, mode))
    # An even window in 'same' mode centres as NumPy does.
    v, even = rng.standard_normal(9), [1., 10, 100, 1000]
    assert np.allclose(rm.correlate(v, even, 'same'), np.correlate(v, even, 'same')) and np.allclose(rm.convolve(v, even, 'same'), np.convolve(v, even, 'same'))
    # Sliding along an image's first two axes contracts its channels with the kernel's, two positions at a time.
    img, kf = rng.standard_normal((9, 8, 3)), rng.standard_normal((3, 3, 3, 4))
    expected = np.tensordot(windows(img, (3, 3), axis=(0, 1))[::2, ::2], kf, axes=([2, 3, 4], [2, 0, 1]))
    assert np.allclose(rm.correlate(img, kf, slide=2, stride=2), expected)
    # Other pairs: maximum over a window of sums is a dilation.
    assert np.allclose(rm.correlate(v, [0., 1, 0], f=rm.maximum, g=rm.add), (windows(v, 3) + [0, 1, 0]).max(-1))
    exact, single = rm.correlate([1, 2, 3, 4], [1, 1]), rm.correlate(np.float32([1, 2, 3]), np.float32([1, 2]))
    assert exact.tolist() == [3, 5, 7] and exact.dtype == 'int64' and single.dtype == 'float32'
    with pytest.raises(ValueError, match="doesn't fit"): rm.correlate([1., 2], [1., 2, 3])
    with pytest.raises(ValueError, match='mode'): rm.convolve(v, even, 'middle')

def test_overlap_add_is_the_transpose_of_windows():
    rng = np.random.default_rng(1)
    x, frames = rng.standard_normal(18), rng.standard_normal((5, 6))
    added = rm.overlap_add(frames, 3)
    expected = np.zeros(18)
    for f in range(5): expected[3 * f:3 * f + 6] += frames[f]
    assert np.allclose(added, expected)
    # Adding frames back is the adjoint of taking windows: both sides give the same inner product.
    assert np.isclose(np.sum(windows(x, 6)[::3] * frames), np.sum(x * np.asarray(added)))
    # Leading axes are separate signals, and `axis` names the frames' axis when other axes follow the frames' items.
    assert np.allclose(rm.overlap_add(np.stack([frames, 2 * frames]), 3), [expected, 2 * expected])
    assert np.allclose(rm.overlap_add(np.stack([frames, 2 * frames], axis=-1), 3, axis=0), np.stack([expected, 2 * expected], axis=-1))
