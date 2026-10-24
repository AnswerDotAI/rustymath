"Correlation and convolution along an array's leading axes, and overlap-add, which adds frames back where their windows were read."
from . import _core
from ._core import add, multiply, overlap_add
from ._naming import _binary_function

def _slid(x, kernel, mode, slide, stride, f, g, flip):
    stride = [stride] if isinstance(stride, int) else list(stride)
    return _core._correlate(_binary_function(f), _binary_function(g), x, kernel, mode, slide, stride, flip)

def correlate(
    x, # The array the kernel slides along
    kernel, # A window size for each sliding axis, then `x`'s other axes, then the outputs at each position
    mode='valid', # 'valid' keeps windows inside `x`, 'full' every overlap, and 'same' one position for each of `x`'s
    slide=None, # How many leading axes of `x` the kernel slides along, by default all of them
    stride=1, # How far the kernel moves along each sliding axis: one int, or one for each axis
    f=add, # Native binary function that folds over each window
    g=multiply, # Native binary function that pairs each item of a window with the matching kernel item
):
    "The correlation of `x` with `kernel`: at each position, `f` over the window there of `g` of each item and the matching kernel item. The window covers the sliding axes and `x`'s other axes. The kernel isn't conjugated."
    return _slid(x, kernel, mode, slide, stride, f, g, False)

def convolve(
    x, # The array the kernel slides along
    kernel, # A window size for each sliding axis, then `x`'s other axes, then the outputs at each position
    mode='valid', # 'valid' keeps windows inside `x`, 'full' every overlap, and 'same' one position for each of `x`'s
    slide=None, # How many leading axes of `x` the kernel slides along, by default all of them
    stride=1, # How far the kernel moves along each sliding axis: one int, or one for each axis
    f=add, # Native binary function that folds over each window
    g=multiply, # Native binary function that pairs each item of a window with the matching kernel item
):
    "`correlate` with the kernel reversed along its window axes"
    return _slid(x, kernel, mode, slide, stride, f, g, True)
