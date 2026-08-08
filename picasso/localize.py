"""
picasso.localize
~~~~~~~~~~~~~~~~

Identify and localize fluorescent single molecules in a frame
sequence.

Spot detection and the localization table live here; the fits themselves are
run by :mod:`picasso.fitting` (Gaussian and cubic-spline PSF models on the CPU
and on CUDA GPUs). This module owns the translation between them:
calibration dicts, initial parameters, the device choice, and the
Cramer-Rao lower bounds that become the reported localization precisions.

References
----------
Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
"Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
Scientific Reports 7, 15722 (2017).
https://doi.org/10.1038/s41598-017-15313-9
Licence (MIT): ``LICENSES/Gpufit-LICENSE.txt``.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import multiprocessing
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from itertools import chain
from typing import Literal, Callable, TypeAlias, Union
from datetime import datetime

import numba
import numpy as np
from numba import cuda
import dask.array as da
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from tqdm import tqdm
from sqlalchemy import create_engine
import matplotlib.gridspec as gridspec
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist

from .ext import bitplane

from . import (
    io,
    lib,
    avgroi,
    postprocess,
    zfit,
    __version__,
)
from .fitting import (
    gaussfit,
    gaussfit_cuda,
    precision,
    splinefit,
    splinefit_cuda,
)

# Check for CUDA availability of the *fitting* backends. Otherwise, CPU is
# used. The CRLB kernels have their own probe in picasso.fitting.precision.
try:
    CUDA_AVAILABLE = bool(cuda.is_available())
except Exception:
    CUDA_AVAILABLE = False


plt.style.use("ggplot")


#: A movie Picasso can read frames from: one loaded by
#: ``picasso.io.load_movie`` (an ``io.AbstractPicassoMovie``, or the
#: ``np.memmap`` of a ``.raw`` file). Not a plain 3D array: the readers are
#: lazy, so a movie is only indexed frame by frame.
LoadedMovie: TypeAlias = Union["io.AbstractPicassoMovie", np.memmap]

#: Anything the identification reads frames from, i.e. a loaded movie or one
#: of the filter wrappers below, which deliberately do not implement
#: ``io.AbstractPicassoMovie`` so that they cannot reach the fit (see
#: ``TemporalMedianMovie``).
MovieLike: TypeAlias = Union[
    LoadedMovie, "TemporalMedianMovie", "GaussianFilteredMovie"
]


MAX_LOCS = int(1e6)

# Axial multi-start. A single in-focus seed leaves a spline fit z-degenerate at
# large |z|. Several seeds spanning the calibration stack are run per spot and the
# one that best explains the data is kept. One seed per ~20 calibration planes,
# bounded; the rule the calibration diagnostic has always used.
_Z_STARTS_PER_PLANES = 20
_Z_STARTS_MIN = 5
_Z_STARTS_MAX = 15


def _default_n_z_starts(calibration: dict) -> int:
    """Number of axial seeds for a spline fit, from the calibration's z depth.

    1 (i.e. no multi-start) for a 2D model, which has no z to be degenerate in,
    and for a calibration whose ``n_data`` does not describe a z axis."""
    if calibration.get("model") == "spline-2d":
        return 1
    n_data = calibration.get("n_data")
    if n_data is None or len(n_data) < 3:
        return 1
    n_z = int(n_data[2])
    return int(
        np.clip(n_z // _Z_STARTS_PER_PLANES, _Z_STARTS_MIN, _Z_STARTS_MAX)
    )


# The columns under base are always available and the keys such as "3D
# only" will be displayed in the save columns dialog in the GUI for
# clarity
LOCALIZATION_COLUMNS = {
    "Base": [
        "frame",
        "x",
        "y",
        "photons",
        "sx",
        "sy",
        "bg",
        "lpx",
        "lpy",
        "ellipticity",
        "net_gradient",
    ],
    "3D only": ["z", "d_zcalib", "lpz"],
    "Rotation only": ["angle", "angle_unc"],
    "Picked spots only": ["n_id"],
    "MLE only": ["log_likelihood", "iterations"],
    "Least squares only": ["chi_square"],
    "Uncertainty": ["photons_unc", "bg_unc", "sx_unc", "sy_unc"],
    "Multichannel only": (
        [f"photons_ch{c}" for c in range(precision._LINK_XYZ_MAX_CHANNELS)]
        + [f"bg_ch{c}" for c in range(precision._LINK_XYZ_MAX_CHANNELS)]
        + [
            f"rel_photons_ch{c}"
            for c in range(precision._LINK_XYZ_MAX_CHANNELS)
        ]
        + ["color"]
    ),
}
# For database:
MEAN_COLS = LOCALIZATION_COLUMNS["Base"] + LOCALIZATION_COLUMNS["3D only"]
SET_COLS = [
    "Frames",
    "Height",
    "Width",
    "Box Size",
    "Min. Net Gradient",
    "Pixelsize",
]
# Memory budget for the cached temporal windows of TemporalMedianMovie.
TEMPORAL_MEDIAN_CACHE_BYTES = 512 * 1024**2
# Gaussian filter for spot identification
GAUSSIAN_FILTER_TRUNCATE = 4.0
GAUSSIAN_FILTER_MODE = "nearest"
# Default bead-detection / matching parameters for `calibrate_affine_transform`.
_AFFINE_MATCH_MAX_DIST_PX = 40.0  # max distance between matched pair
_AFFINE_XCORR_HALF_WIDTH = 18  # half-width of bead crop for xcorr


@numba.jit(nopython=True, nogil=True, cache=False)
def _local_maxima(
    frame: lib.IntArray2D, box: int
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """Find pixels with maximum value within a region of interest.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    box : int
        Size of the box to search for local maxima. Should be an odd
        integer.

    Returns
    -------
    y : lib.IntArray1D
        y-coordinates of the local maxima.
    x : lib.IntArray1D
        x-coordinates of the local maxima.
    """
    Y, X = frame.shape
    maxima_map = np.zeros(frame.shape, np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half : i + box_half + 1,
                j - box_half : j + box_half + 1,
            ]
            flat_max = np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = np.where(maxima_map)
    return y, x


@numba.jit(nopython=True, nogil=True, cache=False)
def _gradient_at(
    frame: lib.IntArray2D,
    y: int,
    x: int,
    i: int,
) -> tuple[float, float]:
    """Calculate the gradient at a specific pixel in the frame.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    y, x : int
        Coordinates of the pixel where the gradient is calculated.
    i : int
        Index of the pixel in the list of maxima. Not used in this
        function.

    Returns
    -------
    gy : float
        Gradient in the y-direction at the pixel (y, x).
    gx : float
        Gradient in the x-direction at the pixel (y, x).
    """
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@numba.jit(nopython=True, nogil=True, cache=False)
def _net_gradient(
    frame: lib.IntArray2D,
    y: lib.IntArray1D,
    x: lib.IntArray1D,
    box: int,
    uy: lib.FloatArray2D,
    ux: lib.FloatArray2D,
) -> lib.FloatArray1D:
    """Calculate the net gradient at the identified maxima in the
    frame.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    y, x : lib.IntArray1D
        Coordinates of the identified maxima in the frame.
    box : int
        Size of the box used for calculating the gradient.
    uy, ux : lib.FloatArray2D
        Arrays of shape (box, box) containing the y and x components
        of the gradient, respectively.

    Returns
    -------
    ng : lib.FloatArray1D
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """
    box_half = int(box / 2)
    ng = np.zeros(len(x), dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(
                range(xi - box_half, xi + box_half + 1)
            ):
                if not (k == yi and m == xi):
                    gy, gx = _gradient_at(frame, k, m, i)
                    ng[i] += (
                        gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
                    )
    return ng


@numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(
    image: lib.IntArray2D,
    minimum_ng: float,
    box: int,
) -> tuple[lib.IntArray1D, lib.IntArray1D, lib.FloatArray1D]:
    """Identify local maxima in the image and calculate the net gradient
    at those maxima.

    Parameters
    ----------
    image : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    minimum_ng : float
        Minimum net gradient value to consider a maximum as valid.
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer.

    Returns
    -------
    y : lib.IntArray1D
        y-coordinates of the identified maxima.
    x : lib.IntArray1D
        x-coordinates of the identified maxima.
    ng : lib.FloatArray1D
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """
    y, x = _local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux**2 + uy**2)
    ux /= unorm
    uy /= unorm
    ng = _net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def _normalize_rect(
    rect: tuple[tuple[int, int], tuple[int, int]],
) -> list[list[int]]:
    """Return a rectangle as ``[[y_min, x_min], [y_max, x_max]]`` with
    integer, correctly ordered corners (the input corners may be given
    in any order)."""
    (y0, x0), (y1, x1) = rect
    return [
        [int(min(y0, y1)), int(min(x0, x1))],
        [int(max(y0, y1)), int(max(x0, x1))],
    ]


def _subtract_rect(
    a: list[list[int]], b: list[list[int]]
) -> list[list[list[int]]]:
    """Subtract rectangle ``b`` from rectangle ``a``.

    Returns the parts of ``a`` not covered by ``b`` as a list of up to
    four disjoint axis-aligned rectangles (a guillotine split into top,
    bottom, left and right bands around the intersection). If the
    rectangles do not overlap, ``[a]`` is returned unchanged. Both
    rectangles must use the ``[[y_min, x_min], [y_max, x_max]]`` format.
    """
    (ay0, ax0), (ay1, ax1) = a
    (by0, bx0), (by1, bx1) = b
    # intersection
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    if iy0 >= iy1 or ix0 >= ix1:
        return [a]  # no overlap
    pieces = []
    if ay0 < iy0:  # top band, full width of a
        pieces.append([[ay0, ax0], [iy0, ax1]])
    if iy1 < ay1:  # bottom band, full width of a
        pieces.append([[iy1, ax0], [ay1, ax1]])
    if ax0 < ix0:  # left band, between the horizontal cuts
        pieces.append([[iy0, ax0], [iy1, ix0]])
    if ix1 < ax1:  # right band, between the horizontal cuts
        pieces.append([[iy0, ix1], [iy1, ax1]])
    return pieces


def clip_rois(
    rois: list[tuple[tuple[int, int], tuple[int, int]]],
    min_size: int = 0,
) -> list[list[list[int]]]:
    """Clip a list of (possibly overlapping) ROIs into a list of
    disjoint rectangles.

    The ROIs are processed in order; each rectangle is trimmed against
    the union of the already-accepted rectangles (via ``_subtract_rect``)
    so that earlier ROIs take precedence and no pixel is covered twice.
    A single ROI may therefore split into several rectangles. Corners
    are normalized and pieces smaller than ``min_size`` in either
    dimension are dropped (pass ``box`` so slivers that cannot hold a
    spot are discarded).

    Parameters
    ----------
    rois : list of rectangles
        Each rectangle is ``((y_min, x_min), (y_max, x_max))`` (corners
        may be in any order).
    min_size : int, optional
        Minimum side length (in pixels) for a clipped piece to be kept.
        Default is 0 (keep any piece with positive area).

    Returns
    -------
    list of rectangles
        Disjoint rectangles in ``[[y_min, x_min], [y_max, x_max]]``
        format whose union equals the union of the inputs (minus dropped
        slivers).
    """
    accepted: list[list[list[int]]] = []
    for rect in rois:
        pieces = [_normalize_rect(rect)]
        for acc in accepted:
            new_pieces: list[list[list[int]]] = []
            for piece in pieces:
                new_pieces.extend(_subtract_rect(piece, acc))
            pieces = new_pieces
        for piece in pieces:
            height = piece[1][0] - piece[0][0]
            width = piece[1][1] - piece[0][1]
            if height > 0 and width > 0:
                if height >= min_size and width >= min_size:
                    accepted.append(piece)
    return accepted


def _as_roi_list(
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None,
) -> list[list[list[int]]] | None:
    """Normalize the ``roi`` argument into a list of rectangles or None.

    Accepts a single rectangle ``((y0, x0), (y1, x1))`` (for backward
    compatibility) or a list of such rectangles. An empty list and
    ``None`` both map to ``None`` (whole frame).
    """
    if roi is None or len(roi) == 0:
        return None
    first = roi[0][0]
    if isinstance(first, (list, tuple, np.ndarray)):
        rois = [_normalize_rect(r) for r in roi]  # list of rectangles
    else:
        rois = [_normalize_rect(roi)]  # single rectangle
    return rois if len(rois) else None


def _as_ng_list(
    minimum_ng: float | list | np.ndarray,
    n_rois: int,
) -> list[float]:
    """Normalize ``minimum_ng`` into one threshold per ROI.

    A scalar (the usual case) applies to every ROI. A sequence gives each
    ROI its own threshold, which is what split-FOV data needs: the regions
    are separate channels imaged through different optics, so their spots
    do not share a brightness scale. A one-element sequence is treated as
    a scalar.

    Parameters
    ----------
    minimum_ng : float or sequence of float
        Minimum net gradient, shared or one per ROI.
    n_rois : int
        Number of ROIs the thresholds have to cover.

    Returns
    -------
    list of float
        ``n_rois`` thresholds.

    Raises
    ------
    ValueError
        If a sequence is given whose length is neither 1 nor ``n_rois``.
    """
    if isinstance(minimum_ng, (list, tuple, np.ndarray, pd.Series)):
        ngs = [float(_) for _ in minimum_ng]
    else:
        ngs = [float(minimum_ng)]
    if len(ngs) == 1:
        return ngs * n_rois
    if len(ngs) != n_rois:
        raise ValueError(
            f"minimum_ng has {len(ngs)} values but there are {n_rois} "
            "ROI(s); give one threshold per ROI or a single shared one."
        )
    return ngs


def _temporal_median(
    frames: np.ndarray, max_stripe_bytes: int = 64 * 1024**2
) -> lib.FloatArray2D:
    """Per-pixel median of a stack of frames, i.e. the median along
    axis 0.

    ``np.partition`` is used instead of ``np.median`` because only the
    middle order statistic is needed and because it keeps the data in its
    native (usually integer) dtype - ``np.median`` promotes to float64,
    which doubles the memory traffic. The stack is processed in stripes of
    rows so that the copy ``np.partition`` makes internally stays bounded
    regardless of the window length and the frame size.

    Parameters
    ----------
    frames : np.ndarray
        Stack of frames of shape (N, Y, X).
    max_stripe_bytes : int, optional
        Approximate memory budget for one stripe. Default is 64 MB.

    Returns
    -------
    lib.FloatArray2D
        Per-pixel median, 2D array of shape (Y, X) and dtype float32.
    """
    n_frames, height, width = frames.shape
    median = np.empty((height, width), dtype=np.float32)
    lower = (n_frames - 1) // 2
    upper = n_frames // 2  # == lower for an odd number of frames
    kth = lower if lower == upper else (lower, upper)
    stripe_rows = max(
        1, int(max_stripe_bytes // max(n_frames * width * frames.itemsize, 1))
    )
    for y0 in range(0, height, stripe_rows):
        y1 = min(y0 + stripe_rows, height)
        stripe = np.partition(frames[:, y0:y1], kth, axis=0)
        if lower == upper:
            median[y0:y1] = stripe[lower]
        else:  # cast before adding, the sum can overflow the input dtype
            median[y0:y1] = 0.5 * (
                stripe[lower].astype(np.float32)
                + stripe[upper].astype(np.float32)
            )
    return median


class _TemporalMedianBlock:
    """One cached temporal window of a ``TemporalMedianMovie``.

    ``ready`` is set once ``median`` (and possibly ``frames``) has been
    filled in, or once the attempt has failed and ``error`` holds the
    exception. Threads that did not win the race to compute the block wait
    on it.
    """

    __slots__ = ("start", "stop", "median", "frames", "ready", "error")

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop
        self.median: lib.FloatArray2D | None = None
        self.frames: np.ndarray | None = None
        self.ready = threading.Event()
        self.error: BaseException | None = None

    @property
    def nbytes(self) -> int:
        """Memory held by this block."""
        total = 0 if self.median is None else self.median.nbytes
        if self.frames is not None:
            total += self.frames.nbytes
        return total


class TemporalMedianMovie:
    """Lazily evaluated, read-only temporal median filtered view of a
    movie.

    Frame ``t`` is ``max(movie[t] - median(movie[window(t)]), 0)`` as
    float32, where ``window(t)`` is a window of ``window`` frames centered
    on ``t``. At the edges of the movie the window is shifted inwards
    rather than truncated, so it always covers ``window`` frames.

    Computing that median for every single frame is far too slow for real
    movies, so the median is only evaluated at *anchor* frames spaced
    ``stride`` apart and shared by the frames in between. The default
    ``stride=window`` gives ``n_frames / window`` medians; ``stride=1``
    reproduces the exact per-frame filter.

    This class is meant for spot identification only - fitting, spot
    cutting and photon conversion must always use the raw movie, since
    the subtracted background would otherwise corrupt the photon counts.
    It deliberately does not implement the ``io.AbstractPicassoMovie``
    interface so that accidentally handing it to ``fit`` trips that
    function's input assertion instead of silently returning wrong
    photon numbers.

    Note that subtracting a per-pixel background changes the scale of the
    net gradient, so the minimum net gradient has to be re-tuned when the
    filter is switched on or off.

    References
    ----------
    Martens, K. J. A., Turkowyd, B. & Endesfelder, U.
    "Raw Data to Results: A Hands-On Introduction and Overview of
    Computational Analysis for Single-Molecule Localization Microscopy."
    Frontiers in Bioinformatics 1, 817254 (2022).
    https://doi.org/10.3389/fbinf.2021.817254

    Parameters
    ----------
    movie : MovieLike
        The raw movie, i.e. any object supporting ``len()`` and integer
        indexing that returns 2D frames.
    window : int, optional
        Number of frames in the temporal window used for the median.
        Clipped to the length of the movie. Default is 51.
    stride : int or None, optional
        Spacing (in frames) between the anchors at which the median is
        evaluated. Clipped to ``[1, window]``. None (the default) uses
        ``window``, which is the fastest setting; 1 evaluates the median
        for every frame.
    roi : tuple, list of tuples or None, optional
        Region(s) of interest that will be identified in, in the same
        format as ``identify``. When given, the median is only computed
        inside their (padded) bounding box and the returned frames are
        zero outside it. Default is None (whole frame).
    roi_pad : int, optional
        Number of pixels the ROI bounding box is grown by, so that the
        gradients of maxima sitting on the ROI border are still computed
        from filtered pixels. Pass ``int(box / 2) + 1``. Default is 0.
    cache_bytes : int, optional
        Memory budget for the cached temporal windows. Default is
        ``TEMPORAL_MEDIAN_CACHE_BYTES``.

    Attributes
    ----------
    raw : lib.IntArray3D
        The underlying unfiltered movie.
    window, stride, roi : as above
    """

    # The wrapper is internally thread safe (see _read_lock), so it tells
    # identify_by_frame_number not to serialize reads on the identification
    # lock - that lock is also what _identify_worker hands out frame
    # numbers with, so holding it across a median would stall every worker.
    supports_concurrent_reads = True

    def __init__(
        self,
        movie: MovieLike,
        window: int = 51,
        *,
        stride: int | None = None,
        roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
        roi_pad: int = 0,
        cache_bytes: int = TEMPORAL_MEDIAN_CACHE_BYTES,
    ) -> None:
        self.raw = movie
        self.n_frames = len(movie)
        if self.n_frames == 0:
            raise ValueError("Cannot temporally filter an empty movie.")
        self.window = max(1, min(int(window), self.n_frames))
        stride = self.window if stride is None else int(stride)
        self.stride = max(1, min(stride, self.window))
        self.roi = roi
        self.roi_pad = int(roi_pad)
        self.cache_bytes = int(cache_bytes)
        # one read to learn the frame geometry; movies do not agree on
        # whether they expose .shape (TiffMap does not) or a usable .dtype
        probe = np.asarray(movie[0])
        self.frame_shape = probe.shape
        self._raw_dtype = probe.dtype
        self._bbox = self._roi_bbox(roi, self.roi_pad)
        self._cache: OrderedDict[int, _TemporalMedianBlock] = OrderedDict()
        self._lock = threading.Lock()  # guards _cache only, never held long
        concurrent = getattr(
            movie, "supports_concurrent_reads", False
        ) or isinstance(movie, np.memmap)
        self._read_lock = None if concurrent else threading.Lock()

    def _roi_bbox(
        self,
        roi: tuple[tuple[int, int], tuple[int, int]] | list | None,
        pad: int,
    ) -> tuple[slice, slice] | None:
        """Bounding box of the ROIs, grown by ``pad`` and clipped to the
        frame, or None to use the whole frame."""
        rois = _as_roi_list(roi)
        if rois is None:
            return None
        height, width = self.frame_shape
        y0 = max(0, min(int(r[0][0]) for r in rois) - pad)
        x0 = max(0, min(int(r[0][1]) for r in rois) - pad)
        y1 = min(height, max(int(r[1][0]) for r in rois) + pad)
        x1 = min(width, max(int(r[1][1]) for r in rois) + pad)
        if y1 <= y0 or x1 <= x0:
            return None
        # the median costs O(window * area), so restricting it only pays
        # off when the bounding box is substantially smaller than the frame
        if (y1 - y0) * (x1 - x0) > 0.5 * height * width:
            return None
        return slice(y0, y1), slice(x0, x1)

    def _block_index(self, frame_number: int) -> int:
        """Index of the temporal window used to filter ``frame_number``.
        Frames are tiled into groups of ``stride``, each sharing one
        median."""
        return frame_number // self.stride

    def _bounds(self, block_index: int) -> tuple[int, int]:
        """Start (inclusive) and stop (exclusive) frame of a block's
        temporal window.

        The window covers the block's own ``stride`` frames and is grown
        symmetrically around them up to ``window`` frames; at the edges of
        the movie it is shifted inwards rather than truncated, so it
        always spans ``window`` frames. Every frame of the block therefore
        lies inside its own window, which is what lets the block serve raw
        frames straight out of its cached window.

        The two extremes are exactly the ones we care about:
        ``stride == window`` tiles the movie into non-overlapping blocks,
        and ``stride == 1`` reduces to a window centered on the frame,
        i.e. the exact per-frame filter.
        """
        start = block_index * self.stride - (self.window - self.stride) // 2
        stop = start + self.window
        if start < 0:
            stop -= start
            start = 0
        if stop > self.n_frames:
            start -= stop - self.n_frames
            stop = self.n_frames
        return max(start, 0), min(stop, self.n_frames)

    def _read_frame(self, index: int) -> np.ndarray:
        if self._read_lock is None:
            return np.asarray(self.raw[index])
        with self._read_lock:
            return np.asarray(self.raw[index])

    def _read(self, start: int, stop: int) -> np.ndarray:
        """Read frames ``[start, stop)`` into one array. Frames are read
        one by one because not every movie class supports slice
        indexing."""
        frames = np.empty(
            (stop - start, *self.frame_shape), dtype=self._raw_dtype
        )
        if self._read_lock is None:
            for i in range(start, stop):
                frames[i - start] = self.raw[i]
        else:
            with self._read_lock:
                for i in range(start, stop):
                    frames[i - start] = self.raw[i]
        return frames

    def _fill(self, block: _TemporalMedianBlock) -> None:
        """Compute a block's median (and keep its frames if they fit in
        the cache budget)."""
        frames = self._read(block.start, block.stop)
        if self._bbox is None:
            block.median = _temporal_median(frames)
        else:
            sy, sx = self._bbox
            block.median = _temporal_median(frames[:, sy, sx])
        # every frame this block serves lies inside its own window (since
        # stride <= window), so keeping the frames means each frame of the
        # movie is read exactly once overall. Two blocks stay resident, so
        # only keep the frames if two windows still fit in the budget.
        if 2 * frames.nbytes <= self.cache_bytes:
            block.frames = frames

    def _evict(self) -> None:
        """Keep the cache within its memory budget. Must be called with
        ``_lock`` held.

        Raw frames are dropped before whole blocks: a block's median is
        tiny (one float32 image) and re-reading a frame is far cheaper
        than recomputing a median. The two most recent blocks are always
        kept intact, since the worker pool straddles at most two of them.
        """
        if len(self._cache) <= 2:
            return
        total = sum(block.nbytes for block in self._cache.values())
        if total <= self.cache_bytes:
            return
        for block in list(self._cache.values())[:-2]:  # oldest first
            if block.frames is not None:
                total -= block.frames.nbytes
                block.frames = None
                if total <= self.cache_bytes:
                    return
        while len(self._cache) > 2 and total > self.cache_bytes:
            total -= self._cache.popitem(last=False)[1].nbytes

    def _block(self, frame_number: int) -> _TemporalMedianBlock:
        """The cached block used to filter ``frame_number``, computing it
        if necessary. Exactly one thread computes a given block; the
        others wait for it without holding the cache lock."""
        index = self._block_index(frame_number)
        with self._lock:
            block = self._cache.get(index)
            owner = block is None
            if owner:
                block = _TemporalMedianBlock(*self._bounds(index))
                self._cache[index] = block
            else:
                self._cache.move_to_end(index)
        if not owner:
            block.ready.wait()
            if block.error is not None:
                raise block.error
            return block
        try:
            self._fill(block)
        except BaseException as error:
            block.error = error
            with self._lock:
                self._cache.pop(index, None)
            raise
        finally:
            block.ready.set()
        with self._lock:
            self._evict()
        return block

    def clear_cache(self) -> None:
        """Drop all cached temporal windows."""
        with self._lock:
            self._cache.clear()

    def __getitem__(self, it):
        if isinstance(it, tuple):
            if len(it) == 1:
                return self[it[0]]
            return self[it[0]][tuple(it[1:])]
        if isinstance(it, slice):
            return np.stack(
                [self[i] for i in range(*it.indices(self.n_frames))]
            )
        index = int(it)
        if index < 0:
            index += self.n_frames
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f"Frame {it} is out of range for a movie with "
                f"{self.n_frames} frames."
            )
        block = self._block(index)
        # bind once: another thread's eviction may clear block.frames
        # between the check and the lookup
        frames = block.frames
        if frames is not None:
            raw = frames[index - block.start]
        else:
            raw = self._read_frame(index)
        if self._bbox is None:
            return np.maximum(raw.astype(np.float32) - block.median, 0)
        sy, sx = self._bbox
        filtered = np.zeros(self.frame_shape, dtype=np.float32)
        filtered[sy, sx] = np.maximum(
            raw[sy, sx].astype(np.float32) - block.median, 0
        )
        return filtered

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self) -> int:
        return self.n_frames

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_frames, *self.frame_shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    def close(self) -> None:
        self.clear_cache()
        close = getattr(self.raw, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def gaussian_filter_radius(
    sigma: float | None, truncate: float = GAUSSIAN_FILTER_TRUNCATE
) -> int:
    """Half-width (in pixels) of the kernel that
    ``scipy.ndimage.gaussian_filter`` actually uses for ``sigma``.

    This is the same ``int(truncate * sd + 0.5)`` expression as
    ``scipy.ndimage.gaussian_filter1d``, so a filtered pixel depends on
    exactly the pixels within this distance and no further.

    Parameters
    ----------
    sigma : float or None
        Standard deviation of the Gaussian kernel, in pixels. None or 0
        means no filtering.
    truncate : float, optional
        Kernel cut-off in units of sigma. Default is
        ``GAUSSIAN_FILTER_TRUNCATE``.

    Returns
    -------
    radius : int
        Kernel half-width in pixels, 0 if no filtering takes place.
    """
    if not sigma or sigma <= 0:
        return 0
    return int(float(truncate) * float(sigma) + 0.5)


def identification_roi_pad(
    box: int, gaussian_filter_sigma: float | None = None
) -> int:
    """Number of pixels a ROI has to be grown by so that every pixel the
    identification actually reads is validly filtered.

    ``identify_in_frame`` computes gradients up to ``int(box / 2) + 1``
    pixels outside a ROI. A Gaussian filter mixes in everything within
    its kernel radius, so with the filter on the valid region has to
    extend that much further still - otherwise the zeros that a
    ``TemporalMedianMovie`` leaves outside its bounding box get smeared
    into the very pixels those gradients are computed from.

    Parameters
    ----------
    box : int
        Box side length used for identification.
    gaussian_filter_sigma : float or None, optional
        Sigma of the spatial Gaussian filter, see
        ``GaussianFilteredMovie``. Default is None (no filtering).

    Returns
    -------
    pad : int
        Padding in pixels.
    """
    return int(box / 2) + 1 + gaussian_filter_radius(gaussian_filter_sigma)


class GaussianFilteredMovie:
    """Lazily evaluated, read-only spatially Gaussian smoothed view of a
    movie.

    Frame ``t`` is ``gaussian_filter(movie[t], sigma)`` as float32.

    Spot identification looks for a single local maximum per spot. A PSF
    may break up into several local maxima, so one molecule can be
    difficult to find.

    This class is meant for spot identification only - fitting, spot
    cutting and photon conversion must always use the raw movie, since
    the smoothed intensities would otherwise corrupt the photon counts.
    It deliberately does not implement the ``io.AbstractPicassoMovie``
    interface so that accidentally handing it to ``fit`` trips that
    function's input assertion instead of silently returning wrong
    photon numbers.

    Note that smoothing lowers gradient magnitudes, so the minimum net
    gradient has to be re-tuned whenever sigma changes.

    Parameters
    ----------
    movie : MovieLike
        The raw movie, i.e. any object supporting ``len()`` and integer
        indexing that returns 2D frames. May itself be a
        ``TemporalMedianMovie``, in which case the median is subtracted
        before smoothing.
    sigma : float
        Standard deviation of the Gaussian kernel, in camera pixels.
        Must be positive; use the unwrapped movie to identify without
        smoothing.
    truncate : float, optional
        Kernel cut-off in units of sigma. Default is
        ``GAUSSIAN_FILTER_TRUNCATE``.
    mode : str, optional
        How the frame borders are extended, see
        ``scipy.ndimage.gaussian_filter``. Default is
        ``GAUSSIAN_FILTER_MODE``.

    Attributes
    ----------
    raw : lib.IntArray3D
        The underlying unsmoothed movie.
    radius : int
        Kernel half-width in pixels, see ``gaussian_filter_radius``.
    sigma, truncate, mode : as above
    """

    # Filtering a frame is stateless (there is no cache, and scipy.ndimage
    # is re-entrant), so reads never have to be serialized on the
    # identification lock.
    supports_concurrent_reads = True

    def __init__(
        self,
        movie: MovieLike,
        sigma: float,
        *,
        truncate: float = GAUSSIAN_FILTER_TRUNCATE,
        mode: str = GAUSSIAN_FILTER_MODE,
    ) -> None:
        self.raw = movie
        self.n_frames = len(movie)
        if self.n_frames == 0:
            raise ValueError("Cannot filter an empty movie.")
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError(
                "The Gaussian filter sigma must be positive; identify on "
                "the unwrapped movie to not filter at all."
            )
        self.truncate = float(truncate)
        self.mode = mode
        # one read to learn the frame geometry; movies do not agree on
        # whether they expose .shape (TiffMap does not)
        self.frame_shape = np.asarray(movie[0]).shape
        concurrent = getattr(
            movie, "supports_concurrent_reads", False
        ) or isinstance(movie, np.memmap)
        self._read_lock = None if concurrent else threading.Lock()

    @property
    def radius(self) -> int:
        """Half-width (in pixels) of the kernel used."""
        return gaussian_filter_radius(self.sigma, self.truncate)

    def _read_frame(self, index: int) -> np.ndarray:
        if self._read_lock is None:
            return np.asarray(self.raw[index])
        with self._read_lock:
            return np.asarray(self.raw[index])

    def clear_cache(self) -> None:
        """Do nothing; this view caches nothing. Kept so that callers can
        treat both identification filters alike."""

    def __getitem__(self, it):
        if isinstance(it, tuple):
            if len(it) == 1:
                return self[it[0]]
            return self[it[0]][tuple(it[1:])]
        if isinstance(it, slice):
            return np.stack(
                [self[i] for i in range(*it.indices(self.n_frames))]
            )
        index = int(it)
        if index < 0:
            index += self.n_frames
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f"Frame {it} is out of range for a movie with "
                f"{self.n_frames} frames."
            )
        raw = self._read_frame(index)
        # gaussian_filter keeps the input dtype unless told otherwise, so
        # a uint16 movie would come back rounded to integers
        return gaussian_filter(
            raw.astype(np.float32, copy=False),
            self.sigma,
            output=np.float32,
            mode=self.mode,
            truncate=self.truncate,
        )

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self) -> int:
        return self.n_frames

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_frames, *self.frame_shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    def close(self) -> None:
        close = getattr(self.raw, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def identify_in_frame(
    frame: lib.IntArray2D,
    minimum_ng: float | list | np.ndarray,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
) -> tuple[lib.IntArray1D, lib.IntArray1D, lib.FloatArray1D]:
    """Identify local maxima in a single frame within optionally
    specified subregion(s) (ROI) and calculate the net gradient at those
    maxima.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    minimum_ng : float or sequence of float
        Minimum net gradient value to consider a maximum as valid. A
        sequence gives each ROI in ``roi`` its own threshold (split-FOV
        regions are separate channels and need not share a brightness
        scale); it must have one value per ROI.
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer.
    roi : tuple or list of tuples, optional
        Region(s) of interest (ROI). A single ROI is a tuple of two
        tuples, where the first contains the start coordinates
        (y_start, x_start) and the second the end coordinates
        (y_end, x_end). A list of such tuples restricts identification to
        several (disjoint) regions. If None, the entire frame is used.
        Note that the origin of the image is in the top-left corner.
        Default is None.

    Returns
    -------
    y : lib.IntArray1D
        y-coordinates of the identified maxima.
    x : lib.IntArray1D
        x-coordinates of the identified maxima.
    net_gradient : lib.FloatArray1D
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """
    rois = _as_roi_list(roi)
    if rois is None:
        image = np.float32(frame)  # otherwise numba goes crazy
        return identify_in_image(image, _as_ng_list(minimum_ng, 1)[0], box)
    minimum_ngs = _as_ng_list(minimum_ng, len(rois))
    height, width = frame.shape
    # pad each ROI to identify at the border
    pad = int(box / 2) + 1
    ys, xs, ngs = [], [], []
    for roi_index, ((y0, x0), (y1, x1)) in enumerate(rois):
        py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
        py1, px1 = min(y1 + pad, height), min(x1 + pad, width)
        image = np.float32(frame[py0:py1, px0:px1])  # numba needs float32!
        y, x, net_gradient = identify_in_image(
            image, minimum_ngs[roi_index], box
        )
        y += py0  # offset back to global frame coordinates
        x += px0
        # keep only maxima centered inside the actual ROI
        inside = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
        ys.append(y[inside])
        xs.append(x[inside])
        ngs.append(net_gradient[inside])
    return np.concatenate(ys), np.concatenate(xs), np.concatenate(ngs)


def identify_by_frame_number(
    movie: MovieLike,
    minimum_ng: float | list | np.ndarray,
    box: int,
    frame_number: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    frame_bounds: tuple[int, int] | list | None = None,
    lock: threading.Lock | None = None,
) -> pd.DataFrame:
    """Identify local maxima in a specific frame of a movie and
    calculate the net gradient at those maxima. Optionally, a lock can
    be used to ensure thread safety when accessing the movie data.

    Parameters
    ----------
    movie : MovieLike
        A 3D array representing the movie of shape (N, Y, X), where N is
        the number of frames, Y is the height, and X is the width.
    minimum_ng : float or sequence of float
        Minimum net gradient value to consider a maximum as valid. A
        sequence gives each ROI its own threshold, one value per ROI (see
        :func:`identify_in_frame`).
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer.
    frame_number : int
        The index of the frame in the movie sequence to be processed.
    roi : tuple or list of tuples, optional
        Region(s) of interest (ROI). A single ROI is a tuple of two
        tuples, where the first contains the start coordinates
        (y_start, x_start) and the second the end coordinates
        (y_end, x_end). A list of such tuples restricts identification to
        several (disjoint) regions. If None, the entire frame is used.
        Note that the origin of the image is in the top-left corner.
        Default is None.
    frame_bounds : tuple, list of tuples, optional
        Frame numbers to consider for the identification. A single
        ``(min, max)`` tuple restricts identification to one contiguous,
        inclusive range; a list of such tuples restricts it to several
        (disjoint) segments, where a frame is processed if it falls in any
        segment. If None, all frames are used. If only min or max is to be
        specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    lock : threading.Lock, optional
        If provided, this lock will be used to ensure thread safety when
        accessing the movie data. This is useful in a multithreaded
        environment. Default is None.

    Returns
    -------
    identifications : pd.DataFrame
        DataFrame containing the frame number, x and y coordinates of
        the identified maxima, and their net gradient.
    """
    # check frame bounds before reading, so that frames that are skipped
    # anyway cost nothing (a TemporalMedianMovie would otherwise compute a
    # whole temporal window for them)
    if not lib.frame_in_bounds(frame_number, frame_bounds, len(movie)):
        return pd.DataFrame(
            {
                "frame": pd.Series(dtype=int),
                "x": pd.Series(dtype=int),
                "y": pd.Series(dtype=int),
                "net_gradient": pd.Series(dtype=np.float32),
            }
        )
    # Movies that read each frame through their own per-thread file
    # handle (TiffMap, STKMovie and the multi-file maps) or a memory map
    # are safe to read concurrently, so they skip the shared lock. This
    # lets several frame reads be in flight at once, which hides per-frame
    # I/O latency on network storage. Formats whose readers are not
    # reentrant stay serialized behind the lock.
    concurrent = getattr(
        movie, "supports_concurrent_reads", False
    ) or isinstance(movie, np.memmap)
    if lock is not None and not concurrent:
        with lock:
            frame = movie[frame_number]
    else:
        frame = movie[frame_number]
    # identify
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
    frame = frame_number * np.ones(len(x))
    identifications = pd.DataFrame(
        {
            "frame": frame.astype(int),
            "x": x.astype(int),
            "y": y.astype(int),
            "net_gradient": net_gradient.astype(np.float32),
        }
    )
    return identifications


def _identify_worker(
    movie: MovieLike,
    current: list[int],
    minimum_ng: float | list | np.ndarray,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None,
    frame_bounds: tuple[int, int] | list | None,
    lock: threading.Lock | None,
) -> list[pd.DataFrame]:
    """Worker function for identifying local maxima in a movie. This
    function is designed to be run in a separate thread and processes
    each frame independently."""
    n_frames = len(movie)
    identifications = []
    while True:
        with lock:
            index = current[0]
            if index == n_frames:
                return identifications
            current[0] += 1
        identifications.append(
            identify_by_frame_number(
                movie,
                minimum_ng,
                box,
                index,
                roi=roi,
                frame_bounds=frame_bounds,
                lock=lock,
            )
        )


def identifications_from_futures(
    futures: list[multiprocessing.pool.Future],
) -> pd.DataFrame:
    """Collect the results from a list of futures and combines them
    into a single ``DataFrame``.

    Parameters
    ----------
    futures : list of multiprocessing.pool.Future's
        A list of futures representing the asynchronous tasks.

    Returns
    -------
    ids : pd.DataFrame
        Data frame containing the combined results from
        all futures. Contains fields ``frame``, ``x``, ``y``, and
        ``net_gradient``.
    """
    ids_list_of_lists = [_.result() for _ in futures]
    ids_list = list(chain(*ids_list_of_lists))
    ids = pd.concat(ids_list, ignore_index=True)
    ids.sort_values(by="frame", kind="quicksort", inplace=True)
    return ids


def identify_async(
    movie: MovieLike,
    minimum_ng: float | list | np.ndarray,
    box: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    frame_bounds: tuple[int, int] | list | None = None,
) -> tuple[list[int], list[multiprocessing.pool.Future]]:
    """Asynchronously (i.e., using multithreading) identify local
    maxima in a movie using multiple threads. This function divides the
    work among a specified number of threads.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    minimum_ng : float or sequence of float
        The minimum net gradient for a spot to be considered. A
        sequence gives each ROI its own threshold, one value per ROI
        (see :func:`identify_in_frame`).
    box : int
        The size of the box to extract around each spot.
    roi : tuple or list of tuples, optional
        Region(s) of interest (ROI). A single ROI is a tuple of two
        tuples, where the first contains the start coordinates
        (y_start, x_start) and the second the end coordinates
        (y_end, x_end). A list of such tuples restricts identification to
        several (disjoint) regions. If None, the entire frame is used.
        Default is None.
    frame_bounds : tuple, list of tuples, optional
        Frame numbers to consider for the identification. A single
        ``(min, max)`` tuple restricts identification to one contiguous,
        inclusive range; a list of such tuples restricts it to several
        (disjoint) segments, where a frame is processed if it falls in any
        segment. If None, all frames are used. If only min or max is to be
        specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.

    Returns
    -------
    current : list[int]
        A list of frame indices representing the current processing
        state.
    f : list[multiprocessing.pool.Future]
            A list of futures representing the asynchronous tasks.
    """
    # Use the user settings to define the number of workers that are being used
    settings = io.load_user_settings()

    # avoid the problem when cpu_utilization is not set
    try:
        cpu_utilization = settings["Localize"]["cpu_utilization"]
    except KeyError:
        cpu_utilization = 0.8

    if isinstance(cpu_utilization, float):
        if cpu_utilization >= 1:
            cpu_utilization = 0.8
    else:
        print("CPU utilization was not set. Setting to 0.8")
        cpu_utilization = 0.8
    settings["Localize"]["cpu_utilization"] = cpu_utilization
    io.save_user_settings(settings)

    n_workers = min(
        60, max(1, int(cpu_utilization * multiprocessing.cpu_count()))
    )  # Python crashes when using >64 cores

    lock = threading.Lock()
    current = [0]
    executor = ThreadPoolExecutor(n_workers)
    f = [
        executor.submit(
            _identify_worker,
            movie,
            current,
            minimum_ng,
            box,
            roi,
            frame_bounds,
            lock,
        )
        for _ in range(n_workers)
    ]
    executor.shutdown(wait=False)
    return current, f


def _identify_threaded(
    movie,
    minimum_ng,
    box,
    roi,
    frame_bounds,
    progress_callback,
    abort_callback,
):
    """Run identify_async and drive its progress loop.

    Returns the identifications, or None if aborted.
    """
    N = len(movie)
    use_tqdm = progress_callback == "console"
    iter_range = (
        tqdm(total=N, desc="Identifying spots", unit="frame")
        if use_tqdm
        else None
    )
    current, futures = identify_async(
        movie, minimum_ng, box, roi=roi, frame_bounds=frame_bounds
    )
    last = 0
    while current[0] < N:
        if abort_callback is not None and abort_callback():
            for f in futures:
                f.cancel()
            if use_tqdm:
                iter_range.close()
            return None
        if use_tqdm:
            iter_range.update(current[0] - last)
            last = current[0]
        elif callable(progress_callback):
            progress_callback(current[0])
        time.sleep(0.2)
    if use_tqdm:
        iter_range.update(N - last)
        iter_range.close()
    return identifications_from_futures(futures)


def _identify_serial(
    movie,
    minimum_ng,
    box,
    roi,
    frame_bounds,
    progress_callback,
):
    """Identify spots frame-by-frame in the current thread."""
    N = len(movie)
    use_tqdm = progress_callback == "console"
    iter_range = (
        tqdm(range(N), desc="Identifying spots", unit="frame")
        if use_tqdm
        else range(N)
    )
    identifications = []
    for i in iter_range:
        identifications.append(
            identify_by_frame_number(
                movie,
                minimum_ng,
                box,
                i,
                roi=roi,
                frame_bounds=frame_bounds,
            )
        )
        if callable(progress_callback):
            progress_callback(i)
    ids = pd.concat(identifications, ignore_index=True)
    ids.sort_values(by="frame", kind="quicksort", inplace=True)
    return ids


def identify(
    movie: MovieLike,
    minimum_ng: float | list | np.ndarray,
    box: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    frame_bounds: tuple[int, int] | list | None = None,
    threaded: bool = True,
    temporal_median_window: int | None = None,
    temporal_median_stride: int | None = None,
    gaussian_filter_sigma: float | None = None,
    progress_callback: (
        Callable[[list[int]], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    return_info: bool = True,  # TODO: remove in v0.12.0
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Identify local maxima in a movie and calculate the net
    gradient at those maxima. This function can run in a threaded or
    non-threaded mode.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    minimum_ng : float or sequence of float
        The minimum net gradient for a spot to be considered. A
        sequence gives each ROI its own threshold, one value per ROI
        (see :func:`identify_in_frame`).
    box : int
        The size of the box to extract around each spot.
    roi : tuple or list of tuples, optional
        Region(s) of interest (ROI). A single ROI is a tuple of two
        tuples, where the first contains the start coordinates
        (y_start, x_start) and the second the end coordinates
        (y_end, x_end). A list of such tuples restricts identification to
        several (disjoint) regions. If None, the entire frame is used.
        Note that the origin of the image is in the top-left corner.
        Default is None.
    frame_bounds : tuple, list of tuples, optional
        Frame numbers to consider for the identification. A single
        ``(min, max)`` tuple restricts identification to one contiguous,
        inclusive range; a list of such tuples restricts it to several
        (disjoint) segments, where a frame is processed if it falls in any
        segment. If None, all frames are used. If only min or max is to be
        specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    threaded : bool, optional
        Whether to use threading for the identification process. Default
        is True.
    temporal_median_window : int or None, optional
        If given (and non-zero), a temporal median background is
        subtracted from every frame before identifying, using a window of
        this many frames, see ``TemporalMedianMovie``. The filter applies
        to the identification only - the returned coordinates refer to
        the raw movie, which is what the spots must be fitted on. Note
        that ``minimum_ng`` has to be re-tuned when this is switched on
        or off, since subtracting a background changes the scale of the
        net gradient. Default is None (no filtering).
    temporal_median_stride : int or None, optional
        Spacing between the frames at which the temporal median is
        evaluated, see ``TemporalMedianMovie``. None (the default) uses
        ``temporal_median_window``, which is the fastest setting.
    gaussian_filter_sigma : float or None, optional
        If given (and non-zero), every frame is smoothed with a Gaussian
        of this standard deviation (in camera pixels) before identifying,
        see ``GaussianFilteredMovie``. This merges the several local
        maxima of a spot that is not Gaussian-shaped into one. Applied
        after the temporal median filter, if both are used. The filter
        applies to the identification only - the returned coordinates
        refer to the raw movie, which is what the spots must be fitted
        on. Note that ``minimum_ng`` has to be re-tuned when this is
        changed, since smoothing lowers gradient magnitudes. Default is
        None (no filtering).
    progress_callback : callable, "console" or None, optional
        A callback function to report the progress of the identification
        process. If "console", progress will be printed to the console.
        If None, no progress will be reported. Default is None.
    abort_callback : callable, optional
        A callable for aborting multiprocessing in the GUI. If a
        callable provided, it must accept no input and return a boolean
        indicating whether the fitting should be aborted. Default is
        None.
    return_info : bool, optional
        Whether to return additional information about the fitting
        process. Default is True. If True, a tuple of (locs, info) is
        returned. In v0.12.0 return_info will be removed and the
        function will always return info.

    Returns
    -------
    ids : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    info : dict, optional
        Additional information about the identification process, such as
        the time taken for identification. Only returned if `return_info`
        is True.
    """
    if not return_info:
        # TODO: remove in v0.12.0
        lib.deprecation_warning(
            "In version 0.12, return_info argument will be removed such "
            "that picasso.localize.localize() will always return both "
            "the localizations and the metadata dictionary."
        )
    roi_pad = identification_roi_pad(box, gaussian_filter_sigma)
    if temporal_median_window:
        # note that identify_async() is not wrapped: callers driving the
        # thread pool themselves build the filtered views explicitly
        movie = TemporalMedianMovie(
            movie,
            temporal_median_window,
            stride=temporal_median_stride,
            roi=roi,
            roi_pad=roi_pad,
        )
    # temporal median first, then smoothing: the Gaussian is meant to merge
    # the maxima of one spot, not those of the background it sits on
    if gaussian_filter_sigma:
        movie = GaussianFilteredMovie(movie, gaussian_filter_sigma)
    if threaded:
        ids = _identify_threaded(
            movie,
            minimum_ng,
            box,
            roi,
            frame_bounds,
            progress_callback,
            abort_callback,
        )
        if ids is None:
            return
    else:
        ids = _identify_serial(
            movie,
            minimum_ng,
            box,
            roi,
            frame_bounds,
            progress_callback,
        )
    if return_info:
        info = {
            "Generated by": f"Picasso: v{__version__} Identify",
            "Min. Net Gradient": minimum_ng,
            "Box Size": box,
            "ROI": roi,
            "Frame Bounds": frame_bounds,
            "Temporal Median Window": int(temporal_median_window or 0),
            "Gaussian Filter Sigma": float(gaussian_filter_sigma or 0.0),
        }
        return ids, info
    else:
        return ids


def picks_to_identifications(
    picks: list[tuple],
    *,
    n_frames: int | None = None,
    drift: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert circular picks (from Picasso: Render) to identifications.
    Only circular picks are allowed.

    Parameters
    ----------
    picks : list of tuples
        List of circular picks positions (centers). See
        ``io.load_picks``.
    n_frames : int, optional
        Number of frames in the acquisition movie. If None is given,
        it will be extracted from the drift file (if provided).
        Otherwise, an error is raised.
    drift : pd.DataFrame or None, optional
        A data frame of length n_frames and with columns 'x' and 'y'.
        Used to adjust the positions of identifications throughout
        acquisition. Only x and y drift is used; if 'z' is present, it
        is ignored.

    Returns
    -------
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`. Note that `net_gradient`
        is a dummy value.

    Raises
    ------
    ValueError
        If `n_frames` and `drift` are not provided.
    """
    assert isinstance(picks, (list, tuple)), "picks must be a list or a tuple."
    assert all([len(_) == 2 for _ in picks]), (
        "Circular picks are required. Each element in 'picks' must "
        "contain two numbers (x and y coordinates)."
    )
    if isinstance(drift, pd.DataFrame):
        assert all(
            col in drift.columns for col in ["x", "y"]
        ), "Drift data frame must contain 'x' and 'y' columns."
    if n_frames is None:
        if drift is None:
            raise ValueError(
                "n_frames must be given if no drift file is provided"
            )
        else:
            n_frames = len(drift)
    else:
        assert isinstance(n_frames, int), "n_frames must be an integer."
        if drift is not None:
            assert n_frames == len(drift), (
                f"{n_frames} frames were provided but the drift suggests"
                f" {len(drift)} frames."
            )
    return _picks_to_identifications(picks, n_frames, drift)


def _picks_to_identifications(
    picks: list[tuple],
    n_frames: int,
    drift: pd.DataFrame | None,
) -> pd.DataFrame:
    """Convert circular picks to identifications, can be drift-corrected.
    Assumes correct inputs. See ``picks_to_identifications`` for more
    details."""
    data = []
    n_id = 0
    for pick_x, pick_y in picks:
        # drifted:
        xloc = np.ones((n_frames,), dtype=float) * pick_x
        yloc = np.ones((n_frames,), dtype=float) * pick_y
        if drift is not None:
            xloc += drift["x"].to_numpy()
            yloc += drift["y"].to_numpy()

        frames = np.arange(n_frames)
        gradient = np.ones(n_frames) + 100
        n_id_all = np.ones(n_frames) + n_id
        temp = np.array([frames, xloc, yloc, gradient, n_id_all])
        data.append([tuple(temp[:, j]) for j in range(temp.shape[1])])
        n_id += 1

    data = [item for sublist in data for item in sublist]
    identifications = pd.DataFrame(
        {
            "frame": [item[0] for item in data],
            "x": [item[1] for item in data],
            "y": [item[2] for item in data],
            "net_gradient": [item[3] for item in data],
            "n_id": [item[4] for item in data],
        }
    )
    identifications.sort_values(
        by="frame",
        inplace=True,
        kind="quicksort",
    )
    return identifications


def locs_to_identifications(
    locs: pd.DataFrame,
    movie_info: list[dict],
    n_frames: int,
) -> pd.DataFrame:
    """Convert localizations to identifications.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    movie_info : list of dicts
        Movie file metadata.
    n_frames : int
        Number of frames around localizations that are to be used for
        extracting identifications.

    Returns
    -------
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`. Note that `net_gradient`
        is a dummy value.
    """
    assert isinstance(
        locs, pd.DataFrame
    ), "Localizations must be a pandas data frame"
    assert (
        isinstance(n_frames, int) and n_frames >= 0
    ), "n_frames must be a non-negative integer"
    max_frames = lib.get_from_metadata(movie_info, "Frames", raise_error=True)
    data = []
    n_id = 0
    for _, element in locs.iterrows():
        currframe = element["frame"]
        if currframe > n_frames and currframe < (max_frames - n_frames):
            xloc = np.ones((2 * n_frames + 1,), dtype=float) * element["x"]
            yloc = np.ones((2 * n_frames + 1,), dtype=float) * element["y"]
            frames = np.arange(
                currframe - n_frames,
                currframe + n_frames + 1,
            )
            gradient = np.ones(2 * n_frames + 1) + 100
            n_id_all = np.ones(2 * n_frames + 1) + n_id
            temp = np.array([frames, xloc, yloc, gradient, n_id_all])
            data.append([tuple(temp[:, j]) for j in range(temp.shape[1])])
        n_id += 1
    data = [item for sublist in data for item in sublist]
    identifications = pd.DataFrame(
        {
            "frame": [item[0] for item in data],
            "x": [item[1] for item in data],
            "y": [item[2] for item in data],
            "net_gradient": [item[3] for item in data],
            "n_id": [item[4] for item in data],
        }
    )
    return identifications


@numba.jit(nopython=True, cache=False)
def _cut_spots_numba_into(
    movie: lib.IntArray3D,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
    spots: lib.IntArray3D,
    start: int,
) -> None:
    """Extract spots out of a movie directly into a preallocated array.

    Spots are written into `spots[start : start + len(ids_x)]`, avoiding
    an intermediate allocation and copy. Used for chunked, progress-aware
    cutting.
    """
    r = int(box / 2)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[start + id] = movie[
            frame, yc - r : yc + r + 1, xc - r : xc + r + 1
        ]


@numba.jit(nopython=True, cache=False)
def _cut_spots_numba(
    movie: lib.IntArray3D,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
) -> lib.IntArray3D:
    """Extract the spots out of a movie using Numba for performance."""
    n_spots = len(ids_x)
    spots = np.zeros((n_spots, box, box), dtype=movie.dtype)
    _cut_spots_numba_into(movie, ids_frame, ids_x, ids_y, box, spots, 0)
    return spots


@numba.jit(nopython=True, cache=False)
def _cut_spots_frame(
    frame: lib.IntArray2D,
    frame_number: int,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    r: int,
    start: int,
    N: int,
    spots: lib.IntArray3D,
) -> int:
    """Extract spots from a movie frame."""
    for j in range(start, N):
        if ids_frame[j] > frame_number:
            break
        if ids_frame[j] < frame_number:
            break
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
    return j


@numba.jit(nopython=True, nogil=True, cache=False)
def _cut_spots_single_frame_into(
    frame: lib.IntArray2D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    r: int,
    start: int,
    end: int,
    spots: lib.IntArray3D,
) -> None:
    """Cut every spot in ``ids_[start:end]`` out of a single 2D frame
    into ``spots[start:end]``.

    ``nogil=True`` lets several threads cut (and, more importantly, read
    their frame) at the same time. Each call writes a disjoint slice of
    ``spots``, so no locking is needed around the writes."""
    for j in range(start, end):
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]


def _n_io_workers() -> int:
    """Number of threads to use for I/O-bound frame reading, derived from
    the same ``cpu_utilization`` user setting as identification."""
    settings = io.load_user_settings()
    try:
        cpu_utilization = settings["Localize"]["cpu_utilization"]
    except KeyError:
        cpu_utilization = 0.8
    if not isinstance(cpu_utilization, float) or cpu_utilization >= 1:
        cpu_utilization = 0.8
    # Python crashes when using >64 cores
    return min(60, max(1, int(cpu_utilization * multiprocessing.cpu_count())))


@numba.jit(nopython=True, cache=False)
def _cut_spots_daskmov(
    movie: MovieLike,
    l_mov: lib.IntArray1D,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
    spots: lib.IntArray3D,
):
    """Extract the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    l_mov : lib.IntArray1D
        Length of the movie, a 1D array with a single element.
    ids_frame, ids_x, ids_y : lib.IntArray1D
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : lib.IntArray3D
        3D array to store the cut spots, with shape (k, box, box),
        where k is the number of spots identified.

    Returns
    -------
    spots : lib.IntArray3D
        3D array with extracted spots of shape (k, box, box), where k is
        the number of spots identified.
    """
    r = int(box / 2)
    N = len(ids_frame)
    start = 0
    for frame_number in range(l_mov[0]):
        frame = movie[frame_number, :, :]
        start = _cut_spots_frame(
            frame,
            frame_number,
            ids_frame,
            ids_x,
            ids_y,
            r,
            start,
            N,
            spots,
        )
    return spots


def _cut_spots_framebyframe(
    movie: MovieLike,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
    spots: lib.IntArray3D,
    progress_callback: Callable[[int], None] | None = None,
):
    """Extract the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    ids_frame, ids_x, ids_y : lib.IntArray1D
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : lib.IntArray3D
        3D array to store the cut spots, with shape (k, box, box),
        where k is the number of spots identified.
    progress_callback : callable or None, optional
        If a callable is provided, it is called after each frame with
        the cumulative number of spots cut so far. Default is None.

    Returns
    -------
    spots : lib.IntArray3D
        3D array with extracted spots of shape (k, box, box), where k is
        the number of spots identified.

    Notes
    -----
    When the movie supports concurrent reads (its frames are read
    through a per-thread file handle), frames are read and cut in
    parallel, which hides per-frame I/O latency the same way threaded
    identification does. ``ids_frame`` is assumed to be sorted (as
    ``identify`` returns it), so each frame maps to one contiguous slice
    of ``spots``.
    """
    r = int(box / 2)
    N = len(ids_frame)
    n_frames = len(movie)

    # Since ids are frame-sorted, frame f's spots are the contiguous
    # slice spots[starts[f]:ends[f]].
    starts = np.searchsorted(ids_frame, np.arange(n_frames), side="left")
    ends = np.append(starts[1:], N)

    if getattr(movie, "supports_concurrent_reads", False):
        done = [0]
        progress_lock = threading.Lock()

        def _read_and_cut(frame_number: int) -> None:
            start = int(starts[frame_number])
            end = int(ends[frame_number])
            frame = movie[frame_number]
            _cut_spots_single_frame_into(
                frame, ids_x, ids_y, r, start, end, spots
            )
            if callable(progress_callback):
                with progress_lock:
                    done[0] += end - start
                    progress_callback(done[0])

        with ThreadPoolExecutor(_n_io_workers()) as executor:
            # consume the iterator so exceptions propagate
            list(executor.map(_read_and_cut, range(n_frames)))
        return spots

    # Serial fallback for movies whose readers are not reentrant
    # (e.g. ND2/CZI/LIF).
    cum = 0
    for frame_number in range(n_frames):
        start = int(starts[frame_number])
        end = int(ends[frame_number])
        frame = movie[frame_number]
        _cut_spots_single_frame_into(frame, ids_x, ids_y, r, start, end, spots)
        cum += end - start
        if callable(progress_callback):
            progress_callback(cum)
    return spots


def _cut_spots(
    movie: MovieLike,
    ids: pd.DataFrame,
    box: int,
    progress_callback: Callable[[int], None] | None = None,
) -> lib.IntArray3D:
    """Cut out spots from a movie based on the identified positions.

    If a callable `progress_callback` is provided, it is called with the
    cumulative number of spots cut so far, allowing the cutting progress
    to be tracked.
    """
    N = len(ids)
    ids_frame = ids["frame"].to_numpy()
    ids_x = ids["x"].to_numpy()
    ids_y = ids["y"].to_numpy()
    if isinstance(movie, np.ndarray):
        if not callable(progress_callback):
            return _cut_spots_numba(movie, ids_frame, ids_x, ids_y, box)
        # cut in chunks so that progress can be reported; spots are
        # written directly into the output array to avoid an extra copy
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        chunk = max(1, N // 100)
        for chunk_start in range(0, N, chunk):
            chunk_end = min(chunk_start + chunk, N)
            _cut_spots_numba_into(
                movie,
                ids_frame[chunk_start:chunk_end],
                ids_x[chunk_start:chunk_end],
                ids_y[chunk_start:chunk_end],
                box,
                spots,
                chunk_start,
            )
            progress_callback(chunk_end)
        return spots
    elif isinstance(movie, io.ND2Movie) and movie.use_dask:
        """Assumes that identifications are in order of frames!"""
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        spots = da.apply_gufunc(
            _cut_spots_daskmov,
            "(p,n,m),(b),(k),(k),(k),(),(k,l,l)->(k,l,l)",
            movie.data,
            np.array([len(movie)]),
            ids_frame,
            ids_x,
            ids_y,
            box,
            spots,
            output_dtypes=[movie.dtype],
            allow_rechunk=True,
        ).compute()
        if callable(progress_callback):
            progress_callback(N)
        return spots
    else:
        """Assumes that identifications are in order of frames!"""
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        spots = _cut_spots_framebyframe(
            movie,
            ids_frame,
            ids_x,
            ids_y,
            box,
            spots,
            progress_callback=progress_callback,
        )
        return spots


def _cut_map(
    image: lib.FloatArray2D, ids: pd.DataFrame, box: int
) -> lib.FloatArray3D:
    """Cut ``(box, box)`` patches out of a full-frame map, one per spot.

    The geometry is that of :func:`_cut_spots_numba_into`
    (``image[yc - r : yc + r + 1, xc - r : xc + r + 1]``), so a patch lines up
    pixel for pixel with the corresponding spot. A camera map has no frame
    axis, so this is a plain fancy-index rather than another ``_cut_spots``
    backend.
    """
    r = box // 2
    offsets = np.arange(box)
    rows = ids["y"].to_numpy()[:, None] - r + offsets  # (k, box)
    cols = ids["x"].to_numpy()[:, None] - r + offsets  # (k, box)
    return np.ascontiguousarray(
        image[rows[:, :, None], cols[:, None, :]], dtype=np.float32
    )


def _sensitivity(
    camera_info: dict, gain_patch: lib.FloatArray3D | None
) -> float | lib.FloatArray3D:
    """Counts-to-photoelectrons factor, scalar or per pixel.

    Picasso's ``Sensitivity`` is electrons per A/D count, i.e. the reciprocal
    of the amplification gain ``g`` a camera calibration measures in ADU per
    photoelectron (Huang et al. 2013, Supplementary Note Section 2.3)."""
    if gain_patch is None:
        return camera_info["Sensitivity"]
    return 1.0 / gain_patch


def _to_photons(
    spots: lib.FloatArray3D,
    camera_info: dict,
    offset: lib.FloatArray3D | None = None,
    gain_patch: lib.FloatArray3D | None = None,
) -> lib.FloatArray3D:
    """Convert the cut spots to photon counts based on camera
    information.

    ``offset`` and ``gain_patch`` are per-spot ``(k, box, box)`` patches of an
    sCMOS camera calibration; each overrides the corresponding scalar
    (``Baseline``, ``Sensitivity``) where it is given.
    """
    spots = np.float32(spots)
    baseline = camera_info["Baseline"] if offset is None else offset
    sensitivity = _sensitivity(camera_info, gain_patch)
    gain = camera_info["Gain"]
    # since v0.6.0: remove quantum efficiency to better reflect precision
    # qe = camera_info["Qe"]
    return (spots - baseline) * sensitivity / (gain)


def _variance_to_photons(
    variance: lib.FloatArray3D,
    camera_info: dict,
    gain_patch: lib.FloatArray3D | None = None,
) -> lib.FloatArray3D:
    """Convert a readout variance from ADU squared to photoelectrons squared.

    :func:`_to_photons` scales counts by ``Sensitivity / Gain``, so a variance
    scales by its square. This is Huang et al.'s ``var / g^2``, the quantity
    the noise model adds to both the data and the model mean.
    """
    sensitivity = _sensitivity(camera_info, gain_patch)
    gain = camera_info["Gain"]
    return np.float32(variance) * (sensitivity / gain) ** 2


def _validate_camera_calibration(
    camera_calibration: dict | None, movie, camera_info: dict
) -> None:
    """Check a camera calibration against the movie it will be applied to.

    The maps are indexed with the identifications' absolute frame coordinates,
    so a calibration recorded with a different camera ROI or binning would
    silently read the wrong pixels. Checked once, before any spot is cut.
    """
    if camera_calibration is None:
        return
    for name in ("offset", "variance"):
        if camera_calibration.get(name) is None:
            raise ValueError(
                f"Invalid camera calibration: missing the '{name}' map. Build "
                "one with picasso.scmos.calibrate_scmos or load it with "
                "picasso.io.load_camera_calibration."
            )
    dims = io._readable_movie_dims(movie)
    height, width = dims.get("Height"), dims.get("Width")
    if height is None or width is None:
        shape = getattr(movie, "shape", None)
        if shape is not None and len(shape) == 3:
            height, width = int(shape[1]), int(shape[2])
    map_shape = np.shape(camera_calibration["offset"])
    if height is not None and width is not None:
        if map_shape != (height, width):
            raise ValueError(
                f"The camera calibration was computed on {map_shape[0]}x"
                f"{map_shape[1]} frames but this movie is {height}x{width}. "
                "Compute the offset/variance maps from a dark movie acquired "
                "with the same camera ROI and binning."
            )
    if (
        camera_calibration.get("gain") is not None
        and camera_info.get("Gain", 1) > 1
    ):
        warnings.warn(
            "A per-pixel camera calibration was supplied together with an EM "
            "gain > 1. The sCMOS noise model of Huang et al. (2013) assumes a "
            "non-multiplying sensor; the EMCCD excess-noise factor of 2 is "
            "still applied to every uncertainty, which double-counts the "
            "noise. Set the EM gain to 1 for an sCMOS camera.",
            RuntimeWarning,
        )


def _mean_readout_variance(
    variance: lib.FloatArray3D | None,
) -> lib.FloatArray1D | float:
    """Per-spot mean readout variance, for the closed-form precisions.

    The Mortensen-family formulas describe a spatially uniform background, so
    a per-pixel map can only enter them through its mean over the fitting box.
    Returns 0.0 when there is no calibration, which leaves those formulas
    exactly as they were."""
    if variance is None:
        return 0.0
    return variance.reshape(len(variance), -1).mean(axis=1)


def _clip_for_mle(
    spots: lib.FloatArray3D, variance: lib.FloatArray3D | None
) -> lib.FloatArray3D:
    """Floor the data where the Poisson likelihood is defined.

    Camera offset subtraction pushes dim pixels below zero, and a Poisson
    likelihood is undefined there. Without a noise model the floor is zero, as
    it has always been. With one, the likelihood is evaluated on the *shifted*
    data ``d + var``, so the floor moves to ``-var``: clipping at zero instead
    would discard exactly the negative excursions readout noise creates, and
    bias the fitted background upward on the noisiest pixels - the opposite of
    what the noise model is for.
    """
    if variance is None:
        return np.maximum(spots, 0)
    return np.maximum(spots, -variance)


def camera_calibration_info(camera_calibration: dict | None) -> dict:
    """Provenance of an sCMOS camera calibration, for the saved metadata.

    Every caller that fits with a calibration must record this, or a
    localization file carries no trace of the noise model that produced it and
    two runs become indistinguishable after the fact. It lives here, rather
    than inline in ``fit``, because Picasso Localize rebuilds its own
    metadata when saving instead of using what ``fit`` returns, and the two
    must not drift apart.

    Returns an empty dict when there is no calibration, so callers can
    ``update()`` unconditionally.
    """
    if not camera_calibration:
        return {}
    info = {
        "Camera calibration path": camera_calibration.get("Path", "N/A"),
        "Camera calibration frames": camera_calibration.get("Frames"),
        "Camera offset source": "per-pixel map",
        "Camera gain source": (
            "per-pixel map"
            if camera_calibration.get("gain") is not None
            else "Sensitivity (scalar)"
        ),
    }
    for key in (
        "Offset median (ADU)",
        "Variance median (ADU^2)",
        "Hot pixels",
    ):
        if key in camera_calibration:
            info[f"Camera calibration {key}"] = camera_calibration[key]
    return info


def _seed_spots(
    spots: lib.FloatArray3D, variance: lib.FloatArray3D | None
) -> lib.FloatArray3D:
    """Spots to estimate the initial fit parameters from.

    The seeds take the background from the dimmest pixel of the ROI and the
    amplitude from the brightest minus the dimmest. That is fine on data
    floored at zero, but :func:`_clip_for_mle` floors at ``-var``, so on a
    noisy pixel the dimmest value can be tens of photons *below* zero. Seeding
    a background there makes the model mean negative across the whole ROI, the
    likelihood is then floored everywhere, the first Hessian is singular and
    the fit aborts - which cost about a third of all spots on a sensor with
    realistic hot pixels.

    The seed is only a starting point, and a negative background is never a
    sensible one, so it is estimated from the zero-floored data. The fit
    itself still runs on the ``-var`` floored data, where the shifted Poisson
    likelihood is defined. This also keeps the seed identical with and without
    a calibration, so any difference between the two fits comes from the noise
    model rather than from where they started.
    """
    if variance is None:
        return spots
    return np.maximum(spots, 0)


def get_spots(
    movie: MovieLike,
    identifications: pd.DataFrame,
    box: int,
    camera_info: dict,
    progress_callback: Callable[[int], None] | None = None,
    camera_calibration: dict | None = None,
    return_variance: bool = False,
) -> lib.FloatArray3D | tuple[lib.FloatArray3D, lib.FloatArray3D | None]:
    """Extract the spots from a movie based on the identified positions
    and convert camera signal to photon counts.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    progress_callback : callable or None, optional
        If a callable is provided, it is called with the cumulative
        number of spots cut so far, allowing the cutting progress to be
        tracked. Default is None.
    camera_calibration : dict or None, optional
        Per-pixel sCMOS camera calibration from ``picasso.scmos``. Its
        ``offset`` map replaces the scalar ``Baseline`` and, if present, its
        ``gain`` map replaces the scalar ``Sensitivity``. Default is None.
    return_variance : bool, optional
        Also return the per-spot readout variance in photoelectrons squared,
        cut from the same ROIs as the spots. None when no calibration was
        given. Default is False.

    Returns
    -------
    spots : lib.FloatArray3D
        A 3D numpy array containing the extracted spots, with shape
        (k, box, box), where k is the number of spots identified.
    variance : lib.FloatArray3D or None
        Only if ``return_variance``.
    """
    spots = _cut_spots(
        movie, identifications, box, progress_callback=progress_callback
    )
    offset = gain_patch = variance = None
    if camera_calibration is not None:
        offset = _cut_map(camera_calibration["offset"], identifications, box)
        if camera_calibration.get("gain") is not None:
            gain_patch = _cut_map(
                camera_calibration["gain"], identifications, box
            )
        if return_variance:
            variance = _variance_to_photons(
                _cut_map(camera_calibration["variance"], identifications, box),
                camera_info,
                gain_patch,
            )
    spots = _to_photons(spots, camera_info, offset, gain_patch)
    if return_variance:
        return spots, variance
    return spots


def locs_from_fits(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    CRLBs: lib.FloatArray2D,
    likelihoods: lib.FloatArray1D,
    iterations: lib.FloatArray1D,
    box: int,
) -> pd.DataFrame:
    """Convert the resulting localizations from the list of Futures
    into a data frame.

    .. deprecated:: 0.11
        Removed in Picasso 1.0. Left over from the old ``gaussmle`` GPU
        pipeline and unused since; use :func:`locs_from_fits_gauss`,
        which takes the fitters' ``theta`` layout directly.

    Parameters
    ----------
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    theta : lib.FloatArray2D
        The fitted Gaussian parameters for each spot (x, y positions,
        photon counts, background, single-emitter image size in x and
        y).
    CRLBs : lib.FloatArray2D
        The Cramer-Rao Lower Bounds for each fitted parameter.
    likelihoods : lib.FloatArray1D
        The log-likelihoods of the fitted models.
    iterations : lib.FloatArray1D
        The number of iterations taken to converge for each spot.
    box : int
        Size of the box used for fitting. Should be an odd integer.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots. The fields include
        `frame`, `x`, `y`, `photons`, `sx`, `sy`, `bg`, `lpx`, `lpy`,
        `net_gradient`, `log_likelihood`, and `iterations`.
    """
    lib.deprecation_warning(
        "picasso.localize.locs_from_fits is deprecated and will be removed "
        "in Picasso 1.0. Use picasso.localize.locs_from_fits_gauss."
    )
    # box_offset = int(box / 2)
    y = theta[:, 0] + identifications["y"]  # - box_offset
    x = theta[:, 1] + identifications["x"]  # - box_offset
    lpy = np.sqrt(CRLBs[:, 0])
    lpx = np.sqrt(CRLBs[:, 1])
    locs = pd.DataFrame(
        {
            "frame": identifications["frame"].astype(np.uint32),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "photons": theta[:, 2].astype(np.float32),
            "sx": theta[:, 5].astype(np.float32),
            "sy": theta[:, 4].astype(np.float32),
            "bg": theta[:, 3].astype(np.float32),
            "lpx": lpx.astype(np.float32),
            "lpy": lpy.astype(np.float32),
            "net_gradient": (
                identifications["net_gradient"].astype(np.float32)
            ),
            "log_likelihood": likelihoods.astype(np.float32),
            "iterations": iterations.astype(np.int32),
        }
    )
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def fit(
    movie: LoadedMovie,
    *,
    camera_info: dict,
    identifications: pd.DataFrame,
    box: int,
    fitting_method: Literal[
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
        "avg",
    ] = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    spline_calibration: dict | None = None,
    camera_calibration: dict | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    cut_progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    """Fit 2D localizations to a movie, given positions of the detected
    spots (identifications).

    Since v0.11.0: renamed from ``fit2D``, which is deprecated and will
    be removed in v0.12.0, together with its unused ``movie_info`` and
    ``mle_method`` arguments. Only the movie is accepted positionally.

    Parameters
    ----------
    movie : LoadedMovie
        The input movie, as loaded by ``picasso.io.load_movie``.
    camera_info : dict
        A dictionary containing camera information: "Baseline",
        "Sensitivity", "Gain" and "Pixelsize".
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    fitting_method : {"gausslq", "gausslq-spherical", "gausslq-rotated", \
            "gausslq-gpu", "gausslq-rotated-gpu", "gausslq-spherical-gpu", \
            "gaussmle", "gaussmle-spherical", "gaussmle-gpu", \
            "gaussmle-rotated-gpu", "gaussmle-spherical-gpu", "spline-gpu", \
            "spline-mle-gpu" or "avg"}, optional
        Which 2D fitting algorithm to use. "gausslq" for least-squares
        fitting of a 2D Gaussian. "gausslq-gpu" for its GPU
        implemntation (if available). "gaussmle" for MLE 2D Gaussian
        fitting (CPU). "gaussmle-gpu" for MLE fitting of a 2D Gaussian
        on the GPU (the Poisson maximum likelihood estimator).
        "gausslq-rotated" for CPU least-squares fitting, and
        "gausslq-rotated-gpu" and "gaussmle-rotated-gpu" for GPU
        least-squares and MLE fitting, respectively, of a rotated
        elliptical Gaussian, whose fitted rotation angle (in degrees)
        is saved in the column "angle". "gausslq-spherical" and
        "gaussmle-spherical" for CPU least-squares and MLE fitting, and
        "gausslq-spherical-gpu" and "gaussmle-spherical-gpu" for their
        GPU counterparts, of a spherical (isotropic) Gaussian with a
        single width; the saved "sx" and "sy" columns are identical.
        "spline" and "spline-mle" for CPU least-squares /
        maximum-likelihood fitting of an experimentally measured
        cubic-spline PSF, and "spline-gpu" and "spline-mle-gpu" for their
        GPU counterparts. All four
        require ``spline_calibration``, and a 3D spline calibration yields
        the fitted ``z`` directly. "avg" for taking the average of each
        spot.
    eps : float or None, optional
        The convergence criterion, honoured by every iterating method on
        either device (all of them except "avg"). None (the default)
        picks the value that suits the method: 0.001 for "gaussmle",
        0.01 for "gausslq" and the GPU Gaussians, and for either spline
        backend 1e-4 with the axial multi-start and 1e-2 without.
    max_it : int or None, optional
        The maximum number of iterations per spot, as ``eps``. None (the
        default) means 100 for "gaussmle", 200 for "gausslq" (MINPACK's
        own default), 20 for the GPU Gaussians and, for either spline
        backend, 100 with the axial multi-start and 20 without.
    spline_calibration : dict or None, optional
        Cubic-spline PSF calibration (see ``io.load_spline_calibration``),
        required for any "spline*" ``fitting_method`` and ignored
        otherwise. For a 3D spline calibration the resulting localizations
        contain the fitted ``z`` directly (no separate z-fitting step is
        needed). Default is None.
    camera_calibration : dict or None, optional
        Per-pixel sCMOS camera calibration (see
        ``io.load_camera_calibration`` and ``scmos.calibrate_scmos``),
        holding the maps "offset" (ADU), "variance" (ADU^2) and,
        optionally, "gain" (ADU/e-). The maps must match the full frame
        shape of ``movie``. When given, they replace the scalar
        "Baseline" (and, if a gain map is present, "Sensitivity") of
        ``camera_info``, and the per-pixel readout variance enters the
        noise model of Huang et al., Nat. Methods 10:653 (2013): every
        MLE fit and every uncertainty estimate ("lpx", "lpy", CRLB) then
        de-emphasises noisy pixels. Least-squares fits are unaffected by
        the variance term itself (the shift cancels), but their
        uncertainties do grow on noisy pixels; prefer an MLE method for
        sCMOS data. Default is None.
    multiprocess: bool, optional
        Whether or not to use multiprocessing. Ignored for GPU fitting.
        Default is True.
    progress_callback : callable, "console" or None, optional
        If a callable provided, it must accept one integer input (number
        of localized spots). If "console", tqdm is used to display
        progress. If None, progress is not tracked.
    abort_callback : callable or None, optional
        A callable for aborting multiprocessing in the GUI. If a
        callable provided, it must accept no input and return a boolean
        indicating whether the fitting should be aborted. Default is
        None.
    cut_progress_callback : callable or None, optional
        If a callable is provided, it is called with the cumulative
        number of spots cut so far while extracting the spots from the
        movie (before fitting). It must accept one integer input.
        Default is None.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots. Returns None if
        fitting was aborted.
    new_info : dict
        New metadata.
    """
    accepted_movie_types = (io.AbstractPicassoMovie, np.memmap)
    if bitplane.IMSWRITER:
        accepted_movie_types += (
            bitplane.MovieMapper,
            bitplane.MovieMapperStack,
        )
    assert isinstance(
        movie, accepted_movie_types
    ), "movie must be a movie loaded by picasso.io.load_movie"
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert isinstance(
        identifications, pd.DataFrame
    ), "identifications must be a DataFrame"
    assert isinstance(box, int) and box > 0, "box must be a positive integer"
    assert fitting_method in FIT_METHODS, (
        f"fitting_method '{fitting_method}' is not one of "
        f"{', '.join(FIT_METHODS)}"
    )
    if fitting_method.startswith("spline"):
        assert isinstance(spline_calibration, dict), (
            "spline_calibration (a spline PSF calibration dict, see "
            "io.load_spline_calibration) is required for spline fitting"
        )
    assert eps is None or (
        isinstance(eps, (int, float)) and eps > 0
    ), "eps must be a positive number or None"
    assert max_it is None or (
        isinstance(max_it, int) and max_it > 0
    ), "max_it must be a positive integer or None"
    assert isinstance(multiprocess, bool), "multiprocess must be a boolean"
    if "Pixelsize" not in camera_info:
        warnings.warn(
            "Camera info in picasso.localize.fit does not contain "
            "'Pixelsize', i.e., effective camera pixel size in nm. "
            "Assuming 130."
        )
        camera_info["Pixelsize"] = 130

    # ``camera_info`` is merged verbatim into the saved YAML at the end, so an
    # array in it would be dumped element by element into every sidecar. The
    # per-pixel maps travel as ``camera_calibration`` precisely so that cannot
    # happen. Checked here rather than at the merge so the message arrives
    # before _to_photons turns it into an opaque broadcast error.
    for _key, _value in camera_info.items():
        if isinstance(_value, np.ndarray):
            raise ValueError(
                f"camera_info['{_key}'] is an array. Per-pixel camera maps "
                "belong in the camera_calibration argument, not in "
                "camera_info, which is written to the metadata file as-is."
            )
    _validate_camera_calibration(camera_calibration, movie, camera_info)
    spots, variance = get_spots(
        movie,
        identifications,
        box,
        camera_info,
        progress_callback=cut_progress_callback,
        camera_calibration=camera_calibration,
        return_variance=True,
    )
    em = camera_info["Gain"] > 1
    gauss_flags = parse_gauss_code(fitting_method)
    if gauss_flags is not None:
        if gauss_flags["use_gpu"] and callable(progress_callback):
            progress_callback(1)
        locs = _fit2d_gauss(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
            tolerance=eps,
            max_iterations=max_it,
            progress_callback=(
                None if gauss_flags["use_gpu"] else progress_callback
            ),
            variance=variance,
            **gauss_flags,
        )
    elif fitting_method in ("spline-gpu", "spline-mle-gpu"):
        if callable(progress_callback):
            progress_callback(1)
        # "spline-mle-gpu" uses the Poisson maximum-likelihood estimator,
        # "spline-gpu" least squares.
        # The GPU fit itself is a single call; progress_callback then tracks
        # the per-spot CRLB / precision computation in locs_from_fits_spline.
        locs = _fit2d_spline_gpu(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
            calibration=spline_calibration,
            mle=fitting_method == "spline-mle-gpu",
            progress_callback=progress_callback,
            tolerance=eps,
            max_iterations=max_it,
            variance=variance,
        )
    elif fitting_method in ("spline", "spline-mle"):
        # The CPU cubic-spline fit (picasso.fitting.splinefit). Unlike the
        # GPU path it is a per-spot loop, so progress_callback tracks the fit
        # itself and the fit can be aborted. eps / max_it override the
        # convergence schedule; None picks the one matching the multi-start.
        locs = _fit2d_spline_cpu(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
            calibration=spline_calibration,
            mle=fitting_method == "spline-mle",
            tolerance=eps,
            max_iterations=max_it,
            multiprocess=multiprocess,
            progress_callback=progress_callback,
            abort_callback=abort_callback,
            variance=variance,
        )
    elif fitting_method == "avg":
        locs = _fit2d_avg(
            spots,
            identifications,
            box,
            em,
            multiprocess,
            progress_callback,
            abort_callback,
            variance=variance,
        )
    # updated metadata
    localize_info = {
        "Generated by": f"Picasso: v{__version__} Fit 2D",
        "Fit method": fitting_method,
    }
    # Record the schedule the fit actually ran with, per method - each
    # backend has its own defaults, and "None" in the caller means "yours".
    if gauss_flags is not None:
        tolerance, max_iterations = gauss_schedule(
            gauss_flags["mle"], gauss_flags["use_gpu"], eps, max_it
        )
        localize_info["Convergence criterion"] = tolerance
        localize_info["Max iterations"] = max_iterations
    if fitting_method.startswith("spline"):
        localize_info["Spline calibration model"] = spline_calibration.get(
            "model"
        )
        localize_info["Spline calibration path"] = spline_calibration.get(
            "Path", "N/A"
        )
        on_gpu = fitting_method.endswith("-gpu")
        localize_info["Spline fit device"] = "GPU" if on_gpu else "CPU"
        localize_info["Spline CRLB device"] = (
            "GPU" if precision.CUDA_AVAILABLE else "CPU"
        )
        # Record what the fit actually used, not what was requested: the
        # schedule depends on whether the axial multi-start ran. Identical
        # on both devices - they share ``_run_splinefit``.
        n_z_starts = _default_n_z_starts(spline_calibration)
        _, apply_seeds = _spline_z_seeds(spline_calibration, n_z_starts)
        tolerance, max_iterations = _spline_schedule(apply_seeds, eps, max_it)
        localize_info["Convergence criterion"] = tolerance
        localize_info["Max iterations"] = max_iterations
        localize_info["Axial seeds"] = n_z_starts if apply_seeds else 1
    localize_info.update(camera_calibration_info(camera_calibration))
    new_info = localize_info | camera_info
    return locs, new_info


# TODO: remove in v0.12.0
def fit2D(
    movie: LoadedMovie,
    movie_info: list[dict] | None = None,
    camera_info: dict | None = None,
    identifications: pd.DataFrame | None = None,
    box: int | None = None,
    fitting_method: str = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    mle_method: Literal["sigma", "sigmaxy"] | None = None,
    **kwargs,
) -> tuple[pd.DataFrame | None, dict]:
    """Deprecated alias for ``fit``.

    .. deprecated:: 0.11.0
        Use ``picasso.localize.fit`` instead. ``fit2D`` will be removed
        in v0.12.0, together with the ``movie_info`` and ``mle_method``
        arguments, neither of which affects the fit.
    """
    lib.deprecation_warning(
        "picasso.localize.fit2D is deprecated and will be removed in "
        "version 0.12; use picasso.localize.fit instead. Its movie_info "
        "and mle_method arguments will be removed with it - neither has "
        "any effect on the fit."
    )
    if mle_method is not None:
        lib.deprecation_warning(
            "The mle_method argument is ignored and will be removed in "
            "version 0.12."
        )
    assert isinstance(movie_info, list), "movie_info must be a list"
    return fit(
        movie=movie,
        camera_info=camera_info,
        identifications=identifications,
        box=box,
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        **kwargs,
    )


def _initial_widths_gauss(
    spots: lib.FloatArray3D,
    size: int,
    background: lib.FloatArray1D,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Seed the Gaussian widths from the second moment of the spot's central
    row and column."""
    half = int(size / 2)
    # float64 throughout: the moment sums photons weighted by a squared
    # distance, which overflows a float32 accumulator on a bright spot.
    d2 = (np.arange(size, dtype=np.float64) - half) ** 2
    # spots is (n, y, x): the central column varies along y, the row along x.
    profile_y = spots[:, :, half].astype(np.float64) - background[:, None]
    profile_x = spots[:, half, :].astype(np.float64) - background[:, None]
    np.clip(profile_y, 0.0, None, out=profile_y)
    np.clip(profile_x, 0.0, None, out=profile_x)
    # A non-finite profile (an inf from a gain map, say) falls through to the
    # fallback below rather than warning here.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        width_y = np.sqrt(profile_y @ d2 / profile_y.sum(axis=1))
        width_x = np.sqrt(profile_x @ d2 / profile_x.sum(axis=1))
    # An empty or single-pixel profile yields 0/0 or a zero width.
    fallback = max(size / 5.0, 1.0)
    width_y[~np.isfinite(width_y)] = fallback
    width_x[~np.isfinite(width_x)] = fallback
    np.minimum(width_y, fallback, out=width_y)
    np.minimum(width_x, fallback, out=width_x)
    np.clip(width_y, 0.5, size / 3.0, out=width_y)
    np.clip(width_x, 0.5, size / 3.0, out=width_x)
    return width_x, width_y


def _initial_parameters_gauss(
    spots: lib.FloatArray3D,
    size: int,
    rotated: bool = False,
    spherical: bool = False,
) -> lib.FloatArray2D:
    """Initialize the parameters for a Gaussian fit - photons, x, y, sx,
    sy, bg (plus the rotation angle if ``rotated``). If ``spherical``,
    a single width is used and the layout is photons, x, y, s, bg
    (the isotropic ``GAUSS_2D`` model)."""
    center = (size / 2.0) - 0.5

    spot_max = np.amax(spots, axis=(1, 2))
    spot_min = np.amin(spots, axis=(1, 2))

    width_x, width_y = _initial_widths_gauss(spots, size, spot_min)

    if spherical:
        # GAUSS_2D: photons, x, y, s (single width), bg.
        initial_parameters = np.empty((len(spots), 5), dtype=np.float32)
        initial_parameters[:, 0] = spot_max - spot_min
        initial_parameters[:, 1] = center
        initial_parameters[:, 2] = center
        initial_parameters[:, 3] = 0.5 * (width_x + width_y)
        initial_parameters[:, 4] = spot_min
        return initial_parameters

    n_parameters = 7 if rotated else 6
    initial_parameters = np.empty((len(spots), n_parameters), dtype=np.float32)

    initial_parameters[:, 0] = spot_max - spot_min
    initial_parameters[:, 1] = center
    initial_parameters[:, 2] = center
    initial_parameters[:, 3] = width_x
    initial_parameters[:, 4] = width_y
    initial_parameters[:, 5] = spot_min
    if rotated:
        # With sx == sy, the rotated Gaussian is independent of the
        # angle, so its derivative is exactly zero and the first LM
        # Hessian is singular - the fit then aborts, returning the
        # initial parameters. Break the symmetry of the widths to keep
        # the angle parameter well-defined.
        initial_parameters[:, 3] *= 1.1
        initial_parameters[:, 4] *= 0.9
        initial_parameters[:, 6] = 0.0

    return initial_parameters


# Per-method convergence schedules. Each backend's own defaults, kept here so
# the resolved values reach both the fit and the saved metadata, and so that
# rerouting a method to a new backend cannot silently change where it stops.
#: Tokens a Gaussian fit code may carry after ``gausslq``/``gaussmle``.
_GAUSS_TOKENS = frozenset({"spherical", "rotated", "gpu"})


def parse_gauss_code(fitting_method: str) -> dict | None:
    """Flags of a Gaussian fit code, or None if it is not one.

    The grammar is ``gauss{lq,mle}[-spherical|-rotated][-gpu]`` and it returns
    ``{"mle", "spherical", "rotated", "use_gpu"}``.

    Returns None for anything that is not a valid Gaussian code, so callers
    can use it as both the parser and the validator.
    """
    tokens = fitting_method.split("-")
    if tokens[0] not in ("gausslq", "gaussmle"):
        return None
    if len(set(tokens[1:])) != len(tokens[1:]):
        return None  # a repeated token, e.g. "gausslq-gpu-gpu"
    flags = {
        "mle": tokens[0] == "gaussmle",
        "spherical": False,
        "rotated": False,
        "use_gpu": False,
    }
    for token in tokens[1:]:
        if token not in _GAUSS_TOKENS:
            return None
        if token == "spherical":
            flags["spherical"] = True
        elif token == "rotated":
            flags["rotated"] = True
        else:  # "gpu"
            flags["use_gpu"] = True
    if flags["spherical"] and flags["rotated"]:
        return None
    return flags


def gauss_fit_methods() -> list[str]:
    """Every Gaussian fit code :func:`parse_gauss_code` accepts.

    Generated from the grammar rather than listed by hand, so a code cannot
    be offered somewhere and rejected here."""
    codes = []
    for estimator in ("gausslq", "gaussmle"):
        for shape in ("", "-spherical", "-rotated"):
            for device in ("", "-gpu"):
                code = f"{estimator}{shape}{device}"
                if parse_gauss_code(code) is not None:
                    codes.append(code)
    return codes


#: Every ``fit`` method. Generated for the Gaussians (see
#: :func:`gauss_fit_methods`) and listed for the rest, which have no grammar.
FIT_METHODS = tuple(
    gauss_fit_methods()
    + ["spline", "spline-mle", "spline-gpu", "spline-mle-gpu", "avg"]
)


_GAUSS_SCHEDULES = {
    # (mle, use_gpu) -> (tolerance, max_iterations)
    #
    # On the CPU each estimator gets a schedule that converges it properly;
    # for least squares that is the historical MINPACK schedule, kept in
    # ``gaussfit`` as ``*_LSQ_CPU``. On the GPU both keep Gpufit's, which is
    # looser - deliberate rather than ideal, since it is what every "-gpu"
    # code has always meant and changing it would silently move existing
    # results. These are only *defaults*: pass ``eps``/``max_it``, or use the
    # Localize parameters dialog, to change them.
    (False, False): (
        gaussfit.TOLERANCE_LSQ_CPU,
        gaussfit.MAX_ITERATIONS_LSQ_CPU,
    ),
    (True, False): (1e-5, 100),
    (False, True): (gaussfit_cuda.TOLERANCE, gaussfit_cuda.MAX_ITERATIONS),
    (True, True): (gaussfit_cuda.TOLERANCE, gaussfit_cuda.MAX_ITERATIONS),
}


def gauss_schedule(
    mle: bool,
    use_gpu: bool,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> tuple:
    """``(tolerance, max_iterations)`` a Gaussian fit uses, explicit values
    winning. ``None`` picks the default of the method, which differs by
    estimator and device - see :data:`_GAUSS_SCHEDULES`."""
    default = _GAUSS_SCHEDULES[(bool(mle), bool(use_gpu))]
    if tolerance is None:
        tolerance = default[0]
    if max_iterations is None:
        max_iterations = default[1]
    return float(tolerance), int(max_iterations)


def _gauss_model(rotated: bool, spherical: bool) -> int:
    """The :mod:`picasso.fitting.gaussfit` model of a method's flags."""
    if spherical:
        return gaussfit.SPHERICAL
    if rotated:
        return gaussfit.ROTATED
    return gaussfit.ELLIPTIC


def fit_spots_gauss(
    spots: lib.FloatArray3D,
    rotated: bool = False,
    mle: bool = False,
    spherical: bool = False,
    use_gpu: bool = False,
    return_stats: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray3D | None = None,
) -> (
    lib.FloatArray2D
    | tuple[
        lib.FloatArray2D,
        lib.FloatArray1D | None,
        lib.FloatArray1D,
        lib.FloatArray1D | None,
    ]
):
    """Fit spots with a 2D Gaussian on the CPU or the GPU.

    The one entry point for every Gaussian method Picasso offers. Both devices
    run the identical Levenberg-Marquardt algorithm
    (:mod:`picasso.fitting.gaussfit` and ``gaussfit_cuda``), so ``use_gpu``
    only affects speed - the arrangement :func:`fit_spots_spline` already uses
    for the spline models.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, box, box)`` photon counts.
    rotated : bool, optional
        Fit a rotated elliptical Gaussian, whose seventh parameter is the
        rotation angle in radians. Cannot be combined with ``spherical``.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
    spherical : bool, optional
        Fit a single shared width. The returned parameters still use the
        elliptical layout with ``sx == sy``, so the rest of the pipeline is
        unchanged.
    use_gpu : bool, optional
        Run on a CUDA GPU. Default False.
    return_stats : bool, optional
        Additionally return ``(log_likelihood, iterations, chi_square)``.
    tolerance, max_iterations : optional
        ``None`` uses the method's own schedule, see :func:`gauss_schedule`.
    progress_callback : callable, "console" or None, optional
        Reported per spot on the CPU; the GPU fit is one launch per chunk.
    variance : lib.FloatArray3D, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid out
        exactly like ``spots`` (from ``get_spots(..., return_variance=True)``).
        Applies Huang et al.'s noise model to the maximum-likelihood
        estimator; least squares is unaffected by construction. Default is
        None.

    Returns
    -------
    parameters : lib.FloatArray2D
        ``[photons, x, y, sx, sy, bg]``, plus the rotation angle (radians) if
        ``rotated``. Positions are box-local.
    log_likelihood, number_iterations, chi_square
        Only if ``return_stats``. ``log_likelihood`` is None for least
        squares, ``chi_square`` is None for maximum likelihood - each
        estimator reports the goodness of fit that means something for it.
    """
    if rotated and spherical:
        raise ValueError("'rotated' and 'spherical' are mutually exclusive.")
    if use_gpu and not CUDA_AVAILABLE:
        raise ImportError(
            "GPU fitting was requested but no CUDA-capable GPU is available."
        )
    model = _gauss_model(rotated, spherical)
    tolerance, max_iterations = gauss_schedule(
        mle, use_gpu, tolerance, max_iterations
    )
    if mle:
        spots = _clip_for_mle(spots, variance)
    size = spots.shape[1]
    initial_parameters = _initial_parameters_gauss(
        _seed_spots(spots, variance) if mle else spots,
        size,
        rotated=rotated,
        spherical=spherical,
    ).astype(np.float64)

    backend = gaussfit_cuda if use_gpu else gaussfit
    parameters, chi_squares, _states, number_iterations = backend.fit_spots(
        model,
        spots,
        initial_parameters,
        mle=mle,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        variance=variance,
    )
    parameters = parameters.astype(np.float32)
    chi_squares = chi_squares.astype(np.float32)

    if spherical:
        # The isotropic models return [amplitude, x, y, s, bg]. Expand to the
        # standard elliptical layout with sx == sy so the rest of the pipeline
        # (CRLB, column building) is unchanged.
        s = parameters[:, 3]
        expanded = np.empty((len(parameters), 6), dtype=parameters.dtype)
        expanded[:, 0] = parameters[:, 0]
        expanded[:, 1] = parameters[:, 1]
        expanded[:, 2] = parameters[:, 2]
        expanded[:, 3] = s
        expanded[:, 4] = s
        expanded[:, 5] = parameters[:, 4]
        expanded[:, 0] *= 2.0 * np.pi * s * s
        parameters = expanded
    else:
        # The models fit a peak height; convert to total photons.
        parameters[:, 0] *= 2.0 * np.pi * parameters[:, 3] * parameters[:, 4]

    if return_stats:
        # The MLE chi-square equals twice the negative Poisson
        # log-likelihood, so -0.5 * chi_square reproduces the CPU MLE
        # fit's log_likelihood (both Stirling-approximated). For least
        # squares the chi-square is the plain residual sum of squares -
        # not a likelihood, since least squares assumes no noise model -
        # and is reported as such, as this fit's goodness-of-fit metric.
        log_likelihood = -0.5 * chi_squares if mle else None
        chi_square = None if mle else chi_squares
        return parameters, log_likelihood, number_iterations, chi_square
    return parameters


def fit_spots_gauss_gpu(
    spots: lib.FloatArray3D,
    rotated: bool = False,
    mle: bool = False,
    spherical: bool = False,
    return_stats: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    variance: lib.FloatArray3D | None = None,
) -> (
    lib.FloatArray2D
    | tuple[
        lib.FloatArray2D,
        lib.FloatArray1D | None,
        lib.FloatArray1D,
        lib.FloatArray1D | None,
    ]
):
    """Fit spots with a 2D Gaussian on the GPU.

    Thin wrapper over :func:`fit_spots_gauss` with ``use_gpu=True``, kept
    because it is the established public name. See there for the arguments
    and the returned layout.
    """
    return fit_spots_gauss(
        spots,
        rotated=rotated,
        mle=mle,
        spherical=spherical,
        use_gpu=True,
        return_stats=return_stats,
        tolerance=tolerance,
        max_iterations=max_iterations,
        variance=variance,
    )


def locs_from_fits_gauss(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
    mle: bool = False,
    log_likelihood: lib.FloatArray1D | None = None,
    iterations: lib.FloatArray1D | None = None,
    spherical: bool = False,
    chi_square: lib.FloatArray1D | None = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame:
    """Convert the fit results from a Gaussian fit into a data frame of
    localizations.

    Backend-agnostic: ``picasso.fitting.gaussfit`` (CPU) and
    ``picasso.fitting.gaussfit_cuda`` (GPU) return the same ``theta``
    layout, so both are converted here.

    Parameters
    ----------
    identifications : pd.DataFrame
        Data frame containing the identifications of the spots,
        including frame numbers, x and y coordinates, and net gradient.
    theta : lib.FloatArray2D
        A 2D array with the optimized parameters for each spot, where
        each row corresponds to a spot and the columns are the
        parameters in the following order: [photons, x, y, sx, sy, bg]
        or, for the rotated elliptical Gaussian,
        [photons, x, y, sx, sy, bg, angle (radians)]. In the latter
        case, the resulting data frame contains the column ``angle``
        (in degrees).
    box : int
        The size of the box used for localization, which is used to
        calculate the offsets for the x and y coordinates.
    em : bool
        Whether EMCCD was used for the localization.
    mle : bool, optional
        Whether ``theta`` came from the maximum-likelihood
        estimator. If True, the localization precisions ``lpx`` / ``lpy``
        and the per-parameter uncertainties (``photons_unc``, ``bg_unc``,
        ``sx_unc``, ``sy_unc`` and, for the rotated model, ``angle_unc``)
        are the Poisson Cramer-Rao bound from the Fisher information of
        the fitted Gaussian model (:func:`precision._gauss_crlb`), matching the CPU
        MLE fit output. If False (least squares), ``lpx`` / ``lpy`` use
        the Mortensen et al. closed form and no per-parameter
        uncertainties are added. Default is False.
    log_likelihood : lib.FloatArray1D, optional
        The per-spot Poisson log-likelihood (from an MLE fit). If
        provided together with ``iterations``, the ``log_likelihood``
        and ``iterations`` columns are added, matching the CPU MLE fit
        output. Default is None.
    iterations : lib.FloatArray1D, optional
        The number of iterations taken to converge for each spot.
        Default is None.
    spherical : bool, optional
        If True, the fit was a spherical (isotropic) Gaussian, so
        ``sx == sy`` and the ellipticity is always 0. The
        ``ellipticity`` column is then omitted as it carries no
        information. Default is False.
    chi_square : lib.FloatArray1D, optional
        The per-spot residual sum of squares at the fit optimum (from a
        least-squares fit). If provided, the ``chi_square`` column is
        added. It is the least-squares counterpart of the MLE fits'
        ``log_likelihood``: a goodness-of-fit measure in photons squared,
        so it scales with the spot brightness and the box size and is
        only comparable between fits of the same box size. Default is
        None.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    """
    box_offset = int(box / 2)
    rotated = theta.shape[1] == 7
    x = theta[:, 1] + identifications["x"] - box_offset
    y = theta[:, 2] + identifications["y"] - box_offset
    if mle:
        # Poisson Cramer-Rao bound from the Fisher information of the
        # point-sampled Gaussian model the fitters optimize. Columns of ``crlb``
        # follow ``theta``: [photons, x, y, sx, sy, bg, (angle)].
        crlb = precision._gauss_crlb(
            theta, box, em, rotated=rotated, variance=variance
        )
        with np.errstate(invalid="ignore"):
            lpx = np.sqrt(crlb[:, 1])
            lpy = np.sqrt(crlb[:, 2])
    else:
        # The closed form has no per-pixel notion, so the readout noise enters
        # as its mean over the box; see precision.localization_precision.
        readout = _mean_readout_variance(variance)
        lpx = precision.localization_precision(
            theta[:, 0],
            theta[:, 3],
            theta[:, 4],
            theta[:, 5],
            em=em,
            readout_variance=readout,
        )
        lpy = precision.localization_precision(
            theta[:, 0],
            theta[:, 4],
            theta[:, 3],
            theta[:, 5],
            em=em,
            readout_variance=readout,
        )
    columns = {
        "frame": identifications["frame"].astype(np.uint32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "photons": theta[:, 0].astype(np.float32),
        "sx": theta[:, 3].astype(np.float32),
        "sy": theta[:, 4].astype(np.float32),
        "bg": theta[:, 5].astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
    }
    if not spherical:
        # For a spherical (isotropic) Gaussian sx == sy, so the
        # ellipticity is always 0 and carries no information.
        a = np.maximum(theta[:, 3], theta[:, 4])
        b = np.minimum(theta[:, 3], theta[:, 4])
        ellipticity = (a - b) / a
        columns["ellipticity"] = ellipticity.astype(np.float32)
    columns["net_gradient"] = identifications["net_gradient"].astype(
        np.float32
    )
    if rotated:  # rotated elliptical Gaussian
        # Normalize to [-90, 90) as the ellipse repeats every half turn.
        angle = -np.rad2deg(theta[:, 6])
        angle = np.mod(angle + 90.0, 180.0) - 90.0
        columns["angle"] = angle.astype(np.float32)
    if mle:
        with np.errstate(invalid="ignore"):
            columns["photons_unc"] = np.sqrt(crlb[:, 0]).astype(np.float32)
            columns["bg_unc"] = np.sqrt(crlb[:, 5]).astype(np.float32)
            columns["sx_unc"] = np.sqrt(crlb[:, 3]).astype(np.float32)
            columns["sy_unc"] = np.sqrt(crlb[:, 4]).astype(np.float32)
            if rotated:
                columns["angle_unc"] = np.rad2deg(np.sqrt(crlb[:, 6])).astype(
                    np.float32
                )
    if log_likelihood is not None:
        columns["log_likelihood"] = log_likelihood.astype(np.float32)
    if iterations is not None:
        columns["iterations"] = iterations.astype(np.int32)
    if chi_square is not None:
        columns["chi_square"] = np.asarray(chi_square).astype(np.float32)
    locs = pd.DataFrame(columns)
    if "n_id" in identifications.columns:
        # The cross-channel link index. Carried through and sorted on, as
        # the spline path does - a multichannel fit needs every channel's
        # localizations in the same order to pair them up.
        locs["n_id"] = np.asarray(identifications["n_id"]).astype(np.uint32)
        locs.sort_values(by="n_id", kind="quicksort", inplace=True)
    else:
        locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def locs_from_fits_gauss_gpu(*args, **kwargs) -> pd.DataFrame:
    """Convert the fit results from a Gaussian fit into a data frame of
    localizations.

    .. deprecated:: 0.11
        Renamed to :func:`locs_from_fits_gauss` and removed under this
        name in Picasso 1.0. The function never was GPU-specific: it
        converts CPU and GPU Gaussian fits alike.
    """
    lib.deprecation_warning(
        "picasso.localize.locs_from_fits_gauss_gpu is deprecated and will "
        "be removed in Picasso 1.0. It handles CPU and GPU fits alike and "
        "was renamed to picasso.localize.locs_from_fits_gauss."
    )
    return locs_from_fits_gauss(*args, **kwargs)


def _fit2d_gauss(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    rotated: bool = False,
    mle: bool = False,
    spherical: bool = False,
    use_gpu: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame:
    """Fit 2D Gaussians with least squares or, if ``mle``, maximum
    likelihood, on the CPU or the GPU. If ``rotated``, a rotated elliptical
    Gaussian is fitted and the resulting localizations contain the fitted
    rotation angle (in degrees) in the column ``angle``. If ``spherical``, an
    isotropic Gaussian with a single width is fitted and the resulting ``sx``
    and ``sy`` columns are identical. See ``fit`` for more details."""
    theta, log_likelihood, iterations, chi_square = fit_spots_gauss(
        spots,
        rotated=rotated,
        mle=mle,
        spherical=spherical,
        use_gpu=use_gpu,
        return_stats=True,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        variance=variance,
    )
    locs = locs_from_fits_gauss(
        identifications,
        theta,
        box,
        em,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        spherical=spherical,
        chi_square=chi_square,
        variance=variance,
    )
    return locs


# ----------------------------------------------------------------------
# Cubic-spline PSF fitting
#
# The spline models fit an experimentally measured PSF (a cubic-spline model
# built from a bead z-stack, see ``spline.calibrate_spline``). Unlike the
# Gaussian models they
# need a coefficient table, which lives inside the calibration dict (see
# ``io.load_spline_calibration``) and is handed to the kernels by
# ``precision._spline_coeff_reshaped``.
# ----------------------------------------------------------------------


def _as_link_xyz_calibration(calibration: dict) -> dict:
    """Shallow copy of a ``spline-3d-multichannel`` calibration
    re-tagged for the photon-decoupled (link-XYZ) fit. Validates the
    supported channel range."""

    if calibration.get("model") != "spline-3d-multichannel":
        raise ValueError(
            "Photon-decoupled (link-XYZ) fitting requires a "
            "'spline-3d-multichannel' calibration."
        )
    n_channels = precision._spline_n_channels(calibration)
    if not 2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS:
        raise ValueError(
            "Photon decoupling (link-XYZ) supports 2 to "
            f"{precision._LINK_XYZ_MAX_CHANNELS} channels; this calibration has "
            f"{n_channels}. The limit is the per-thread device memory the "
            "fit kernel needs, which grows as the square of the parameter "
            "count. Keep photons linked - the shared-amplitude model works "
            "for any number of channels."
        )
    cal = dict(calibration)
    cal["model"] = precision._LINK_XYZ_MODEL
    return cal


def crop_spline_calibration(calibration: dict, box: int) -> dict:
    """Adapt a spline calibration to a smaller lateral fit box.

    This derives a smaller-box calibration by cropping the coefficient
    interval grid to the **central** ``box x box`` lateral region -
    centered on the PSF, so a spot centered in its ROI still starts
    (and converges) at ``x_shift = y_shift = 0`` and the reconstructed
    x/y carry no global shift. The axial (z) grid is untouched.

    ``box`` must be a positive integer no larger than the calibration's box; a
    ``box`` equal to the calibration's box returns the calibration unchanged. A
    smaller box of the opposite parity is allowed - the crop is then off-center
    by at most half a pixel, a harmless constant shift of all localizations.
    """
    model = calibration["model"]
    n_data = list(calibration["n_data"])
    cal_box = int(n_data[0])
    box = int(box)
    if box == cal_box:
        return calibration
    if box <= 0 or box > cal_box:
        raise ValueError(
            f"Fit box ({box}) must be a positive integer no larger than the "
            f"spline calibration's box ({cal_box})."
        )
    off = (cal_box - box) // 2  # centered offset (floored for odd size diffs)
    ni = box - 1  # lateral intervals after cropping
    lat = slice(off, off + ni)
    coeff = np.ascontiguousarray(calibration["coefficients"], dtype=np.float32)

    if model == "spline-2d":
        _, nix, niy = coeff.shape
        phys = coeff.ravel(order="C").reshape(niy, nix, 4, 4)
        phys_c = np.ascontiguousarray(phys[lat, lat, :, :])
        new_coeff = phys_c.ravel(order="C").reshape(16, ni, ni)
        new_n_intervals = [ni, ni]
        new_n_data = [box, box]
    elif model == "spline-3d":
        _, nix, niy, niz = coeff.shape
        phys = coeff.ravel(order="C").reshape(niz, niy, nix, 4, 4, 4)
        phys_c = np.ascontiguousarray(phys[:, lat, lat, :, :, :])
        new_coeff = phys_c.ravel(order="C").reshape(64, ni, ni, niz)
        new_n_intervals = [ni, ni, int(niz)]
        new_n_data = [box, box, int(n_data[2])]
    elif model in ("spline-3d-multichannel", precision._LINK_XYZ_MODEL):
        _, nix, niy, niz, n_channels = coeff.shape
        new_coeff = np.empty((64, ni, ni, niz, n_channels), dtype=np.float32)
        for c in range(n_channels):
            sub = np.ascontiguousarray(coeff[..., c])
            phys = sub.ravel(order="C").reshape(niz, niy, nix, 4, 4, 4)
            phys_c = np.ascontiguousarray(phys[:, lat, lat, :, :, :])
            new_coeff[..., c] = phys_c.ravel(order="C").reshape(
                64, ni, ni, niz
            )
        new_n_intervals = [ni, ni, int(niz)]
        new_n_data = [box, box, int(n_data[2])]
    else:
        raise ValueError(f"Unknown spline model '{model}'.")

    cropped = dict(calibration)
    cropped["coefficients"] = np.ascontiguousarray(new_coeff, dtype=np.float32)
    cropped["n_intervals"] = new_n_intervals
    cropped["n_data"] = new_n_data
    cropped["box"] = box
    return cropped


def _initial_parameters_spline(
    spots: lib.FloatArray3D, calibration: dict
) -> lib.FloatArray2D:
    """Initialize spline fit parameters per spot.

    Parameter order matches the spline models:
    ``[amplitude, x_shift, y_shift, offset]`` (2D) or
    ``[amplitude, x_shift, y_shift, z_shift, offset]`` (3D and 3D
    multichannel). The spline model evaluates the spline at
    ``position = pixel_index - parameter`` (see ``spline_3d.cuh``), so:

    - x_shift/y_shift are the emitter's lateral offset from the (centered)
      template, i.e. 0 for a spot centered in its ROI.

    For the multichannel model ``spots`` is channel-stacked
    ``(n, box, box, n_channels)``; amplitude/offset are estimated across all
    channels."""
    model = calibration["model"]
    if model == precision._LINK_XYZ_MODEL:
        # Photon-decoupled (link-XYZ) model: parameters
        # [x_shift, y_shift, z_shift, N_0..N_{c-1}, bg_0..bg_{c-1}], with the
        # photon amplitude and background estimated PER CHANNEL (that is the
        # whole point). spots is (n, box, box, n_channels).
        n_channels = precision._spline_n_channels(calibration)
        per_ch = np.asarray(spots)  # (n, box, box, n_channels)
        ch_max = np.amax(per_ch, axis=(1, 2))  # (n, n_channels)
        ch_min = np.amin(per_ch, axis=(1, 2))  # (n, n_channels)
        z_init = float(
            calibration.get("z_init", calibration.get("z_center", 0.0))
        )
        initial = np.zeros((len(spots), 3 + 2 * n_channels), dtype=np.float32)
        # x_shift (0), y_shift (1) start at 0 (spot centered); z_shift (2).
        initial[:, 2] = -z_init
        initial[:, 3 : 3 + n_channels] = ch_max - ch_min  # per-channel photons
        initial[:, 3 + n_channels :] = ch_min  # per-channel background
        return initial
    if model == "spline-2d":
        n_parameters = 4
    else:
        n_parameters = 5
    # spots is (n, box, box) or, for multichannel, (n, box, box, n_channels)
    reduce_axes = tuple(range(1, spots.ndim))
    spot_max = np.amax(spots, axis=reduce_axes)
    spot_min = np.amin(spots, axis=reduce_axes)
    initial = np.zeros((len(spots), n_parameters), dtype=np.float32)
    initial[:, 0] = spot_max - spot_min  # amplitude
    # x_shift (col 1) and y_shift (col 2) start at 0 (spot centered in the ROI).
    if model == "spline-2d":
        initial[:, 3] = spot_min  # offset
    else:
        z_init = float(
            calibration.get("z_init", calibration.get("z_center", 0.0))
        )
        initial[:, 3] = -z_init  # z_shift (in-focus start; see docstring)
        initial[:, 4] = spot_min  # offset
    return initial


# ----------------------------------------------------------------------
# CPU cubic-spline fitting
#
# The numerical core lives in ``picasso.fitting.splinefit``, a numba port of
# Gpufit's
# Levenberg-Marquardt driver and its spline models). What follows is the
# translation layer: calibration dict in, plain arrays out, plus the
# device-agnostic entry points the multichannel fitters call so that a single
# ``use_gpu`` flag selects the backend.
# ----------------------------------------------------------------------


def _spline_kind(model: str) -> int:
    """``picasso.fitting.splinefit`` model kind for a spline calibration
    ``model``."""
    if model == "spline-2d":
        return splinefit.KIND_2D
    if model in ("spline-3d", "spline-3d-multichannel"):
        return splinefit.KIND_3D
    if model == precision._LINK_XYZ_MODEL:
        return splinefit.KIND_LINK_XYZ
    raise ValueError(
        f"Unknown spline calibration model '{model}'. Expected one of "
        "'spline-2d', 'spline-3d', 'spline-3d-multichannel', "
        f"'{precision._LINK_XYZ_MODEL}'."
    )


def _spline_z_seeds(calibration: dict, n_z_starts: int) -> tuple:
    """Axial seeds for the multi-start, as ``(z_seeds, apply_seeds)``.

    The seeds span the calibration z-stack (``z_shift = -z_plane``, so
    ``[-(n_z - 1), 0]``); both devices get the same grid, so they explore
    the same axial minima."""
    n_seeds = max(1, int(n_z_starts))
    if n_seeds == 1 or calibration["model"] == "spline-2d":
        return np.zeros(1), False
    n_z = int(calibration["n_data"][2])
    return np.linspace(-(n_z - 1), 0.0, n_seeds), True


def _spline_schedule(
    apply_seeds: bool,
    tolerance: float | None,
    max_iterations: int | None,
) -> tuple:
    """Resolve a spline fit's convergence schedule. Same on either device.

    A multi-start has to rank its seeds on the chi-square and therefore needs a
    much tighter stop than a single start. ``None`` picks whichever applies;
    explicit values always win."""
    return splinefit.resolve_schedule(apply_seeds, tolerance, max_iterations)


def _run_splinefit(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int | None = None,
    residuals: np.ndarray | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    use_gpu: bool = False,
    variance: lib.FloatArray3D | None = None,
) -> tuple | None:
    """Low-level spline fit: unpack the calibration and run the kernels.

    Returns ``(theta, chi_squares, states, iterations)``, or None if
    ``abort_callback`` asked to stop. The axial multi-start runs inside the
    per-spot kernel on both devices, so every seed is tried while that spot's
    data is still in cache and progress is reported once per spot rather than
    once per pass.

    ``use_gpu`` selects :mod:`picasso.fitting.splinefit_cuda` over
    :mod:`picasso.fitting.splinefit`. The two backends are driven from *this*
    function
    rather than from separate translation layers deliberately: everything above
    the dispatch - the box crop, the channel-major reshape, the coefficient
    view, the affines, the ROI residuals, the initial parameters and the
    schedule - is computed once, so both devices are guaranteed to see
    byte-identical inputs. That is what makes a CPU/GPU comparison meaningful
    rather than a test of two translation layers agreeing.

    ``multiprocess`` keeps ``fit``'s argument name, but as for ``gaussmle``
    it selects a **thread** pool: the CPU kernels are ``nogil``, so the workers
    run concurrently while sharing the spots and the coefficient table rather
    than pickling a copy of each into a subprocess. False runs the fit serially
    in the calling thread, which is what the tests use for reproducibility. It
    is ignored on the GPU, where one launch fits every spot.
    """
    box = spots.shape[1]
    # Fit a smaller-than-calibration box against a centered crop, exactly as
    # locs_from_fits_spline crops identically itself, so its CRLB matches
    # the fit geometry.
    calibration = crop_spline_calibration(calibration, box)
    model = calibration["model"]
    kind = _spline_kind(model)
    n_channels = precision._spline_n_channels(calibration)
    if mle:
        # A Poisson likelihood is undefined for negative counts. Same clip as
        # on the GPU; see :func:`_clip_for_mle`.
        spots = _clip_for_mle(spots, variance)
    fit_data = precision._spline_channel_major(spots, n_channels)
    # The variance rides through the same reshape, so it stays aligned with
    # the spots pixel for pixel on both devices.
    fit_variance = (
        None
        if variance is None
        else precision._spline_channel_major(variance, n_channels)
    )
    initial = np.ascontiguousarray(
        _initial_parameters_spline(
            _seed_spots(spots, variance) if mle else spots, calibration
        ),
        dtype=np.float64,
    )
    coefficients = precision._spline_coeff_reshaped(calibration)
    affines = precision._spline_channel_affines(calibration, n_channels)
    roi_residuals = precision._spline_crlb_residuals(
        residuals, len(spots), n_channels
    )
    if n_z_starts is None:
        n_z_starts = _default_n_z_starts(calibration)
    z_seeds, apply_seeds = _spline_z_seeds(calibration, n_z_starts)
    tolerance, max_iterations = _spline_schedule(
        apply_seeds, tolerance, max_iterations
    )

    args = (
        kind,
        fit_data,
        coefficients,
        affines,
        roi_residuals,
        initial,
        z_seeds,
        apply_seeds,
    )
    kwargs = {
        "mle": mle,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "variance": fit_variance,
    }
    aborted = callable(abort_callback) and abort_callback()
    if aborted:
        return None
    if use_gpu:
        stopped_early = False

        def _abort() -> bool:
            nonlocal stopped_early
            if abort_callback():
                stopped_early = True
                return True
            return False

        result = splinefit_cuda.fit_spots(
            *args,
            progress_callback=progress_callback,
            abort_callback=_abort if callable(abort_callback) else None,
            **kwargs,
        )
        return None if stopped_early else result
    if not multiprocess or len(spots) == 0:
        return splinefit.fit_spots(
            *args, progress_callback=progress_callback, **kwargs
        )

    n_spots = len(spots)
    fit = splinefit.fit_spots_async(*args, **kwargs)
    use_tqdm = progress_callback == "console"
    iter_range = (
        tqdm(total=n_spots, desc="Fitting", unit="spot") if use_tqdm else None
    )
    last = 0
    while fit.current[0] < n_spots:
        if callable(abort_callback) and abort_callback():
            fit.stop()
            aborted = True
            break
        if use_tqdm:
            iter_range.update(fit.current[0] - last)
            last = fit.current[0]
        elif callable(progress_callback):
            progress_callback(fit.current[0])
        # The workers write into preallocated arrays and nothing ever collects
        # their futures, so a worker that died would leave the counter frozen
        # and this loop spinning forever. Surface the error instead.
        fit.raise_errors()
        if fit.finished() and fit.current[0] < n_spots:
            raise RuntimeError(
                "The spline fitting workers stopped after "
                f"{fit.current[0]} of {n_spots} spots."
            )
        time.sleep(0.2)
    while not fit.finished():
        # A spot is claimed before it is fitted, so the last few may still be
        # in flight once the counter reaches n_spots. Aborted runs wait too, so
        # no thread is left writing into the arrays after this returns.
        fit.raise_errors()
        time.sleep(0.05)
    fit.raise_errors()
    if use_tqdm:
        if not aborted:
            iter_range.update(n_spots - last)
        iter_range.close()
    if aborted:
        return None
    if callable(progress_callback):
        # Report completion explicitly: a fit short enough to finish before the
        # first poll would otherwise never call back at all.
        progress_callback(n_spots)
    return fit.results()


def fit_spots_splinefit(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int | None = None,
    return_stats: bool = False,
    residuals: np.ndarray | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    use_gpu: bool = False,
    variance: lib.FloatArray3D | None = None,
) -> np.ndarray | tuple | None:
    """Fit multiple spots with a cubic-spline PSF model using the numba kernels.

    Runs on the CPU by default and on the GPU with ``use_gpu``; the two are the
    same algorithm, so the choice only affects speed. Same arguments, parameter
    conventions and return shape as :func:`fit_spots_spline_gpu`, so all
    three are interchangeable (see :func:`fit_spots_spline`, which picks between
    them). Every spline model is supported: ``spline-2d``, ``spline-3d``,
    ``spline-3d-multichannel`` and the photon-decoupled
    ``spline-3d-multichannel-link-xyz``.

    Progress is reported per spot on the CPU and per chunk on the GPU; see
    ``tolerance``/``max_iterations`` for the convergence schedule and
    :func:`_run_splinefit` for the multi-start.

    Returns None if ``abort_callback`` asked to stop.
    """
    result = _run_splinefit(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        residuals=residuals,
        tolerance=tolerance,
        max_iterations=max_iterations,
        multiprocess=multiprocess,
        progress_callback=progress_callback,
        abort_callback=abort_callback,
        use_gpu=use_gpu,
        variance=variance,
    )
    if result is None:
        return None
    theta, chi_squares, _states, iterations = result
    theta = theta.astype(np.float32)
    if return_stats:
        # As in fit_spots_spline_gpu: the maximum-likelihood chi-square is
        # twice the negative Poisson log-likelihood, the least-squares one is
        # the residual sum of squares.
        log_likelihood = (
            (-0.5 * chi_squares).astype(np.float32) if mle else None
        )
        chi_square = None if mle else chi_squares.astype(np.float32)
        return theta, log_likelihood, iterations, chi_square
    return theta


def _fit_splinefit_multistart(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int = 1,
    residuals: np.ndarray | None = None,
    use_gpu: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    variance: lib.FloatArray3D | None = None,
) -> tuple:
    """Axial multi-start with the numba kernels, on either device.

    Returns ``(parameters, chi_squares, converged, n_iterations)``.
    :func:`fit_spline_multichannel_ratiometric` uses this form because it ranks
    photon-ratio hypotheses on the chi-square and needs to know which fits
    converged.

    There is no separate seed loop here: both kernels run the multi-start
    per spot internally and return the winning seed, so this only has to
    translate the fit state into the ``converged`` mask."""
    theta, chi_squares, states, iterations = _run_splinefit(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        residuals=residuals,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
        variance=variance,
    )
    finite = np.isfinite(theta).all(axis=1) & np.isfinite(chi_squares)
    converged = finite & (
        (states == splinefit.FIT_STATE_CONVERGED) if mle else True
    )
    return theta.astype(np.float32), chi_squares, converged, iterations


def _spline_use_gpu(use_gpu: bool | None) -> bool:
    """Resolve a ``use_gpu`` flag: None means "whatever is available".

    Raises if the GPU was explicitly asked for but is unusable, so an explicit
    request never silently becomes a (much slower) CPU fit."""
    if use_gpu is None:
        return CUDA_AVAILABLE
    if use_gpu and not CUDA_AVAILABLE:
        raise ImportError(
            "GPU spline fitting was requested but no CUDA-capable GPU is "
            "available. Pass use_gpu=False to fit the spline PSF on the CPU "
            "instead."
        )
    return bool(use_gpu)


def fit_spots_spline(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int | None = None,
    return_stats: bool = False,
    residuals: np.ndarray | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray3D | None = None,
) -> np.ndarray | tuple | None:
    """Fit spots with a cubic-spline PSF model on the available device.

    A thin wrapper over :func:`fit_spots_splinefit` that resolves the device:
    ``use_gpu`` None (the default) uses the GPU when one is available. Both
    devices run the same algorithm, so the choice only affects speed.

    ``progress_callback`` is reported per spot on the CPU and per chunk on the
    GPU."""
    return fit_spots_splinefit(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        return_stats=return_stats,
        residuals=residuals,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        use_gpu=_spline_use_gpu(use_gpu),
        variance=variance,
    )


def _fit_spline_multistart(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int = 1,
    residuals: np.ndarray | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    variance: lib.FloatArray3D | None = None,
) -> tuple:
    """Axial multi-start on whichever device is available.

    See :func:`_fit_splinefit_multistart`, which does the work on either
    device; this only resolves which one."""
    return _fit_splinefit_multistart(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        residuals=residuals,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_gpu=_spline_use_gpu(use_gpu),
    )


def _locs_from_fits_spline_link_xyz(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    calibration: dict,
    mle: bool = False,
    em: bool = False,
    log_likelihood: lib.FloatArray1D | None = None,
    iterations: lib.FloatArray1D | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    residuals: np.ndarray | None = None,
    chi_square: lib.FloatArray1D | None = None,
    variance: lib.FloatArray4D | None = None,
) -> pd.DataFrame:
    """Localizations from a photon-decoupled (link-XYZ) multichannel spline fit.

    ``theta`` columns are ``[x_shift, y_shift, z_shift, N_0..N_{c-1},
    bg_0..bg_{c-1}]``. Emits the shared ``x, y, z`` plus per-channel photon and
    background columns ``photons_ch{c}`` / ``bg_ch{c}``, their totals in
    ``photons`` / ``bg``, and ``rel_photons_ch{c}`` = that channel's share of
    the total photons (the continuous ratiometric readout that the free photon
    ratio provides; the shares sum to 1). ``calibration`` is assumed already
    cropped to ``box``."""
    n_channels = precision._spline_n_channels(calibration)
    variance = precision._crlb_variance_channel_major(variance, n_channels)
    oversampling = float(calibration.get("oversampling", 1.0))
    box_offset = int(box / 2)
    center = (box - 1) / 2.0

    theta = np.asarray(theta, dtype=np.float64)
    x_shift = theta[:, 0]
    y_shift = theta[:, 1]
    z_shift = theta[:, 2]
    amp = theta[:, 3 : 3 + n_channels]  # per-channel amplitude
    bg_ch = theta[
        :, 3 + n_channels : 3 + 2 * n_channels
    ]  # per-channel background

    ps = _photon_scales(calibration, n_channels)  # (n_channels,)
    photons_ch = amp * ps[None, :]  # per-channel photon counts
    photons = photons_ch.sum(axis=1)
    bg_total = bg_ch.sum(axis=1)

    ids_x = np.asarray(identifications["x"], dtype=np.float64)
    ids_y = np.asarray(identifications["y"], dtype=np.float64)
    x = x_shift / oversampling + center + ids_x - box_offset
    y = y_shift / oversampling + center + ids_y - box_offset

    crlb = precision._spline_link_xyz_crlb(
        theta,
        calibration,
        box,
        mle=mle,
        em=em,
        progress_callback=progress_callback,
        residuals=residuals,
        variance=variance,
    )  # variances [x, y, z, N_0.., bg_0..]
    var_amp = crlb[:, 3 : 3 + n_channels]
    var_bg = crlb[:, 3 + n_channels : 3 + 2 * n_channels]
    with np.errstate(invalid="ignore"):
        lpx = np.sqrt(crlb[:, 0]) / oversampling
        lpy = np.sqrt(crlb[:, 1]) / oversampling
        # total-photon uncertainty: independent per-channel photon variances add
        photons_unc = np.sqrt(np.sum(var_amp * (ps[None, :] ** 2), axis=1))
        bg_unc = np.sqrt(np.sum(var_bg, axis=1))

    z_center = float(calibration.get("z_center", 0.0))
    z_init = float(calibration.get("z_init", z_center))
    z_step_nm = float(calibration.get("z_step_nm", 1.0))
    magnification_factor = float(calibration.get("magnification_factor", 1.0))
    z = (z_shift + z_init) * z_step_nm * magnification_factor + (
        z_center - z_init
    ) * z_step_nm
    with np.errstate(invalid="ignore", divide="ignore"):
        lpz = np.sqrt(crlb[:, 2]) * z_step_nm * magnification_factor
        # Each channel's share of the total photons; sums to 1 per spot.
        rel_photons = np.where(
            photons[:, None] > 0, photons_ch / photons[:, None], np.nan
        )

    columns = {
        "frame": np.asarray(identifications["frame"]).astype(np.uint32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "z": z.astype(np.float32),
        "photons": photons.astype(np.float32),
        "bg": bg_total.astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
        "lpz": lpz.astype(np.float32),
        "net_gradient": np.asarray(
            identifications["net_gradient"], dtype=np.float32
        ),
        "photons_unc": photons_unc.astype(np.float32),
        "bg_unc": bg_unc.astype(np.float32),
    }
    for c in range(n_channels):
        columns[f"photons_ch{c}"] = photons_ch[:, c].astype(np.float32)
        columns[f"bg_ch{c}"] = bg_ch[:, c].astype(np.float32)
        columns[f"rel_photons_ch{c}"] = rel_photons[:, c].astype(np.float32)
    if log_likelihood is not None:
        columns["log_likelihood"] = np.asarray(log_likelihood).astype(
            np.float32
        )
    if iterations is not None:
        columns["iterations"] = np.asarray(iterations).astype(np.int32)
    if chi_square is not None:
        columns["chi_square"] = np.asarray(chi_square).astype(np.float32)
    locs = pd.DataFrame(columns)
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def locs_from_fits_spline(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
    calibration: dict,
    mle: bool = True,
    log_likelihood: lib.FloatArray1D | None = None,
    iterations: lib.FloatArray1D | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    residuals: np.ndarray | None = None,
    chi_square: lib.FloatArray1D | None = None,
    variance: lib.FloatArray4D | None = None,
) -> pd.DataFrame:
    """Convert spline fit results into a localizations data frame.

    ``theta`` columns are ``[amplitude, x_shift, y_shift, offset]`` (2D) or
    ``[amplitude, x_shift, y_shift, z_shift, offset]`` (3D). Localization
    precisions (``lpx``, ``lpy``, ``lpz``) and the ``photons`` / ``bg``
    uncertainties come from :func:`precision._spline_crlb`: the Poisson Cramer-Rao bound
    for maximum-likelihood fits (``mle`` True) or the least-squares sandwich
    covariance for ``spline-gpu`` least-squares fits (``mle`` False). ``mle``
    must match the estimator that produced ``theta``. ``em`` doubles those
    variances for EMCCD excess noise, as in the Gaussian fits.
    ``progress_callback`` is forwarded to :func:`precision._spline_crlb`.

    ``log_likelihood`` (MLE) and ``chi_square`` (the least-squares residual
    sum of squares at the optimum) are the per-estimator goodness-of-fit
    metrics; each becomes a column when given. See
    :func:`locs_from_fits_gauss` for how to read ``chi_square``."""
    calibration = crop_spline_calibration(calibration, box)
    model = calibration["model"]
    if model == precision._LINK_XYZ_MODEL:
        # Photon-decoupled model: 3 + 2*n_channels parameters
        # [x, y, z, N_0.., bg_0..] with per-channel photons/background and a
        # continuous per-channel relative-photon readout.
        return _locs_from_fits_spline_link_xyz(
            identifications,
            theta,
            box,
            calibration,
            mle=mle,
            em=em,
            log_likelihood=log_likelihood,
            iterations=iterations,
            progress_callback=progress_callback,
            residuals=residuals,
            chi_square=chi_square,
            variance=variance,
        )
    is_3d = model != "spline-2d"
    box_offset = int(box / 2)
    oversampling = float(calibration.get("oversampling", 1.0))

    amplitude = np.asarray(theta[:, 0])
    x_shift = np.asarray(theta[:, 1])
    y_shift = np.asarray(theta[:, 2])
    offset = np.asarray(theta[:, -1])
    center = (box - 1) / 2.0
    x = x_shift / oversampling + center + identifications["x"] - box_offset
    y = y_shift / oversampling + center + identifications["y"] - box_offset

    # photon_scale converts the fitted (shared) amplitude to a photon count.
    # A multichannel calibration may store a per-channel array; the shared
    # amplitude then maps to the TOTAL photons across channels (their sum). A
    # scalar (single-channel, or older multichannel calibrations) is unchanged.
    photon_scale_raw = calibration.get("photon_scale", 1.0)
    if np.ndim(photon_scale_raw) > 0:
        photon_scale = float(np.sum(np.asarray(photon_scale_raw, dtype=float)))
    else:
        photon_scale = float(photon_scale_raw)
    photons = amplitude * photon_scale

    # CRLB / LSQ variances
    crlb = precision._spline_crlb(
        theta,
        calibration,
        box,
        mle=mle,
        em=em,
        progress_callback=progress_callback,
        residuals=residuals,
        variance=variance,
    )
    amp_var, off_var = crlb[:, -2], crlb[:, -1]
    with np.errstate(invalid="ignore"):
        lpx = np.sqrt(crlb[:, 0]) / oversampling
        lpy = np.sqrt(crlb[:, 1]) / oversampling
        photons_unc = np.sqrt(amp_var) * photon_scale
        bg_unc = np.sqrt(off_var)

    columns = {
        "frame": identifications["frame"].astype(np.uint32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "photons": photons.astype(np.float32),
        "bg": offset.astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
        "net_gradient": identifications["net_gradient"].astype(np.float32),
    }
    if is_3d:
        z_shift = np.asarray(theta[:, 3])
        z_center = float(calibration.get("z_center", 0.0))
        z_init = float(calibration.get("z_init", z_center))
        z_step_nm = float(calibration.get("z_step_nm", 1.0))
        magnification_factor = float(
            calibration.get("magnification_factor", 1.0)
        )
        z_position = (z_shift + z_init) * z_step_nm * magnification_factor
        z_offset_nm = (z_center - z_init) * z_step_nm  # raw stage nm, no mag
        z = z_position + z_offset_nm
        columns["z"] = z.astype(np.float32)
        with np.errstate(invalid="ignore"):
            # var(z_shift) -> nm via the same z-step scaling used for z
            lpz = np.sqrt(crlb[:, 2]) * z_step_nm * magnification_factor
        columns["lpz"] = lpz.astype(np.float32)
    columns["photons_unc"] = photons_unc.astype(np.float32)
    columns["bg_unc"] = bg_unc.astype(np.float32)
    if log_likelihood is not None:
        columns["log_likelihood"] = log_likelihood.astype(np.float32)
    if iterations is not None:
        columns["iterations"] = iterations.astype(np.int32)
    if chi_square is not None:
        columns["chi_square"] = np.asarray(chi_square).astype(np.float32)
    locs = pd.DataFrame(columns)
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    if precision._spline_n_channels(calibration) > 1:
        return locs
    return lib.apply_affine_transforms(locs, calibration)


def _fit2d_spline_gpu(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    calibration: dict,
    mle: bool = False,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    n_z_starts: int | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame:
    """Fit an experimentally measured cubic-spline PSF on the GPU. For a 3D
    calibration the localizations contain the fitted ``z`` directly. See
    ``fit`` for more details. ``progress_callback`` tracks the per-spot CRLB
    computation in ``locs_from_fits_spline``.

    ``n_z_starts`` is the axial multi-start (see
    :func:`fit_spots_splinefit`); ``None`` picks it from the calibration.
    Pass 1 for the single in-focus start."""
    theta, log_likelihood, iterations, chi_square = fit_spots_splinefit(
        spots,
        calibration,
        mle=mle,
        return_stats=True,
        n_z_starts=n_z_starts,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_gpu=True,
        variance=variance,
    )
    locs = locs_from_fits_spline(
        identifications,
        theta,
        box,
        em,
        calibration,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        progress_callback=progress_callback,
        chi_square=chi_square,
        variance=variance,
    )
    return locs


def _fit2d_spline_cpu(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame | None:
    """Fit an experimentally measured cubic-spline PSF on the CPU. For a 3D
    calibration the localizations contain the fitted ``z`` directly. See
    ``fit`` for more details.

    Unlike the GPU path, whose fit is one launch per chunk,
    ``progress_callback`` here tracks the fit itself, one step per spot. The
    per-spot CRLB pass in ``locs_from_fits_spline`` is a second sweep over the
    same localizations, so it only gets the callback in ``"console"`` mode -
    where it draws its own labelled progress bar - rather than rewinding a
    GUI's counter back to zero.

    Returns None if ``abort_callback`` asked to stop."""
    result = fit_spots_splinefit(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        return_stats=True,
        tolerance=tolerance,
        max_iterations=max_iterations,
        multiprocess=multiprocess,
        progress_callback=progress_callback,
        abort_callback=abort_callback,
        variance=variance,
    )
    if result is None:
        return None
    theta, log_likelihood, iterations, chi_square = result
    return locs_from_fits_spline(
        identifications,
        theta,
        box,
        em,
        calibration,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        progress_callback=(
            "console" if progress_callback == "console" else None
        ),
        chi_square=chi_square,
        variance=variance,
    )


# ----------------------------------------------------------------------
# Multichannel cubic-spline PSF fitting (shared-amplitude 3D model)
#
# Several spatially-registered channels (separate movies, e.g. biplane
# microscopy) are fit simultaneously with shared x, y, z. A detection in
# the reference channel is mapped into every channel via a per-channel affine
# transform (stored in the calibration), the box ROIs are cut from each
# channel and stacked, and the stack is fitted against the per-channel spline
# coefficients. There is no affine-transform machinery elsewhere in Picasso,
# so the small least-squares helpers below are self-contained.
#
# This is the global-fitting scheme of globLoc: Li, Y., Shi, W., Liu, S.,
# Cavka, I., Wu, Y.-L., Matti, U., Wu, D., Koehler, S. & Ries, J. "Global
# fitting for high-accuracy multi-channel single-molecule localization."
# Nature Communications 13, 3133 (2022). DOI: 10.1038/s41467-022-30719-4.
# Linked parameters (shared x, y, z), the optional per-channel photon /
# background decoupling (``_as_link_xyz_calibration``) and the ratiometric
# color assignment (``fit_spline_multichannel_ratiometric``) all follow that
# work; the calibration side lives in ``picasso.spline``.
# ----------------------------------------------------------------------


def estimate_affine_transform(
    src_xy: lib.FloatArray2D, dst_xy: lib.FloatArray2D
) -> lib.FloatArray2D:
    """Least-squares 2D affine transform mapping ``src_xy`` to ``dst_xy``.

    Both inputs are ``(n, 2)`` arrays of matching point correspondences (e.g.
    the same beads seen in two channels). Returns a ``(2, 3)`` matrix ``M``
    such that ``dst ≈ src @ M[:, :2].T + M[:, 2]`` (see
    ``apply_affine_transform``). At least 3 non-collinear correspondences are
    required."""
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    if src.shape[0] < 3:
        raise ValueError(
            "At least 3 point correspondences are required to estimate an "
            "affine transform."
        )
    a = np.hstack([src, np.ones((len(src), 1))])  # (n, 3): [x, y, 1]
    solution, *_ = np.linalg.lstsq(a, dst, rcond=None)  # (3, 2)
    return solution.T.astype(np.float64)  # (2, 3)


def apply_affine_transform(
    xy: lib.FloatArray2D, transform: lib.FloatArray2D
) -> lib.FloatArray2D:
    """Apply a ``(2, 3)`` affine ``transform`` to ``(n, 2)`` points."""
    xy = np.asarray(xy, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    return xy @ transform[:, :2].T + transform[:, 2]


def _region_origin_xy(
    rect: tuple[tuple[int, int], tuple[int, int]] | list,
) -> np.ndarray:
    """``(x, y)`` top-left origin of a ``[[y_a, x_a], [y_b, x_b]]`` rectangle."""
    (ya, xa), (yb, xb) = rect
    return np.array([float(min(xa, xb)), float(min(ya, yb))], dtype=np.float64)


def decompose_region_affines(
    region_rects: list, transforms: list
) -> list[np.ndarray]:
    """Region-local channel affines from absolute channel transforms (split-FOV).

    Each absolute ``transform`` maps reference-channel **absolute** chip
    coordinates to channel-``c`` **absolute** coordinates. This strips out the
    region placement and returns the affine ``A_c`` that maps
    reference-**region-local** coordinates (relative to the reference region's
    top-left) to channel-``c``-region-local coordinates - the *inter-channel*
    registration, independent of where the regions sit on the chip (identity for
    a perfectly aligned, same-orientation split; ``A_0`` is the identity).

    This is the ROI-agnostic form stored in the calibration: the coarse region
    offset lives in the ROI positions (chosen at fit time), while ``A_c`` carries
    only the fine sub-pixel/rotation/scale registration. Inverse of
    :func:`compose_region_transforms`.
    """
    o0 = _region_origin_xy(region_rects[0])
    affines = []
    for rect, transform in zip(region_rects, transforms):
        t = np.asarray(transform, dtype=np.float64)
        linear, offset = t[:, :2], t[:, 2]
        oc = _region_origin_xy(rect)
        local_t = offset - oc + linear @ o0
        affines.append(np.hstack([linear, local_t[:, None]]))
    return affines


def compose_region_transforms(
    region_rects: list, affines: list
) -> list[np.ndarray]:
    """Absolute channel transforms from region-local affines + region positions.

    Inverse of :func:`decompose_region_affines`: given the region rectangles in
    use (e.g. re-drawn at fit time) and the stored region-local ``affines``,
    rebuild the absolute reference->channel transforms placed at those regions.
    Only the region *origins* enter, so fit-time regions may differ in size from
    the calibration ones - the placement follows their top-left corners.
    """
    o0 = _region_origin_xy(region_rects[0])
    transforms = []
    for rect, affine in zip(region_rects, affines):
        a = np.asarray(affine, dtype=np.float64)
        linear, local_t = a[:, :2], a[:, 2]
        oc = _region_origin_xy(rect)
        offset = oc - linear @ o0 + local_t
        transforms.append(np.hstack([linear, offset[:, None]]))
    return transforms


def multichannel_inbounds_ids(
    identifications: pd.DataFrame,
    box: int,
    movies: list,
    transforms: list,
) -> pd.DataFrame:
    """Filter reference detections to those whose full ``box`` fits inside
    every channel's frame after mapping through the per-channel transforms.
    """
    r = box // 2
    ref_xy = np.column_stack(
        [
            np.asarray(identifications["x"], dtype=np.float64),
            np.asarray(identifications["y"], dtype=np.float64),
        ]
    )
    inside = np.ones(len(ref_xy), dtype=bool)
    for c, movie in enumerate(movies):
        xy = (
            ref_xy if c == 0 else apply_affine_transform(ref_xy, transforms[c])
        )
        x = np.rint(xy[:, 0]).astype(np.int64)
        y = np.rint(xy[:, 1]).astype(np.int64)
        height, width = int(movie.shape[1]), int(movie.shape[2])
        inside &= (
            (x - r >= 0) & (x + r < width) & (y - r >= 0) & (y + r < height)
        )
    if inside.all():
        return identifications
    n_total = len(inside)
    n_dropped = int((~inside).sum())
    # Dropping a few edge detections is normal; dropping a large fraction almost
    # always means the inter-channel registration is wrong (detections map off
    # the frame in another channel), so make that visible instead of silently
    # returning a tiny, edge-clustered subset.
    if n_total and n_dropped / n_total >= 0.2:
        warnings.warn(
            f"Multichannel spot extraction dropped {n_dropped} of {n_total} "
            f"detections ({100 * n_dropped / n_total:.0f}%) whose box falls "
            "outside a channel after mapping - the split-FOV registration is "
            "likely off (re-register the channels or check the drawn ROIs).",
            stacklevel=2,
        )
    return identifications.iloc[np.flatnonzero(inside)].reset_index(drop=True)


def _matched_mask_within_tol(
    pred_xy: np.ndarray, target_xy: np.ndarray, tol: float
) -> np.ndarray:
    """Nearest-neighbor pairing within ``tol``, one-to-one.

    ``pred_xy`` are reference detections predicted into a channel (via the
    calibration transform), ``target_xy`` are that channel's own detections.
    Each reference proposes its nearest target; conflicts are resolved in order
    of increasing distance so every reference and every target is used at most
    once. Returns a boolean mask over ``pred_xy`` marking the paired rows - the
    same pairing the viewer's link colors use (``_nearest_unique_match`` in
    ``picasso.gui.localize``), reduced to "was this reference spot matched".
    """
    n_pred = len(pred_xy)
    matched = np.zeros(n_pred, dtype=bool)
    if n_pred == 0 or len(target_xy) == 0:
        return matched
    dists = np.sqrt(
        ((pred_xy[:, None, :] - target_xy[None, :, :]) ** 2).sum(axis=2)
    )  # (n_pred, n_target)
    nearest = np.argmin(dists, axis=1)
    best = dists[np.arange(n_pred), nearest]
    candidates = np.flatnonzero(best <= tol)
    if len(candidates) == 0:
        return matched
    # closest pair wins; a target already claimed cannot be reused
    used_target: set[int] = set()
    for k in candidates[np.argsort(best[candidates], kind="stable")]:
        j = int(nearest[k])
        if j in used_target:
            continue
        used_target.add(j)
        matched[k] = True
    return matched


def link_identifications_multichannel(
    identifications_per_channel: list,
    transforms: list,
    tol: float,
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Boolean mask over the reference channel's detections marking those that
    are also detected in *every* other channel.

    A molecule that the joint multichannel model can describe must be present in
    all channels: the fit ties one shared ``x, y, z`` to the *relative*
    intensities across channels. A reference detection with no counterpart in
    some channel is fitted there against background only, which biases the
    shared parameters. Filtering on this mask keeps only the
    cross-channel-linked molecules.

    Parameters
    ----------
    identifications_per_channel : list of pd.DataFrame
        One identification table per channel, ``[0]`` being the reference. Each
        needs ``frame``, ``x``, ``y``. ``None`` or empty entries mean that
        channel was not identified; it is then skipped (not counted as a
        missing match), so a partially identified set degrades to linking
        against the channels that are available.
    transforms : list
        One ``(2, 3)`` affine per channel mapping reference coordinates into
        that channel (as stored in the calibration's ``channel_transforms``).
    tol : float
        Pairing radius in camera pixels (the GUI preview uses ``1.5 * box``).
    progress_callback : callable, optional
        Called with the cumulative number of reference detections processed.

    Returns
    -------
    linked : np.ndarray
        Boolean mask over ``identifications_per_channel[0]`` rows. All ``False``
        if no other channel carries identifications.
    """
    reference = identifications_per_channel[0]
    n_ref = 0 if reference is None else len(reference)
    if n_ref == 0:
        return np.zeros(0, dtype=bool)
    ref_frames = np.asarray(reference["frame"], dtype=np.int64)
    ref_xy = np.column_stack(
        [
            np.asarray(reference["x"], dtype=np.float64),
            np.asarray(reference["y"], dtype=np.float64),
        ]
    )
    # frame-sorted view of the reference rows, so each frame is a slice
    ref_order = np.argsort(ref_frames, kind="stable")
    ref_frames_sorted = ref_frames[ref_order]
    frames, frame_starts = np.unique(ref_frames_sorted, return_index=True)
    frame_stops = np.append(frame_starts[1:], n_ref)

    # channels that can be linked against (an unidentified channel is skipped)
    check = [
        c
        for c in range(1, len(identifications_per_channel))
        if identifications_per_channel[c] is not None
        and len(identifications_per_channel[c])
    ]
    n_checked = len(check)
    if n_checked == 0:
        return np.zeros(n_ref, dtype=bool)

    match_count = np.zeros(n_ref, dtype=np.int64)
    for i_c, c in enumerate(check):
        ids_c = identifications_per_channel[c]
        pred = apply_affine_transform(
            ref_xy, np.asarray(transforms[c], dtype=np.float64)
        )[ref_order]
        c_frames = np.asarray(ids_c["frame"], dtype=np.int64)
        c_xy = np.column_stack(
            [
                np.asarray(ids_c["x"], dtype=np.float64),
                np.asarray(ids_c["y"], dtype=np.float64),
            ]
        )
        c_order = np.argsort(c_frames, kind="stable")
        c_frames_sorted = c_frames[c_order]
        c_xy_sorted = c_xy[c_order]
        # this channel's rows for each reference frame, by binary search
        c_lo = np.searchsorted(c_frames_sorted, frames, side="left")
        c_hi = np.searchsorted(c_frames_sorted, frames, side="right")
        done = 0
        for f in range(len(frames)):
            start, stop = frame_starts[f], frame_stops[f]
            lo, hi = c_lo[f], c_hi[f]
            if hi > lo:
                matched = _matched_mask_within_tol(
                    pred[start:stop], c_xy_sorted[lo:hi], tol
                )
                match_count[ref_order[start:stop][matched]] += 1
            done += stop - start
            # one monotone 0 -> n_ref progression across all checked channels
            if callable(progress_callback) and (f % 2000 == 0):
                progress_callback((i_c * n_ref + done) // n_checked)
    if callable(progress_callback):
        progress_callback(n_ref)
    return match_count == n_checked


def filter_linked_identifications(
    identifications_per_channel: list,
    transforms: list,
    box: int,
    tol: float | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Keep only the reference detections linked across *all* channels.

    Thin wrapper around :func:`link_identifications_multichannel` returning the
    filtered reference table plus ``(n_kept, n_total)``. If no other channel has
    identifications, the reference table is returned unchanged (with
    ``n_kept == n_total``) so an un-identified set degrades to the previous
    behaviour instead of fitting nothing.
    """
    reference = identifications_per_channel[0]
    n_total = 0 if reference is None else len(reference)
    others = [
        ids
        for ids in identifications_per_channel[1:]
        if ids is not None and len(ids)
    ]
    if n_total == 0 or not others:
        return reference, n_total, n_total
    if tol is None:
        tol = 1.5 * float(box)
    linked = link_identifications_multichannel(
        identifications_per_channel,
        transforms,
        tol,
        progress_callback=progress_callback,
    )
    n_kept = int(linked.sum())
    # Keeping almost nothing means the inter-channel registration (or the
    # pairing radius) is off rather than that the sample is empty - the same
    # failure mode ``multichannel_inbounds_ids`` warns about.
    if n_total and n_kept / n_total <= 0.05:
        warnings.warn(
            f"Cross-channel linking kept only {n_kept} of {n_total} "
            f"reference detections ({100 * n_kept / n_total:.1f}%) - the "
            "channel registration is likely off, or the other channels were "
            "identified with a much higher threshold.",
            stacklevel=2,
        )
    return (
        reference.iloc[np.flatnonzero(linked)].reset_index(drop=True),
        n_kept,
        n_total,
    )


def get_spots_multichannel(
    movies: list,
    identifications: pd.DataFrame,
    box: int,
    camera_infos: list[dict],
    transforms: list,
    progress_callback: Callable[[int], None] | None = None,
    return_residuals: bool = False,
    camera_calibrations: list[dict | None] | None = None,
    return_variance: bool = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Extract channel-stacked spots for multichannel spline fitting.

    For each identification (given in the reference channel's coordinates),
    the position is mapped into every channel via its affine ``transform`` and
    the box ROI is cut from that channel's movie. The per-channel ROIs are
    stacked along a new trailing axis.

    Parameters
    ----------
    movies : list
        One movie per channel (as loaded by ``io.load_movie``). ``movies[0]``
        is the reference channel.
    identifications : pd.DataFrame
        Detections in the reference channel (``frame``, ``x``, ``y``,
        ``net_gradient``).
    box : int
        Box side length (camera pixels).
    camera_infos : list of dict
        One camera-info dict per channel (for the photon conversion).
    transforms : list
        One ``(2, 3)`` affine transform per channel mapping reference-channel
        coordinates to that channel; ``transforms[0]`` is the identity.
    progress_callback : callable, optional
        Forwarded to ``get_spots`` for the reference channel.
    return_residuals : bool, optional
        Also return the sub-pixel ROI-placement residuals, i.e. the fractional
        part discarded when each channel's box is snapped to an integer pixel.
        The fit models need these to evaluate the spline where the data
        actually is; see :func:`channel_roi_residuals`. Default False.
    camera_calibrations : list of dict or None, optional
        One per-pixel sCMOS camera calibration per channel, or None for no
        calibration at all. Individual entries may be None when only some
        channels are on a characterized camera. Default None.
    return_variance : bool, optional
        Also return the per-spot readout variance in photoelectrons squared,
        channel-stacked exactly like ``spots``. None when no calibration was
        given. Default False.

    Returns
    -------
    spots : np.ndarray
        Array of shape ``(n_spots, box, box, n_channels)`` in photon units.
    residuals : np.ndarray
        Only if ``return_residuals``. ``(n_spots, n_channels, 2)`` in ``[x,
        y]`` order; channel 0 is exactly zero.
    variance : np.ndarray or None
        Only if ``return_variance``. Same shape as ``spots``.
    """
    n_channels = len(movies)
    if not (len(camera_infos) == len(transforms) == n_channels):
        raise ValueError(
            "movies, camera_infos and transforms must have the same length "
            "(one per channel)."
        )
    if camera_calibrations is None:
        camera_calibrations = [None] * n_channels
    elif len(camera_calibrations) != n_channels:
        raise ValueError(
            "camera_calibrations must have one entry per channel "
            f"({n_channels}), got {len(camera_calibrations)}."
        )
    ref_xy = np.column_stack(
        [
            np.asarray(identifications["x"], dtype=np.float64),
            np.asarray(identifications["y"], dtype=np.float64),
        ]
    )
    channel_spots = []
    channel_variance = []
    residuals = np.zeros((len(ref_xy), n_channels, 2), dtype=np.float32)
    for c in range(n_channels):
        if c == 0:
            ids_c = identifications
        else:
            mapped = apply_affine_transform(ref_xy, transforms[c])
            ids_c = identifications.copy()
            # get_spots/_cut_spots cut an INTEGER-pixel box, so the box origin
            # cannot express the fractional part of the mapped position. Keep
            # it: the fit model subtracts it from the evaluation position (see
            # channel_roi_residuals). Channel 0 is the reference and its box
            # sits on the (integer) detection itself, so its residual is 0.
            rounded = np.rint(mapped)
            residuals[:, c, :] = (mapped - rounded).astype(np.float32)
            ids_c["x"] = rounded[:, 0].astype(np.int64)
            ids_c["y"] = rounded[:, 1].astype(np.int64)
        # ``ids_c`` carries this channel's *rounded* box origins, so cutting
        # the calibration maps from it here - rather than from the reference
        # identifications - is what keeps a channel's variance patch aligned
        # with the spot it accompanies.
        spots_c, variance_c = get_spots(
            movies[c],
            ids_c,
            box,
            camera_infos[c],
            progress_callback=progress_callback if c == 0 else None,
            camera_calibration=camera_calibrations[c],
            return_variance=True,
        )
        channel_spots.append(spots_c)
        if variance_c is None:
            # A channel on an uncharacterized camera keeps the plain Poisson
            # model, which is what a zero readout variance means.
            variance_c = np.zeros_like(spots_c)
        channel_variance.append(variance_c)
    spots = np.stack(channel_spots, axis=-1)
    variance = (
        np.stack(channel_variance, axis=-1)
        if any(c is not None for c in camera_calibrations)
        else None
    )
    result = (spots,)
    if return_residuals:
        result += (residuals,)
    if return_variance:
        result += (variance,)
    return result[0] if len(result) == 1 else result


def channel_roi_residuals(
    identifications: pd.DataFrame, transforms: list
) -> np.ndarray:
    """Sub-pixel ROI-placement residual per localization and channel.

    :func:`get_spots_multichannel` cuts each channel's box at an integer pixel
    (``rint`` of the mapped position), so the fractional part of the mapping is
    not representable by the box origin. That leftover, ``mapped -
    rint(mapped)``, is what this returns; the multichannel spline models
    subtract it from the position at which they evaluate the spline (see the
    residual block in ``spline_3d_multichannel.cuh``).

    Detections sit on integer pixels, so for a channel transform
    ``A = I + E`` the residual is ``(E@x + b) - rint(E@x + b)``.

    :func:`get_spots_multichannel` computes the same quantity as a by-product
    of cutting the ROIs; prefer its ``return_residuals=True`` form when you are
    extracting spots anyway. This function is for callers that already have
    spots and only need the residuals.

    Parameters
    ----------
    identifications : pd.DataFrame
        Detections in the reference channel, with integer ``x``/``y`` columns -
        the same frame the ROIs are cut in.
    transforms : list
        One ``(2, 3)`` affine per channel mapping reference-channel coordinates
        to that channel, ``transforms[0]`` the identity (as stored in the
        calibration's ``channel_transforms``).

    Returns
    -------
    residuals : np.ndarray
        ``(n_spots, n_channels, 2)`` float32 in ``[x, y]`` order, matching the
        model's ``[fit][channel][x, y]`` residual block. Channel 0 is the
        reference (its box is cut on the detection itself) and is exactly zero.
    """
    ref_xy = np.column_stack(
        [
            np.asarray(identifications["x"], dtype=np.float64),
            np.asarray(identifications["y"], dtype=np.float64),
        ]
    )
    residuals = np.zeros((len(ref_xy), len(transforms), 2), dtype=np.float32)
    for c in range(1, len(transforms)):
        mapped = apply_affine_transform(ref_xy, transforms[c])
        residuals[:, c, :] = (mapped - np.rint(mapped)).astype(np.float32)
    return residuals


def fit_spline_multichannel(
    movies: list,
    camera_infos: list[dict],
    identifications: pd.DataFrame,
    box: int,
    calibration: dict,
    mle: bool = False,
    link_photons: bool = True,
    progress_callback: Callable[[int], None] | None = None,
    apply_roi_residuals: bool = True,
    n_z_starts: int | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    camera_calibrations: list[dict | None] | None = None,
) -> pd.DataFrame:
    """Fit a multichannel cubic-spline PSF across several registered channels.

    Global fit in the sense of globLoc (Li et al., Nat. Commun. 13, 3133,
    2022): every channel contributes to one fit with linked x, y and z.

    Ties ``get_spots_multichannel`` (extraction via the calibration's stored
    ``channel_transforms``) to ``fit_spots_spline`` and
    ``locs_from_fits_spline``. The resulting localizations are in the
    reference channel's coordinates and contain the fitted ``z`` directly.

    Parameters
    ----------
    movies : list
        One movie per channel; ``movies[0]`` is the reference channel and its
        order must match the calibration's channels.
    camera_infos : list of dict
        One camera-info dict per channel.
    identifications : pd.DataFrame
        Detections in the reference channel.
    box : int
        Box side length (camera pixels), must match the calibration.
    calibration : dict
        A ``"spline-3d-multichannel"`` calibration (see ``picasso.spline``).
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator. Default False.
    use_gpu : bool or None, optional
        Fit on the GPU (``picasso.fitting.splinefit_cuda``) or on the CPU
        (``picasso.fitting.splinefit``). None (the default) uses the GPU when
        one is
        available. Both compute the same quantity, so this only affects speed.
    link_photons : bool, optional
        If True (default), the shared-amplitude model links one photon
        amplitude and one background across all channels. If False, use
        the photon-decoupled model: x, y, z stay shared but each channel
        gets its own photon count and background, reported as
        ``photons_ch{c}`` / ``bg_ch{c}`` / ``rel_photons_ch{c}``.
        Available for 2 to 6 channels. See :func:`_as_link_xyz_calibration`.
    apply_roi_residuals : bool, optional
        Hand the sub-pixel ROI-placement residuals to the fit model (default
        True), so each channel's spline is evaluated where its data actually
        sits rather than at the nearest whole pixel. See
        :func:`channel_roi_residuals` for why this matters and by how much. Set
        False to reproduce results from before this correction, or to A/B the
        two on the same data.
    camera_calibrations : list of dict or None, optional
        One per-pixel sCMOS camera calibration per channel (from
        ``picasso.scmos`` or ``io.load_camera_calibration``), or None for none
        at all; individual entries may be None when only some channels sit on
        a characterized camera. Each channel's maps are cut at that channel's
        own mapped, rounded box origin, so a calibration follows its channel
        through the affine registration. Default None.
    """
    if calibration.get("model") != "spline-3d-multichannel":
        raise ValueError(
            "fit_spline_multichannel requires a 'spline-3d-multichannel' "
            "calibration."
        )
    if not link_photons:
        calibration = _as_link_xyz_calibration(calibration)
    transforms = calibration["channel_transforms"]
    if len(movies) != len(transforms):
        raise ValueError(
            f"Got {len(movies)} channels but the calibration has "
            f"{len(transforms)} channel transforms."
        )
    identifications = multichannel_inbounds_ids(
        identifications, box, movies, transforms
    )
    spots, residuals, variance = get_spots_multichannel(
        movies,
        identifications,
        box,
        camera_infos,
        transforms,
        progress_callback=progress_callback,
        return_residuals=True,
        camera_calibrations=camera_calibrations,
        return_variance=True,
    )
    theta, log_likelihood, iterations, chi_square = fit_spots_spline(
        spots,
        calibration,
        mle=mle,
        return_stats=True,
        residuals=residuals if apply_roi_residuals else None,
        n_z_starts=n_z_starts,
        use_gpu=use_gpu,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        variance=variance,
    )
    em = camera_infos[0].get("Gain", 1) > 1
    return locs_from_fits_spline(
        identifications,
        theta,
        box,
        em,
        calibration,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        progress_callback=progress_callback,
        residuals=residuals if apply_roi_residuals else None,
        chi_square=chi_square,
        variance=variance,
    )


def scale_channel_blocks(
    coefficients: np.ndarray, ratios: lib.FloatArray1D
) -> np.ndarray:
    """Scale each channel's spline coefficient block by a per-channel factor.

    ``coefficients`` is a multichannel table
    ``(64, n_int_x, n_int_y, n_int_z, n_channels)``. Because the cubic spline is
    linear in its coefficients, multiplying channel ``c``'s block by ``r[c]``
    scales that channel's model exactly: ``mu_c = offset + amplitude * r[c] *
    phi_c``. This is how a fixed per-channel photon **ratio** is imposed for
    ratiometric color assignment (and how an unequal biplane photon split is
    baked in) without changing the model itself. Only the *relative*
    ratios matter - the shared amplitude absorbs any overall scale.

    Returns a new ``float32`` array; the input is not modified.
    """
    coeff = np.array(coefficients, dtype=np.float32, copy=True)
    if coeff.ndim != 5:
        raise ValueError(
            "scale_channel_blocks expects a multichannel coefficient table "
            "(64, n_int_x, n_int_y, n_int_z, n_channels)."
        )
    ratios = np.asarray(ratios, dtype=np.float32)
    if coeff.shape[-1] != len(ratios):
        raise ValueError(
            f"Got {len(ratios)} ratios but the coefficient table has "
            f"{coeff.shape[-1]} channels."
        )
    for c in range(coeff.shape[-1]):
        coeff[..., c] *= ratios[c]
    return coeff


def _photon_scales(calibration: dict, n_channels: int) -> np.ndarray:
    """Per-channel ``photon_scale`` as a length-``n_channels`` array.

    Accepts either a per-channel array or a single scalar (broadcast to every
    channel, i.e. the equal-brightness assumption of older calibrations)."""
    ps = np.asarray(calibration.get("photon_scale", 1.0), dtype=float).ravel()
    if ps.size == 1:
        ps = np.repeat(ps, n_channels)
    if ps.size != n_channels:
        raise ValueError(
            f"photon_scale has {ps.size} entries but the calibration has "
            f"{n_channels} channels."
        )
    return ps


def fit_spline_multichannel_ratiometric(
    movies: list,
    camera_infos: list[dict],
    identifications: pd.DataFrame,
    box: int,
    calibration: dict,
    photon_ratios: lib.FloatArray2D | None = None,
    mle: bool = False,
    progress_callback: Callable[[int], None] | None = None,
    apply_roi_residuals: bool = True,
    n_z_starts: int | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    camera_calibrations: list[dict | None] | None = None,
) -> pd.DataFrame:
    """Ratiometric multichannel spline fit with photon-ratio color assignment.

    Implements globLoc's ratiometric scheme (Li et al., Nat. Commun. 13, 3133,
    2022) on top of the existing ``spline-3d-multichannel`` model: for each
    candidate per-channel photon **ratio** (one per dye/color), the per-channel
    coefficient blocks are scaled by that ratio (see
    :func:`scale_channel_blocks`) and the channel-stacked spots are fit. Each
    spot is then assigned the ratio whose fit best explains the data (lowest
    residual / highest likelihood); the winning ratio index is the color.

    ``photon_ratios`` is a ``(n_hypotheses, n_channels)`` array; if omitted it
    is taken from the calibration's ``"photon_ratios"`` field. Only the relative
    per-channel values matter (the shared amplitude absorbs the overall scale).

    Selection uses the **least-squares** residual by default (``mle=False``).
    The maximum-likelihood chi-square can still be unavailable for a spot whose
    fit diverged, so when ``mle=True`` the ranking is restricted to converged
    fits. (Before the Numba CUDA port this was a much bigger effect: Gpufit
    abandoned any fit whose model rang negative, which was a large fraction of
    them - see ``splinefit.MU_FLOOR``.)

    Returns localizations in the reference channel's coordinates with an added
    integer ``color`` column (the winning ratio index) and per-channel photon
    columns ``photons_ch{c}`` (``photons`` is their sum).

    ``apply_roi_residuals`` (default True) is as in
    :func:`fit_spline_multichannel`.
    camera_calibrations : list of dict or None, optional
        One per-pixel sCMOS camera calibration per channel (from
        ``picasso.scmos`` or ``io.load_camera_calibration``), or None for none
        at all; individual entries may be None when only some channels sit on
        a characterized camera. Each channel's maps are cut at that channel's
        own mapped, rounded box origin, so a calibration follows its channel
        through the affine registration. Default None.
    """
    if calibration.get("model") != "spline-3d-multichannel":
        raise ValueError(
            "fit_spline_multichannel_ratiometric requires a "
            "'spline-3d-multichannel' calibration."
        )
    if photon_ratios is None:
        photon_ratios = calibration.get("photon_ratios")
    if photon_ratios is None:
        raise ValueError(
            "No photon_ratios given and none stored in the calibration; "
            "provide a (n_hypotheses, n_channels) array of candidate ratios."
        )
    photon_ratios = np.atleast_2d(np.asarray(photon_ratios, dtype=np.float64))
    transforms = calibration["channel_transforms"]
    n_channels = len(transforms)
    if len(movies) != n_channels:
        raise ValueError(
            f"Got {len(movies)} channels but the calibration has "
            f"{n_channels} channel transforms."
        )
    if photon_ratios.shape[1] != n_channels:
        raise ValueError(
            f"photon_ratios has {photon_ratios.shape[1]} channels but the "
            f"calibration has {n_channels}."
        )

    identifications = multichannel_inbounds_ids(
        identifications, box, movies, transforms
    )
    spots, roi_residuals, variance = get_spots_multichannel(
        movies,
        identifications,
        box,
        camera_infos,
        transforms,
        progress_callback=progress_callback,
        return_residuals=True,
        camera_calibrations=camera_calibrations,
        return_variance=True,
    )
    if not apply_roi_residuals:
        roi_residuals = None
    n_spots = len(spots)
    n_hyp = len(photon_ratios)
    # Normalize each hypothesis so the shared amplitude keeps a total-photon
    # meaning; the ranking is unaffected by the overall scale.
    ratios_norm = photon_ratios / photon_ratios.sum(axis=1, keepdims=True)

    # Fit every hypothesis; keep per-spot parameters, fit state and score.
    # Every hypothesis gets the same axial multi-start, so the scores that
    # rank them are comparable - a hypothesis must not win by having landed in
    # a better axial minimum by luck.
    if n_z_starts is None:
        n_z_starts = _default_n_z_starts(calibration)
    thetas = []
    chis = []  # raw per-hypothesis chi-squares, kept for the saved column
    scores = np.full((n_hyp, n_spots), np.inf)
    valid = np.zeros((n_hyp, n_spots), dtype=bool)
    for k in range(n_hyp):
        calib_k = dict(calibration)
        calib_k["coefficients"] = scale_channel_blocks(
            calibration["coefficients"], ratios_norm[k]
        )
        params, chi_squares, converged, _n_it = _fit_spline_multistart(
            spots,
            calib_k,
            mle=mle,
            n_z_starts=n_z_starts,
            residuals=roi_residuals,
            tolerance=tolerance,
            max_iterations=max_iterations,
            use_gpu=use_gpu,
            variance=variance,
        )
        thetas.append(params)
        chis.append(np.asarray(chi_squares))
        finite = np.isfinite(params).all(axis=1) & np.isfinite(chi_squares)
        # LSE is robust here; for MLE keep only converged fits since its
        # chi-square is unreliable on the frequent negative-curvature exits.
        valid[k] = finite & (converged if mle else True)
        scores[k] = np.where(finite, chi_squares, np.inf)

    # Per spot: the best VALID hypothesis (lowest residual / chi2), falling back
    # to the best finite score if none was flagged valid.
    best_k = np.argmin(np.where(valid, scores, np.inf), axis=0)
    none_valid = ~valid.any(axis=0)
    if none_valid.any():
        best_k[none_valid] = np.argmin(scores[:, none_valid], axis=0)

    # Build localizations per winning-hypothesis group so z-conversion and CRLB
    # use that hypothesis's (scaled) calibration. Index-aligned column
    # assignment keeps per-channel photons correct across the internal
    # frame-sort of locs_from_fits_spline.
    em = camera_infos[0].get("Gain", 1) > 1
    parts = []
    for k in range(n_hyp):
        rows = np.where(best_k == k)[0]
        if len(rows) == 0:
            continue
        calib_k = dict(calibration)
        calib_k["coefficients"] = scale_channel_blocks(
            calibration["coefficients"], ratios_norm[k]
        )
        ids_k = identifications.iloc[rows]
        theta_k = np.asarray(thetas[k])[rows]
        locs_k = locs_from_fits_spline(
            ids_k,
            theta_k,
            box,
            em,
            calib_k,
            mle=mle,
            residuals=(None if roi_residuals is None else roi_residuals[rows]),
            # The winning hypothesis's own score. Only for least squares:
            # under MLE this chi-square is a likelihood, and the frequent
            # negative-curvature exits make it unreliable anyway (see above).
            chi_square=(None if mle else chis[k][rows]),
            variance=(None if variance is None else variance[rows]),
        )
        amp = pd.Series(np.asarray(theta_k[:, 0]), index=ids_k.index)
        ps = _photon_scales(calib_k, n_channels)
        total = None
        for c in range(n_channels):
            pc = amp * float(ratios_norm[k, c]) * float(ps[c])
            locs_k[f"photons_ch{c}"] = pc.astype(np.float32)
            total = pc if total is None else total + pc
        locs_k["photons"] = total.astype(np.float32)
        locs_k["color"] = np.int32(k)
        parts.append(locs_k)

    locs = pd.concat(parts) if parts else pd.DataFrame()
    if len(locs):
        locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def _split_fov_channel_affines(calibration: dict) -> list | None:
    """Region-local channel affines for a split-FOV calibration.

    Uses the stored ``channel_affines`` when present; otherwise (older
    calibrations) derives them from the stored absolute ``channel_transforms``
    and default ``regions`` so those calibrations can also be re-placed at
    fit time. Returns None if neither is available."""
    affines = calibration.get("channel_affines")
    if affines is not None:
        return [np.asarray(a, dtype=np.float64) for a in affines]
    regions = calibration.get("regions")
    transforms = calibration.get("channel_transforms")
    if regions and transforms:
        return decompose_region_affines(regions, transforms)
    return None


def split_fov_fit_geometry(
    calibration: dict, regions: list | None = None
) -> tuple[list, int, list]:
    """Where a split-FOV fit's channels sit, and how they map onto each other.

    The channels are placed at ``regions`` (e.g. the ROIs drawn for *this*
    data) when given, else at the calibration's own regions; the inter-channel
    affine is the same either way (see :func:`compose_region_transforms`).

    Parameters
    ----------
    calibration : dict
        Split-FOV spline calibration (``spline.calibrate_spline_split_fov``).
    regions : list, optional
        One ``[[y_min, x_min], [y_max, x_max]]`` per channel, reference first.

    Returns
    -------
    fit_regions : list
        Normalized channel rectangles actually in use.
    reference : int
        Index of the reference channel in ``fit_regions``/``transforms``.
    transforms : list
        One ``(2, 3)`` affine per channel mapping reference-channel
        coordinates into that channel, placed at ``fit_regions``.
    """
    if not calibration.get("split_fov"):
        raise ValueError(
            "A split-FOV fit requires a split-FOV calibration (built with "
            "spline.calibrate_spline_split_fov / a 'regions' argument)."
        )
    calib_regions = calibration.get("regions")
    if not calib_regions:
        raise ValueError("Split-FOV calibration is missing 'regions'.")
    n_channels = len(calib_regions)

    if regions is not None:
        if len(regions) != n_channels:
            raise ValueError(
                f"Got {len(regions)} regions but the calibration has "
                f"{n_channels} channels; draw one ROI per channel "
                "(reference first)."
            )
        fit_regions = [_normalize_rect(r) for r in regions]
        reference = 0  # fit-time ROIs are drawn reference-first
        affines = _split_fov_channel_affines(calibration)
        if affines is None:
            raise ValueError(
                "Calibration has no channel_affines to re-place at the given "
                "regions."
            )
        transforms = compose_region_transforms(fit_regions, affines)
    else:
        fit_regions = [_normalize_rect(r) for r in calib_regions]
        reference = int(calibration.get("reference", 0))
        transforms = [
            np.asarray(t, dtype=np.float64)
            for t in calibration["channel_transforms"]
        ]
    if len(transforms) != n_channels:
        raise ValueError(
            f"Calibration has {n_channels} channels but {len(transforms)} "
            "channel transforms."
        )
    return fit_regions, reference, transforms


def confine_to_region(
    identifications: pd.DataFrame, region: list
) -> pd.DataFrame:
    """The detections inside one ``[[y_min, x_min], [y_max, x_max]]`` rect."""
    (y0, x0), (y1, x1) = _normalize_rect(region)
    x = np.asarray(identifications["x"], dtype=np.float64)
    y = np.asarray(identifications["y"], dtype=np.float64)
    inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return identifications.iloc[np.flatnonzero(inside)].reset_index(drop=True)


def filter_linked_identifications_split_fov(
    identifications: pd.DataFrame,
    calibration: dict,
    box: int,
    regions: list | None = None,
    tol: float | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Split-FOV counterpart of :func:`filter_linked_identifications`.

    A split-FOV movie is identified as a whole, so one table holds the
    detections of every channel. They are split by region and paired across
    regions exactly as separate channel movies are, keeping only the
    reference-region detections found in *all* other regions - the molecules
    the joint fit can describe.

    Parameters
    ----------
    identifications : pd.DataFrame
        Movie-wide detections (all regions), as identified in the GUI.
    calibration : dict
        Split-FOV spline calibration.
    box : int
        Box side length; sets the default pairing radius (``1.5 * box``).
    regions : list, optional
        Channel ROIs for this data (reference first); see
        :func:`split_fov_fit_geometry`.

    Returns
    -------
    linked : pd.DataFrame
        The reference-region detections that link across all regions.
    n_kept, n_total : int
        Linked and total *reference-region* detections.
    """
    fit_regions, reference, transforms = split_fov_fit_geometry(
        calibration, regions
    )
    # reference first, as ``filter_linked_identifications`` expects; the
    # transforms already map reference coordinates into each region
    order = [reference] + [
        c for c in range(len(fit_regions)) if c != reference
    ]
    ids_per_region = [
        confine_to_region(identifications, fit_regions[c]) for c in order
    ]
    return filter_linked_identifications(
        ids_per_region,
        [transforms[c] for c in order],
        box,
        tol=tol,
        progress_callback=progress_callback,
    )


def fit_spline_split_fov(
    movie,
    camera_info: dict,
    identifications: pd.DataFrame,
    box: int,
    calibration: dict,
    regions: list | None = None,
    photon_ratios: lib.FloatArray2D | None = None,
    mle: bool = False,
    link_photons: bool = True,
    confine_to_reference: bool = True,
    progress_callback: Callable[[int], None] | None = None,
    apply_roi_residuals: bool = True,
    n_z_starts: int | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    camera_calibration: dict | None = None,
) -> pd.DataFrame:
    """Fit a split-FOV multichannel spline PSF from a *single* movie whose
    rectangular sub-regions are the channels.

    Global fit as in globLoc (Li et al., Nat. Commun. 13, 3133, 2022), for the
    single-camera split-FOV geometry.

    The calibration (built by :func:`picasso.spline.calibrate_spline_split_fov`)
    stores the *inter-channel* registration as region-local ``channel_affines``
    (see :func:`decompose_region_affines`), independent of where the channels sit
    on the chip. This function repeats the one ``movie``/``camera_info`` once per
    channel and delegates to the standard multichannel fitters. The model is
    chosen exactly as in the GUI ``MultichannelSplineFitWorker``: ratiometric if
    photon ratios are present, otherwise the plain linked fit.

    Parameters
    ----------
    movie, camera_info
        The single loaded movie and its camera info dict.
    identifications : pd.DataFrame
        Detections; when ``confine_to_reference`` is True (default) they are
        filtered to the reference region so each molecule yields one spot that is
        mapped into the other regions via the transforms.
    regions : list, optional
        The channel ROIs *for this data* (one ``[[y_min, x_min], [y_max,
        x_max]]`` per channel, reference first), e.g. re-drawn in the GUI. When
        given, the absolute channel transforms are rebuilt at these positions via
        the stored region-local affines - so the same calibration can be applied
        to data whose split sits at a different position. When omitted, the
        calibration's own ``regions`` (the calibration-time positions) are used.
    photon_ratios : lib.FloatArray2D, optional
        Candidate per-channel ratios for the ratiometric path (else taken from
        the calibration).

    Remaining parameters are as in the underlying multichannel fitters. The
    localizations are in the reference region's coordinates.
    camera_calibration : dict or None, optional
        A per-pixel sCMOS camera calibration for the (single) camera. All
        split-FOV regions are read from one sensor and the maps are indexed by
        absolute frame coordinates, so the same full-frame calibration serves
        every region. Default None.
    """
    fit_regions, reference, transforms = split_fov_fit_geometry(
        calibration, regions
    )
    n_channels = len(fit_regions)

    ids = identifications
    if confine_to_reference:
        ids = confine_to_region(ids, fit_regions[reference])

    # downstream fitters read channel_transforms off the calibration; hand them
    # the transforms placed at the regions actually in use
    calibration = dict(calibration)
    calibration["channel_transforms"] = [
        np.asarray(t, dtype=np.float64) for t in transforms
    ]

    movies = [movie] * n_channels
    camera_infos = [camera_info] * n_channels
    # Split-FOV is one physical camera, so every region reads the same maps;
    # _cut_map indexes them with absolute frame coordinates, which is exactly
    # what makes one full-frame calibration serve all regions.
    camera_calibrations = (
        None
        if camera_calibration is None
        else [camera_calibration] * n_channels
    )

    if (
        not link_photons
        and 2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS
    ):
        return fit_spline_multichannel(
            movies,
            camera_infos,
            ids,
            box,
            calibration,
            mle=mle,
            link_photons=False,
            progress_callback=progress_callback,
            apply_roi_residuals=apply_roi_residuals,
            n_z_starts=n_z_starts,
            use_gpu=use_gpu,
            tolerance=tolerance,
            max_iterations=max_iterations,
            camera_calibrations=camera_calibrations,
        )
    if (
        photon_ratios is not None
        or calibration.get("photon_ratios") is not None
    ):
        return fit_spline_multichannel_ratiometric(
            movies,
            camera_infos,
            ids,
            box,
            calibration,
            photon_ratios=photon_ratios,
            mle=mle,
            progress_callback=progress_callback,
            apply_roi_residuals=apply_roi_residuals,
            n_z_starts=n_z_starts,
            use_gpu=use_gpu,
            tolerance=tolerance,
            max_iterations=max_iterations,
            camera_calibrations=camera_calibrations,
        )
    return fit_spline_multichannel(
        movies,
        camera_infos,
        ids,
        box,
        calibration,
        mle=mle,
        link_photons=link_photons,
        progress_callback=progress_callback,
        apply_roi_residuals=apply_roi_residuals,
        n_z_starts=n_z_starts,
        use_gpu=use_gpu,
        tolerance=tolerance,
        max_iterations=max_iterations,
        camera_calibrations=camera_calibrations,
    )


def _fit2d_avg(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame | None:
    """Take localizations at the average value of the spots, see
    ``fit_2D`` for more details."""
    N = len(identifications)
    if multiprocess:
        fs = avgroi.fit_spots_parallel(spots, asynch=True)
        theta = _process_fitting_futures(
            fs, N, progress_callback, abort_callback
        )
        if theta is None:
            return
    else:
        theta = avgroi.fit_spots(spots, progress_callback)
    locs = avgroi.locs_from_fits(
        identifications,
        theta,
        box,
        em,
        readout_variance=_mean_readout_variance(variance),
    )
    return locs


def _process_fitting_futures(
    fs: list[Future],
    N: int,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> lib.FloatArray2D | None:
    """Convenience function for processing progress of fitting using
    multiprocessing. See ``_fit2d_gauss``, ``_fit2d_avg``."""
    n_tasks = len(fs)
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(total=N, desc="Fitting", unit="spot")

    while lib.n_futures_done(fs) < n_tasks:
        # check for abort
        if callable(abort_callback) and abort_callback():
            for f in fs:
                f.cancel()
            if use_tqdm:
                iter_range.close()
            return

        # update progress
        n_finished = round(N * lib.n_futures_done(fs) / n_tasks)
        if use_tqdm:
            iter_range.update(n_finished - iter_range.n)
        elif callable(progress_callback):
            progress_callback(n_finished)
        time.sleep(0.2)
    if use_tqdm:
        iter_range.update(N - iter_range.n)
        iter_range.close()
    theta = avgroi.fits_from_futures(fs)
    return theta


def localize(
    movie: LoadedMovie,
    # TODO: remove in v0.12.0 - only movie may be passed positionally, and
    # camera_info / identification_parameters become keyword-only
    *args,
    camera_info: dict | None = None,
    identification_parameters: dict | None = None,
    parameters: dict | None = None,  # TODO: remove in v0.12.0 (renamed)
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    movie_info: list[dict] | None = None,
    fitting_method: Literal[
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
        "avg",
    ] = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    mle_method: Literal["sigma", "sigmaxy"] | None = None,  # TODO: rm v0.12.0
    spline_calibration: dict | None = None,
    calibration_3d: dict | str | None = None,
    affine_calibration: dict | list | None = None,
    camera_calibration: dict | None = None,
    threaded: bool = True,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_z_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    return_info: bool = True,  # TODO: remove in v0.12.0
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    """Localize (i.e., identify and fit) spots in a movie using the
    specified parameters.

    Fits in 2D, unless an astigmatism calibration is given in
    ``calibration_3d``, in which case a z position is fitted on top of
    the 2D fit, see Huang, et al. Science, 2008. A 3D
    ``spline_calibration`` yields z directly from the fit itself and
    needs no ``calibration_3d``.

    Since v0.10.0: support for frame bounds and ROI for identification +
    all fitting methods.

    Since v0.11.0: astigmatic 3D fitting via ``calibration_3d``, which
    replaces the deprecated ``localize_3D``.

    Parameters
    ----------
    movie : LoadedMovie
        The input movie, as loaded by ``picasso.io.load_movie``.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    identification_parameters : dict
        A dictionary containing spot identification parameters,
        including:

        - `Min. Net Gradient`: Minimum net gradient for spot
          identification.
        - `Box Size`: Size of the box to cut out around each spot.
        - `Temporal Median Window`: optional, window length (in frames)
          of the temporal median filter applied before identification;
          0 or missing disables it. Fitting always uses the raw movie.
        - `Gaussian Filter Sigma`: optional, standard deviation (in
          camera pixels) of a spatial Gaussian filter applied to every
          frame before identification, see ``GaussianFilteredMovie``. It
          merges the several local maxima of a spot that is not
          Gaussian-shaped into one. Applied after the temporal median
          filter, if both are used. The filter applies to the
          identification only - the spots are always cut out of and
          fitted on the raw movie. Note that the minimum net gradient
          has to be re-tuned when this is changed, since smoothing
          lowers gradient magnitudes. 0 or missing disables it.
    parameters : dict, optional
        Deprecated alias for ``identification_parameters``, removed in
        v0.12.0.
    threaded : bool, optional
        Whether to use multithreading/multiprocessing. Default is True.
    movie_info : list[dict], optional
        Movie metadata. If None, an empty list is used. Default is None.
    roi : tuple, optional
        Region of interest (ROI) defined as a tuple of two tuples,
        where the first tuple contains the start coordinates
        (y_start, x_start) and the second tuple contains the end
        coordinates (y_end, x_end). If None, the entire frame is used.
        Default is None.
    frame_bounds : tuple, optional
        Minimum and maximum frame numbers to consider for the
        identification. If None, all frames are used. Default is None.
    fitting_method : {"gausslq", "gausslq-spherical", "gausslq-rotated", \
            "gausslq-gpu", "gausslq-rotated-gpu", "gausslq-spherical-gpu", \
            "gaussmle", "gaussmle-spherical", "gaussmle-gpu", \
            "gaussmle-rotated-gpu", "gaussmle-spherical-gpu" or "avg"}, \
            optional
        Which 2D fitting algorithm to use, see ``fit``. Default is
        "gausslq".
    eps : float or None, optional
        The convergence criterion, honoured by every iterating method on
        either device (all of them except "avg"). None (the default)
        picks the value that suits the method, see ``fit``.
    max_it : int or None, optional
        The maximum number of iterations per spot, as ``eps``. None (the
        default) picks the value that suits the method, see ``fit``.
    mle_method : Literal["sigma", "sigmaxy"] or None, optional
        Deprecated and ignored, removed in v0.12.0. Specify the
        fitting_method instead.
    calibration_3d : dict, str or None, optional
        Astigmatism calibration for fitting z on top of the 2D fit,
        either an already loaded calibration dictionary or a path to a
        YAML file holding one, with the keys:

        - "X Coefficients": list of 7 floats, polynomial coefficients
          for the x-axis calibration curve;
        - "Y Coefficients": list of 7 floats, polynomial coefficients
          for the y-axis calibration curve;
        - "Magnification factor": float, magnification factor of the
          microscope, i.e., the ratio between the actual z position of
          the calibration sample and the estimated z position from the
          localization data.

        Ignored for the "spline*" fitting methods, which fit z
        themselves from ``spline_calibration``. Not supported for "avg",
        which fits no Gaussian widths, nor for the "*-spherical" methods,
        which constrain sx == sy and so carry no astigmatism. The
        "*-rotated" methods report sx and sy along the rotated principal
        axes, whereas the astigmatism calibration assumes the camera
        axes, so use them for z fitting with care. Default is None (2D
        localization).
    affine_calibration : dict or list or None, optional
        Additional lateral (x, y) affine corrections to apply after
        fitting, e.g. a standalone chromatic-aberration calibration used
        on its own in a 2D experiment. Either a calibration dictionary
        carrying an ``"Affine transforms"`` list or the list itself; they
        are applied in order. Corrections stored in ``spline_calibration``
        or ``calibration_3d`` are applied by the fit itself, so they must
        not be repeated here. Default is None.
    camera_calibration : dict or None, optional
        A per-pixel sCMOS camera calibration for the (single) camera. All
        split-FOV regions are read from one sensor and the maps are indexed by
        absolute frame coordinates, so the same full-frame calibration serves
        every region. Default None.
    identification_progress_callback : callable or "console" or None
        A callback for progress updates during identification. If
        "console", progress will be printed to the console. If None,
        progress is not reported. Default is None.
    fit_progress_callback : callable or "console" or None
        A callback for progress updates during fitting. If "console",
        progress will be printed to the console. If None, progress is
        not reported. Default is None.
    fit_z_progress_callback : callable or "console" or None
        As ``fit_progress_callback``, for the astigmatic z fitting.
        Ignored unless ``calibration_3d`` is given. Default is None.
    return_info : bool, optional
        Whether to return additional information about the fitting
        process. Default is True. If True, a tuple of (locs, info) is
        returned. In v0.12.0 return_info will be removed and the
        function will always return info.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    info : list[dict], optional
        A list of dictionaries containing metadata about the movie and
        the fitting process. Only returned if `return_info` is True.
    """
    if not return_info:
        # TODO: remove in v0.12.0
        lib.deprecation_warning(
            "In version 0.12, return_info argument will be removed such "
            "that picasso.localize.localize() will always return both "
            "the localizations and the metadata dictionary."
        )
    camera_info, identification_parameters = _localize_legacy_arguments(
        args, camera_info, identification_parameters, parameters, mle_method
    )
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert isinstance(
        identification_parameters, dict
    ), "identification_parameters must be a dict"
    fit_z = _validate_calibration_3d(calibration_3d, fitting_method)

    # Use empty list as default for movie_info
    if movie_info is None:
        movie_info = []

    # Identify spots
    identifications, identify_info = identify(
        movie,
        identification_parameters["Min. Net Gradient"],
        identification_parameters["Box Size"],
        roi=roi,
        frame_bounds=frame_bounds,
        threaded=threaded,
        temporal_median_window=identification_parameters.get(
            "Temporal Median Window", 0
        ),
        gaussian_filter_sigma=identification_parameters.get(
            "Gaussian Filter Sigma", None
        ),
        progress_callback=identification_progress_callback,
    )

    # Fit spots
    locs, fit_info = fit(
        movie=movie,
        camera_info=camera_info,
        identifications=identifications,
        box=identification_parameters["Box Size"],
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        spline_calibration=spline_calibration,
        camera_calibration=camera_calibration,
        multiprocess=threaded,
        progress_callback=fit_progress_callback,
    )
    info = movie_info + [identify_info] + [fit_info]

    if fit_z:
        # Astigmatic z fitting on top of the 2D fit, see Huang, et al.
        # Science, 2008. zfit only knows gausslq/gaussmle; map the GPU /
        # rotated codes to the corresponding CPU noise model.
        locs, info = zfit.zfit(
            locs=locs,
            info=info,
            calibration=calibration_3d,
            fitting_method=(
                "gaussmle"
                if fitting_method.startswith("gaussmle")
                else "gausslq"
            ),
            filter=0,
            multiprocess=threaded,
            progress_callback=fit_z_progress_callback,
        )
        # The astigmatism calibration's own affine corrections were applied
        # by zfit, so they are not repeated here.
        return _localize_return(
            *_apply_extra_affine(
                locs, info, affine_calibration, applied=calibration_3d
            ),
            return_info=return_info,
        )

    # Standalone affine corrections (e.g. a chromatic one used without a 3D
    # calibration); those carried by the spline calibration were already
    # applied by the fit, so they are dropped here rather than applied twice.
    extra, duplicates = lib.drop_duplicate_affine_transforms(
        affine_calibration, spline_calibration
    )
    _warn_duplicate_affine(duplicates)
    if extra:
        locs = lib.apply_affine_transforms(locs, extra)
        fit_info["Affine corrections applied"] = (
            lib.describe_affine_transforms(extra)
        )
    return _localize_return(locs, info, return_info=return_info)


# TODO: remove in v0.12.0 (return_info is removed, info is always returned)
def _localize_return(
    locs: pd.DataFrame,
    info: list[dict],
    return_info: bool,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    """``localize``'s return value, honouring the deprecated
    ``return_info``."""
    if return_info:
        return locs, info
    return locs


# TODO: remove in v0.12.0, together with the *args, ``parameters`` and
# ``mle_method`` arguments of ``localize`` that it resolves
def _localize_legacy_arguments(
    args: tuple,
    camera_info: dict | None,
    identification_parameters: dict | None,
    parameters: dict | None,
    mle_method: str | None,
) -> tuple[dict | None, dict | None]:
    """Map ``localize``'s pre-v0.11.0 calling conventions onto the current
    arguments, warning about each one, and return the resolved
    ``(camera_info, identification_parameters)``."""
    if args:
        lib.deprecation_warning(
            "In version 0.12, picasso.localize.localize() will only accept "
            "the movie as a positional argument; pass camera_info and "
            "identification_parameters as keyword arguments."
        )
        if len(args) > 2:
            raise TypeError(
                "localize() takes at most 3 positional arguments "
                f"(movie, camera_info, identification_parameters), "
                f"{len(args) + 1} given"
            )
        if camera_info is not None:
            raise TypeError("localize() got multiple values for camera_info")
        camera_info = args[0]
        if len(args) == 2:
            if identification_parameters is not None or parameters is not None:
                raise TypeError(
                    "localize() got multiple values for "
                    "identification_parameters"
                )
            identification_parameters = args[1]
    if parameters is not None:
        lib.deprecation_warning(
            "The parameters argument of picasso.localize.localize() was "
            "renamed to identification_parameters and will be removed in "
            "version 0.12."
        )
        if identification_parameters is None:
            identification_parameters = parameters
    if mle_method is not None:
        lib.deprecation_warning(
            "The mle_method argument of picasso.localize.localize() is "
            "ignored and will be removed in version 0.12."
        )
    return camera_info, identification_parameters


def _validate_calibration_3d(
    calibration_3d: dict | str | None,
    fitting_method: str,
) -> bool:
    """Whether an astigmatic z fit is to be run after the 2D fit, i.e.
    ``calibration_3d`` was given and the fitting method supports it."""
    if calibration_3d is None:
        return False
    assert isinstance(
        calibration_3d, (dict, str)
    ), "calibration_3d must be a dict or a path to a YAML file"
    if fitting_method.startswith("spline"):
        # The spline PSF fit recovers z itself, from spline_calibration.
        warnings.warn(
            "Ignoring calibration_3d: the spline PSF fit recovers z itself, "
            "using spline_calibration.",
            stacklevel=3,
        )
        return False
    assert fitting_method != "avg", (
        "astigmatic z fitting (calibration_3d) requires fitted Gaussian "
        "widths, which 'avg' does not provide"
    )
    assert "spherical" not in fitting_method, (
        "astigmatic z fitting (calibration_3d) is not possible with the "
        "spherical Gaussian methods, which constrain sx == sy and thus "
        "carry no astigmatism"
    )
    return True


# TODO: remove in v0.12.0 - superseded by localize(calibration_3d=...)
def localize_3D(
    movie: LoadedMovie,
    *,
    movie_info: list[dict],
    camera_info: dict,
    box: int,
    minimum_ng: float,
    calibration_3d: dict,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    fitting_method: Literal[
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
    ] = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    spline_calibration: dict | None = None,
    affine_calibration: dict | list | None = None,
    camera_calibration: dict | None = None,
    multiprocess: bool = True,
    temporal_median_window: int | None = None,
    gaussian_filter_sigma: float | None = None,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_z_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Localize (i.e., identify and fit) spots in 3D in a movie using
    the specified parameters.

    .. deprecated:: 0.11.0
        Use ``picasso.localize.localize`` with its ``calibration_3d``
        argument instead - the two functions differ only in the astigmatic
        z fitting. ``localize_3D`` will be removed in v0.12.0.

    For the Gaussian ``fitting_method`` values this first runs 2D
    localizations, followed by z position fitting assuming astigmatism, see
    Huang, et al. Science, 2008 (``calibration_3d`` holds the astigmatism
    polynomials). For ``"spline-gpu"`` a cubic-spline PSF fit recovers z
    directly in the 2D fit, so no separate z-fitting step is run and
    ``spline_calibration`` is used instead of ``calibration_3d``.

    Parameters
    ----------
    movie : LoadedMovie
        The input movie, read frame by frame.
    movie_info : list of dicts
        Movie metadata.
    camera_info : dict
        A dictionary containing camera information: "Baseline",
        "Sensitivity", "Gain" and "Pixelsize".
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    minimum_ng : float
        Minimum net gradient for spot identification.
    calibration_3d : path or dict
        Either a path to a YAML file containing the calibration data or
        an already loaded calibration dictionary containing the
        following keys:

        - "X Coefficients": list of 7 floats, polynomial coefficients
            for the x-axis calibration curve;
        - "Y Coefficients": list of 7 floats, polynomial coefficients
            for the y-axis calibration curve;
        - "Magnification factor": float, magnification factor of the
            microscope, i.e., the ratio between the actual z position of
            the calibration sample and the estimated z position from the
            localization data.
    roi : tuple, optional
        Region of interest (ROI) defined as a tuple of two tuples,
        where the first tuple contains the start coordinates
        (y_start, x_start) and the second tuple contains the end
        coordinates (y_end, x_end). If None, the entire frame is used.
        Default is None.
    frame_bounds : tuple, optional
        Minimum and maximum frame numbers to consider for the
        identification. If None, all frames are used. If only min or max
        is to be specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    fitting_method : {"gausslq", "gausslq-spherical", "gausslq-rotated", \
            "gausslq-gpu", "gausslq-rotated-gpu", "gausslq-spherical-gpu", \
            "gaussmle", "gaussmle-spherical", "gaussmle-gpu", \
            "gaussmle-rotated-gpu" or "gaussmle-spherical-gpu"}, optional
        Which 2D fitting algorithm to use, see ``fit``. "avg" is not
        supported since z fitting requires the fitted Gaussian sigmas.
        Note that the rotated elliptical Gaussian methods report sx and
        sy along the rotated principal axes, whereas the astigmatism
        calibration assumes the camera axes, so use them for z fitting
        with care. The spherical Gaussian methods constrain sx == sy, so
        they carry no astigmatism and are unsuitable for z fitting.
        Default is "gausslq".
    eps : float, optional
        The convergence criterion for CPU MLE fitting. Ignored for
        other methods (GPU fitting uses its own convergence
        settings). Default is 0.001.
    max_it : int, optional
        The maximum number of iterations for CPU MLE fitting. Ignored
        for other methods. Default is 100.
    mle_method : Literal["sigma", "sigmaxy"], optional
        The method used for CPU MLE fitting (impose same sigma in x and
        y or not, respectively). Default is "sigmaxy".
    affine_calibration : dict or list or None, optional
        Lateral (x, y) affine corrections to apply on top of those the
        fit's own calibration carries, e.g. a standalone
        chromatic-aberration calibration combined with an astigmatism
        transform stored in ``calibration_3d``. Applied last, in list
        order. Default is None.
    camera_calibration : dict or None, optional
        A per-pixel sCMOS camera calibration for the (single) camera. All
        split-FOV regions are read from one sensor and the maps are indexed by
        absolute frame coordinates, so the same full-frame calibration serves
        every region. Default None.
    multiprocess: bool, optional
        Whether or not to use multiprocessing. Ignored for GPU fitting.
        Default is True.
    temporal_median_window : int or None, optional
        If given (and non-zero), a temporal median background is
        subtracted from every frame before identifying, using a window of
        this many frames, see ``TemporalMedianMovie``. The filter applies
        to the identification only - the spots are always cut out of and
        fitted on the raw movie. Note that ``minimum_ng`` has to be
        re-tuned when this is switched on or off, since subtracting a
        background changes the scale of the net gradient. Default is None
        (no filtering).
    gaussian_filter_sigma : float or None, optional
        Standard deviation (in camera pixels) of a spatial Gaussian
        filter applied to every frame before identifying, see
        ``GaussianFilteredMovie``. It merges the several local maxima of
        a spot that is not Gaussian-shaped into one. Applied after the
        temporal median filter, if both are used. The filter applies to
        the identification only - the spots are always cut out of and
        fitted on the raw movie. Note that ``minimum_ng`` has to be
        re-tuned when this is changed, since smoothing lowers gradient
        magnitudes. Default is None (no filtering).
    progress_callbacks : callable, "console" or None, optional
        If a callable provided, it must accept one integer input (number
        of movie frames, or spots for identifying and fitting callbacks,
        respectively). If "console", tqdm is used to display
        progress. If None, progress is not tracked.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots in 3D.
    info : list[dict]
        A list of dictionaries containing metadata about the movie and
        the fitting processes.
    """
    lib.deprecation_warning(
        "picasso.localize.localize_3D is deprecated and will be removed in "
        "version 0.12; use picasso.localize.localize with its calibration_3d "
        "argument instead."
    )
    assert isinstance(
        movie, (np.ndarray, io.ND2Movie)
    ), "movie must be a numpy array or ND2Movie"
    assert isinstance(movie_info, list), "movie_info must be a list"
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert (
        isinstance(box, int) and box > 0 and box % 2 == 1
    ), "box must be a positive odd integer"
    assert isinstance(
        minimum_ng, (int, float, list, tuple, np.ndarray)
    ), "minimum_ng must be a number or one number per ROI"
    assert fitting_method in [
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
    ], (
        "fitting_method must be one of 'gausslq', 'gausslq-spherical',"
        " 'gausslq-rotated', 'gausslq-gpu', 'gausslq-rotated-gpu',"
        " 'gausslq-spherical-gpu', 'gaussmle', 'gaussmle-spherical',"
        " 'gaussmle-gpu', 'gaussmle-rotated-gpu', 'gaussmle-spherical-gpu',"
        " 'spline-gpu', or 'spline-mle-gpu'"
    )
    if fitting_method.startswith("spline"):
        # The spline PSF fit recovers z itself; it uses a spline calibration
        # instead of the astigmatism polynomials in calibration_3d.
        assert isinstance(spline_calibration, dict), (
            "spline_calibration (a spline PSF calibration dict, see "
            "io.load_spline_calibration) is required for spline 3D "
            "localization"
        )
    else:
        assert isinstance(
            calibration_3d, (dict, str)
        ), "calibration_3d must be a dict or a path to a YAML file"
    assert eps is None or (
        isinstance(eps, (int, float)) and eps > 0
    ), "eps must be a positive number or None"
    assert max_it is None or (
        isinstance(max_it, int) and max_it > 0
    ), "max_it must be a positive integer or None"
    assert mle_method in [
        "sigma",
        "sigmaxy",
    ], "mle_method must be 'sigma' or 'sigmaxy'"
    assert isinstance(multiprocess, bool), "multiprocess must be a boolean"
    return _localize_3D(
        movie=movie,
        movie_info=movie_info,
        camera_info=camera_info,
        box=box,
        minimum_ng=minimum_ng,
        calibration_3d=calibration_3d,
        roi=roi,
        frame_bounds=frame_bounds,
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        mle_method=mle_method,
        spline_calibration=spline_calibration,
        affine_calibration=affine_calibration,
        camera_calibration=camera_calibration,
        multiprocess=multiprocess,
        temporal_median_window=temporal_median_window,
        gaussian_filter_sigma=gaussian_filter_sigma,
        identification_progress_callback=identification_progress_callback,
        fit_progress_callback=fit_progress_callback,
        fit_z_progress_callback=fit_z_progress_callback,
    )


def _localize_3D(
    movie: LoadedMovie,
    *,
    movie_info: list[dict],
    camera_info: dict,
    box: int,
    minimum_ng: float,
    calibration_3d: dict,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    fitting_method: Literal[
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
    ] = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    spline_calibration: dict | None = None,
    affine_calibration: dict | list | None = None,
    multiprocess: bool = True,
    temporal_median_window: int | None = None,
    gaussian_filter_sigma: float | None = None,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_z_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    camera_calibration: dict | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Internal function for `localize_3D`, assumes validated inputs.

    A thin wrapper around ``localize``, which does the astigmatic z
    fitting itself since v0.11.0. ``mle_method`` is not passed on: it has
    no effect on the fit and warns in ``localize``.
    """
    return localize(
        movie=movie,
        camera_info=camera_info,
        identification_parameters={
            "Min. Net Gradient": minimum_ng,
            "Box Size": box,
            "Temporal Median Window": temporal_median_window,
            "Gaussian Filter Sigma": gaussian_filter_sigma,
        },
        roi=roi,
        frame_bounds=frame_bounds,
        movie_info=movie_info,
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        spline_calibration=spline_calibration,
        # The spline fit recovers z itself, from spline_calibration; the
        # astigmatism polynomials only apply to the Gaussian methods.
        calibration_3d=(
            None if fitting_method.startswith("spline") else calibration_3d
        ),
        affine_calibration=affine_calibration,
        camera_calibration=camera_calibration,
        threaded=multiprocess,
        identification_progress_callback=identification_progress_callback,
        fit_progress_callback=fit_progress_callback,
        fit_z_progress_callback=fit_z_progress_callback,
    )


def _warn_duplicate_affine(duplicates: list) -> None:
    """Warn that corrections the fit's own calibration already applies were
    dropped, so they are not applied twice."""
    if duplicates:
        warnings.warn(
            "Skipping "
            + ", ".join(lib.describe_affine_transforms(duplicates))
            + ": the calibration used for fitting already carries this "
            "affine correction and applies it itself. Applying it again "
            "would correct the coordinates twice.",
            stacklevel=3,
        )


def _apply_extra_affine(
    locs: pd.DataFrame,
    info: list[dict],
    affine_calibration: dict | list | None,
    applied: dict | list | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply affine corrections that are not carried by the fit's own
    calibration (e.g. a standalone chromatic one) and record them in the
    metadata. Corrections ``applied`` already covers are dropped instead of
    applied a second time. A no-op when there are none."""
    extra, duplicates = lib.drop_duplicate_affine_transforms(
        affine_calibration, applied
    )
    _warn_duplicate_affine(duplicates)
    if not extra:
        return locs, info
    locs = lib.apply_affine_transforms(locs, extra)
    info = info + [
        {
            "Generated by": f"Picasso: v{__version__} Affine correction",
            "Affine corrections applied": lib.describe_affine_transforms(
                extra
            ),
        }
    ]
    return locs, info


def check_nena(
    locs: pd.DataFrame,
    info: None,
    callback: Callable[[int], None] = None,
) -> float:
    """Calculate the NeNA (experimental localization precision) from
    localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    info : None
        Not used.
    callback : Callable[[int], None], optional
        A callback function that can be used to report progress. It
        should accept an integer argument representing the current
        step or frame number. Default is None.

    Returns
    -------
    nena_px : float
        The NeNA value in pixels, representing the experimental
        localization precision.
    """
    print("Calculating NeNA.. ", end="")
    locs = locs[0:MAX_LOCS]
    try:
        result, nena_px = postprocess.nena(locs, info, callback=callback)
    except Exception as e:
        print(e)
        nena_px = float("nan")
    print(f"{nena_px:.2f} px.")
    return nena_px


def check_kinetics(locs: pd.DataFrame, info: list[dict]) -> float:
    """Calculate the mean length of binding events from localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    info : list of dicts
        A list of dictionaries containing metadata about the movie.

    Returns
    -------
    len_mean : float
        The mean length of binding events in frames.
    """
    print("Linking.. ", end="")
    locs = locs.iloc[0:MAX_LOCS]
    locs = postprocess.link(locs, info=info)
    len_mean = locs.len.mean()
    print(f"Mean length {len_mean:.2f} frames.")
    return len_mean


def check_drift(
    locs: pd.DataFrame,
    info: list[dict],
    callback: Callable[[int], None] = None,
) -> tuple[float, float]:
    """Estimate the drift of localizations in x and y directions.

    Parameters
    ----------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    info : list[dict]
        A list of dictionaries containing metadata about the movie.
    callback : Callable[[int], None], optional
        A callback function that can be used to report progress. It
        should accept an integer argument representing the current
        step or frame number. Default is None.

    Returns
    -------
    drift_x : float
        The estimated drift in the x direction.
    drift_y : float
        The estimated drift in the y direction.
    """
    steps = int(len(locs) // (MAX_LOCS))
    steps = max(1, steps)
    locs = locs[::steps]

    n_frames = lib.get_from_metadata(info, "Frames", raise_error=True)
    segmentation = max(1, int(n_frames // 10))

    print(f"Estimating drift with segmentation {segmentation}")
    drift, locs = postprocess.undrift(
        locs,
        info,
        segmentation,
        display=False,
        rcc_callback=callback,
    )
    drift_x = float(drift["x"].mean())
    drift_y = float(drift["y"].mean())

    print(f"Drift is X: {drift_x:.2f}, Y: {drift_y:.2f}.")

    return (drift_x, drift_y)


def get_file_summary(
    file: str,
    file_hdf: str,
    drift: tuple[float, float] | None = None,
    len_mean: float | None = None,
    nena: float | None = None,
) -> dict:
    """Generate a summary of the localization file, including metadata
    and statistics about the localizations.

    Parameters
    ----------
    file : str
        The path to the localization file (HDF5 format).
    file_hdf : str
        The path to the HDF5 file containing localizations.
    drift : tuple[float, float] | None, optional
        A tuple containing the drift in x and y directions. If None,
        the drift will be calculated from the localizations.
    len_mean : float | None, optional
        The mean length of binding events in frames. If None, it will
        be calculated from the localizations.
    nena : float | None, optional
        The NeNA value in pixels. If None, it will be calculated from
        the localizations.

    Returns
    -------
    summary : dict
        A dictionary containing the summary of the localization file,
        including metadata and statistics about the localizations.
    """
    if file_hdf is None:
        base, ext = os.path.splitext(file)
        file_hdf = base + "_locs.hdf5"

    locs, info = io.load_locs(file_hdf)

    summary = {}

    for col in MEAN_COLS:
        try:
            summary[col + "_mean"] = locs[col].mean()
            summary[col + "_std"] = locs[col].std()
        except KeyError:
            summary[col + "_mean"] = float("nan")
            summary[col + "_std"] = float("nan")

    for col in SET_COLS:
        col_ = col.lower()
        for inf in info:
            if col in inf:
                summary[col_] = inf[col]

    for col in SET_COLS:
        col_ = col.lower()
        if col_ not in summary:
            summary[col_] = float("nan")

    nena_px = check_nena(locs, info) if nena is None else nena
    len_mean = check_kinetics(locs, info) if len_mean is None else len_mean
    drift_x, drift_y = check_drift(locs, info) if drift is None else drift

    summary["len_mean"] = len_mean
    summary["n_locs"] = len(locs)
    summary["locs_frame"] = len(locs) / summary["frames"]
    summary["drift_x"] = drift_x
    summary["drift_y"] = drift_y
    summary["nena_px"] = nena_px
    summary["nena_nm"] = nena_px * summary["pixelsize"]
    summary["filename"] = os.path.normpath(file)
    summary["filename_hdf"] = file_hdf
    summary["file_created"] = datetime.fromtimestamp(os.path.getmtime(file))
    summary["entry_created"] = datetime.now()
    return summary


def _db_filename() -> str:
    """Return the path to the SQLite database file used for storing
    localization summaries. The database is stored in the user's home
    directory under the ``.picasso`` folder."""
    home = os.path.expanduser("~")
    picasso_dir = os.path.join(home, ".picasso")
    os.makedirs(picasso_dir, exist_ok=True)
    return os.path.abspath(os.path.join(picasso_dir, "app_0410.db"))


def _save_file_summary(summary: dict) -> None:
    """Save the summary of a localization file to a SQLite database."""
    engine = create_engine("sqlite:///" + _db_filename(), echo=False)
    s = pd.Series(summary, index=summary.keys()).to_frame().T
    s.to_sql("files", con=engine, if_exists="append", index=False)


def add_file_to_db(
    file: str,
    file_hdf: str,
    drift: tuple[float, float] | None = None,
    len_mean: float | None = None,
    nena: float | None = None,
) -> None:
    """Add a localization file summary to the SQLite database."""
    summary = get_file_summary(file, file_hdf, drift, len_mean, nena)
    _save_file_summary(summary)


def _movie_to_image(movie) -> np.ndarray:
    """Collapse a picasso movie to a single 2D float32 image on the
    original intensity (raw-count) scale. Multi-frame movies are
    averaged; single-frame movies are passed through. Keeping the raw
    scale means the net gradient computed during bead detection is
    comparable to the "Min. Net Gradient" used for normal localization.
    Frames are read one-at-a-time so the lazy-loading movie classes in
    ``picasso.io`` don't have to materialise the full stack at once."""
    n = len(movie)
    if n == 0:
        raise ValueError("Movie has zero frames.")
    if n == 1:
        return np.asarray(movie[0], dtype=np.float32)
    acc = np.zeros(np.asarray(movie[0]).shape, dtype=np.float64)
    for i in range(n):
        acc += np.asarray(movie[i], dtype=np.float64)
    return (acc / n).astype(np.float32)


def _affine_detect_beads(
    image: np.ndarray, box: int, minimum_ng: float
) -> np.ndarray:
    """Detect bead candidates using the standard spot identification
    (local maxima above a minimum net gradient).

    Parameters
    ----------
    image : np.ndarray
        2D image to detect beads in.
    box : int
        Box size used by ``identify_in_image`` (also sets the minimum
        distance between two detected beads). Should be an odd integer.
    minimum_ng : float
        Minimum net gradient for a local maximum to be kept.

    Returns
    -------
    np.ndarray
        (N, 2) array of [row, col] integer coordinates.
    """
    y, x, _ = identify_in_image(image, minimum_ng, box)
    return np.column_stack((y, x))


def _affine_refine_bead_positions(
    image: np.ndarray, coarse: np.ndarray, box: int
) -> np.ndarray:
    """Refine coarse bead positions to sub-pixel accuracy using the
    standard 2D Gaussian least-squares spot fitting.

    Parameters
    ----------
    image : np.ndarray
        2D image the beads were detected in.
    coarse : np.ndarray
        (N, 2) array of integer [row, col] bead positions.
    box : int
        Box size cut out around each bead for fitting. Should match the
        detection box so every spot lies fully inside the image.

    Returns
    -------
    np.ndarray
        (M, 2) array of refined [row, col] coordinates (M <= N; spots
        whose fit did not converge to a finite position are dropped).
    """
    if len(coarse) == 0:
        return np.empty((0, 2))
    ids = pd.DataFrame(
        {
            "frame": np.zeros(len(coarse), dtype=np.int32),
            "x": coarse[:, 1].astype(np.int32),
            "y": coarse[:, 0].astype(np.int32),
            "net_gradient": np.ones(len(coarse), dtype=np.float32),
        }
    )
    # Treat the single image as a one-frame movie; an identity camera
    # leaves the pixel values unchanged (the fit only needs relative
    # intensities to localize each bead).
    camera_info = {"Baseline": 0, "Sensitivity": 1.0, "Gain": 1}
    spots = get_spots(image[np.newaxis], ids, box, camera_info)
    theta = fit_spots_gauss(spots.astype(np.float32))
    locs = locs_from_fits_gauss(ids, theta, box, em=False)
    refined = np.column_stack((locs["y"].to_numpy(), locs["x"].to_numpy()))
    return refined[np.isfinite(refined).all(axis=1)]


def _affine_match_bead_pairs(
    coords_ref: np.ndarray,
    coords_mov: np.ndarray,
    return_indices: bool = False,
) -> tuple:
    """Match beads via mutual nearest-neighbour with a distance threshold.
    Returns (pairs_ref, pairs_mov), each (M, 2).

    With ``return_indices``, the indices the pairs have in ``coords_ref``
    and ``coords_mov`` are returned as well, so a caller can tell which
    detections stayed unmatched - the Localize viewer greys those out when
    it draws the pairing (see ``Window.draw_affine_pairing``)."""
    if len(coords_ref) == 0 or len(coords_mov) == 0:
        empty = np.empty((0, 2))
        if return_indices:
            idx = np.empty(0, dtype=int)
            return empty, empty, idx, idx
        return empty, empty
    D = cdist(coords_ref, coords_mov)
    nn_r2m = np.argmin(D, axis=1)
    nn_m2r = np.argmin(D, axis=0)
    pairs_r, pairs_m, idx_r, idx_m = [], [], [], []
    for i, j in enumerate(nn_r2m):
        if D[i, j] < _AFFINE_MATCH_MAX_DIST_PX and nn_m2r[j] == i:
            pairs_r.append(coords_ref[i])
            pairs_m.append(coords_mov[j])
            idx_r.append(i)
            idx_m.append(j)
    pairs_ref = np.array(pairs_r) if pairs_r else np.empty((0, 2))
    pairs_mov = np.array(pairs_m) if pairs_m else np.empty((0, 2))
    if return_indices:
        return (
            pairs_ref,
            pairs_mov,
            np.asarray(idx_r, dtype=int),
            np.asarray(idx_m, dtype=int),
        )
    return pairs_ref, pairs_mov


def _affine_estimate_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate a 6-DOF 2D affine transform mapping src -> dst from
    [row, col] correspondences. Returns a 3x3 homogeneous matrix in the
    (x=col, y=row) convention."""
    N = len(src)
    if N < 3:
        if N == 0:
            return np.eye(3)
        tx = float(np.mean(dst[:, 1] - src[:, 1]))
        ty = float(np.mean(dst[:, 0] - src[:, 0]))
        return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    src_xy = src[:, ::-1].copy()
    dst_xy = dst[:, ::-1].copy()
    A_mat = np.zeros((2 * N, 6))
    b_vec = np.zeros(2 * N)
    for i, ((xs, ys), (xd, yd)) in enumerate(zip(src_xy, dst_xy)):
        A_mat[2 * i, :] = [xs, ys, 1, 0, 0, 0]
        A_mat[2 * i + 1, :] = [0, 0, 0, xs, ys, 1]
        b_vec[2 * i] = xd
        b_vec[2 * i + 1] = yd
    result, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    a, b, tx, c, d, ty = result
    return np.array([[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]])


def _affine_decompose(M: np.ndarray, pixelsize: float | None = None) -> dict:
    """Decompose the 2x2 linear part of the affine matrix into
    rotation, anisotropic scaling, and shear via QR factorisation.

    Translations are always reported in pixels (``tx_px``, ``ty_px``).
    If ``pixelsize`` is given, the nanometre equivalents (``tx_nm``,
    ``ty_nm``) are added too.
    """
    A = M[:2, :2]
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    Q = Q * signs
    R = R * signs[np.newaxis, :]
    rot_deg = np.degrees(np.arctan2(Q[1, 0], Q[0, 0]))
    scale_x = R[0, 0]
    scale_y = R[1, 1]
    shear_deg = np.degrees(np.arctan2(R[0, 1], R[1, 1]))
    out = {
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "rotation_deg": float(rot_deg),
        "shear_deg": float(shear_deg),
        "tx_px": float(M[0, 2]),
        "ty_px": float(M[1, 2]),
    }
    if pixelsize is not None:
        out["tx_nm"] = float(M[0, 2] * pixelsize)
        out["ty_nm"] = float(M[1, 2] * pixelsize)
    return out


def _affine_apply(image: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Warp ``image`` by the affine transform ``M`` (x = col, y = row)
    using ``scipy.ndimage.affine_transform``. ``M`` is the forward map
    mov -> ref; we invert it in [row, col] space to feed scipy's pull
    convention."""
    M_rc = np.array(
        [
            [M[1, 1], M[1, 0], M[1, 2]],
            [M[0, 1], M[0, 0], M[0, 2]],
            [0.0, 0.0, 1.0],
        ]
    )
    M_rc_inv = np.linalg.inv(M_rc)
    A = M_rc_inv[:2, :2]
    offset = M_rc_inv[:2, 2]
    return affine_transform(
        image,
        A,
        offset=offset,
        output_shape=image.shape,
        order=3,
        mode="constant",
        cval=0.0,
    )


def _affine_plot_alignment(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    img_cor: np.ndarray,
    pairs_ref: np.ndarray,
    decomp: dict,
    n_pairs: int,
    pixelsize: float | None = None,
    save_path: str = "",
    ref_path: str = "",
    target_path: str = "",
    transform_type: str = "astigmatism",
) -> None:
    """Four-panel QC figure: overlay before/after correction and mean
    per-bead cross-correlation before/after correction.

    If ``pixelsize`` is None, axes are labelled in pixels; otherwise
    they are scaled to nm and labelled accordingly.
    """
    nm = pixelsize if pixelsize is not None else 1.0
    unit = "nm" if pixelsize is not None else "px"

    def norm(img):
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn + 1e-12)

    ref_n = norm(img_ref)
    mov_n = norm(img_mov)
    cor_n = norm(img_cor)

    crop_r = _AFFINE_XCORR_HALF_WIDTH

    def bead_xcorr_mean(frame_a, coords_a, frame_b, coords_b):
        acc, count = None, 0
        ny, nx = frame_a.shape
        for (ry_a, rx_a), (ry_b, rx_b) in zip(
            coords_a.astype(int), coords_b.astype(int)
        ):
            ya0 = max(0, ry_a - crop_r)
            ya1 = min(ny, ry_a + crop_r)
            xa0 = max(0, rx_a - crop_r)
            xa1 = min(nx, rx_a + crop_r)
            yb0 = max(0, ry_b - crop_r)
            yb1 = min(ny, ry_b + crop_r)
            xb0 = max(0, rx_b - crop_r)
            xb1 = min(nx, rx_b + crop_r)
            pa = frame_a[ya0:ya1, xa0:xa1]
            pb = frame_b[yb0:yb1, xb0:xb1]
            if pa.shape[0] < 4 or pb.shape[0] < 4:
                continue

            def prep(x):
                x = x - x.mean()
                s = x.std()
                return x / (s + 1e-12)

            cc = fftconvolve(prep(pa), prep(pb[::-1, ::-1]), mode="full")
            cc -= cc.min()
            if acc is None:
                acc = np.zeros_like(cc)
            if cc.shape == acc.shape:
                acc += cc
                count += 1
        if count == 0 or acc is None:
            s = 4 * crop_r - 1
            return np.zeros((s, s)), 0
        return acc / count, count

    def peak_nm(cc):
        py, px = np.unravel_index(np.argmax(cc), cc.shape)
        cy_cc, cx_cc = cc.shape[0] // 2, cc.shape[1] // 2
        r = 5
        y0 = max(0, py - r)
        y1 = min(cc.shape[0], py + r + 1)
        x0 = max(0, px - r)
        x1 = min(cc.shape[1], px + r + 1)
        patch = cc[y0:y1, x0:x1]
        nyp, nxp = patch.shape
        yg, xg = np.mgrid[0:nyp, 0:nxp].astype(float)

        def g2d(xy, x0, y0, sx, sy, amp, bg):
            x, y = xy
            return bg + amp * np.exp(
                -((x - x0) ** 2 / (2 * sx**2) + (y - y0) ** 2 / (2 * sy**2))
            )

        try:
            popt, _ = curve_fit(
                g2d,
                (xg.ravel(), yg.ravel()),
                patch.ravel(),
                p0=[
                    nxp / 2,
                    nyp / 2,
                    2.0,
                    2.0,
                    patch.max() - patch.min(),
                    patch.min(),
                ],
                maxfev=400,
            )
            sub_px = x0 + popt[0] - cx_cc
            sub_py = y0 + popt[1] - cy_cc
        except Exception:
            sub_px = float(px - cx_cc)
            sub_py = float(py - cy_cc)
        return sub_py * nm, sub_px * nm

    cc_raw, n_raw = bead_xcorr_mean(ref_n, pairs_ref, mov_n, pairs_ref)
    cc_cor, n_cor = bead_xcorr_mean(ref_n, pairs_ref, cor_n, pairs_ref)

    dy_raw, dx_raw = peak_nm(cc_raw) if n_raw > 0 else (0.0, 0.0)
    dy_cor, dx_cor = peak_nm(cc_cor) if n_cor > 0 else (0.0, 0.0)
    off_raw = np.hypot(dy_raw, dx_raw)
    off_cor = np.hypot(dy_cor, dx_cor)

    fig = plt.figure(figsize=(11, 11))
    if pixelsize is not None:
        trans_str = f"Tx={decomp['tx_nm']:.1f} nm  Ty={decomp['ty_nm']:.1f} nm"
    else:
        trans_str = f"Tx={decomp['tx_px']:.3f} px  Ty={decomp['ty_px']:.3f} px"
    title = (
        f"Alignment check  |  {n_pairs} bead pairs  |  "
        f"Scale X={decomp['scale_x']:.5f}  Y={decomp['scale_y']:.5f}  "
        f"Rot={decomp['rotation_deg']:.4f}°  " + trans_str
    )
    title = f"{transform_type.capitalize()}  |  " + title
    if ref_path or target_path:
        title += (
            f"\nref: {os.path.basename(ref_path)}   "
            f"target: {os.path.basename(target_path)}"
        )
    fig.suptitle(title, fontsize=10, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.25)

    ext = [0, img_ref.shape[1] * nm, img_ref.shape[0] * nm, 0]

    ax = fig.add_subplot(gs[0])
    ax.imshow(
        np.clip(np.stack([mov_n, ref_n, mov_n], axis=-1), 0, 1), extent=ext
    )
    ax.set_title(
        "Overlay  BEFORE\nRef (green) | Target (magenta)",
        fontsize=9,
        fontweight="bold",
    )
    ax.axis("off")

    ax = fig.add_subplot(gs[1])
    ax.imshow(
        np.clip(np.stack([cor_n, ref_n, cor_n], axis=-1), 0, 1), extent=ext
    )
    ax.set_title(
        "Overlay  AFTER\nRef (green) | Corrected (magenta)",
        fontsize=9,
        fontweight="bold",
    )
    ax.axis("off")

    lag = (np.arange(cc_raw.shape[1]) - cc_raw.shape[1] // 2) * nm
    e_cc = [lag[0], lag[-1], lag[-1], lag[0]]
    vmax = max(cc_raw.max(), cc_cor.max(), 1e-12)

    ax = fig.add_subplot(gs[2])
    im = ax.imshow(
        cc_raw,
        cmap="hot",
        extent=e_cc,
        vmin=0,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    ax.grid(False)
    ax.axhline(0, color="cyan", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0, color="cyan", lw=1.0, ls="--", alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(
        f"Mean bead cross-corr  BEFORE\n"
        f"peak ({dx_raw:.1f}, {dy_raw:.1f}) {unit}  "
        f"|offset| = {off_raw:.1f} {unit}",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel(f"Δx ({unit})")
    ax.set_ylabel(f"Δy ({unit})")

    ax = fig.add_subplot(gs[3])
    im2 = ax.imshow(
        cc_cor,
        cmap="hot",
        extent=e_cc,
        vmin=0,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    ax.grid(False)
    ax.axhline(0, color="cyan", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0, color="cyan", lw=1.0, ls="--", alpha=0.7)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(
        f"Mean bead cross-corr  AFTER\n"
        f"peak ({dx_cor:.1f}, {dy_cor:.1f}) {unit}  "
        f"|offset| = {off_cor:.1f} {unit}",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel(f"Δx ({unit})")
    ax.set_ylabel(f"Δy ({unit})")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def fit_affine_transform(
    movie_ref,
    movie_target,
    calibration: dict,
    box: int,
    minimum_ng: float,
    pixelsize: float | None = None,
    transform_type: str = "astigmatism",
    ref_path: str = "",
    target_path: str = "",
) -> tuple[dict, dict]:
    """Fit the target -> reference affine transform and append it to
    ``calibration``'s ordered list of affine corrections.

    This is the computational half of :func:`calibrate_affine_transform`.
    It touches no matplotlib state, so it is safe to call from a worker
    thread; the returned ``qc`` dict carries everything
    :func:`plot_affine_calibration` needs to draw the diagnostic figure
    afterwards (on the GUI thread, where matplotlib must be driven from).

    Parameters
    ----------
    See :func:`calibrate_affine_transform`; ``plot_path`` is the only
    argument not accepted here.

    Returns
    -------
    calibration : dict
        The input calibration, with the transform appended to its
        ``"Affine transforms"`` list (an existing entry of the same type is
        replaced). Save it with ``io.save_any_calibration``.
    qc : dict
        Inputs for :func:`plot_affine_calibration`: the reference, target
        and corrected images, the matched reference bead positions, the
        decomposition, the number of pairs, the pixel size, the transform
        type and the source paths.

    Raises
    ------
    ValueError
        If ``transform_type`` is unknown, if ``calibration`` is a
        multichannel spline calibration (affine corrections are
        single-channel only), or if fewer than 3 bead pairs match.
    """
    if transform_type not in lib.AFFINE_TRANSFORM_TYPES:
        raise ValueError(
            f"Unknown affine transform type '{transform_type}'; expected "
            f"one of {lib.AFFINE_TRANSFORM_TYPES}."
        )
    if calibration.get("model") in (
        "spline-3d-multichannel",
        precision._LINK_XYZ_MODEL,
    ):
        raise ValueError(
            "Affine corrections apply to single-channel data only, but "
            f"this is a '{calibration['model']}' calibration, which "
            "registers its channels itself. Append the transform to a "
            "single-channel calibration, or save it as a standalone "
            "affine calibration."
        )
    img_ref = _movie_to_image(movie_ref)
    img_target = _movie_to_image(movie_target)

    coarse_ref = _affine_detect_beads(img_ref, box, minimum_ng)
    coarse_target = _affine_detect_beads(img_target, box, minimum_ng)
    refined_ref = _affine_refine_bead_positions(img_ref, coarse_ref, box)
    refined_target = _affine_refine_bead_positions(
        img_target, coarse_target, box
    )
    pairs_ref, pairs_target, idx_ref, idx_target = _affine_match_bead_pairs(
        refined_ref, refined_target, return_indices=True
    )

    if len(pairs_ref) < 3:
        raise ValueError(
            f"Only {len(pairs_ref)} matched bead pair(s) — need >= 3 to "
            "fit an affine transform. Check the input images / detection "
            "parameters."
        )

    M = _affine_estimate_2d(pairs_target, pairs_ref)
    decomp = _affine_decompose(M, pixelsize)

    # What the two images are called depends on what is being corrected;
    # the fit and the way the transform is applied are identical.
    source = (
        "cylindrical" if transform_type == "astigmatism" else "target channel"
    )
    affine_entry = {
        "Type": transform_type,
        "Matrix": [[float(v) for v in row] for row in M],
        "Direction": f"{source} -> reference (x = col, y = row)",
        "Reference image": ref_path or "N/A",
        "Target image": target_path or "N/A",
        "Bead pairs": int(len(pairs_ref)),
        "Decomposition": decomp,
    }
    if pixelsize is not None:
        affine_entry["Pixelsize (nm)"] = float(pixelsize)
    lib.append_affine_transform(calibration, affine_entry)

    qc = {
        "img_ref": img_ref,
        "img_target": img_target,
        # the warp is part of the fit's output, not of the drawing, so it
        # is computed here and only displayed by the plotting function
        "img_cor": _affine_apply(img_target, M),
        "pairs_ref": pairs_ref,
        # Every detection in each image plus the indices of the matched
        # ones (pair k is (idx_ref[k], idx_target[k])), so the Localize
        # viewer can draw the pairing as color-coded identification boxes.
        "beads_ref": refined_ref,
        "beads_target": refined_target,
        "idx_ref": idx_ref,
        "idx_target": idx_target,
        "box": int(box),
        "decomposition": decomp,
        "n_pairs": int(len(pairs_ref)),
        "pixelsize": pixelsize,
        "transform_type": transform_type,
        "ref_path": ref_path,
        "target_path": target_path,
    }
    return calibration, qc


def plot_affine_calibration(qc: dict, save_path: str = "") -> None:
    """Draw the affine-calibration diagnostic figure from the ``qc`` dict
    returned by :func:`fit_affine_transform`.

    Kept separate from the fit so a GUI can run the fit in a worker thread
    and still draw from the main thread. ``save_path`` writes the figure to
    disk; it is always shown interactively.
    """
    _affine_plot_alignment(
        qc["img_ref"],
        qc["img_target"],
        qc["img_cor"],
        qc["pairs_ref"],
        qc["decomposition"],
        n_pairs=qc["n_pairs"],
        pixelsize=qc["pixelsize"],
        save_path=save_path,
        ref_path=qc.get("ref_path", ""),
        target_path=qc.get("target_path", ""),
        transform_type=qc.get("transform_type", "astigmatism"),
    )


def calibrate_affine_transform(
    movie_ref,
    movie_target,
    calibration: dict,
    box: int,
    minimum_ng: float,
    pixelsize: float | None = None,
    transform_type: str = "astigmatism",
    ref_path: str = "",
    target_path: str = "",
    plot_path: str = "",
) -> dict:
    """Fit a 6-DOF affine transform that maps a bead image into a
    reference frame and append it to any calibration dict.

    The same calibration serves two corrections, selected by
    ``transform_type``:

    - ``"astigmatism"``: the cylindrical-lens image is mapped into the
      reference (no-lens) frame, undoing the lateral distortion the
      cylindrical lens introduces.
    - ``"chromatic"``: one color channel is mapped into the reference
      color channel, correcting chromatic aberration.

    Both are stored as entries of an ordered ``"Affine transforms"`` list
    and applied to ``x``/``y`` in that order after fitting, so a 3D
    two-color experiment can chain the astigmatism correction and the
    chromatic one. The list is read from whatever calibration the fit uses
    - Gaussian astigmatism (YAML) or cubic-spline PSF (HDF5) - and an empty
    ``calibration`` dict starts a standalone affine calibration file, which
    is what a purely 2D chromatic correction needs.

    This is a **single-channel** correction: it maps one movie into a
    reference frame. A multichannel spline calibration registers its
    channels itself, so appending a transform to one raises ``ValueError``.

    The fit is performed in pixel coordinates on a per-pixel mean of
    each movie. Bead candidates are found by Gaussian-blur + local-max,
    refined to sub-pixel accuracy by a 2D Gaussian fit, then matched
    between the two images by mutual nearest neighbour. The affine
    matrix is solved by 6-DOF linear least squares and decomposed into
    rotation / anisotropic scale / shear via QR.

    Parameters
    ----------
    movie_ref, movie_target : AbstractPicassoMovie
        In-focus bead movies of the reference and of the frame to be
        corrected: without / with the cylindrical lens for
        ``"astigmatism"``, reference / other color channel for
        ``"chromatic"``. If a movie has multiple frames they are
        averaged; a single-frame movie is used as-is.
    calibration : dict
        Calibration the transform is appended to; may be a Gaussian
        astigmatism calibration, a single-channel spline PSF calibration,
        or ``{}`` to start a standalone affine calibration. An existing
        entry of the same ``transform_type`` is replaced. A multichannel
        spline calibration is rejected (see above).
    box : int
        Box size used to identify bead candidates (also sets the minimum
        distance between two detected beads). Should be an odd integer.
    minimum_ng : float
        Minimum net gradient for a bead candidate to be kept.
    pixelsize : float, optional
        Camera pixel size in nm. If given, decomposition translations
        and the diagnostic plot are converted from pixels to nm. If
        None (default), values are reported in pixels. Default is None.
    transform_type : {"astigmatism", "chromatic"}, optional
        What the transform corrects; recorded in the entry and used to
        decide which existing entry it replaces. Default is
        "astigmatism".
    ref_path, target_path : str, optional
        Paths to the source images, recorded in the calibration for
        traceability and shown in the diagnostic plot title. Default
        is "".
    plot_path : str, optional
        If given, the diagnostic figure is saved to this path. The
        figure is always shown interactively. Default is "".

    Returns
    -------
    calibration : dict
        The input calibration with the transform appended to its
        ``"Affine transforms"`` list. Use ``io.save_any_calibration`` to
        save the result to YAML or HDF5, whichever the calibration is.

    Notes
    -----
    Fit and figure are also available separately as
    :func:`fit_affine_transform` and :func:`plot_affine_calibration`, for
    callers that must not touch matplotlib from the thread doing the fit.
    """
    calibration, qc = fit_affine_transform(
        movie_ref,
        movie_target,
        calibration,
        box=box,
        minimum_ng=minimum_ng,
        pixelsize=pixelsize,
        transform_type=transform_type,
        ref_path=ref_path,
        target_path=target_path,
    )
    plot_affine_calibration(qc, save_path=plot_path)
    return calibration
