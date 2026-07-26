"""
picasso.localize
~~~~~~~~~~~~~~~~

Identify and localize fluorescent single molecules in a frame
sequence.

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
from concurrent.futures import ThreadPoolExecutor, Future
from itertools import chain
from typing import Literal
from typing import Callable
from datetime import datetime

import numba
import numpy as np
import dask.array as da
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sqlalchemy import create_engine

from .ext import bitplane

from . import (
    io,
    lib,
    gausslq,
    gaussmle,
    avgroi,
    postprocess,
    zfit,
    __version__,
)

try:
    from .ext.pygpufit import gpufit as gf

    GPUFIT_INSTALLED = bool(gf.cuda_available())
except Exception:
    GPUFIT_INSTALLED = False

try:
    from .ext.pygpuspline import gpuspline as gs  # noqa: F401

    # gpuspline is a plain CPU coefficient library (despite the name, it does
    # not use CUDA/GPU); the import succeeds only if the compiled splines
    # library can be loaded. It is required only to *generate* a spline PSF
    # calibration (spline_coefficients), not to fit an existing one - fitting
    # needs only Gpufit. Exposed as ``localize.gs`` for the calibration
    # builder.
    GPUSPLINE_INSTALLED = True
except Exception:
    GPUSPLINE_INSTALLED = False

plt.style.use("ggplot")


MAX_LOCS = int(1e6)
_SPLINE_CRLB_MU_FLOOR = 1e-3  # photons; floors 1 / mu in the Fisher weight
_GAUSS_CRLB_MU_FLOOR = (
    1e-3  # photons; floors 1 / mu in the Poisson Fisher weight
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
    "Uncertainty": ["photons_unc", "bg_unc", "sx_unc", "sy_unc"],
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


def identify_in_frame(
    frame: lib.IntArray2D,
    minimum_ng: float,
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
    minimum_ng : float
        Minimum net gradient value to consider a maximum as valid.
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
        return identify_in_image(image, minimum_ng, box)
    height, width = frame.shape
    # pad each ROI to identify at the border
    pad = int(box / 2) + 1
    ys, xs, ngs = [], [], []
    for (y0, x0), (y1, x1) in rois:
        py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
        py1, px1 = min(y1 + pad, height), min(x1 + pad, width)
        image = np.float32(frame[py0:py1, px0:px1])  # numba needs float32!
        y, x, net_gradient = identify_in_image(image, minimum_ng, box)
        y += py0  # offset back to global frame coordinates
        x += px0
        # keep only maxima centered inside the actual ROI
        inside = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
        ys.append(y[inside])
        xs.append(x[inside])
        ngs.append(net_gradient[inside])
    return np.concatenate(ys), np.concatenate(xs), np.concatenate(ngs)


def identify_by_frame_number(
    movie: lib.IntArray3D,
    minimum_ng: float,
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
    movie : lib.IntArray3D
        A 3D array representing the movie of shape (N, Y, X), where N is
        the number of frames, Y is the height, and X is the width.
    minimum_ng : float
        Minimum net gradient value to consider a maximum as valid.
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
    # check frame bounds
    if not lib.frame_in_bounds(frame_number, frame_bounds, len(movie)):
        return pd.DataFrame(
            {
                "frame": pd.Series(dtype=int),
                "x": pd.Series(dtype=int),
                "y": pd.Series(dtype=int),
                "net_gradient": pd.Series(dtype=np.float32),
            }
        )
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
    movie: lib.IntArray3D,
    current: list[int],
    minimum_ng: float,
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
    movie: lib.IntArray3D,
    minimum_ng: float,
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
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    minimum_ng : float
        The minimum net gradient for a spot to be considered.
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
    movie: lib.IntArray3D,
    minimum_ng: float,
    box: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    frame_bounds: tuple[int, int] | list | None = None,
    threaded: bool = True,
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
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    minimum_ng : float
        The minimum net gradient for a spot to be considered.
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
    movie: lib.IntArray3D,
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
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
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
    movie: lib.IntArray3D,
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
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
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
    movie: lib.IntArray3D,
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


def _to_photons(
    spots: lib.FloatArray3D, camera_info: dict
) -> lib.FloatArray3D:
    """Convert the cut spots to photon counts based on camera
    information."""
    spots = np.float32(spots)
    baseline = camera_info["Baseline"]
    sensitivity = camera_info["Sensitivity"]
    gain = camera_info["Gain"]
    # since v0.6.0: remove quantum efficiency to better reflect precision
    # qe = camera_info["Qe"]
    return (spots - baseline) * sensitivity / (gain)


def get_spots(
    movie: lib.IntArray3D,
    identifications: pd.DataFrame,
    box: int,
    camera_info: dict,
    progress_callback: Callable[[int], None] | None = None,
) -> lib.FloatArray3D:
    """Extract the spots from a movie based on the identified positions
    and convert camera signal to photon counts.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
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

    Returns
    -------
    spots : lib.FloatArray3D
        A 3D numpy array containing the extracted spots, with shape
        (k, box, box), where k is the number of spots identified.
    """
    spots = _cut_spots(
        movie, identifications, box, progress_callback=progress_callback
    )
    return _to_photons(spots, camera_info)


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
        `net_gradient`, `likelihood`, and `iterations`.
    """
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
            "likelihood": likelihoods.astype(np.float32),
            "iterations": iterations.astype(np.int32),
        }
    )
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def fit2D(
    movie: lib.IntArray3D,
    movie_info: list[dict],
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
        "spline-gpu",
        "spline-mle-gpu",
        "avg",
    ] = "gausslq",
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    spline_calibration: dict | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    cut_progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    """Fit 2D localizations to a movie, given positions of the detected
    spots (identifications).

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    movie_info : list of dicts
        Movie metadata.
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
        on the GPU (Gpufit's maximum likelihood estimator).
        "gausslq-rotated" for CPU least-squares fitting, and
        "gausslq-rotated-gpu" and "gaussmle-rotated-gpu" for GPU
        least-squares and MLE fitting, respectively, of a rotated
        elliptical Gaussian, whose fitted rotation angle (in degrees)
        is saved in the column "angle". "gausslq-spherical" and
        "gaussmle-spherical" for CPU least-squares and MLE fitting, and
        "gausslq-spherical-gpu" and "gaussmle-spherical-gpu" for their
        GPU counterparts, of a spherical (isotropic) Gaussian with a
        single width; the saved "sx" and "sy" columns are identical.
        "spline-gpu" and "spline-mle-gpu"
        for GPU least-squares / maximum-likelihood fitting of an
        experimentally measured cubic-spline PSF (Gpufit's SPLINE_2D /
        SPLINE_3D models); both require ``spline_calibration``, and a 3D
        spline calibration yields the fitted ``z`` directly. "avg" for
        taking the average of each spot.
    eps : float, optional
        The convergence criterion for CPU MLE fitting. Ignored for
        other methods (GPU fitting uses Gpufit's own convergence
        settings). Default is 0.001.
    max_it : int, optional
        The maximum number of iterations for CPU MLE fitting. Ignored
        for other methods (GPU fitting uses Gpufit's own convergence
        settings). Default is 100.
    mle_method : Literal["sigma", "sigmaxy"], optional
        The method used for CPU MLE fitting (impose same sigma in x and
        y or not, respectively). Default is "sigmaxy".
    spline_calibration : dict or None, optional
        Cubic-spline PSF calibration (see ``io.load_spline_calibration``),
        required when ``fitting_method`` is "spline-gpu" and ignored
        otherwise. For a 3D spline calibration the resulting localizations
        contain the fitted ``z`` directly (no separate z-fitting step is
        needed). Default is None.
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
    assert isinstance(movie_info, list), "movie_info must be a list"
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert isinstance(
        identifications, pd.DataFrame
    ), "identifications must be a DataFrame"
    assert isinstance(box, int) and box > 0, "box must be a positive integer"
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
        "spline-gpu",
        "spline-mle-gpu",
        "avg",
    ], (
        "fitting_method must be one of 'gausslq', 'gausslq-spherical',"
        " 'gausslq-rotated', 'gausslq-gpu', 'gausslq-rotated-gpu',"
        " 'gausslq-spherical-gpu', 'gaussmle', 'gaussmle-spherical',"
        " 'gaussmle-gpu', 'gaussmle-rotated-gpu', 'gaussmle-spherical-gpu',"
        " 'spline-gpu', 'spline-mle-gpu', or 'avg'"
    )
    if fitting_method.startswith("spline"):
        assert isinstance(spline_calibration, dict), (
            "spline_calibration (a spline PSF calibration dict, see "
            "io.load_spline_calibration) is required for spline fitting"
        )
    assert (
        isinstance(eps, (int, float)) and eps > 0
    ), "eps must be a positive number"
    assert (
        isinstance(max_it, int) and max_it > 0
    ), "max_it must be a positive integer"
    assert mle_method in [
        "sigma",
        "sigmaxy",
    ], "mle_method must be 'sigma' or 'sigmaxy'"
    assert isinstance(multiprocess, bool), "multiprocess must be a boolean"
    if "Pixelsize" not in camera_info:
        warnings.warn(
            "Camera info in picasso.localize.fit2D does not contain "
            "'Pixelsize', i.e., effective camera pixel size in nm. "
            "Assuming 130."
        )
        camera_info["Pixelsize"] = 130

    spots = get_spots(
        movie,
        identifications,
        box,
        camera_info,
        progress_callback=cut_progress_callback,
    )
    em = camera_info["Gain"] > 1
    if fitting_method in ("gausslq", "gausslq-spherical", "gausslq-rotated"):
        locs = _fit2d_gausslq(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
            multiprocess=multiprocess,
            progress_callback=progress_callback,
            abort_callback=abort_callback,
            spherical=fitting_method == "gausslq-spherical",
            rotated=fitting_method == "gausslq-rotated",
        )
    elif fitting_method in (
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
    ):
        if callable(progress_callback):
            progress_callback(1)
        locs = _fit2d_gauss_gpu(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
            rotated="-rotated-" in fitting_method,
            mle=fitting_method.startswith("gaussmle"),
            spherical="-spherical-" in fitting_method,
        )
    elif fitting_method in ("spline-gpu", "spline-mle-gpu"):
        if callable(progress_callback):
            progress_callback(1)
        # "spline-mle-gpu" uses Gpufit's Poisson maximum-likelihood estimator,
        # "spline-gpu" least squares (as in the reference pyGpufit examples).
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
        )
    elif fitting_method in ("gaussmle", "gaussmle-spherical"):
        # "gaussmle-spherical" forces the isotropic single-width ("sigma")
        # MLE path, which already outputs equal sx/sy; the plain "gaussmle"
        # code respects the separate mle_method argument.
        locs = _fit2d_gaussmle(
            spots=spots,
            identifications=identifications,
            box=box,
            eps=eps,
            max_it=max_it,
            mle_method=(
                "sigma"
                if fitting_method == "gaussmle-spherical"
                else mle_method
            ),
            multiprocess=multiprocess,
            progress_callback=progress_callback,
            abort_callback=abort_callback,
            spherical=fitting_method == "gaussmle-spherical",
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
        )
    # updated metadata
    localize_info = {
        "Generated by": f"Picasso: v{__version__} Fit 2D",
        "Fit method": fitting_method,
    }
    if fitting_method in ("gaussmle", "gaussmle-spherical"):
        localize_info["Convergence criterion"] = eps
        localize_info["Max iterations"] = max_it
    if fitting_method.startswith("spline"):
        localize_info["Spline calibration model"] = spline_calibration.get(
            "model"
        )
        localize_info["Spline calibration path"] = spline_calibration.get(
            "Path", "N/A"
        )
    new_info = localize_info | camera_info
    return locs, new_info


def _fit2d_gausslq(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    spherical: bool = False,
    rotated: bool = False,
) -> pd.DataFrame | None:
    """Fit 2D Gaussians using least-squares fitting (CPU). If
    ``spherical``, an isotropic Gaussian with a single width is fitted
    and the resulting ``sx`` and ``sy`` columns are identical. If
    ``rotated``, a rotated elliptical Gaussian is fitted and the
    resulting localizations contain the fitted rotation angle (in
    degrees) in the column ``angle``. See ``fit_2D`` for more details."""
    N = len(identifications)
    if multiprocess:
        fs = gausslq.fit_spots_parallel(
            spots, asynch=True, spherical=spherical, rotated=rotated
        )
        theta = _process_fitting_futures(
            fs, N, progress_callback, abort_callback
        )
        if theta is None:
            return
    else:
        theta = gausslq.fit_spots(
            spots, progress_callback, spherical=spherical, rotated=rotated
        )
    locs = gausslq.locs_from_fits(
        identifications,
        theta,
        box,
        em,
        spherical=spherical,
    )
    return locs


def _initial_parameters_gpufit(
    spots: lib.FloatArray3D,
    size: int,
    rotated: bool = False,
    spherical: bool = False,
) -> lib.FloatArray2D:
    """Initialize the parameters for the GPU fit - photons, x, y, sx,
    sy, bg (plus the rotation angle if ``rotated``). If ``spherical``,
    a single width is used and the layout is photons, x, y, s, bg
    (Gpufit's isotropic ``GAUSS_2D`` model)."""
    center = (size / 2.0) - 0.5
    initial_width = np.amax([size / 5.0, 1.0])

    spot_max = np.amax(spots, axis=(1, 2))
    spot_min = np.amin(spots, axis=(1, 2))

    if spherical:
        # GAUSS_2D: photons, x, y, s (single width), bg.
        initial_parameters = np.empty((len(spots), 5), dtype=np.float32)
        initial_parameters[:, 0] = spot_max - spot_min
        initial_parameters[:, 1] = center
        initial_parameters[:, 2] = center
        initial_parameters[:, 3] = initial_width
        initial_parameters[:, 4] = spot_min
        return initial_parameters

    n_parameters = 7 if rotated else 6
    initial_parameters = np.empty((len(spots), n_parameters), dtype=np.float32)

    initial_parameters[:, 0] = spot_max - spot_min
    initial_parameters[:, 1] = center
    initial_parameters[:, 2] = center
    initial_parameters[:, 3] = initial_width
    initial_parameters[:, 4] = initial_width
    initial_parameters[:, 5] = spot_min
    if rotated:
        # With sx == sy, the rotated Gaussian is independent of the
        # angle, so its derivative is exactly zero and the first LM
        # Hessian is singular - Gpufit then aborts, returning the
        # initial parameters. Break the symmetry of the widths to keep
        # the angle parameter well-defined.
        initial_parameters[:, 3] *= 1.1
        initial_parameters[:, 4] *= 0.9
        initial_parameters[:, 6] = 0.0

    return initial_parameters


def fit_spots_gpufit(
    spots: lib.FloatArray3D,
    rotated: bool = False,
    mle: bool = False,
    spherical: bool = False,
    return_stats: bool = False,
) -> (
    lib.FloatArray2D
    | tuple[lib.FloatArray2D, lib.FloatArray1D | None, lib.FloatArray1D]
):
    """Fit multiple spots using GPU-based Gaussian fitting. Each spot is
    a 2D array representing the pixel values of the spot image. The
    function returns a 2D array with the optimized parameters for each
    spot, where each row corresponds to a spot and the columns are the
    parameters in the following order: [photons, x, y, sx, sy, bg] or,
    for the rotated elliptical Gaussian, [photons, x, y, sx, sy, bg,
    angle], where angle is the rotation angle in radians.

    Picasso vendors pyGPUfit under picasso/ext/pygpufit where the
    License can be found too.

    Only Windows with a CUDA-capable GPU is supported. Linux users
    need to build the code first, see Localize documentation.

    Cite: Przybylski, et al. Scientific Reports, 2017.
    DOI: 10.1038/s41598-017-15313-9

    Parameters
    ----------
    spots : lib.FloatArray3D
        A 3D array of shape (n_spots, size, size), where n_spots is the
        number of spots and size is the length of one side of the square
        spot image. Each slice along the first axis represents a single
        spot image.
    rotated : bool, optional
        If True, fit a rotated elliptical Gaussian (Gpufit's
        GAUSS_2D_ROTATED model) whose seventh parameter is the rotation
        angle. Default is False.
    mle : bool, optional
        If True, use Gpufit's maximum likelihood estimator (Poisson
        noise model) instead of least squares. Default is False.
    spherical : bool, optional
        If True, fit a spherical (isotropic) Gaussian with a single
        width using Gpufit's ``GAUSS_2D`` model. The returned parameters
        still use the standard elliptical layout with ``sx == sy`` so
        the rest of the pipeline is unchanged. Cannot be combined with
        ``rotated``. Default is False.
    return_stats : bool, optional
        If True, additionally return the per-spot fit diagnostics
        (log-likelihood and iteration counts) reported by Gpufit.
        Default is False.

    Returns
    -------
    parameters : lib.FloatArray2D
        A 2D array with the optimized parameters for each spot. The
        columns correspond to [photons, x, y, sx, sy, bg] or
        [photons, x, y, sx, sy, bg, angle] if ``rotated``.
    log_likelihood : lib.FloatArray1D or None
        Only returned if ``return_stats``. The per-spot Poisson
        log-likelihood when ``mle`` (derived from Gpufit's chi-square,
        whose MLE value equals twice the negative log-likelihood), or
        None for least squares, where Gpufit reports a residual sum of
        squares rather than a likelihood.
    number_iterations : lib.FloatArray1D
        Only returned if ``return_stats``. The number of iterations
        taken to converge for each spot.
    """
    if not GPUFIT_INSTALLED:
        raise ImportError(
            "GPUfit could not be found, CUDA-capable GPU is required."
        )
    if rotated and spherical:
        raise ValueError("'rotated' and 'spherical' are mutually exclusive.")
    if mle:
        spots = np.maximum(spots, 0)
    size = spots.shape[1]
    initial_parameters = _initial_parameters_gpufit(
        spots, size, rotated=rotated, spherical=spherical
    )
    if spherical:
        model_id = gf.ModelID.GAUSS_2D
    elif rotated:
        model_id = gf.ModelID.GAUSS_2D_ROTATED
    else:
        model_id = gf.ModelID.GAUSS_2D_ELLIPTIC
    estimator_id = gf.EstimatorID.MLE if mle else gf.EstimatorID.LSE

    parameters, states, chi_squares, number_iterations, exec_time = gf.fit(
        spots.reshape((len(spots), (size * size))),
        None,
        model_id,
        initial_parameters,
        tolerance=1e-2,
        max_number_iterations=20,
        estimator_id=estimator_id,
    )

    if spherical:
        # GAUSS_2D returns [amp, x, y, s, bg]. Expand to the standard
        # elliptical layout [photons, x, y, sx, sy, bg] with sx == sy so
        # the rest of the pipeline (CRLB, column building) is unchanged.
        s = parameters[:, 3]
        expanded = np.empty((len(parameters), 6), dtype=parameters.dtype)
        expanded[:, 0] = parameters[:, 0] * 2.0 * np.pi * s * s
        expanded[:, 1] = parameters[:, 1]
        expanded[:, 2] = parameters[:, 2]
        expanded[:, 3] = s
        expanded[:, 4] = s
        expanded[:, 5] = parameters[:, 4]
        parameters = expanded
    else:
        parameters[:, 0] *= 2.0 * np.pi * parameters[:, 3] * parameters[:, 4]

    if return_stats:
        # Gpufit's MLE chi-square equals twice the negative Poisson
        # log-likelihood, so -0.5 * chi_square reproduces the CPU MLE
        # fit's log_likelihood (both Stirling-approximated). For least
        # squares the chi-square is a residual sum of squares, not a
        # likelihood, so there is nothing meaningful to return.
        log_likelihood = -0.5 * chi_squares if mle else None
        return parameters, log_likelihood, number_iterations
    return parameters


def _gauss_crlb(
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
    rotated: bool = False,
) -> lib.FloatArray2D:
    """Poisson Cramer-Rao lower bound for gpufit MLE Gaussian fits.

    Builds the Fisher information matrix ``I = Σ g gᵀ / μ`` (``g = ∂μ/∂θ``) of
    the point-sampled Gaussian model gpufit actually optimizes
    (``GAUSS_2D_ELLIPTIC`` / ``GAUSS_2D_ROTATED``) and returns the diagonal of
    its inverse — the variance an efficient maximum-likelihood estimator
    attains. Evaluated at the fitted parameters; the spot data is not needed.
    Mirrors :func:`_spline_crlb` and ``gaussmle._mlefit_sigmaxy_crlb`` but for
    the point-sampled Gaussian gpufit fits rather than the erf-integrated CPU
    model.

    Model (photon units, spots are gain-converted before fitting)::

        mu(i, j) = N / (2 pi sx sy) * E + bg

    where ``i`` indexes the x (column) coordinate, ``j`` the y (row)
    coordinate, and ``E`` is the (optionally rotated) unit-height Gaussian.
    Parametrizing the amplitude directly as the total photon count ``N``
    (Picasso's reported ``photons``) makes the returned variances line up with
    the reported columns, with the ``N``/``sx``/``sy`` coupling of ``mu`` folded
    into the derivatives.

    Parameters
    ----------
    theta : lib.FloatArray2D
        Fitted parameters in Picasso/gpufit order ``[photons (total N), x, y,
        sx, sy, bg]`` (elliptic) or ``[..., bg, angle (radians)]`` (rotated).
        Positions are box-local (pixel = parameter, matching
        :func:`_initial_parameters_gpufit`).
    box : int
        Fit box side length (pixels).
    em : bool
        EMCCD excess noise: doubles every parameter's variance (halves the
        Fisher weight), as in :func:`gausslq.localization_precision`.
    rotated : bool, optional
        If True, ``theta`` carries the seventh (angle) column and the CRLB
        includes it. Default False.

    Returns
    -------
    crlb : lib.FloatArray2D
        ``(n_locs, n_params)`` parameter variances (float64) in the same column
        order as ``theta`` (angle variance in rad²). Non-converged and
        numerically singular fits are NaN.
    """
    theta = np.asarray(theta, dtype=np.float64)
    n_locs = len(theta)
    n_params = 7 if rotated else 6

    N = theta[:, 0]
    x = theta[:, 1]
    y = theta[:, 2]
    sx = theta[:, 3]
    sy = theta[:, 4]
    ang = theta[:, 6] if rotated else None
    finite = np.isfinite(theta).all(axis=1)

    grid = np.arange(box, dtype=np.float64)
    crlb = np.full((n_locs, n_params), np.nan)
    if n_locs == 0:
        return crlb

    # One kernel spans all localizations; chunk only to bound peak memory of the
    # (chunk, n_params, box, box) gradient tensor.
    chunk = max(1, min(n_locs, 50_000))
    for start in range(0, n_locs, chunk):
        stop = min(start + chunk, n_locs)
        sl = slice(start, stop)
        # Per-pixel coordinates relative to the fitted center. Axis 1 = x
        # (column) pixel index, axis 2 = y (row) pixel index.
        Nc = N[sl][:, None, None]
        sxc = sx[sl][:, None, None]
        syc = sy[sl][:, None, None]
        dx = grid[None, :, None] - x[sl][:, None, None]
        dy = grid[None, None, :] - y[sl][:, None, None]

        if rotated:
            ct = np.cos(ang[sl])[:, None, None]
            st = np.sin(ang[sl])[:, None, None]
            u = dx * ct - dy * st
            w = dx * st + dy * ct
            E = np.exp(-0.5 * (u**2 / sxc**2 + w**2 / syc**2))
            s = Nc / (2.0 * np.pi * sxc * syc) * E  # signal = mu - bg
            gx = s * (u * ct / sxc**2 + w * st / syc**2)
            gy = s * (-u * st / sxc**2 + w * ct / syc**2)
            gsx = s * (u**2 / sxc**3 - 1.0 / sxc)
            gsy = s * (w**2 / syc**3 - 1.0 / syc)
            gang = s * (u * w * (1.0 / sxc**2 - 1.0 / syc**2))
        else:
            E = np.exp(-0.5 * (dx**2 / sxc**2 + dy**2 / syc**2))
            s = Nc / (2.0 * np.pi * sxc * syc) * E  # signal = mu - bg
            gx = s * (dx / sxc**2)
            gy = s * (dy / syc**2)
            gsx = s * (dx**2 / sxc**3 - 1.0 / sxc)
            gsy = s * (dy**2 / syc**3 - 1.0 / syc)

        gN = s / Nc
        gbg = np.ones_like(s)
        grads = [gN, gx, gy, gsx, gsy, gbg]
        if rotated:
            grads.append(gang)
        g = np.stack(grads, axis=1)  # (m, n_params, box, box)

        mu = np.maximum(s + theta[sl, 5][:, None, None], _GAUSS_CRLB_MU_FLOOR)
        gw = g / mu[:, None, :, :]
        fisher = np.einsum("mpij,mqij->mpq", gw, g)  # (m, n_params, n_params)

        # Non-converged rows carry NaN parameters (hence NaN Fisher); set them to
        # the identity so the batched pinv stays well-defined, then mask below.
        bad = ~finite[sl]
        fisher[bad] = np.eye(n_params)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = np.linalg.pinv(fisher)
            var = np.diagonal(cov, axis1=1, axis2=2).copy()
        var[bad] = np.nan
        crlb[sl] = var

    if em:
        # EMCCD excess noise doubles every pixel's variance, hence the CRLB
        # (matches the factor-2 in gausslq.localization_precision).
        crlb *= 2.0
    crlb = np.where(crlb > 0.0, crlb, np.nan)
    return crlb


def locs_from_fits_gpufit(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
    mle: bool = False,
    log_likelihood: lib.FloatArray1D | None = None,
    iterations: lib.FloatArray1D | None = None,
    spherical: bool = False,
) -> pd.DataFrame:
    """Convert the fit results from GPU-based fitting (Gaussian) into a
    data frame array of localizations.

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
        Whether ``theta`` came from Gpufit's maximum-likelihood
        estimator. If True, the localization precisions ``lpx`` / ``lpy``
        and the per-parameter uncertainties (``photons_unc``, ``bg_unc``,
        ``sx_unc``, ``sy_unc`` and, for the rotated model, ``angle_unc``)
        are the Poisson Cramer-Rao bound from the Fisher information of
        the fitted Gaussian model (:func:`_gauss_crlb`), matching the CPU
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
        # point-sampled Gaussian model Gpufit optimizes. Columns of ``crlb``
        # follow ``theta``: [photons, x, y, sx, sy, bg, (angle)].
        crlb = _gauss_crlb(theta, box, em, rotated=rotated)
        with np.errstate(invalid="ignore"):
            lpx = np.sqrt(crlb[:, 1])
            lpy = np.sqrt(crlb[:, 2])
    else:
        lpx = gausslq.localization_precision(
            theta[:, 0], theta[:, 3], theta[:, 4], theta[:, 5], em=em
        )
        lpy = gausslq.localization_precision(
            theta[:, 0], theta[:, 4], theta[:, 3], theta[:, 5], em=em
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
    locs = pd.DataFrame(columns)
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def _fit2d_gauss_gpu(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    rotated: bool = False,
    mle: bool = False,
    spherical: bool = False,
) -> pd.DataFrame:
    """Fit 2D Gaussians on the GPU using least squares or, if ``mle``,
    maximum likelihood estimation. If ``rotated``, a rotated elliptical
    Gaussian is fitted and the resulting localizations contain the
    fitted rotation angle (in degrees) in the column ``angle``. If
    ``spherical``, an isotropic Gaussian with a single width is fitted
    and the resulting ``sx`` and ``sy`` columns are identical. See
    ``fit_2D`` for more details."""
    theta, log_likelihood, iterations = fit_spots_gpufit(
        spots, rotated=rotated, mle=mle, spherical=spherical, return_stats=True
    )
    locs = locs_from_fits_gpufit(
        identifications,
        theta,
        box,
        em,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        spherical=spherical,
    )
    return locs


# ----------------------------------------------------------------------
# Cubic-spline PSF fitting (Gpufit SPLINE_2D / SPLINE_3D models)
#
# The spline models fit an experimentally measured PSF (a cubic-spline model
# built from a bead z-stack with Gpuspline) instead of a Gaussian. Unlike the
# Gaussian models, they need the spline coefficient table passed through
# Gpufit's ``user_info`` argument; everything else mirrors ``fit_spots_gpufit``.
# The 3D model recovers x, y, *z*, photons and background in a single fit, so
# no separate ``zfit`` step is needed. The coefficients live inside the
# calibration dict (see ``io.load_spline_calibration``).
# ----------------------------------------------------------------------


def _spline_model_id(model: str) -> int:
    """Map a spline calibration ``model`` string to a Gpufit ModelID."""
    if model == "spline-2d":
        return gf.ModelID.SPLINE_2D
    if model == "spline-3d":
        return gf.ModelID.SPLINE_3D
    if model == "spline-3d-multichannel":
        return gf.ModelID.SPLINE_3D_MULTICHANNEL
    raise ValueError(
        f"Unknown spline calibration model '{model}'. Expected one of "
        "'spline-2d', 'spline-3d', 'spline-3d-multichannel'."
    )


def _reorder_spline_coefficients_for_gpufit(
    coefficients: np.ndarray, model: str
) -> np.ndarray:
    """Reorder a spline coefficient table into the axis order Gpufit indexes.

    Gpuspline's ``spline_coefficients`` Python binding calls the C library with
    its spatial axes **reversed** (it passes ``data.shape`` as
    ``(size[-1], ..., size[0])``). The flat coefficient buffer it returns is
    therefore interval-major but in reversed spatial-axis order, while NumPy
    merely *labels* the array ``(4**d, *n_intervals)`` - the label does not
    match the physical memory order. Gpufit's ``spline_2d.cuh`` /
    ``spline_3d.cuh`` read coefficients in **forward** order
    (``coeff[(i*niy*niz + j*niz + k)*64 + oi*16+oj*4+ok]``, with ``i`` the fast
    pixel index = column = x). At the C level Gpuspline's own
    ``Spline{2,3}D::calculate_value`` uses the identical formula, so the *only*
    discrepancy is the binding's reversal.

    Passing the buffer through unchanged (as the reference pyGpufit examples
    and older Picasso did) makes Gpufit read every interval's coefficients from
    the wrong location: the fit then diverges even when seeded at the true
    minimum, while a CPU least-squares fit over the *same* coefficients (via
    ``gpuspline.spline_values``) recovers it. This undoes the reversal by
    transposing both the interval axes and the per-interval power axes back to
    forward ``(x, y[, z])`` order. Returns a flat ``float32`` array ready to
    drop into ``user_info``.
    """
    coeff = np.ascontiguousarray(coefficients, dtype=np.float32)
    if model == "spline-2d":
        # label (16, nix, niy); the raw buffer reshapes to (niy, nix, 4, 4)
        # = (y_interval, x_interval, y_power, x_power). Swap to forward (x, y).
        _, nix, niy = coeff.shape
        phys = coeff.ravel(order="C").reshape(niy, nix, 4, 4)
        return np.ascontiguousarray(phys.transpose(1, 0, 3, 2)).ravel(
            order="C"
        )
    if model == "spline-3d":
        # label (64, nix, niy, niz); the raw buffer reshapes to
        # (niz, niy, nix, 4, 4, 4) = (z, y, x, z_power, y_power, x_power).
        # Swap to forward (x, y, z).
        _, nix, niy, niz = coeff.shape
        phys = coeff.ravel(order="C").reshape(niz, niy, nix, 4, 4, 4)
        return np.ascontiguousarray(phys.transpose(2, 1, 0, 5, 4, 3)).ravel(
            order="C"
        )
    if model == "spline-3d-multichannel":
        # label (64, nix, niy, niz, n_channels); Gpufit expects the channel as
        # the OUTERMOST axis, each channel a plain forward 3D table. Reorder
        # each channel's block like the single-channel 3D case, concatenated
        # channel-major.
        _, nix, niy, niz, n_channels = coeff.shape
        blocks = []
        for c in range(n_channels):
            sub = np.ascontiguousarray(coeff[..., c])
            phys = sub.ravel(order="C").reshape(niz, niy, nix, 4, 4, 4)
            blocks.append(
                np.ascontiguousarray(phys.transpose(2, 1, 0, 5, 4, 3)).ravel(
                    order="C"
                )
            )
        return np.concatenate(blocks).astype(np.float32)
    raise ValueError(f"Unknown spline model '{model}'.")


def _pack_spline_user_info(calibration: dict) -> lib.FloatArray1D:
    """Pack a spline calibration into Gpufit's ``user_info`` blob:

    - 2D: ``[n_data_x, n_data_y, n_int_x, n_int_y, coefficients...]``
    - 3D: ``[n_data_x, n_data_y, n_data_z(=1), n_int_x, n_int_y, n_int_z,
      coefficients...]``
    - 3D multichannel: ``[n_channels, n_data_x, n_data_y, n_data_z(=1),
      n_int_x, n_int_y, n_int_z, coefficients...]``

    The coefficient block is the calibration's ``coefficients`` array reordered
    by :func:`_reorder_spline_coefficients_for_gpufit` from Gpuspline's
    (binding-reversed) layout into the forward axis order Gpufit's spline
    models index. This reorder is essential: without it Gpufit reads scrambled
    coefficients and the fit diverges (see that function's docstring). It is
    what the reference pyGpufit examples get wrong.

    IMPORTANT: the whole blob is cast to ``float32`` to match the
    single-precision (``REAL=float``) Gpufit build shipped with Picasso. The
    reference examples build ``user_info`` with ``np.hstack`` of mixed ints and
    float32 arrays, which NumPy promotes to float64 - that only works against a
    double-precision Gpufit build. Passing float64 to a single-precision build
    makes Gpufit misread the coefficients, so the explicit ``float32`` cast
    here is deliberate.
    """
    model = calibration["model"]
    coefficients = np.ascontiguousarray(
        calibration["coefficients"], dtype=np.float32
    )
    n_data = list(calibration["n_data"])
    n_intervals = list(calibration["n_intervals"])
    if model == "spline-2d":
        header = [n_data[0], n_data[1], n_intervals[0], n_intervals[1]]
    elif model == "spline-3d-multichannel":
        # The multichannel model prepends the number of channels; the
        # coefficient table is (64, n_int_x, n_int_y, n_int_z, n_channels).
        n_channels = int(calibration.get("n_channels", coefficients.shape[-1]))
        header = [
            n_channels,
            n_data[0],
            n_data[1],
            1,
            n_intervals[0],
            n_intervals[1],
            n_intervals[2],
        ]
    else:  # 3D
        # A single camera frame is fitted, so the number of data points in z
        # is 1; the spline recovers the continuous z position.
        header = [
            n_data[0],
            n_data[1],
            1,
            n_intervals[0],
            n_intervals[1],
            n_intervals[2],
        ]
    coefficient_block = _reorder_spline_coefficients_for_gpufit(
        coefficients, model
    )
    user_info = np.hstack(
        (
            np.asarray(header, dtype=np.float32),
            coefficient_block,
        )
    ).astype(np.float32)
    return user_info


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
    elif model == "spline-3d-multichannel":
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

    Parameter order matches the Gpufit spline models:
    ``[amplitude, x_shift, y_shift, offset]`` (2D) or
    ``[amplitude, x_shift, y_shift, z_shift, offset]`` (3D and 3D
    multichannel). The Gpufit spline model evaluates the spline at
    ``position = pixel_index - parameter`` (see ``spline_3d.cuh``), so:

    - x_shift/y_shift are the emitter's lateral offset from the (centered)
      template, i.e. 0 for a spot centered in its ROI.

    For the multichannel model ``spots`` is channel-stacked
    ``(n, box, box, n_channels)``; amplitude/offset are estimated across all
    channels."""
    model = calibration["model"]
    n_parameters = 4 if model == "spline-2d" else 5
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


def fit_spots_gpufit_spline(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    return_stats: bool = False,
) -> (
    lib.FloatArray2D
    | tuple[lib.FloatArray2D, lib.FloatArray1D | None, lib.FloatArray1D]
):
    """Fit multiple spots with a cubic-spline PSF model on the GPU.

    Mirrors ``fit_spots_gpufit`` but uses Gpufit's SPLINE_2D / SPLINE_3D model
    with the coefficient table from ``calibration`` passed via ``user_info``.

    Parameters
    ----------
    spots : lib.FloatArray3D
        Spot stack of shape ``(n_spots, box, box)``, or, for the multichannel
        model, channel-stacked ``(n_spots, box, box, n_channels)`` (see
        ``get_spots_multichannel``).
    calibration : dict
        Spline PSF calibration (see ``io.load_spline_calibration``). Must
        contain ``model``, ``coefficients``, ``n_data`` and ``n_intervals``.
    mle : bool, optional
        Use Gpufit's maximum likelihood (Poisson) estimator. Default False.
    return_stats : bool, optional
        Also return ``(log_likelihood, number_iterations)``. Default False.

    Returns
    -------
    parameters : lib.FloatArray2D
        Fitted parameters, columns ``[amplitude, x_shift, y_shift, offset]``
        (2D) or ``[amplitude, x_shift, y_shift, z_shift, offset]`` (3D).
    log_likelihood, number_iterations
        Only if ``return_stats`` (log_likelihood is None for least squares).
    """
    if not GPUFIT_INSTALLED:
        raise ImportError(
            "GPUfit could not be found, CUDA-capable GPU is required."
        )
    box = spots.shape[1]
    # Fit a smaller-than-calibration box against a centered crop of the
    # calibration (equal box: no-op; larger box: raises). This makes the fitter
    # self-sufficient - any caller may pass a smaller box without pre-cropping.
    # NOTE: locs_from_fits_spline must be given the SAME box so its
    # CRLB/reconstruction crop identically (it does so itself).
    calibration = crop_spline_calibration(calibration, box)
    model = calibration["model"]
    is_multichannel = model == "spline-3d-multichannel"
    if is_multichannel:
        # Multichannel spots are channel-stacked (n, box, box, n_channels);
        # the per-fit data is the channels concatenated pixel-major, exactly
        # as in the reference splinefit_3d_multi_channel example.
        if spots.ndim != 4:
            raise ValueError(
                "Multichannel spline fitting expects spots of shape "
                "(n_spots, box, box, n_channels)."
            )
        n_channels = int(
            calibration.get(
                "n_channels",
                np.asarray(calibration["coefficients"]).shape[-1],
            )
        )
        if spots.shape[3] != n_channels:
            raise ValueError(
                f"Spots have {spots.shape[3]} channels but the calibration "
                f"has {n_channels}."
            )
        n_points = box * box * n_channels
    else:
        n_points = box * box
    if mle:
        spots = np.maximum(spots, 0)
    initial_parameters = _initial_parameters_spline(spots, calibration)
    user_info = _pack_spline_user_info(calibration)
    model_id = _spline_model_id(model)
    estimator_id = gf.EstimatorID.MLE if mle else gf.EstimatorID.LSE

    if is_multichannel:
        # Gpufit's multichannel model is channel-MAJOR: each channel occupies a
        # contiguous box*box block (channel_id = point_index // (box*box), see
        # spline_3d_multichannel.cuh). Picasso's spots are (n, box, box,
        # n_channels) = channel-minor, so move the channel axis in front of the
        # pixels before flattening.
        fit_data = np.ascontiguousarray(spots.transpose(0, 3, 1, 2)).reshape(
            (len(spots), n_points)
        )
    else:
        fit_data = spots.reshape((len(spots), n_points))

    parameters, states, chi_squares, number_iterations, exec_time = gf.fit(
        fit_data,
        None,
        model_id,
        initial_parameters,
        tolerance=1e-2,
        max_number_iterations=20,
        estimator_id=estimator_id,
        user_info=user_info,
    )
    if return_stats:
        # As in fit_spots_gpufit: Gpufit's MLE chi-square is twice the
        # negative Poisson log-likelihood; for least squares it is a residual
        # sum of squares, so there is no likelihood to report.
        log_likelihood = -0.5 * chi_squares if mle else None
        return parameters, log_likelihood, number_iterations
    return parameters


def _spline_model_and_grad(
    coeff: np.ndarray,
    box: int,
    x_shift: np.ndarray,
    y_shift: np.ndarray,
    z_eval: np.ndarray | None,
) -> tuple:
    """Vectorized cubic-spline model image and its analytic spatial derivatives.

    Reimplements ``gpuspline.spline_values`` in NumPy for a batch of ``M``
    localizations and, because a tricubic/bicubic is a polynomial in the local
    fractional coordinate, also returns its exact spatial derivatives by
    differentiating that polynomial term-by-term (the closed form Gpufit uses on
    the GPU). This is the readable reference for the tricubic evaluation; the
    production CRLB uses the equivalent parallel numba kernels
    (:func:`_spline_infomats_3d` / :func:`_spline_infomats_2d`). It is kept for tests
    and to validate the coefficient layout against ``gpuspline.spline_values``.

    ``coeff`` is one channel's raw calibration coefficient table, ``(16, nix,
    niy)`` (2D) or ``(64, nix, niy, niz)`` (3D). Its flat C-order buffer is the
    Gpuspline-binding layout - it reshapes to ``(niy, nix, yp, xp)`` /
    ``(niz, niy, nix, zp, yp, xp)`` (see
    ``_reorder_spline_coefficients_for_gpufit``). The model pixel ``(i, j)``
    (column ``i`` = x, row ``j`` = y) samples the template at ``x = i -
    x_shift`` and ``y = j - y_shift`` ("position = pixel - parameter", see
    ``_initial_parameters_spline``); ``z_eval`` is the native z coordinate.

    Returns ``(phi, dphi_dx, dphi_dy, dphi_dz)``, each ``(M, box, box)`` indexed
    ``[loc, x-pixel, y-pixel]`` (``dphi_dz`` is None for 2D). Derivatives are
    w.r.t. the native coordinate; the shift derivative is their negative
    (irrelevant to the sign-invariant CRLB diagonal).
    """
    x_shift = np.asarray(x_shift, dtype=np.float32)
    y_shift = np.asarray(y_shift, dtype=np.float32)
    grid = np.arange(box, dtype=np.float32)
    xc = grid[None, :] - x_shift[:, None]  # (M, box) native x per pixel
    yc = grid[None, :] - y_shift[:, None]

    def _axis_basis(coords, n_int):
        # interval index (clamped, so out-of-range extrapolates like Gpuspline)
        # + value/derivative power bases [1,f,f^2,f^3] / [0,1,2f,3f^2].
        idx = np.clip(np.floor(coords), 0, n_int - 1).astype(np.intp)
        f = coords - idx
        ones, zeros = np.ones_like(f), np.zeros_like(f)
        p = np.stack([ones, f, f * f, f * f * f], axis=-1)
        dp = np.stack([zeros, ones, 2.0 * f, 3.0 * f * f], axis=-1)
        return idx, p, dp

    coeff = np.ascontiguousarray(coeff, dtype=np.float32)
    if z_eval is None:  # 2D bicubic
        _, nix, niy = coeff.shape
        c = coeff.reshape(-1).reshape(niy, nix, 4, 4)  # (yi, xi, yp, xp)
        xi, px, dpx = _axis_basis(xc, nix)
        yi, py, dpy = _axis_basis(yc, niy)
        cg = c[yi[:, None, :], xi[:, :, None]]  # (M, box_x, box_y, yp, xp)
        phi = np.einsum("mijyx,mjy,mix->mij", cg, py, px)
        dphi_dx = np.einsum("mijyx,mjy,mix->mij", cg, py, dpx)
        dphi_dy = np.einsum("mijyx,mjy,mix->mij", cg, dpy, px)
        return phi, dphi_dx, dphi_dy, None

    # 3D tricubic
    _, nix, niy, niz = coeff.shape
    c = coeff.reshape(-1).reshape(
        niz, niy, nix, 4, 4, 4
    )  # (zi,yi,xi,zp,yp,xp)
    xi, px, dpx = _axis_basis(xc, nix)
    yi, py, dpy = _axis_basis(yc, niy)
    zc = np.asarray(z_eval, dtype=np.float32)
    zidx = np.clip(np.floor(zc), 0, niz - 1).astype(np.intp)
    fz = zc - zidx
    pz = np.stack([np.ones_like(fz), fz, fz * fz, fz * fz * fz], axis=-1)
    dpz = np.stack(
        [np.zeros_like(fz), np.ones_like(fz), 2.0 * fz, 3.0 * fz * fz], axis=-1
    )
    # z is one slice per loc: contract the z power first, then gather per pixel.
    cz = c[zidx]  # (M, niy, nix, 4, 4, 4)
    c_val = np.einsum("mYXzyx,mz->mYXyx", cz, pz)  # (M, niy, nix, yp, xp)
    c_dz = np.einsum("mYXzyx,mz->mYXyx", cz, dpz)
    mm = np.arange(len(x_shift))[:, None, None]
    cg = c_val[mm, yi[:, None, :], xi[:, :, None]]  # (M, box_x, box_y, yp, xp)
    cg_dz = c_dz[mm, yi[:, None, :], xi[:, :, None]]
    phi = np.einsum("mijyx,mjy,mix->mij", cg, py, px)
    dphi_dx = np.einsum("mijyx,mjy,mix->mij", cg, py, dpx)
    dphi_dy = np.einsum("mijyx,mjy,mix->mij", cg, dpy, px)
    dphi_dz = np.einsum("mijyx,mjy,mix->mij", cg_dz, py, px)
    return phi, dphi_dx, dphi_dy, dphi_dz


@numba.njit(parallel=True, cache=True, fastmath=True)
def _spline_infomats_3d(
    coeff,
    box,
    amp,
    x_shift,
    y_shift,
    z_eval,
    offset,
    finite,
    mu_floor,
    mle,
    bread,
    meat,
):
    """Fill the per-localization information matrices of the 3D cubic-spline
    model. Parallel per-spot numba kernel; the readable reference is
    :func:`_spline_model_and_grad`. ``coeff`` is
    ``(n_channels, niz, niy, nix, 4, 4, 4)``. Non-converged rows are skipped
    (left as preset by the caller). Parameter order [x, y, z, amplitude, offset].

    With ``mle`` True, ``bread`` (n, 5, 5) receives the Poisson Fisher matrix
    ``I = Σ g gᵀ / μ`` (its inverse is the MLE Cramer-Rao bound) and ``meat``
    is left untouched (weight 0). With ``mle`` False, the two matrices form the
    unweighted-least-squares sandwich covariance ``J⁻¹ M J⁻¹``: ``bread`` = the
    Gauss-Newton normal matrix ``J = Σ g gᵀ`` and ``meat`` = ``M = Σ μ g gᵀ``
    (Poisson pixel variance ``σ² = μ``). ``g = ∂μ/∂θ``.
    """
    n_channels, niz, niy, nix = (
        coeff.shape[0],
        coeff.shape[1],
        (coeff.shape[2]),
        coeff.shape[3],
    )
    n_locs = amp.shape[0]
    for m in numba.prange(n_locs):
        if not finite[m]:
            continue
        a = amp[m]
        o = offset[m]
        # bread accumulators (f*): Fisher when mle else Gauss-Newton normal J.
        f00 = f01 = f02 = f03 = f04 = 0.0
        f11 = f12 = f13 = f14 = 0.0
        f22 = f23 = f24 = 0.0
        f33 = f34 = 0.0
        f44 = 0.0
        # meat accumulators (s*): least-squares sandwich M = Σ μ g gᵀ (0 if mle).
        s00 = s01 = s02 = s03 = s04 = 0.0
        s11 = s12 = s13 = s14 = 0.0
        s22 = s23 = s24 = 0.0
        s33 = s34 = 0.0
        s44 = 0.0
        # z basis (one slice per localization)
        zc = z_eval[m]
        zi = int(np.floor(zc))
        zi = 0 if zi < 0 else (niz - 1 if zi > niz - 1 else zi)
        fz = zc - zi
        pz0, pz1, pz2, pz3 = 1.0, fz, fz * fz, fz * fz * fz
        dz1, dz2, dz3 = 1.0, 2.0 * fz, 3.0 * fz * fz
        for ch in range(n_channels):
            for i in range(box):
                xco = i - x_shift[m]
                xi = int(np.floor(xco))
                xi = 0 if xi < 0 else (nix - 1 if xi > nix - 1 else xi)
                fx = xco - xi
                px0, px1, px2, px3 = 1.0, fx, fx * fx, fx * fx * fx
                dx1, dx2, dx3 = 1.0, 2.0 * fx, 3.0 * fx * fx
                for j in range(box):
                    yco = j - y_shift[m]
                    yi = int(np.floor(yco))
                    yi = 0 if yi < 0 else (niy - 1 if yi > niy - 1 else yi)
                    fy = yco - yi
                    py0, py1, py2, py3 = 1.0, fy, fy * fy, fy * fy * fy
                    dy1, dy2, dy3 = 1.0, 2.0 * fy, 3.0 * fy * fy
                    phi = gx = gy = gz = 0.0
                    for zp in range(4):
                        pzv = (
                            pz0
                            if zp == 0
                            else (
                                pz1 if zp == 1 else (pz2 if zp == 2 else pz3)
                            )
                        )
                        dzv = (
                            0.0
                            if zp == 0
                            else (
                                dz1 if zp == 1 else (dz2 if zp == 2 else dz3)
                            )
                        )
                        for yp in range(4):
                            pyv = (
                                py0
                                if yp == 0
                                else (
                                    py1
                                    if yp == 1
                                    else (py2 if yp == 2 else py3)
                                )
                            )
                            dyv = (
                                0.0
                                if yp == 0
                                else (
                                    dy1
                                    if yp == 1
                                    else (dy2 if yp == 2 else dy3)
                                )
                            )
                            for xp in range(4):
                                cf = coeff[ch, zi, yi, xi, zp, yp, xp]
                                pxv = (
                                    px0
                                    if xp == 0
                                    else (
                                        px1
                                        if xp == 1
                                        else (px2 if xp == 2 else px3)
                                    )
                                )
                                dxv = (
                                    0.0
                                    if xp == 0
                                    else (
                                        dx1
                                        if xp == 1
                                        else (dx2 if xp == 2 else dx3)
                                    )
                                )
                                phi += cf * pzv * pyv * pxv
                                gx += cf * pzv * pyv * dxv
                                gy += cf * pzv * dyv * pxv
                                gz += cf * dzv * pyv * pxv
                    mu = o + a * phi
                    if mu < mu_floor:
                        mu = mu_floor
                    # bread weight wa (1/μ Fisher, else 1) and meat weight wb
                    # (μ for the least-squares sandwich, else unused).
                    if mle:
                        wa = 1.0 / mu
                        wb = 0.0
                    else:
                        wa = 1.0
                        wb = mu
                    # d(mu)/d(param); the CRLB diagonal is sign-invariant per
                    # parameter, so native-coordinate vs shift sign is irrelevant.
                    d0, d1, d2, d3 = a * gx, a * gy, a * gz, phi
                    f00 += d0 * d0 * wa
                    f01 += d0 * d1 * wa
                    f02 += d0 * d2 * wa
                    f03 += d0 * d3 * wa
                    f04 += d0 * wa
                    f11 += d1 * d1 * wa
                    f12 += d1 * d2 * wa
                    f13 += d1 * d3 * wa
                    f14 += d1 * wa
                    f22 += d2 * d2 * wa
                    f23 += d2 * d3 * wa
                    f24 += d2 * wa
                    f33 += d3 * d3 * wa
                    f34 += d3 * wa
                    f44 += wa
                    s00 += d0 * d0 * wb
                    s01 += d0 * d1 * wb
                    s02 += d0 * d2 * wb
                    s03 += d0 * d3 * wb
                    s04 += d0 * wb
                    s11 += d1 * d1 * wb
                    s12 += d1 * d2 * wb
                    s13 += d1 * d3 * wb
                    s14 += d1 * wb
                    s22 += d2 * d2 * wb
                    s23 += d2 * d3 * wb
                    s24 += d2 * wb
                    s33 += d3 * d3 * wb
                    s34 += d3 * wb
                    s44 += wb
        bread[m, 0, 0] = f00
        bread[m, 0, 1] = bread[m, 1, 0] = f01
        bread[m, 0, 2] = bread[m, 2, 0] = f02
        bread[m, 0, 3] = bread[m, 3, 0] = f03
        bread[m, 0, 4] = bread[m, 4, 0] = f04
        bread[m, 1, 1] = f11
        bread[m, 1, 2] = bread[m, 2, 1] = f12
        bread[m, 1, 3] = bread[m, 3, 1] = f13
        bread[m, 1, 4] = bread[m, 4, 1] = f14
        bread[m, 2, 2] = f22
        bread[m, 2, 3] = bread[m, 3, 2] = f23
        bread[m, 2, 4] = bread[m, 4, 2] = f24
        bread[m, 3, 3] = f33
        bread[m, 3, 4] = bread[m, 4, 3] = f34
        bread[m, 4, 4] = f44
        if not mle:
            meat[m, 0, 0] = s00
            meat[m, 0, 1] = meat[m, 1, 0] = s01
            meat[m, 0, 2] = meat[m, 2, 0] = s02
            meat[m, 0, 3] = meat[m, 3, 0] = s03
            meat[m, 0, 4] = meat[m, 4, 0] = s04
            meat[m, 1, 1] = s11
            meat[m, 1, 2] = meat[m, 2, 1] = s12
            meat[m, 1, 3] = meat[m, 3, 1] = s13
            meat[m, 1, 4] = meat[m, 4, 1] = s14
            meat[m, 2, 2] = s22
            meat[m, 2, 3] = meat[m, 3, 2] = s23
            meat[m, 2, 4] = meat[m, 4, 2] = s24
            meat[m, 3, 3] = s33
            meat[m, 3, 4] = meat[m, 4, 3] = s34
            meat[m, 4, 4] = s44


@numba.njit(parallel=True, cache=True, fastmath=True)
def _spline_infomats_2d(
    coeff,
    box,
    amp,
    x_shift,
    y_shift,
    offset,
    finite,
    mu_floor,
    mle,
    bread,
    meat,
):
    """2D analogue of :func:`_spline_infomats_3d`. ``coeff`` is
    ``(n_channels, niy, nix, 4, 4)``; parameter order [x, y, amplitude, offset].
    """
    n_channels, niy, nix = coeff.shape[0], coeff.shape[1], coeff.shape[2]
    n_locs = amp.shape[0]
    for m in numba.prange(n_locs):
        if not finite[m]:
            continue
        a = amp[m]
        o = offset[m]
        # bread accumulators (f*): Fisher when mle else Gauss-Newton normal J.
        f00 = f01 = f02 = f03 = 0.0
        f11 = f12 = f13 = 0.0
        f22 = f23 = 0.0
        f33 = 0.0
        # meat accumulators (s*): least-squares sandwich M = Σ μ g gᵀ (0 if mle).
        s00 = s01 = s02 = s03 = 0.0
        s11 = s12 = s13 = 0.0
        s22 = s23 = 0.0
        s33 = 0.0
        for ch in range(n_channels):
            for i in range(box):
                xco = i - x_shift[m]
                xi = int(np.floor(xco))
                xi = 0 if xi < 0 else (nix - 1 if xi > nix - 1 else xi)
                fx = xco - xi
                px0, px1, px2, px3 = 1.0, fx, fx * fx, fx * fx * fx
                dx1, dx2, dx3 = 1.0, 2.0 * fx, 3.0 * fx * fx
                for j in range(box):
                    yco = j - y_shift[m]
                    yi = int(np.floor(yco))
                    yi = 0 if yi < 0 else (niy - 1 if yi > niy - 1 else yi)
                    fy = yco - yi
                    py0, py1, py2, py3 = 1.0, fy, fy * fy, fy * fy * fy
                    dy1, dy2, dy3 = 1.0, 2.0 * fy, 3.0 * fy * fy
                    phi = gx = gy = 0.0
                    for yp in range(4):
                        pyv = (
                            py0
                            if yp == 0
                            else (
                                py1 if yp == 1 else (py2 if yp == 2 else py3)
                            )
                        )
                        dyv = (
                            0.0
                            if yp == 0
                            else (
                                dy1 if yp == 1 else (dy2 if yp == 2 else dy3)
                            )
                        )
                        for xp in range(4):
                            cf = coeff[ch, yi, xi, yp, xp]
                            pxv = (
                                px0
                                if xp == 0
                                else (
                                    px1
                                    if xp == 1
                                    else (px2 if xp == 2 else px3)
                                )
                            )
                            dxv = (
                                0.0
                                if xp == 0
                                else (
                                    dx1
                                    if xp == 1
                                    else (dx2 if xp == 2 else dx3)
                                )
                            )
                            phi += cf * pyv * pxv
                            gx += cf * pyv * dxv
                            gy += cf * dyv * pxv
                    mu = o + a * phi
                    if mu < mu_floor:
                        mu = mu_floor
                    if mle:
                        wa = 1.0 / mu
                        wb = 0.0
                    else:
                        wa = 1.0
                        wb = mu
                    d0, d1, d2 = a * gx, a * gy, phi
                    f00 += d0 * d0 * wa
                    f01 += d0 * d1 * wa
                    f02 += d0 * d2 * wa
                    f03 += d0 * wa
                    f11 += d1 * d1 * wa
                    f12 += d1 * d2 * wa
                    f13 += d1 * wa
                    f22 += d2 * d2 * wa
                    f23 += d2 * wa
                    f33 += wa
                    s00 += d0 * d0 * wb
                    s01 += d0 * d1 * wb
                    s02 += d0 * d2 * wb
                    s03 += d0 * wb
                    s11 += d1 * d1 * wb
                    s12 += d1 * d2 * wb
                    s13 += d1 * wb
                    s22 += d2 * d2 * wb
                    s23 += d2 * wb
                    s33 += wb
        bread[m, 0, 0] = f00
        bread[m, 0, 1] = bread[m, 1, 0] = f01
        bread[m, 0, 2] = bread[m, 2, 0] = f02
        bread[m, 0, 3] = bread[m, 3, 0] = f03
        bread[m, 1, 1] = f11
        bread[m, 1, 2] = bread[m, 2, 1] = f12
        bread[m, 1, 3] = bread[m, 3, 1] = f13
        bread[m, 2, 2] = f22
        bread[m, 2, 3] = bread[m, 3, 2] = f23
        bread[m, 3, 3] = f33
        if not mle:
            meat[m, 0, 0] = s00
            meat[m, 0, 1] = meat[m, 1, 0] = s01
            meat[m, 0, 2] = meat[m, 2, 0] = s02
            meat[m, 0, 3] = meat[m, 3, 0] = s03
            meat[m, 1, 1] = s11
            meat[m, 1, 2] = meat[m, 2, 1] = s12
            meat[m, 1, 3] = meat[m, 3, 1] = s13
            meat[m, 2, 2] = s22
            meat[m, 2, 3] = meat[m, 3, 2] = s23
            meat[m, 3, 3] = s33


def _spline_coeff_reshaped(calibration: dict) -> np.ndarray:
    """Raw calibration coefficients as ``(n_channels, niz, niy, nix, 4, 4, 4)``
    (3D) or ``(n_channels, niy, nix, 4, 4)`` (2D), float64 for the numba kernels.
    The flat C-order buffer is the Gpuspline-binding layout (see
    :func:`_spline_model_and_grad`)."""
    model = calibration["model"]
    coeff = np.ascontiguousarray(calibration["coefficients"], dtype=np.float64)
    if model == "spline-3d-multichannel":
        _, nix, niy, niz, n_channels = coeff.shape
        return np.stack(
            [
                np.ascontiguousarray(coeff[..., c]).reshape(
                    niz, niy, nix, 4, 4, 4
                )
                for c in range(n_channels)
            ]
        )
    if model == "spline-2d":
        _, nix, niy = coeff.shape
        return coeff.reshape(niy, nix, 4, 4)[None]
    _, nix, niy, niz = coeff.shape
    return coeff.reshape(niz, niy, nix, 4, 4, 4)[None]


def _spline_crlb(
    theta: lib.FloatArray2D,
    calibration: dict,
    box: int,
    mle: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> lib.FloatArray2D:
    """Parameter-variance estimates for spline-fitted localizations.

    Evaluates the estimator covariance at the fitted parameters, using the
    cubic-spline PSF model ``mu = offset + amplitude * Phi`` and its analytic
    spatial derivatives.

    With ``mle`` True the result is the Cramer-Rao lower bound: the diagonal of
    the inverse Poisson Fisher-information matrix ``I = Σ g gᵀ / μ`` (``g =
    ∂μ/∂θ``), which an efficient maximum-likelihood estimator attains. Mirrors
    ``gaussmle._mlefit_sigmaxy_crlb``.

    With ``mle`` False the result is the covariance of the *unweighted*
    least-squares estimator (Gpufit's ``spline-gpu`` LSE mode), the Huber
    sandwich ``J⁻¹ M J⁻¹`` with normal matrix ``J = Σ g gᵀ`` and, for Poisson
    pixel noise (``σ² = μ``), meat ``M = Σ μ g gᵀ``. This is ≥ the Cramer-Rao
    bound elementwise (least squares is not efficient for Poisson data), so it
    is the honest precision for LSQ fits rather than the optimistic MLE floor.

    Parameters
    ----------
    theta : lib.FloatArray2D
        Fitted parameters, columns ``[amplitude, x_shift, y_shift, offset]``
        (2D) or ``[amplitude, x_shift, y_shift, z_shift, offset]`` (3D and 3D
        multichannel). Photon units (spots are gain-converted before fitting),
        so ``mu`` is an expected photon count and the Poisson noise model
        applies directly.
    calibration : dict
        The spline PSF calibration (see ``io.load_spline_calibration``).
    box : int
        Fit box side length (camera pixels).
    mle : bool, optional
        If True (default), return the Poisson Cramer-Rao bound (for
        maximum-likelihood fits). If False, return the least-squares sandwich
        covariance (for ``spline-gpu`` least-squares fits).
    progress_callback : callable, "console" or None, optional
        Progress over localization chunks. ``"console"`` shows a tqdm bar; a
        callable is invoked with the cumulative number of localizations done.

    Returns
    -------
    crlb : lib.FloatArray2D
        ``(n_locs, n_params)`` array of parameter variances (float64) in native
        parameter order ``[x_shift, y_shift, (z_shift,) amplitude, offset]``
        (pixels, (z-slices,) photons, photons). Non-converged fits and
        numerically singular problems are NaN.
    """
    model = calibration["model"]
    is_3d = model != "spline-2d"
    n_params = 5 if is_3d else 4

    theta = np.asarray(theta, dtype=np.float64)
    n_locs = len(theta)

    coeff = _spline_coeff_reshaped(calibration)
    amplitude = np.ascontiguousarray(theta[:, 0])
    x_shift = np.ascontiguousarray(theta[:, 1])
    y_shift = np.ascontiguousarray(theta[:, 2])
    offset = np.ascontiguousarray(theta[:, -1])
    # Native z sampling coordinate = -z_shift (single-frame Gpufit
    # "position = pixel_index - parameter", pixel_index_z = 0). The kernel
    # clamps the z-interval, so no pre-clamping is needed here.
    z_eval = np.ascontiguousarray(-theta[:, 3]) if is_3d else None
    finite = np.isfinite(theta).all(axis=1)

    # Per-localization information matrices (float64). ``bread`` is the Fisher
    # matrix (mle) or the least-squares normal matrix J; non-converged rows stay
    # the identity so the batched pinv is well-defined (they become NaN below).
    # ``meat`` M is only filled for the least-squares sandwich (stays 0 for mle).
    bread = np.tile(np.eye(n_params), (max(n_locs, 1), 1, 1))
    meat = np.zeros((max(n_locs, 1), n_params, n_params))

    use_tqdm = progress_callback == "console"
    do_callback = callable(progress_callback)
    # One kernel call spans all localizations; chunk only to report progress.
    chunk = (
        max(1, min(n_locs, 100_000)) if (use_tqdm or do_callback) else n_locs
    )
    starts = range(0, n_locs, chunk) if n_locs else []
    if use_tqdm:
        starts = tqdm(starts, desc="Computing spline CRLB")

    for start in starts:
        stop = min(start + chunk, n_locs)
        sl = slice(start, stop)
        if is_3d:
            _spline_infomats_3d(
                coeff,
                box,
                amplitude[sl],
                x_shift[sl],
                y_shift[sl],
                z_eval[sl],
                offset[sl],
                finite[sl],
                _SPLINE_CRLB_MU_FLOOR,
                mle,
                bread[sl],
                meat[sl],
            )
        else:
            _spline_infomats_2d(
                coeff,
                box,
                amplitude[sl],
                x_shift[sl],
                y_shift[sl],
                offset[sl],
                finite[sl],
                _SPLINE_CRLB_MU_FLOOR,
                mle,
                bread[sl],
                meat[sl],
            )
        if do_callback:
            progress_callback(stop)

    with np.errstate(invalid="ignore", divide="ignore"):
        bread_inv = np.linalg.pinv(bread)
        if mle:
            # cov = I⁻¹ (Cramer-Rao bound); bread is the Fisher matrix.
            cov = bread_inv
        else:
            # cov = J⁻¹ M J⁻¹ (unweighted-least-squares sandwich).
            cov = bread_inv @ meat @ bread_inv
        crlb = np.diagonal(cov, axis1=1, axis2=2).copy()
    crlb = crlb[:n_locs]
    crlb[~finite] = np.nan
    crlb = np.where(crlb > 0.0, crlb, np.nan)
    return crlb


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
) -> pd.DataFrame:
    """Convert spline fit results into a localizations data frame.

    ``theta`` columns are ``[amplitude, x_shift, y_shift, offset]`` (2D) or
    ``[amplitude, x_shift, y_shift, z_shift, offset]`` (3D). Localization
    precisions (``lpx``, ``lpy``, ``lpz``) and the ``photons`` / ``bg``
    uncertainties come from :func:`_spline_crlb`: the Poisson Cramer-Rao bound
    for maximum-likelihood fits (``mle`` True) or the least-squares sandwich
    covariance for ``spline-gpu`` least-squares fits (``mle`` False). ``mle``
    must match the estimator that produced ``theta``. ``progress_callback`` is
    forwarded to :func:`_spline_crlb` (that per-localization loop is the slow
    part)."""
    calibration = crop_spline_calibration(calibration, box)
    model = calibration["model"]
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

    photon_scale = float(calibration.get("photon_scale", 1.0))
    photons = amplitude * photon_scale

    # CRLB / LSQ variances in native order [x_shift, y_shift, (z_shift,)
    # amplitude, offset]; amplitude/offset are always the last two columns.
    crlb = _spline_crlb(
        theta, calibration, box, mle=mle, progress_callback=progress_callback
    )
    with np.errstate(invalid="ignore"):
        lpx = np.sqrt(crlb[:, 0]) / oversampling
        lpy = np.sqrt(crlb[:, 1]) / oversampling
        photons_unc = np.sqrt(crlb[:, -2]) * photon_scale
        bg_unc = np.sqrt(crlb[:, -1])

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
    locs = pd.DataFrame(columns)
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


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
) -> pd.DataFrame:
    """Fit an experimentally measured cubic-spline PSF on the GPU. For a 3D
    calibration the localizations contain the fitted ``z`` directly. See
    ``fit2D`` for more details. ``progress_callback`` tracks the per-spot CRLB
    computation in ``locs_from_fits_spline``."""
    theta, log_likelihood, iterations = fit_spots_gpufit_spline(
        spots, calibration, mle=mle, return_stats=True
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
    )
    return locs


# ----------------------------------------------------------------------
# Multichannel cubic-spline PSF fitting (Gpufit SPLINE_3D_MULTICHANNEL)
#
# Several spatially-registered channels (separate movies, e.g. biplane /
# 4Pi microscopy) are fit simultaneously with shared x, y, z. A detection in
# the reference channel is mapped into every channel via a per-channel affine
# transform (stored in the calibration), the box ROIs are cut from each
# channel and stacked, and the stack is fitted against the per-channel spline
# coefficients. There is no affine-transform machinery elsewhere in Picasso,
# so the small least-squares helpers below are self-contained.
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


def get_spots_multichannel(
    movies: list,
    identifications: pd.DataFrame,
    box: int,
    camera_infos: list[dict],
    transforms: list,
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
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

    Returns
    -------
    spots : np.ndarray
        Array of shape ``(n_spots, box, box, n_channels)`` in photon units.
    """
    n_channels = len(movies)
    if not (len(camera_infos) == len(transforms) == n_channels):
        raise ValueError(
            "movies, camera_infos and transforms must have the same length "
            "(one per channel)."
        )
    ref_xy = np.column_stack(
        [
            np.asarray(identifications["x"], dtype=np.float64),
            np.asarray(identifications["y"], dtype=np.float64),
        ]
    )
    channel_spots = []
    for c in range(n_channels):
        if c == 0:
            ids_c = identifications
        else:
            mapped = apply_affine_transform(ref_xy, transforms[c])
            ids_c = identifications.copy()
            # get_spots/_cut_spots cut an integer-pixel box; the fixed
            # fractional per-channel offset is absorbed consistently because
            # the calibration is built with the same extractor.
            ids_c["x"] = np.rint(mapped[:, 0]).astype(np.int64)
            ids_c["y"] = np.rint(mapped[:, 1]).astype(np.int64)
        spots_c = get_spots(
            movies[c],
            ids_c,
            box,
            camera_infos[c],
            progress_callback=progress_callback if c == 0 else None,
        )
        channel_spots.append(spots_c)
    return np.stack(channel_spots, axis=-1)


def fit_spline_multichannel(
    movies: list,
    camera_infos: list[dict],
    identifications: pd.DataFrame,
    box: int,
    calibration: dict,
    mle: bool = False,
    progress_callback: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """Fit a multichannel cubic-spline PSF across several registered channels.

    Ties ``get_spots_multichannel`` (extraction via the calibration's stored
    ``channel_transforms``) to ``fit_spots_gpufit_spline`` and
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
        Use Gpufit's maximum-likelihood estimator. Default False.
    """
    if calibration.get("model") != "spline-3d-multichannel":
        raise ValueError(
            "fit_spline_multichannel requires a 'spline-3d-multichannel' "
            "calibration."
        )
    transforms = calibration["channel_transforms"]
    if len(movies) != len(transforms):
        raise ValueError(
            f"Got {len(movies)} channels but the calibration has "
            f"{len(transforms)} channel transforms."
        )
    spots = get_spots_multichannel(
        movies,
        identifications,
        box,
        camera_infos,
        transforms,
        progress_callback=progress_callback,
    )
    theta, log_likelihood, iterations = fit_spots_gpufit_spline(
        spots, calibration, mle=mle, return_stats=True
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
    )


def _fit2d_gaussmle(
    spots,
    identifications: pd.DataFrame,
    box: int,
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    spherical: bool = False,
) -> pd.DataFrame | None:
    """Fit 2D Gaussians using MLE fitting. If ``spherical``, an isotropic
    Gaussian with a single width is fitted (``sx == sy``) and the
    ellipticity column is omitted as it is always 0. See ``fit_2D`` for
    more details."""
    N = len(identifications)
    # MLE API is a bit different (at least for now) so we cannot use
    # _process_fitting_futures here
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(total=N, desc="Fitting", unit="spot")
    if multiprocess:
        curr, thetas, CRLBs, llhoods, iterations = gaussmle.gaussmle_async(
            spots, eps, max_it, method=mle_method
        )
        last = 0
        while curr[0] < N:
            # abort check
            if callable(abort_callback) and abort_callback():
                if use_tqdm:
                    iter_range.close()
                return

            # progress update
            if use_tqdm:
                iter_range.update(curr[0] - last)
                last = curr[0]
            elif callable(progress_callback):
                progress_callback(curr[0])
            time.sleep(0.2)
        if use_tqdm:
            iter_range.update(N - last)
            iter_range.close()
    else:
        thetas, CRLBs, llhoods, iterations = gaussmle.gaussmle(
            spots, eps, max_it, mle_method, progress_callback
        )
    locs = gaussmle.locs_from_fits(
        identifications,
        thetas,
        CRLBs,
        llhoods,
        iterations,
        box,
        spherical=spherical,
    )
    return locs


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
    multiprocessing. See ``_fit2d_gausslq``, ``_fit2d_avg``."""
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
    movie: lib.IntArray3D,
    camera_info: dict,
    parameters: dict,
    *,
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
        "spline-gpu",
        "spline-mle-gpu",
        "avg",
    ] = "gausslq",
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    spline_calibration: dict | None = None,
    threaded: bool = True,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    return_info: bool = True,  # TODO: remove in v0.12.0
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    """Localize (i.e., identify and fit) spots in 2D in a movie using
    the specified parameters.

    Since v0.10.0: support for frame bounds and ROI for identification +
    all fitting methods.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    parameters : dict
        A dictionary containing localization parameters, including:
        - `Min. Net Gradient`: Minimum net gradient for spot
          identification.
        - `Box Size`: Size of the box to cut out around each spot.
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
        Which 2D fitting algorithm to use, see ``fit2D``. Default is
        "gausslq".
    eps : float, optional
        The convergence criterion for MLE fitting. Default is 0.001.
    max_it : int, optional
        The maximum number of iterations for MLE fitting. Default is
        100.
    mle_method : Literal["sigma", "sigmaxy"], optional
        The method used for MLE fitting. Default is "sigmaxy".
    identification_progress_callback : callable or "console" or None
        A callback for progress updates during identification. If
        "console", progress will be printed to the console. If None,
        progress is not reported. Default is None.
    fit_progress_callback : callable or "console" or None
        A callback for progress updates during fitting. If "console",
        progress will be printed to the console. If None, progress is
        not reported. Default is None.
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

    # Use empty list as default for movie_info
    if movie_info is None:
        movie_info = []

    # Identify spots
    identifications, identify_info = identify(
        movie,
        parameters["Min. Net Gradient"],
        parameters["Box Size"],
        roi=roi,
        frame_bounds=frame_bounds,
        threaded=threaded,
        progress_callback=identification_progress_callback,
    )

    # Fit spots
    locs, fit_info = fit2D(
        movie=movie,
        movie_info=movie_info,
        camera_info=camera_info,
        identifications=identifications,
        box=parameters["Box Size"],
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        mle_method=mle_method,
        spline_calibration=spline_calibration,
        multiprocess=threaded,
        progress_callback=fit_progress_callback,
    )
    info = movie_info + [identify_info] + [fit_info]
    if return_info:
        return locs, info
    return locs


def localize_3D(
    movie: lib.IntArray3D,
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
        "spline-gpu",
        "spline-mle-gpu",
    ] = "gausslq",
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    spline_calibration: dict | None = None,
    multiprocess: bool = True,
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

    For the Gaussian ``fitting_method`` values this first runs 2D
    localizations, followed by z position fitting assuming astigmatism, see
    Huang, et al. Science, 2008 (``calibration_3d`` holds the astigmatism
    polynomials). For ``"spline-gpu"`` a cubic-spline PSF fit recovers z
    directly in the 2D fit, so no separate z-fitting step is run and
    ``spline_calibration`` is used instead of ``calibration_3d``.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
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
        Which 2D fitting algorithm to use, see ``fit2D``. "avg" is not
        supported since z fitting requires the fitted Gaussian sigmas.
        Note that the rotated elliptical Gaussian methods report sx and
        sy along the rotated principal axes, whereas the astigmatism
        calibration assumes the camera axes, so use them for z fitting
        with care. The spherical Gaussian methods constrain sx == sy, so
        they carry no astigmatism and are unsuitable for z fitting.
        Default is "gausslq".
    eps : float, optional
        The convergence criterion for CPU MLE fitting. Ignored for
        other methods (GPU fitting uses Gpufit's own convergence
        settings). Default is 0.001.
    max_it : int, optional
        The maximum number of iterations for CPU MLE fitting. Ignored
        for other methods. Default is 100.
    mle_method : Literal["sigma", "sigmaxy"], optional
        The method used for CPU MLE fitting (impose same sigma in x and
        y or not, respectively). Default is "sigmaxy".
    multiprocess: bool, optional
        Whether or not to use multiprocessing. Ignored for GPU fitting.
        Default is True.
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
    assert isinstance(
        movie, (np.ndarray, io.ND2Movie)
    ), "movie must be a numpy array or ND2Movie"
    assert isinstance(movie_info, list), "movie_info must be a list"
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert (
        isinstance(box, int) and box > 0 and box % 2 == 1
    ), "box must be a positive odd integer"
    assert isinstance(minimum_ng, (int, float)), "minimum_ng must be a number"
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
    assert (
        isinstance(eps, (int, float)) and eps > 0
    ), "eps must be a positive number"
    assert (
        isinstance(max_it, int) and max_it > 0
    ), "max_it must be a positive integer"
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
        multiprocess=multiprocess,
        identification_progress_callback=identification_progress_callback,
        fit_progress_callback=fit_progress_callback,
        fit_z_progress_callback=fit_z_progress_callback,
    )


def _localize_3D(
    movie: lib.IntArray3D,
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
        "spline-gpu",
        "spline-mle-gpu",
    ] = "gausslq",
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    spline_calibration: dict | None = None,
    multiprocess: bool = True,
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
    """Internal function for `localize_3D`, assumes validated inputs."""
    locs, info = localize(
        movie=movie,
        camera_info=camera_info,
        parameters={
            "Min. Net Gradient": minimum_ng,
            "Box Size": box,
        },
        roi=roi,
        frame_bounds=frame_bounds,
        movie_info=movie_info,
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        mle_method=mle_method,
        spline_calibration=spline_calibration,
        threaded=multiprocess,
        identification_progress_callback=identification_progress_callback,
        fit_progress_callback=fit_progress_callback,
        return_info=True,  # TODO: remove in v0.12.0
    )
    if fitting_method.startswith("spline"):
        # The 3D cubic-spline fit already produced the z column directly, so
        # there is no separate astigmatism z-fitting step to run.
        return locs, info
    # zfit only knows gausslq/gaussmle; map the GPU/rotated codes to the
    # corresponding CPU noise model
    fitting_method_3d = (
        "gaussmle" if fitting_method.startswith("gaussmle") else "gausslq"
    )
    locs, info = zfit.zfit(
        locs=locs,
        info=info,
        calibration=calibration_3d,
        fitting_method=fitting_method_3d,
        filter=0,
        multiprocess=multiprocess,
        progress_callback=fit_z_progress_callback,
    )
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
