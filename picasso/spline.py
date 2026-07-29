"""
picasso.spline
~~~~~~~~~~~~~~

Generate cubic-spline PSF calibrations from a bead z-stack.

A calibration bead sample (e.g., fluorescent/gold beads) is imaged while the
stage is scanned through z. This module averages the beads into a clean,
3D-registered PSF volume, normalizes it, and computes cubic-spline
coefficients with Gpuspline. The resulting calibration (coefficients +
metadata) is saved via ``picasso.io.save_spline_calibration`` and later fitted
per spot with Gpufit's SPLINE_2D / SPLINE_3D models (see
``picasso.localize.fit_spots_gpufit_spline``).

Note: Gpuspline is a CPU library (no GPU/CUDA); only the subsequent fitting
step needs a CUDA GPU. Building a calibration therefore only needs Gpuspline,
which is exposed as ``picasso.localize.gs`` and gated by
``picasso.localize.GPUSPLINE_INSTALLED``.

The frame -> z-step binning mirrors ``picasso.zfit.calibrate_z`` so that
multiple fields of view per z position (``frame_order``, ``frames_per_step``,
``frame_bounds``) are supported.

Bead alignment and preparation is done according to Li et al. (2018);
spline fitting follows Gpufit, see References.

The multichannel calibration (``calibrate_spline_multichannel`` /
``calibrate_spline_split_fov``, used for biplane, split-FOV and ratiometric
multicolor data) follows the global-fitting approach of globLoc (Li et al.,
2022): one PSF per channel, the channels registered to a reference by an
affine transform, and all channels fitted jointly with shared (linked)
coordinates.

References
----------
- Li, Y., Mund, M., Hoess, P., Deschamps, J., Matti, U., Nijmeijer, B.,
  Sabinina, V. J., Ellenberg, J., Schoen, I. & Ries, J. "Real-time 3D
  single-molecule localization using experimental point spread functions."
  Nature Methods 15, 367-369 (2018).
- Li, Y., Shi, W., Liu, S., Cavka, I., Wu, Y.-L., Matti, U., Wu, D.,
  Koehler, S. & Ries, J. "Global fitting for high-accuracy multi-channel
  single-molecule localization." Nature Communications 13, 3133 (2022).
  DOI: 10.1038/s41467-022-30719-4. (globLoc)
- Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
  "Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
  Scientific Reports 7, 15722 (2017).

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
from itertools import combinations
from typing import Callable, Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import shift as _ndi_shift, zoom as _ndi_zoom
from scipy.spatial import KDTree

from . import io, lib, gausslq, localize, __version__


def _step_of_frame(
    n_frames: int,
    d: float,
    frames_per_step: int,
    frame_order: Literal["fov", "z"],
    frame_bounds: tuple[int, int] | list | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map every movie frame to the index of its z (stage) position.

    Identical binning to ``picasso.zfit.calibrate_z`` (so multi-FOV z-stacks
    behave the same). Returns ``(step_of_frame, z_of_step, step_range)`` where
    ``step_of_frame[f]`` is the z-step of frame ``f`` (or -1 if the frame is
    ignored), ``z_of_step`` the z position (nm) of each step, and
    ``step_range`` the sorted steps that actually receive at least one frame.
    """
    frames_per_step = max(1, int(frames_per_step))
    n_steps = n_frames // frames_per_step
    if n_steps < 1:
        raise ValueError(
            "Number of frames per step is larger than the number of frames "
            "in the movie."
        )

    all_frames = np.arange(n_frames)
    valid = all_frames < n_steps * frames_per_step
    if frame_order == "z":
        step_of_frame = all_frames % n_steps
    else:  # "fov": consecutive frames share the same z position
        step_of_frame = all_frames // frames_per_step
    step_of_frame = np.where(valid, step_of_frame, -1)

    # z position of each step; negative so a bottom-to-top scan starts positive
    z_span = (n_steps - 1) * d
    z_of_step = -(np.arange(n_steps) * d - z_span / 2)

    if frame_bounds is not None:
        segments = lib.normalize_frame_bounds(frame_bounds, n_frames - 1)
        in_bounds = np.zeros(n_frames, dtype=bool)
        for frame_min, frame_max in segments:
            in_bounds |= (all_frames >= frame_min) & (all_frames <= frame_max)
        step_of_frame = np.where(in_bounds, step_of_frame, -1)

    step_range = np.unique(step_of_frame[step_of_frame >= 0])
    return step_of_frame, z_of_step, step_range


def _fov_of_frame(
    n_frames: int,
    frames_per_step: int,
    frame_order: Literal["fov", "z"],
) -> np.ndarray:
    """Map every movie frame to the index of its field of view (FOV).

    When several frames are acquired per z (stage) position. Returns
    ``fov_of_frame[f]`` = the FOV index of frame ``f`` (-1 for trailing
    frames that do not complete a step).
    """
    frames_per_step = max(1, int(frames_per_step))
    n_steps = n_frames // frames_per_step
    all_frames = np.arange(n_frames)
    valid = all_frames < n_steps * frames_per_step
    if frame_order == "z":
        # each FOV is a full z stack: frames [k*n_steps, (k+1)*n_steps)
        fov_of_frame = all_frames // n_steps
    else:  # "fov": consecutive frames are the different FOVs at one z position
        fov_of_frame = all_frames % frames_per_step
    return np.where(valid, fov_of_frame, -1)


def _mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a boolean frame mask to a list of inclusive ``(lo, hi)``
    contiguous-run segments (the frame-bounds form ``localize.identify``
    accepts)."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _dedupe_beads(
    x: np.ndarray, y: np.ndarray, min_separation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse detections that are closer than ``min_separation`` pixels into
    a single bead.

    The same physical bead is detected on every reference frame and its
    sub-pixel jitter rounds to slightly different integer positions, so exact
    de-duplication would count it several times. We greedily keep detections
    in scan order, dropping any that fall within ``min_separation`` of an
    already-kept bead.
    """
    order = np.lexsort((y, x))
    keep = np.ones(len(order), dtype=bool)
    kept_xy: list[tuple[int, int]] = []
    sq = min_separation * min_separation
    for pos, i in enumerate(order):
        xi, yi = x[i], y[i]
        if any((xi - kx) ** 2 + (yi - ky) ** 2 < sq for kx, ky in kept_xy):
            keep[pos] = False
        else:
            kept_xy.append((xi, yi))
    kept = order[keep]
    return x[kept], y[kept]


def _detect_bead_positions(
    movie: lib.IntArray3D,
    minimum_ng: float,
    box: int,
    ref_frame_bounds: tuple[int, int],
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    threaded: bool = True,
    min_separation: float | None = None,
    fov_of_frame: np.ndarray | None = None,
) -> pd.DataFrame:
    """Detect bead centers (integer pixel positions) from a set of reference
    frames (ideally the in-focus ones, where beads are brightest).

    Beads are static in x/y (only the stage moves in z), so we detect them
    once and reuse the positions across all z-steps. Detections are rounded
    to the pixel grid and de-duplicated spatially (detections within
    ``min_separation`` pixels - defaulting to the box size - are treated as
    the same bead); beads whose box would fall outside the frame are dropped.

    If ``roi`` is given, only detections inside the ROI(s) are kept; an empty
    list or None means the whole frame.

    Returns a data frame with integer ``x``/``y`` columns (one row per bead),
    plus a ``fov`` column when ``fov_of_frame`` is given.
    """
    if min_separation is None:
        min_separation = box
    ids, _ = localize.identify(
        movie,
        minimum_ng,
        box,
        roi=roi,
        frame_bounds=ref_frame_bounds,
        threaded=threaded,
    )
    if len(ids) == 0:
        raise ValueError(
            "No beads detected for the spline calibration. Lower the minimum "
            "net gradient or check the reference frames."
        )

    x = np.rint(np.asarray(ids["x"])).astype(int)
    y = np.rint(np.asarray(ids["y"])).astype(int)
    frame = np.asarray(ids["frame"]).astype(int)
    # keep beads whose full box fits inside the frame
    height, width = movie.shape[1], movie.shape[2]
    half = box // 2
    inside = (
        (x - half >= 0)
        & (x + half < width)
        & (y - half >= 0)
        & (y + half < height)
    )
    x, y, frame = x[inside], y[inside], frame[inside]

    if fov_of_frame is not None:
        # multi-FOV: de-duplicate within each FOV so beads from
        # different fields are never merged, and tag each bead with its FOV.
        fov = np.asarray(fov_of_frame)[frame]
        xs, ys, fs = [], [], []
        for k in np.unique(fov):
            if k < 0:
                continue
            m = fov == k
            xk, yk = _dedupe_beads(x[m], y[m], min_separation)
            xs.append(xk)
            ys.append(yk)
            fs.append(np.full(len(xk), int(k), dtype=int))
        beads = pd.DataFrame(
            {
                "x": (np.concatenate(xs) if xs else np.array([], dtype=int)),
                "y": (np.concatenate(ys) if ys else np.array([], dtype=int)),
                "fov": (np.concatenate(fs) if fs else np.array([], dtype=int)),
            }
        )
    else:
        # merge detections of the same physical bead across reference frames
        x, y = _dedupe_beads(x, y, min_separation)
        beads = pd.DataFrame({"x": x, "y": y})

    beads = beads.reset_index(drop=True)
    if len(beads) == 0:
        raise ValueError(
            "All detected beads are too close to the frame edge for the "
            "requested box size."
        )
    return beads


def _bead_volumes(
    movie: lib.IntArray3D,
    camera_info: dict,
    beads: pd.DataFrame,
    box: int,
    step_of_frame: np.ndarray,
    step_range: np.ndarray,
    return_spots: bool = False,
    fov_of_frame: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a PSF volume ``(box, box, n_steps)`` in photon units for
    every bead, returning ``(n_beads, box, box, n_steps)``.

    **Multi-FOV** (``beads`` has a ``fov`` column and ``fov_of_frame`` is
    given): each bead belongs to one field of view and its volume is built
    only from that FOV's frames - exactly one frame per z-step.

    **Single field / repeats** (no ``fov`` column): every z-slice is the mean
    over all frames assigned to that step.

    If ``return_spots`` is True, also returns the individual per-frame spots
    flattened to ``(n_spots, box, box)``, for each spot the position of its
    z-step within ``step_range`` (``(n_spots,)``), and for each spot the row of
    ``beads`` it came from (``(n_spots,)``), so the axial precision can be
    measured by fitting every single-frame spot separately (the realistic,
    single-frame shot-noise regime)."""
    n_beads = len(beads)
    n_steps = len(step_range)
    step_to_pos = {int(s): i for i, s in enumerate(step_range)}
    bead_x = np.asarray(beads["x"], dtype=np.int64)
    bead_y = np.asarray(beads["y"], dtype=np.int64)

    if fov_of_frame is not None and "fov" in beads.columns:
        fov_of_frame = np.asarray(fov_of_frame)
        bead_fov = np.asarray(beads["fov"], dtype=int)
        # frame index for each (fov, z-step position); -1 = no such frame
        n_fov = int(
            max(fov_of_frame.max(initial=-1), bead_fov.max(initial=-1))
        )
        frame_of = np.full((n_fov + 1, n_steps), -1, dtype=np.int64)
        valid_frames = np.where(step_of_frame >= 0)[0]
        for f in valid_frames:
            kf = int(fov_of_frame[f])
            s = int(step_of_frame[f])
            if kf >= 0 and s in step_to_pos:
                frame_of[kf, step_to_pos[s]] = f
        # (bead, z-step position) frame grid; extract only existing spots
        frames_bp = frame_of[bead_fov]  # (n_beads, n_steps)
        bidx, pidx = np.where(frames_bp >= 0)
        frames_flat = frames_bp[bidx, pidx]
        # ``localize.get_spots`` cuts lazy movies (TiffMap/TiffMultiMap, ND2,
        # ...) frame by frame and REQUIRES the identifications to be sorted by
        # frame (it maps each frame to a contiguous slice of the output).
        # Our per-FOV grid is bead-major, so sort by frame here; without this
        # the spots are scattered to the wrong rows for every non-ndarray
        # movie (the numpy path is order-agnostic and hides the bug).
        order = np.argsort(frames_flat, kind="stable")
        bidx, pidx, frames_flat = (
            bidx[order],
            pidx[order],
            frames_flat[order],
        )
        ids = pd.DataFrame(
            {
                "frame": frames_flat,
                "x": bead_x[bidx],
                "y": bead_y[bidx],
                "net_gradient": np.ones(len(bidx), dtype=np.float32),
            }
        )
        got = localize.get_spots(movie, ids, box, camera_info)
        volumes = np.zeros((n_beads, box, box, n_steps), dtype=np.float32)
        volumes[bidx, :, :, pidx] = got
        if return_spots:
            return (
                volumes,
                got.astype(np.float32),
                pidx.astype(int),
                bidx.astype(int),
            )
        return volumes

    # single field / repeats: average all frames of a step (pixelwise)
    valid_frames = np.where(step_of_frame >= 0)[0]
    n_valid = len(valid_frames)
    # identifications for every (frame, bead) pair, frame-major so the spot
    # stack reshapes cleanly to (n_valid, n_beads, box, box)
    frame_col = np.repeat(valid_frames, n_beads)
    x_col = np.tile(bead_x, n_valid)
    y_col = np.tile(bead_y, n_valid)
    ids = pd.DataFrame(
        {
            "frame": frame_col.astype(np.int64),
            "x": x_col.astype(np.int64),
            "y": y_col.astype(np.int64),
            "net_gradient": np.ones(n_valid * n_beads, dtype=np.float32),
        }
    )
    spots = localize.get_spots(movie, ids, box, camera_info)
    spots = spots.reshape(n_valid, n_beads, box, box)

    steps_of_valid = step_of_frame[valid_frames]
    volumes = np.zeros((n_beads, box, box, n_steps), dtype=np.float32)
    for i, s in enumerate(step_range):
        mask = steps_of_valid == s
        # mean over the frames belonging to this step -> (n_beads, box, box)
        volumes[:, :, :, i] = spots[mask].mean(axis=0)
    if return_spots:
        # flatten to (n_valid * n_beads, box, box) (frame-major, bead-minor)
        # and give each spot the position of its z-step within step_range
        pos_of_valid = np.array(
            [step_to_pos[int(s)] for s in steps_of_valid], dtype=int
        )
        spots_flat = spots.reshape(n_valid * n_beads, box, box)
        spot_step_pos = np.repeat(pos_of_valid, n_beads)
        spot_bead_idx = np.tile(np.arange(n_beads, dtype=int), n_valid)
        return volumes, spots_flat, spot_step_pos, spot_bead_idx
    return volumes


def _focus_step(volume: np.ndarray) -> tuple[int, float]:
    """Return ``(z_center, effective_sigma)``: the sharpest z-slice of a
    ``(box, box, n_steps)`` PSF volume (smallest fitted Gaussian sigma) and
    the mean sigma there, using the least-squares single-spot fitter."""
    n_steps = volume.shape[2]
    sigmas = np.full(n_steps, np.inf, dtype=np.float32)
    for k in range(n_steps):
        theta = gausslq.fit_spot(np.ascontiguousarray(volume[:, :, k]))
        sx, sy = abs(theta[4]), abs(theta[5])
        if np.isfinite(sx) and np.isfinite(sy):
            sigmas[k] = np.sqrt(sx * sy)
    z_center = int(np.argmin(sigmas))
    return z_center, float(sigmas[z_center])


def _fft_cross_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Zero-mean 3D cross-correlation of two equally shaped volumes.

    Returned fft-shifted so a zero relative shift sits at the array center; the
    offset of the peak from the center is then the shift that aligns ``b`` onto
    ``a`` (apply it with ``scipy.ndimage.shift(b, shift)``). Both volumes are
    mean-subtracted first so the correlation is insensitive to a constant
    background offset.
    """
    a = a - a.mean()
    b = b - b.mean()
    cc = np.fft.ifftn(np.fft.fftn(a) * np.conj(np.fft.fftn(b))).real
    return np.fft.fftshift(cc)


def _subpixel_shift(
    cc: np.ndarray,
    max_shift: np.ndarray,
    upsample: int = 20,
    radius: int = 3,
) -> tuple[np.ndarray, float]:
    """Locate a cross-correlation peak with sub-voxel precision.

    The integer peak is found first, within ``+/- max_shift`` voxels (per axis)
    of the array center. A small window (``radius`` voxels) around it is then
    upsampled ``upsample``-fold by cubic-spline interpolation and the
    interpolated maximum gives the fractional part - mirroring the reference
    implementation (Li et al., Nat. Methods 2018), which scales up the central
    part of the cross-correlation by a factor of 20 by cubic-spline
    interpolation and reads the x, y, z shift off the position of the maximum.
    Returns ``(shift, peak_value)`` with ``shift`` the ``(row, col, z)``
    displacement from the center voxel.
    """
    shape = np.array(cc.shape)
    center = shape // 2
    max_shift = np.minimum(np.asarray(max_shift, dtype=int), center)
    lo = center - max_shift
    hi = center + max_shift + 1
    region = cc[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    ip = np.array(np.unravel_index(int(np.argmax(region)), region.shape)) + lo
    peak_value = float(cc[ip[0], ip[1], ip[2]])

    # sub-voxel refinement: cubic-spline upsample a small window around the peak
    wlo = np.maximum(ip - radius, 0)
    whi = np.minimum(ip + radius + 1, shape)
    window = cc[wlo[0] : whi[0], wlo[1] : whi[1], wlo[2] : whi[2]]
    win_center = ip - wlo
    order = int(min(3, min(window.shape) - 1))
    if order >= 2:
        zoomed = _ndi_zoom(window, upsample, order=order, mode="nearest")
        zshape = zoomed.shape
        peak = np.unravel_index(int(np.argmax(zoomed)), zshape)
        # zoom (grid_mode=False) maps corners to corners: the input coordinate
        # of output index p is p * (L - 1) / (M - 1)
        frac = np.array(
            [
                peak[ax] * (window.shape[ax] - 1) / (zshape[ax] - 1)
                - win_center[ax]
                for ax in range(cc.ndim)
            ]
        )
    else:
        frac = np.zeros(cc.ndim)
    return (ip - center) + frac, peak_value


def _keep_inliers(
    ncc: np.ndarray, mse: np.ndarray, k: float = 2.5
) -> np.ndarray:
    """Boolean mask of beads to keep during iterative averaging.

    A bead is dropped when it is dissimilar from the running average - either
    its normalized cross-correlation with the average falls far below, or its
    mean-square error rises far above, the robust spread (median ``+/- k``
    scaled MADs) of the population. At least half of the beads (and never fewer
    than three) are always kept, falling back to the lowest-MSE beads if the
    two criteria together would reject too many.
    """
    ncc = np.asarray(ncc, dtype=np.float64)
    mse = np.asarray(mse, dtype=np.float64)
    n = len(mse)
    if n <= 3:
        return np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    for values, high_is_bad in ((ncc, False), (mse, True)):
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad <= 0:
            continue
        t = 1.4826 * k * mad
        keep &= values <= med + t if high_is_bad else values >= med - t
    min_keep = max(3, n // 2)
    if keep.sum() < min_keep:
        keep = np.zeros(n, dtype=bool)
        keep[np.argsort(mse)[:min_keep]] = True
    return keep


# Sub-pixel lateral offsets below this are left alone rather than paying for an
# interpolation that cannot move the volume anyway.
_RECENTER_MIN_SHIFT = 1e-3
# Largest offset worth correcting, in pixels. The physical worst case is 1.0: a
# bead is detected on the integer grid, so its average sits within half a pixel
# of the box centre, plus (for a mapped channel) its sub-pixel ROI residual,
# another half. The cap sits just above that so a genuine worst-case offset is
# not refused on fit noise, while a plainly wrong estimate still is - and past
# roughly here the correction cannot be delivered anyway: on a 7 px box a 1.0
# px shift lands within 0.01 px of the centre and a 1.3 px one within 0.04, but
# a 1.7 px one leaves 0.44 px behind, because the PSF runs off the edge and
# ``mode="nearest"`` smears it. Refusing to shift leaves the template as it
# was, which is the safe direction.
_RECENTER_MAX_SHIFT = 1.25


def _focus_center_offset(
    volume: np.ndarray, z_center: int
) -> tuple[float, float]:
    """Lateral ``(row, col)`` offset of the in-focus PSF centre from the box
    centre, in pixels.

    Uses the same least-squares Gaussian fitter as :func:`_focus_step` (whose
    sigma already defines "in focus" here), falling back to the
    background-subtracted intensity centroid when the fit does not converge -
    a PSF a single Gaussian cannot describe, such as a double helix, still has
    a well-defined centroid. Returns ``(0.0, 0.0)`` if neither estimate is
    usable, so the caller simply leaves the volume alone."""
    focus = np.ascontiguousarray(volume[:, :, int(z_center)], dtype=np.float32)
    try:
        theta = gausslq.fit_spot(focus)
        # gausslq returns [x, y, ...] as offsets from the box centre, x the
        # column and y the row (see gausslq._sum_and_center_of_mass).
        d_col, d_row = float(theta[0]), float(theta[1])
    except Exception:
        # a degenerate slice can divide by zero inside the least-squares
        # model; that is a reason to fall back, not to fail the calibration
        d_col = d_row = np.nan
    if not (np.isfinite(d_row) and np.isfinite(d_col)):
        weights = np.clip(focus - focus.min(), 0.0, None).astype(np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            return 0.0, 0.0
        rows, cols = np.mgrid[0 : focus.shape[0], 0 : focus.shape[1]]
        d_row = (
            float((weights * rows).sum() / total) - (focus.shape[0] - 1) / 2.0
        )
        d_col = (
            float((weights * cols).sum() / total) - (focus.shape[1] - 1) / 2.0
        )
    if not (np.isfinite(d_row) and np.isfinite(d_col)):
        return 0.0, 0.0
    if max(abs(d_row), abs(d_col)) > _RECENTER_MAX_SHIFT:
        return 0.0, 0.0
    return d_row, d_col


def _register_and_average(
    volumes: np.ndarray,
    z_center: int,
    return_registered: bool = False,
    n_iter: int = 3,
    upsample: int = 20,
):
    """Register every bead volume to a common 3D center and average.

    Following Li et al. (Nat. Methods 2018), the beads are aligned to one
    another in all three dimensions by 3D cross-correlation: each bead's
    ``(row, col, z)`` sub-voxel shift is read from the cross-correlation
    peak (upsampled ``upsample``-fold by cubic-spline interpolation,
    see ``_subpixel_shift``) and the whole volume is shifted there by
    cubic-spline interpolation. Being model-free, this also centers
    non-Gaussian PSFs (e.g. double-helix), and the axial alignment
    should remove the per-bead focus jitter caused by coverslip tilt /
    bead-height variation.

    The first round aligns all beads to a single (brightest in-focus)
    bead; each subsequent round realigns the original volumes to the
    average of the aligned beads and excludes beads that are dissimilar
    from that average (by cross-correlation peak and mean-square error,
    see ``_keep_inliers``) before recomputing it. The correlation uses
    only a central z-band (the sharp, high SNR, in-focus region) so
    defocus rings cannot create spurious axial matches.

    Returns the mean PSF volume ``(box, box, n_steps)`` (photon units). If
    ``return_registered`` is True, also returns the stack of the retained
    individual registered bead volumes (photon units, shape
    ``(n_used, box, box, n_steps)``) so the averaged model can be compared
    against the individual beads (goodness of fit)."""
    n_beads, box, _, n_steps = volumes.shape
    vols = volumes.astype(np.float64)
    z_center = int(z_center)

    # per-axis integer search range: small laterally (beads are already
    # detected on the pixel grid) and wider axially (beads reach focus at
    # slightly different stage positions)
    max_shift = np.array(
        [max(2, box // 4), max(2, box // 4), max(2, n_steps // 4)], dtype=int
    )
    # correlate only a central z-band around focus (sharp, high-SNR slices)
    band = max(3, n_steps // 4)
    z0 = max(0, z_center - band)
    z1 = min(n_steps, z_center + band + 1)
    cmax = max_shift.copy()
    cmax[2] = min(cmax[2], (z1 - z0) // 2)

    def central(v):
        return v[:, :, z0:z1]

    # round-0 reference: the single brightest in-focus bead
    ref = vols[int(np.argmax(vols[:, :, :, z_center].max(axis=(1, 2))))]

    keep = np.ones(n_beads, dtype=bool)
    aligned = vols
    for _ in range(n_iter):
        ref_c = central(ref)
        ref_z = ref_c - ref_c.mean()
        ref_energy = float((ref_z * ref_z).sum())
        aligned = np.empty_like(vols)
        ncc = np.zeros(n_beads)
        for b in range(n_beads):
            vb = central(vols[b])
            cc = _fft_cross_correlation(ref_c, vb)
            shift, peak = _subpixel_shift(cc, cmax, upsample=upsample)
            aligned[b] = _ndi_shift(
                vols[b], shift=shift, order=3, mode="nearest"
            )
            # energy-normalized peak correlation, one dissimilarity measure
            vb_z = vb - vb.mean()
            denom = np.sqrt(ref_energy * float((vb_z * vb_z).sum()))
            ncc[b] = peak / denom if denom > 0 else 0.0

        # mean-square error of each (amplitude-matched) bead vs the running
        # average, over the central band, as the second dissimilarity measure
        avg_c = central(aligned[keep].mean(axis=0))
        avg_energy = float((avg_c * avg_c).sum()) or 1.0
        mse = np.empty(n_beads)
        for b in range(n_beads):
            bead_c = central(aligned[b])
            scale = float((bead_c * avg_c).sum()) / avg_energy
            mse[b] = float(((bead_c - scale * avg_c) ** 2).mean())

        keep = _keep_inliers(ncc, mse)
        ref = aligned[keep].mean(axis=0)

    if not keep.any():
        raise ValueError(
            "No usable beads after registration; the calibration failed."
        )
    # The rounds above align every bead to ONE anchor - the brightest in-focus
    # bead of round 0 - so the average inherits that bead's own sub-pixel
    # offset instead of sitting at the box centre. Put it back on the centre,
    # which is where the fit model's zero lateral shift evaluates the spline.
    # Left in, the offset is a constant lateral bias: harmless for a single
    # channel, where it just translates the whole reconstruction, but not
    # across channels - each picks its own anchor, so a linked multichannel fit
    # sees a fixed inter-channel misregistration it cannot represent.
    d_row, d_col = _focus_center_offset(ref, z_center)
    if max(abs(d_row), abs(d_col)) > _RECENTER_MIN_SHIFT:
        shift = (-d_row, -d_col, 0.0)
        ref = _ndi_shift(ref, shift=shift, order=3, mode="nearest")
        if return_registered:
            # the individual beads feed the goodness of fit, which fits only
            # amplitude and background - so they have to move with the template
            aligned = _ndi_shift(
                aligned,
                shift=(0.0, -d_row, -d_col, 0.0),
                order=3,
                mode="nearest",
            )
    mean_volume = ref.astype(np.float32)
    if return_registered:
        return mean_volume, aligned[keep].astype(np.float32)
    return mean_volume


def _smooth_z(volume: np.ndarray) -> np.ndarray:
    """Regularize a PSF volume by smoothing each voxel's axial profile.

    Following Li et al. (Nat. Methods 2018), the ``(box, box, n_steps)`` bead
    average is denoised along z with a smoothing cubic B-spline applied
    independently to every lateral voxel's intensity-vs-z curve. The spline's
    penalty is chosen automatically by generalized cross-validation
    (``scipy.interpolate.make_smoothing_spline``), so it removes shot noise
    without washing out the real axial variation that encodes z. Returns the
    smoothed volume (same shape/dtype); the volume is returned unchanged if
    there are too few z-steps to fit a cubic smoothing spline.
    """
    box, _, n_steps = volume.shape
    if n_steps < 5:  # a cubic smoothing spline needs a few points to be stable
        return volume
    z = np.arange(n_steps, dtype=np.float64)
    smoothed = np.empty_like(volume, dtype=np.float64)
    for i in range(box):
        for j in range(box):
            profile = volume[i, j, :].astype(np.float64)
            try:
                smoothed[i, j, :] = make_smoothing_spline(z, profile)(z)
            except Exception:
                # a degenerate (e.g. constant) profile keeps its raw values
                smoothed[i, j, :] = profile
    return smoothed.astype(volume.dtype, copy=False)


def _normalize_template(
    volume: np.ndarray, z_center: int
) -> tuple[np.ndarray, float, float, float]:
    """Normalize a PSF volume to a unit-peak template ``(psf - bg) / amp``.

    Background is the minimum of the (z-smoothed) volume - as in Li et al.
    (Nat. Methods 2018), which eliminates the background by subtracting the
    minimum of the bead stack; amplitude is the peak of the
    (background-subtracted) in-focus slice. Returns
    ``(template, background, amplitude, photon_scale)`` where ``photon_scale``
    is the integral of the in-focus normalized slice (used to convert a fitted
    amplitude into integrated photons). Unit-peak scaling is kept (rather than
    the paper's central-slice-sum) so the Gpufit spline fit's amplitude
    initialization (``spot_max - spot_min``) stays valid without change."""
    background = float(np.min(volume))
    focus = volume[:, :, z_center] - background
    amplitude = float(np.max(focus))
    if amplitude <= 0:
        raise ValueError(
            "Non-positive PSF amplitude; the calibration failed (check the "
            "bead brightness and background)."
        )
    template = ((volume - background) / amplitude).astype(np.float32)
    photon_scale = float(np.clip(template[:, :, z_center], 0, None).sum())
    return template, background, amplitude, photon_scale


def _goodness_of_fit(registered: np.ndarray, template: np.ndarray) -> dict:
    """Quantify how well the averaged PSF template reproduces the individual
    measured beads.

    The spline is an (interpolating) representation of ``template``, so it
    reproduces the template exactly at its nodes; the only independent "data"
    to compare against is the individual beads that were averaged into it. For
    each registered bead volume a single amplitude ``a`` and background
    ``o`` are least-squares fitted to the unit-peak ``template`` (i.e.
    ``bead ~= a * template + o`` over the whole volume) and the residual is
    measured. Returns ``r2`` (per-bead coefficient of determination),
    ``r2_median``, ``nrmse_pct`` (residual RMSE as a percentage of the fitted
    peak amplitude, pooled over all beads) and ``residual_profile_pct`` (the
    same normalized RMSE resolved per z-slice - where in z the single PSF model
    describes the beads best/worst).
    """
    n_beads = int(registered.shape[0])
    nz = int(registered.shape[2 + 1])
    model = template.reshape(-1, nz).astype(np.float64)  # (n_pixels, nz)
    m_flat = model.ravel()
    m_mean = m_flat.mean()
    m_c = m_flat - m_mean
    denom = float(m_c @ m_c)

    r2: list[float] = []
    slice_sq = np.zeros(nz, dtype=np.float64)
    slice_cnt = np.zeros(nz, dtype=np.float64)
    for b in range(n_beads):
        bead = registered[b].reshape(-1, nz).astype(np.float64)
        v_flat = bead.ravel()
        v_mean = v_flat.mean()
        if denom <= 0:
            continue
        # closed-form linear fit v ~= a * model + o
        a = float(m_c @ (v_flat - v_mean)) / denom
        o = v_mean - a * m_mean
        if not np.isfinite(a) or a <= 0:
            continue
        resid = bead - (a * model + o)  # (n_pixels, nz), photon units
        ss_res = float((resid.ravel() ** 2).sum())
        ss_tot = float(((v_flat - v_mean) ** 2).sum())
        if ss_tot > 0:
            r2.append(1.0 - ss_res / ss_tot)
        # amplitude-normalized squared residual, accumulated per z-slice
        norm_sq = (resid / a) ** 2
        slice_sq += norm_sq.sum(axis=0)
        slice_cnt += bead.shape[0]

    with np.errstate(invalid="ignore", divide="ignore"):
        profile_pct = 100.0 * np.sqrt(slice_sq / slice_cnt)
    total_cnt = float(slice_cnt.sum())
    nrmse_pct = (
        100.0 * float(np.sqrt(slice_sq.sum() / total_cnt))
        if total_cnt > 0
        else float("nan")
    )
    return {
        "r2": np.asarray(r2, dtype=np.float64),
        "r2_median": float(np.median(r2)) if r2 else float("nan"),
        "nrmse_pct": nrmse_pct,
        "residual_profile_pct": profile_pct,
        "n_used": len(r2),
    }


def _axial_intensity_focus(template: np.ndarray) -> float:
    """Sub-slice z index where the PSF's axial intensity profile peaks.

    The profile is the brightest (normalized) pixel per z-slice - the same
    curve drawn in the diagnostic plot's "Axial intensity profile" panel. A
    parabolic interpolation around the discrete maximum gives sub-slice
    precision. Used (optionally) to define z = 0 at this intensity focus,
    correcting a potential z bias in the raw stage scan. This only makes sense
    when the PSF has a single, well-defined axial intensity peak (e.g.
    astigmatism)."""
    profile = template.max(axis=(0, 1)).astype(np.float64)
    k = int(np.argmax(profile))
    if 0 < k < len(profile) - 1:
        y0, y1, y2 = profile[k - 1], profile[k], profile[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            # vertex of the parabola through the three points around the peak
            return k + 0.5 * (y0 - y2) / denom
    return float(k)


def _scan_center_index(z_of_step: np.ndarray) -> float:
    """Find the z position where z == 0 using linear interpolation."""
    zos = np.asarray(z_of_step, dtype=float)
    n = len(zos)
    if n < 2:
        return 0.0
    dz = (zos[-1] - zos[0]) / (n - 1)
    if dz == 0.0:
        return 0.0
    return float(-zos[0] / dz)  # solve zos[0] + k * dz == 0


def _reference_frame_segments(
    step_of_frame: np.ndarray, step_range: np.ndarray
) -> list[tuple[int, int]]:
    """In-focus reference frames (the middle third of the scan, brightest),
    used for bead detection, as inclusive ``(lo, hi)`` frame segments."""
    n_steps = len(step_range)
    lo = step_range[n_steps // 3]
    hi = step_range[min(n_steps - 1, 2 * n_steps // 3)]
    mask = (step_of_frame >= lo) & (step_of_frame <= hi)
    return _mask_to_segments(mask)


def _reference_mid_frame(
    step_of_frame: np.ndarray, step_range: np.ndarray
) -> int:
    """A single representative in-focus frame (first frame at the central
    z-step), used to coarsely cross-correlate channels for registration."""
    focus_step = int(step_range[len(step_range) // 2])
    candidates = np.where(step_of_frame == focus_step)[0]
    return int(candidates[0]) if len(candidates) else 0


def build_psf_template(
    movie: lib.IntArray3D,
    camera_info: dict,
    box: int,
    minimum_ng: float,
    d: float,
    frames_per_step: int = 1,
    frame_bounds: tuple[int, int] | list | None = None,
    frame_order: Literal["fov", "z"] = "fov",
    threaded: bool = True,
    beads: pd.DataFrame | None = None,
    return_spots: bool = False,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
) -> dict:
    """Build a normalized PSF template volume from a bead z-stack.

    This is the GPU-independent part of the calibration (no Gpuspline needed),
    factored out so it can be unit-tested. Returns a dict with keys
    ``template`` (box, box, n_steps), ``z_center``, ``effective_sigma``,
    ``background``, ``amplitude``, ``photon_scale``, ``n_beads``,
    ``z_of_step``, ``gof`` and ``registered`` (the 3D-registered individual
    bead volumes, ``(n_used, box, box, n_steps)``, photon units, used for the
    goodness-of-fit).

    If ``return_spots`` is True, the dict also carries ``spots`` (every
    individual per-frame bead spot, ``(n_spots, box, box)``, photon units),
    ``spot_step_idx`` (the index into the template z-axis of each spot's stage
    step) and ``spot_bead_idx`` (the row of ``beads`` each spot came from),
    which ``_axial_precision`` fits one by one to measure the axial precision
    in the realistic single-frame regime.

    If ``beads`` (a data frame with integer ``x``/``y`` columns) is given, it
    is used instead of detecting beads on this movie - this lets the
    multichannel calibration reuse the same physical beads, mapped into each
    channel, across all channels. Otherwise beads are detected on this movie,
    restricted to ``roi`` if given (see ``_detect_bead_positions``).
    """
    n_frames = int(movie.shape[0])
    step_of_frame, z_of_step, step_range = _step_of_frame(
        n_frames, d, frames_per_step, frame_order, frame_bounds
    )
    # FOV of each frame: with several frames per z position they may be
    # genuinely different fields (different beads), which are detected and
    # extracted per FOV rather than averaged together (see _bead_volumes).
    fov_of_frame = _fov_of_frame(n_frames, frames_per_step, frame_order)

    if beads is None:
        ref_segments = _reference_frame_segments(step_of_frame, step_range)
        beads = _detect_bead_positions(
            movie,
            minimum_ng,
            box,
            ref_segments,
            roi=roi,
            threaded=threaded,
            fov_of_frame=fov_of_frame,
        )
    if return_spots:
        volumes, spots, spot_step_pos, spot_bead_idx = _bead_volumes(
            movie,
            camera_info,
            beads,
            box,
            step_of_frame,
            step_range,
            return_spots=True,
            fov_of_frame=fov_of_frame,
        )
    else:
        volumes = _bead_volumes(
            movie,
            camera_info,
            beads,
            box,
            step_of_frame,
            step_range,
            fov_of_frame=fov_of_frame,
        )
    # first pass on the raw bead-average to locate focus, then register
    z_center, _ = _focus_step(volumes.mean(axis=0))
    mean_volume, registered = _register_and_average(
        volumes, z_center, return_registered=True
    )
    # regularize the averaged PSF by smoothing along z (paper's smoothing
    # B-spline), then (re)locate focus and normalize on the cleaned volume
    mean_volume = _smooth_z(mean_volume)
    z_center, effective_sigma = _focus_step(mean_volume)
    template, background, amplitude, photon_scale = _normalize_template(
        mean_volume, z_center
    )
    # how well the averaged PSF model represents the individual beads
    gof = _goodness_of_fit(registered, template)
    # fractional z-slice of the axial intensity peak; used (optionally) to
    # define z = 0 at the intensity focus (see ``calibrate_spline``'s
    # ``correct_z_bias``)
    z_focus = _axial_intensity_focus(template)
    result = {
        "template": template,
        "z_center": z_center,
        "z_focus": z_focus,
        "effective_sigma": effective_sigma,
        "background": background,
        "amplitude": amplitude,
        "photon_scale": photon_scale,
        "n_beads": int(len(beads)),
        "z_of_step": z_of_step[step_range],
        "gof": gof,
        "registered": registered,
    }
    if return_spots:
        # every individual per-frame bead spot, flattened to (n_spots, box,
        # box), with for each spot the index into the template z-axis
        # (0..n_steps-1) of its stage step. z_of_step[step_idx] is then the
        # spot's known stage position (see _axial_precision).
        result["spots"] = spots
        result["spot_step_idx"] = spot_step_pos
        # which bead each spot came from, so a caller that knows where the
        # beads sit can attach per-spot geometry (e.g. the multichannel ROI
        # residuals) without re-deriving the flattening order
        result["spot_bead_idx"] = spot_bead_idx
    return result


def calibrate_spline(
    movie: lib.IntArray3D,
    info: list[dict],
    camera_info: dict,
    box: int,
    minimum_ng: float,
    d: float,
    frames_per_step: int = 1,
    frame_bounds: tuple[int, int] | list | None = None,
    frame_order: Literal["fov", "z"] = "fov",
    model: Literal["spline-2d", "spline-3d"] = "spline-3d",
    magnification_factor: float = 0.79,
    correct_z_bias: bool = False,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    path: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """Generate a cubic-spline PSF calibration from a bead z-stack movie.

    Parameters
    ----------
    movie : lib.IntArray3D
        The bead z-stack movie (as loaded by ``picasso.io.load_movie``).
    info : list of dicts
        Movie metadata.
    camera_info : dict
        Camera information ("Baseline", "Sensitivity", "Gain", "Pixelsize").
    box : int
        Lateral ROI size (camera pixels). The resulting calibration expects
        fits with this same box size.
    minimum_ng : float
        Minimum net gradient for bead detection.
    d : float
        Step size in nm between consecutive z (stage) positions.
    frames_per_step : int, optional
        Number of frames acquired at each z position (multi-FOV). Default 1.
    frame_bounds : tuple, list of tuples, optional
        Frame numbers to consider (see ``zfit.calibrate_z``). Default None.
    frame_order : {"fov", "z"}, optional
        Acquisition order when ``frames_per_step`` > 1 (see
        ``zfit.calibrate_z``). Default "fov".
    model : {"spline-2d", "spline-3d"}, optional
        Whether to build a 3D spline PSF (recovers z) or a single-plane 2D
        spline PSF (the in-focus slice only). Default "spline-3d".
    magnification_factor : float, optional
        Ratio between the actual axial position and the stage travel of the
        calibration scan (refractive-index mismatch), applied to the fitted
        z at localization time (as in ``picasso.zfit``). Stored in the
        calibration. Default 0.79.
    correct_z_bias : bool, optional
        If True, define z = 0 at the axial intensity peak of the averaged PSF
        (the peak of the diagnostic plot's axial intensity profile) instead of
        at the raw stage-scan center (``z_of_step == 0``). Analogous to the
        astigmatism calibration pinning z = 0 to the sx == sy crossing
        (see ``picasso.zfit.calibrate_z``); only meaningful for a PSF
        with a single, well-defined intensity focus (e.g. astigmatism).
        Default is False.
    roi : tuple or list of tuples, optional
        Region(s) of interest for bead detection, in the same
        ``[[y_min, x_min], [y_max, x_max]]`` form as ``localize.identify`` and
        the GUI's ``view.rois``. Only beads inside the ROI(s) are used for the
        calibration; None (or an empty list) uses the whole frame. Default None.
    path : str, optional
        Where to save the calibration (HDF5) and a diagnostic PNG. If None,
        nothing is written. Default is None.
    progress_callback : callable, optional
        Called with an integer step count (0..3) as the calibration proceeds.

    Returns
    -------
    calibration : dict
        The spline PSF calibration (see ``io.save_spline_calibration`` /
        ``io.load_spline_calibration`` and ``localize.fit_spots_gpufit_spline``).
    """
    if not localize.GPUSPLINE_INSTALLED:
        raise ImportError(
            "Gpuspline is required to build a spline PSF calibration but "
            "could not be loaded. See picasso/ext/pygpuspline/README.txt."
        )
    assert model in (
        "spline-2d",
        "spline-3d",
    ), "model must be 'spline-2d' or 'spline-3d'"

    if callable(progress_callback):
        progress_callback(0)
    built = build_psf_template(
        movie,
        camera_info,
        box,
        minimum_ng,
        d,
        frames_per_step=frames_per_step,
        frame_bounds=frame_bounds,
        frame_order=frame_order,
        roi=roi,
        # keep the individual per-frame spots so the diagnostic PNG can measure
        # the axial precision by fitting each spot (only needed when we plot)
        return_spots=path is not None,
    )
    template = built["template"]  # (box, box, n_steps)
    z_center = built["z_center"]
    # Two distinct z references (see also ``localize._initial_parameters_spline``
    # / ``locs_from_fits_spline``):
    #  - z_init: the sharpest (in-focus) slice. The fit's z_shift is initialized
    #    to -z_init. This is the numerically sound, convention-INDEPENDENT start
    #    and must NOT depend on ``correct_z_bias`` - otherwise the two
    #    calibrations start the fit from different z, converge to different
    #    (x, y, z) minima (an astigmatic PSF has flat gradients off-focus), and
    #    the z=0 shift leaks into x/y instead of being a clean constant.
    #  - z_origin: the z = 0 reference used only when converting the fitted
    #    z_shift to physical z (output). Without correction it is the raw
    #    stage-scan zero (the scan center); ``correct_z_bias`` moves it to the
    #    axial intensity focus. The difference is the raw-scan z bias (the
    #    focus's offset from the scan center, ~100s of nm), applied as a
    #    constant shift of the output z and nothing else.
    z_init = float(z_center)
    z_origin = (
        built["z_focus"]
        if correct_z_bias
        else _scan_center_index(built["z_of_step"])
    )

    if callable(progress_callback):
        progress_callback(1)

    gs = localize.gs
    # The template is (row=y, col=x, z), but at fit time spots are flattened
    # C-order so the Gpufit spline model's fast pixel index (point_index_x) is
    # the movie column (x). The spline's first axis must therefore be x, so we
    # swap the lateral axes before computing coefficients. Skipping this
    # transposes the PSF laterally, which corrupts an astigmatic PSF's z
    # encoding (z barely recovers) and mis-fits amplitude/position. It is a
    # no-op for a laterally symmetric PSF.
    if model == "spline-2d":
        slab = np.ascontiguousarray(template[:, :, z_center].T)
        coefficients = gs.spline_coefficients(slab)
        n_intervals = [int(i) for i in (np.array(slab.shape) - 1)]
        coefficients = np.reshape(coefficients, [16] + n_intervals).astype(
            np.float32
        )
        n_data = [box, box]
    else:
        template_xyz = np.ascontiguousarray(template.transpose(1, 0, 2))
        coefficients = gs.spline_coefficients(template_xyz)
        n_intervals = [int(i) for i in (np.array(template_xyz.shape) - 1)]
        coefficients = np.reshape(coefficients, [64] + n_intervals).astype(
            np.float32
        )
        n_data = [int(s) for s in template_xyz.shape]

    if callable(progress_callback):
        progress_callback(2)

    pixelsize = camera_info.get("Pixelsize", 130)
    calibration = {
        "model": model,
        "coefficients": coefficients,
        "n_data": n_data,
        "n_intervals": n_intervals,
        # lateral template sampling equals the camera pixel grid, so shifts
        # are already in camera pixels
        "oversampling": 1.0,
        # the template is centred on the box (see _register_and_average), so a
        # zero fitted lateral shift means "emitter at the box centre"
        "lateral_centered": True,
        "z_center": float(z_origin),
        "z_init": float(z_init),
        "z_step_nm": float(d),
        "magnification_factor": float(magnification_factor),
        "correct_z_bias": bool(correct_z_bias),
        "effective_sigma": float(built["effective_sigma"]),
        "photon_scale": float(built["photon_scale"]),
        "box": int(box),
        "pixelsize": float(pixelsize),
        "n_channels": 1,
        "n_beads": int(built["n_beads"]),
        "Frames per step": int(frames_per_step),
        "Frame order": frame_order,
        "Frame bounds": frame_bounds,
        "Generated by": f"Picasso: v{__version__} Spline PSF calibration",
        "Path": path if path is not None else "N/A",
    }

    if path is not None:
        io.save_spline_calibration(path, calibration)
        # empirical axial precision (z RMSD to the true stage position vs z)
        # from fitting every individual per-frame bead spot through the
        # just-built spline model; None when no GPU fitter is available or for a
        # 2D calibration (the plot then falls back to the model-vs-data RMSE)
        precision = _axial_precision(built, calibration)
        _save_diagnostic_plot(built, calibration, path, precision=precision)

    if callable(progress_callback):
        progress_callback(3)
    return calibration


# ----------------------------------------------------------------------
# Multichannel calibration (SPLINE_3D_MULTICHANNEL)
#
# Global (multi-channel) fitting as introduced by globLoc: Li, Shi, Liu,
# et al. "Global fitting for high-accuracy multi-channel single-molecule
# localization." Nature Communications 13, 3133 (2022).
# DOI: 10.1038/s41467-022-30719-4.
# ----------------------------------------------------------------------


def _normalized_region(
    rect: tuple[tuple[int, int], tuple[int, int]] | list,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Normalize a ``[[y_a, x_a], [y_b, x_b]]`` rectangle (as produced by the
    GUI ROI tool) into ``((y_min, x_min), (y_max, x_max))`` with integer,
    correctly ordered corners."""
    (ya, xa), (yb, xb) = rect
    y0, y1 = sorted((int(ya), int(yb)))
    x0, x1 = sorted((int(xa), int(xb)))
    return (y0, x0), (y1, x1)


def _similarity_from_two(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> list[np.ndarray]:
    """Candidate similarity transforms mapping ``a -> b`` from two point pairs.

    A similarity (translation + rotation + isotropic scale, optionally a
    reflection) is fixed by two correspondences up to the reflection ambiguity,
    so both the proper-rotation and the reflected solution are returned as
    ``(2, 3)`` affines. Using a *similarity* (4 DOF) as the RANSAC minimal model -
    rather than a full 6-DOF affine, which three points always fit exactly - keeps
    a spare bead to validate the sample, so correct correspondences can be told
    from wrong ones even with only three beads. Empty if the two reference
    points coincide."""
    va, vb = a1 - a0, b1 - b0
    na = float(np.hypot(va[0], va[1]))
    if na < 1e-9:
        return []
    s = float(np.hypot(vb[0], vb[1])) / na
    ang_a = np.arctan2(va[1], va[0])
    ang_b = np.arctan2(vb[1], vb[0])
    out = []
    # proper rotation (angle b - angle a) and reflection (across the a/b bisector)
    th = ang_b - ang_a
    r_rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    two_alpha = ang_a + ang_b
    r_ref = np.array(
        [
            [np.cos(two_alpha), np.sin(two_alpha)],
            [np.sin(two_alpha), -np.cos(two_alpha)],
        ]
    )
    for r in (r_rot, r_ref):
        A = s * r
        t = b0 - A @ a0
        out.append(np.hstack([A, t[:, None]]))
    return out


def _ransac_match(
    ref_xy: np.ndarray,
    c_xy: np.ndarray,
    aligned_c: np.ndarray,
    inlier_tol: float,
    radius: float,
    max_iter: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly match beads across channels via RANSAC on a similarity
    transform.

    Correspondence candidates are proposed from ``aligned_c`` (``c_xy`` coarsely
    overlaid onto the reference frame - a flip orientation plus an approximate
    shift), but the transform is fit on the original **absolute** ``ref_xy`` /
    ``c_xy``. Two candidate pairs are sampled, the similarity transforms they
    imply (see :func:`_similarity_from_two`) are formed, and the beads each maps
    within ``inlier_tol`` are counted; the largest consensus wins and its
    inliers (unique nearest-neighbour assignment) are returned. Because only the
    *candidate proposal* uses the coarse overlay - not the fit - an inaccurate
    overlay (e.g. an imperfectly placed split-FOV ROI) cannot mis-pair beads
    and corrupt the transform, which otherwise makes the calibration
    hypersensitive to ROI placement. Returns ``(ref_idx, c_idx)`` inliers.
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    c_xy = np.asarray(c_xy, dtype=np.float64)
    empty = (np.array([], dtype=int), np.array([], dtype=int))
    if min(len(ref_xy), len(c_xy)) < 3:
        return empty

    # candidate (ref_i, c_j) pairs: c beads near ref_i in the coarse overlay
    overlay_tree = KDTree(np.asarray(aligned_c, dtype=np.float64))
    pairs = [
        (i, j)
        for i in range(len(ref_xy))
        for j in overlay_tree.query_ball_point(ref_xy[i], radius)
    ]
    if len(pairs) < 2:
        return empty
    pairs = np.asarray(pairs, dtype=int)

    c_tree = KDTree(c_xy)
    n_samples = len(pairs) * (len(pairs) - 1) // 2
    if n_samples > max_iter:
        rs = np.random.RandomState(0)  # deterministic for reproducible calib
        samples = rs.randint(0, len(pairs), size=(max_iter, 2))
    else:
        samples = np.asarray(
            list(combinations(range(len(pairs)), 2)), dtype=int
        )

    best_M, best_count = None, 0
    for a, b in samples:
        (i0, j0), (i1, j1) = pairs[a], pairs[b]
        if i0 == i1 or j0 == j1:  # need two distinct ref and channel beads
            continue
        for M in _similarity_from_two(
            ref_xy[i0], ref_xy[i1], c_xy[j0], c_xy[j1]
        ):
            pred = localize.apply_affine_transform(ref_xy, M)
            dist, _ = c_tree.query(pred, k=1)
            count = int(np.count_nonzero(dist <= inlier_tol))
            if count > best_count:
                best_count, best_M = count, M

    if best_M is None:
        return empty
    pred = localize.apply_affine_transform(ref_xy, best_M)
    return _match_beads(pred, c_xy, inlier_tol)


def _match_beads(
    ref_xy: np.ndarray, other_xy: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor match beads between two channels.

    Returns ``(ref_idx, other_idx)`` index arrays of matched pairs within
    ``max_distance``; each ``other`` bead is used at most once (closest match
    wins)."""
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    other_xy = np.asarray(other_xy, dtype=np.float64)
    if len(ref_xy) == 0 or len(other_xy) == 0:
        empty = np.array([], dtype=int)
        return empty, empty
    tree = KDTree(other_xy)
    dist, idx = tree.query(ref_xy, k=1)
    keep = np.where(dist <= max_distance)[0]
    # resolve duplicate targets: assign each target to its closest reference
    order = keep[np.argsort(dist[keep])]
    seen: set[int] = set()
    ref_idx, other_idx = [], []
    for r in order:
        o = int(idx[r])
        if o in seen:
            continue
        seen.add(o)
        ref_idx.append(int(r))
        other_idx.append(o)
    return np.array(ref_idx, dtype=int), np.array(other_idx, dtype=int)


def _estimate_channel_transform(
    movie_ref,
    movie_c,
    beads_ref: pd.DataFrame,
    minimum_ng: float,
    box: int,
    ref_bounds: tuple[int, int] | list,
    mid_frame: int,
    max_distance: float,
    channel_roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    coarse_shift: tuple[float, float] | np.ndarray | None = None,
    return_matches: bool = False,
) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Estimate the affine transform mapping reference-channel coordinates to
    channel ``c``.

    Beads are detected in channel ``c``, coarsely aligned to the reference,
    matched to the reference beads and an affine transform is fitted to the
    correspondences. Returns ``(transform (2, 3), n_matches)``.

    Two coarse-alignment paths:

    * **Separate movies** (default): channel ``c`` beads are detected over the
      whole frame and the coarse shift is measured by image cross-correlation
      of a mid (in-focus) frame; both shift signs are tried.
    * **Split-FOV** (``channel_roi`` and ``coarse_shift`` given): the two
      "channels" are two regions of the *same* movie, so channel ``c`` beads are
      detected inside ``channel_roi`` and pre-aligned by the known region-origin
      offset ``coarse_shift`` (added to the channel beads to overlay them on the
      reference region). ``coarse_shift`` is ``(x0_ref - x0_c, y0_ref - y0_c)``.
    """
    from . import imageprocess

    beads_c = _detect_bead_positions(
        movie_c, minimum_ng, box, ref_bounds, roi=channel_roi
    )
    ref_xy = beads_ref[["x", "y"]].to_numpy(dtype=np.float64)
    c_xy = beads_c[["x", "y"]].to_numpy(dtype=np.float64)

    # Coarse pre-alignment for bead matching only: the affine below is fit on the
    # original (untransformed) channel coordinates, so it absorbs whatever
    # orientation the matching candidate implied. Tries flipping if it
    # matches more beads
    flips = (
        ("none", 1.0, 1.0),
        ("flip-x", -1.0, 1.0),
        ("flip-y", 1.0, -1.0),
        ("flip-xy", -1.0, -1.0),
    )
    candidates = []  # (label, channel coords pre-aligned onto the reference)
    if coarse_shift is not None:
        # split-FOV: the two channels are regions of one movie, so the flip is
        # taken about the channel region (its size == the reference region's)
        # and the known region-origin offset places it on the reference region.
        (cy0, cx0), (cy1, cx1) = _normalized_region(channel_roi)
        h, w = float(cy1 - cy0), float(cx1 - cx0)
        x0_ref = cx0 + float(coarse_shift[0])
        y0_ref = cy0 + float(coarse_shift[1])
        xl = c_xy[:, 0] - cx0
        yl = c_xy[:, 1] - cy0
        for label, sx, sy in flips:
            fx = (w - xl) if sx < 0 else xl
            fy = (h - yl) if sy < 0 else yl
            candidates.append(
                (label, np.column_stack([fx + x0_ref, fy + y0_ref]))
            )
    else:
        # separate movies: no known geometry, so for every candidate orientation
        # the coarse translation is measured directly by cross-correlating the
        # reference mid-frame against the (identically mirrored) channel
        # mid-frame
        try:
            img_ref = np.asarray(movie_ref[mid_frame], dtype=np.float32)
            img_c = np.asarray(movie_c[mid_frame], dtype=np.float32)
        except Exception:
            img_ref = img_c = None
        height, width = int(movie_c.shape[1]), int(movie_c.shape[2])
        ref_centroid = ref_xy.mean(axis=0)
        for label, sx, sy in flips:
            # mirror the channel beads about the frame (consistently with the
            # image flip below) so the same orientation is applied to both
            fx = (width - 1 - c_xy[:, 0]) if sx < 0 else c_xy[:, 0]
            fy = (height - 1 - c_xy[:, 1]) if sy < 0 else c_xy[:, 1]
            flipped = np.column_stack([fx, fy])
            if img_ref is not None:
                img_cf = img_c
                if sx < 0:
                    img_cf = img_cf[:, ::-1]
                if sy < 0:
                    img_cf = img_cf[::-1, :]
                try:
                    dy, dx = imageprocess.get_image_shift(img_ref, img_cf, 5)
                except Exception:
                    dx = dy = 0.0
                # the cross-correlation sign convention can go either way, so try
                # both and let the match count decide
                for sign in (1.0, -1.0):
                    candidates.append(
                        (label, flipped + sign * np.array([dx, dy]))
                    )
            else:
                # a flip plus an arbitrary translation is recovered by aligning
                # centroids (coarser, needs the bead sets to overlap)
                candidates.append(
                    (label, flipped + (ref_centroid - flipped.mean(axis=0)))
                )

    # Match against each coarse orientation with RANSAC and keep the orientation
    # with the most inliers. The coarse overlay only proposes candidate pairs
    # (a generous radius covers an imperfectly placed ROI); the transform is fit
    # on absolute coordinates and mismatches are rejected, so the registration
    # is independent of exact ROI placement (see _ransac_match).
    radius = max(3.0 * box, float(max_distance))
    inlier_tol = max(3.0, 0.25 * box)
    best_ref_idx, best_c_idx = np.array([], int), np.array([], int)
    for _label, aligned in candidates:
        ri, ci = _ransac_match(ref_xy, c_xy, aligned, inlier_tol, radius)
        if len(ri) > len(best_ref_idx):
            best_ref_idx, best_c_idx = ri, ci

    if len(best_ref_idx) < 3:
        raise ValueError(
            f"Only {len(best_ref_idx)} bead correspondences found between the "
            "reference channel and another channel; cannot estimate an affine "
            "transform. Increase the bead count or the match distance."
        )
    transform = localize.estimate_affine_transform(
        ref_xy[best_ref_idx], c_xy[best_c_idx]
    )
    n_matches = int(len(best_ref_idx))
    if return_matches:
        # the matched reference / channel bead coordinates, for the
        # registration diagnostic (residual = transform(ref) - channel)
        return transform, n_matches, ref_xy[best_ref_idx], c_xy[best_c_idx]
    return transform, n_matches


def calibrate_spline_multichannel(
    movies: list,
    infos: list,
    camera_infos: list[dict],
    box: int,
    minimum_ng: float,
    d: float,
    frames_per_step: int = 1,
    frame_bounds: tuple[int, int] | list | None = None,
    frame_order: Literal["fov", "z"] = "fov",
    magnification_factor: float = 0.79,
    correct_z_bias: bool = False,
    max_match_distance: float | None = None,
    photon_ratios: np.ndarray | list | None = None,
    link_photons: bool = True,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    regions: list | None = None,
    reference: int = 0,
    path: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """Generate a multichannel cubic-spline PSF calibration from registered
    bead z-stacks (one movie per channel).

    This is the calibration side of globLoc-style global fitting (Li et al.,
    Nat. Commun. 13, 3133, 2022): a separate experimental PSF per channel plus
    the channel-to-channel registration, so that the channels can later be
    fitted jointly with shared coordinates by
    :func:`localize.fit_spline_multichannel`.

    Detects beads in the reference channel (``movies[0]``), estimates an affine
    transform from the reference channel to each other channel from bead
    correspondences, builds a per-channel PSF template using the same physical
    beads (mapped into each channel), and assembles the per-channel spline
    coefficients into a ``(64, n_int_x, n_int_y, n_int_z, n_channels)`` table.
    The transforms are stored in the calibration and reused at fit time by
    ``localize.fit_spline_multichannel`` / ``get_spots_multichannel``.

    Parameters are as in ``calibrate_spline`` but per-channel lists; all movies
    must share the same frame layout (z scan). ``roi`` (in reference-channel
    coordinates) restricts which reference-channel beads are calibrated on; the
    channel-to-channel transform is still estimated from all detected beads.
    Returns a ``"spline-3d-multichannel"`` calibration dict.

    **Split-FOV mode** (``regions`` given): the channels are rectangular
    sub-regions of a *single* movie (all ``movies`` entries are the same movie).
    ``regions`` is a list of ``[[y_min, x_min], [y_max, x_max]]`` rectangles,
    one per channel, all the same size; ``regions[reference]`` is the reference
    channel. Reference beads are detected inside the reference region and each
    channel-to-channel transform is estimated from beads inside that channel's
    region, pre-aligned by the known region-origin offset (see
    :func:`_estimate_channel_transform`). The regions and reference index are
    stored in the calibration so the fit path can rebuild the single-movie
    channel stack. Prefer the :func:`calibrate_spline_split_fov` wrapper, which
    builds the repeated per-channel lists for you.
    """
    if not localize.GPUSPLINE_INSTALLED:
        raise ImportError(
            "Gpuspline is required to build a spline PSF calibration but "
            "could not be loaded. See picasso/ext/pygpuspline/README.txt."
        )
    n_channels = len(movies)
    if n_channels < 2:
        raise ValueError(
            "Multichannel calibration needs at least 2 channels; use "
            "calibrate_spline for a single channel."
        )
    if not (len(camera_infos) == len(infos) == n_channels):
        raise ValueError(
            "movies, infos and camera_infos must have the same length."
        )
    if max_match_distance is None:
        max_match_distance = float(box)

    # Split-FOV: the channels are rectangular sub-regions of one movie. Put the
    # reference region first (channel 0), require all regions to share a size,
    # and precompute the known region-origin offsets used to pre-align beads.
    split_fov = regions is not None
    region_rects = None
    coarse_shifts = None
    ref_roi = roi
    if split_fov:
        if len(regions) != n_channels:
            raise ValueError(
                f"Got {n_channels} channels (movies) but {len(regions)} "
                "regions; they must match for a split-FOV calibration."
            )
        if not (0 <= reference < n_channels):
            raise ValueError(
                f"reference={reference} is out of range for {n_channels} "
                "regions."
            )
        region_rects = [_normalized_region(r) for r in regions]
        # reference region must be channel 0 (identity) for the transform and
        # template conventions; reorder so it comes first.
        order = [reference] + [c for c in range(n_channels) if c != reference]
        region_rects = [region_rects[c] for c in order]
        sizes = {(r[1][0] - r[0][0], r[1][1] - r[0][1]) for r in region_rects}
        if len(sizes) != 1:
            raise ValueError(
                "All split-FOV regions must have the same size (height, "
                f"width); got {sorted(sizes)}."
            )
        y0_ref, x0_ref = region_rects[0][0]
        # shift added to a channel's beads to overlay them on the reference
        # region: (x0_ref - x0_c, y0_ref - y0_c)
        coarse_shifts = [
            (float(x0_ref - r[0][1]), float(y0_ref - r[0][0]))
            for r in region_rects
        ]
        ref_roi = region_rects[0]

    if callable(progress_callback):
        progress_callback(0)

    n_frames = int(movies[0].shape[0])
    step_of_frame, _, step_range = _step_of_frame(
        n_frames, d, frames_per_step, frame_order, frame_bounds
    )
    fov_of_frame = _fov_of_frame(n_frames, frames_per_step, frame_order)
    ref_bounds = _reference_frame_segments(step_of_frame, step_range)
    mid_frame = _reference_mid_frame(step_of_frame, step_range)

    # reference beads tagged with their FOV: genuine multi-FOV z-stacks hold
    # different beads per field, extracted per FOV (see _bead_volumes).
    beads_ref = _detect_bead_positions(
        movies[0],
        minimum_ng,
        box,
        ref_bounds,
        roi=ref_roi,
        fov_of_frame=fov_of_frame,
    )

    # channel transforms (channel 0 is the identity reference)
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transforms = [identity]
    reg_info = (
        []
    )  # per non-reference channel: matched beads for the diagnostic
    for c in range(1, n_channels):
        transform, n_matches, ref_m, c_m = _estimate_channel_transform(
            movies[0],
            movies[c],
            beads_ref,
            minimum_ng,
            box,
            ref_bounds,
            mid_frame,
            max_match_distance,
            channel_roi=region_rects[c] if split_fov else None,
            coarse_shift=coarse_shifts[c] if split_fov else None,
            return_matches=True,
        )
        transforms.append(transform)
        reg_info.append(
            {
                "channel": c,
                "n_matches": n_matches,
                "ref_xy": ref_m,
                "c_xy": c_m,
                "transform": transform,
            }
        )

    if callable(progress_callback):
        progress_callback(1)

    # per-channel PSF templates from the same physical beads
    ref_xy = beads_ref[["x", "y"]].to_numpy(dtype=np.float64)
    gs = localize.gs
    per_channel = []
    for c in range(n_channels):
        if c == 0:
            beads_c = beads_ref
        else:
            mapped = localize.apply_affine_transform(ref_xy, transforms[c])
            beads_c = pd.DataFrame(
                {
                    "x": np.rint(mapped[:, 0]).astype(int),
                    "y": np.rint(mapped[:, 1]).astype(int),
                    # same physical beads, so carry each bead's FOV over so
                    # this channel is also extracted per FOV
                    "fov": beads_ref["fov"].to_numpy(),
                }
            )
        built = build_psf_template(
            movies[c],
            camera_infos[c],
            box,
            minimum_ng,
            d,
            frames_per_step=frames_per_step,
            frame_bounds=frame_bounds,
            frame_order=frame_order,
            beads=beads_c,
            return_spots=True,  # per-channel axial-precision diagnostic
        )
        per_channel.append(built)

    if callable(progress_callback):
        progress_callback(2)

    # assemble coefficients (64, n_int_x, n_int_y, n_int_z, n_channels).
    # Swap the lateral axes (row=y, col=x) -> (x, y) so the spline's first axis
    # is x, matching the model's fast pixel index (see calibrate_spline).
    templates = [
        np.ascontiguousarray(p["template"].transpose(1, 0, 2))
        for p in per_channel
    ]
    n_intervals = [int(i) for i in (np.array(templates[0].shape) - 1)]
    coefficients = np.zeros(
        [64] + n_intervals + [n_channels], dtype=np.float32
    )
    for c, template in enumerate(templates):
        coeff_c = gs.spline_coefficients(template)
        coefficients[..., c] = np.reshape(coeff_c, [64] + n_intervals)

    ref = per_channel[0]
    # z_init (sharpest slice, fit initialization) vs z_origin (output z = 0
    # reference: raw stage-scan zero, or the intensity focus with
    # correct_z_bias); see ``calibrate_spline`` for why they must be decoupled.
    z_init = float(ref["z_center"])
    z_origin = (
        ref["z_focus"]
        if correct_z_bias
        else _scan_center_index(ref["z_of_step"])
    )
    pixelsize = camera_infos[0].get("Pixelsize", 130)
    calibration = {
        "model": "spline-3d-multichannel",
        "coefficients": coefficients,
        "n_data": [int(s) for s in templates[0].shape],
        "n_intervals": n_intervals,
        "n_channels": n_channels,
        "channel_transforms": [t.tolist() for t in transforms],
        "oversampling": 1.0,
        # every channel's template is centred on its own box, so the channels
        # share one lateral origin and the linked fit is free of the constant
        # inter-channel offset an anchor-bead-centred template would carry
        "lateral_centered": True,
        "z_center": float(z_origin),
        "z_init": float(z_init),
        "z_step_nm": float(d),
        "magnification_factor": float(magnification_factor),
        "correct_z_bias": bool(correct_z_bias),
        "link_photons": bool(link_photons),
        "effective_sigma": float(ref["effective_sigma"]),
        # Per-channel amplitude->photon conversion
        "photon_scale": [float(p["photon_scale"]) for p in per_channel],
        # Per-channel focus offset
        "plane_offsets": [
            float((p["z_center"] - ref["z_center"]) * d) for p in per_channel
        ],
        "box": int(box),
        "pixelsize": float(pixelsize),
        "n_beads": int(ref["n_beads"]),
        "Frames per step": int(frames_per_step),
        "Frame order": frame_order,
        "Frame bounds": frame_bounds,
        "Generated by": (
            f"Picasso: v{__version__} Spline PSF calibration (multichannel)"
        ),
        "Path": path if path is not None else "N/A",
    }

    if photon_ratios is not None:
        ratios = np.atleast_2d(np.asarray(photon_ratios, dtype=float))
        if ratios.shape[1] != n_channels:
            raise ValueError(
                f"photon_ratios has {ratios.shape[1]} channels but the "
                f"calibration has {n_channels}."
            )
        calibration["photon_ratios"] = ratios.tolist()

    if split_fov:
        # single-movie channels: the fit path rebuilds the channel stack from
        # one movie using these regions (reference region first).
        calibration["split_fov"] = True
        calibration["reference"] = 0
        calibration["regions"] = [
            [[int(r[0][0]), int(r[0][1])], [int(r[1][0]), int(r[1][1])]]
            for r in region_rects
        ]
        # ROI-agnostic registration: store the inter-channel affine relative to
        # the region origins so the channels can be re-placed at fit time by
        # re-drawing the ROIs (the ``regions`` above are only the defaults). The
        # absolute ``channel_transforms`` are recomputed from these + the ROIs.
        channel_affines = localize.decompose_region_affines(
            region_rects, transforms
        )
        calibration["channel_affines"] = [a.tolist() for a in channel_affines]
        h = region_rects[0][1][0] - region_rects[0][0][0]
        w = region_rects[0][1][1] - region_rects[0][0][1]
        calibration["region_size"] = [int(h), int(w)]

    if path is not None:
        io.save_spline_calibration(path, calibration)
        # diagnostic PNGs: one per-channel PSF plot (reusing the single-channel
        # diagnostic, with the axial-precision panels when a GPU fitter is
        # present) plus one channel-registration plot. Never fatal.
        try:
            _save_multichannel_diagnostics(
                per_channel,
                calibration,
                coefficients,
                reg_info,
                path,
                spot_residuals=_spot_roi_residuals(
                    per_channel, ref_xy, transforms
                ),
            )
        except Exception:
            pass

    if callable(progress_callback):
        progress_callback(3)
    return calibration


# palette shared by the multichannel diagnostic figures (one color per channel)
_CHANNEL_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
]


def _decompose_affine(transform, pixelsize: float) -> dict:
    """Human-readable decomposition of a ``(2, 3)`` registration affine.

    Splits the linear part (via SVD) into a rotation, two principal scales and a
    mirror flag, plus the translation in nm. A reflected channel (biplane /
    spectral splitters) has ``mirror=True``; the reported rotation then has the
    canonical axis flip removed (the axis that minimizes the residual rotation is
    named in ``flip_axis``) so a pure mirror reads as ~0 degrees, not ~180.
    """
    M = np.asarray(transform, dtype=float)
    A = M[:, :2]
    tx = float(M[0, 2]) * pixelsize
    ty = float(M[1, 2]) * pixelsize
    det = float(np.linalg.det(A))
    mirror = det < 0
    U, S, Vt = np.linalg.svd(A)
    scale_major, scale_minor = float(S[0]), float(S[1])
    R = U @ Vt  # orthogonal part (determinant +/-1)
    flip_axis = None
    if mirror:
        best = None
        for axis, D in (
            ("x", np.array([[-1.0, 0.0], [0.0, 1.0]])),
            ("y", np.array([[1.0, 0.0], [0.0, -1.0]])),
        ):
            Rr = R @ D
            ang = float(np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0])))
            if best is None or abs(ang) < abs(best[0]):
                best = (ang, axis)
        rotation, flip_axis = best
    else:
        rotation = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return {
        "tx_nm": tx,
        "ty_nm": ty,
        "rotation_deg": rotation,
        "scale_major": scale_major,
        "scale_minor": scale_minor,
        "mirror": mirror,
        "flip_axis": flip_axis,
    }


def _save_multichannel_summary_plot(
    per_channel: list[dict],
    calibration: dict,
    joint_precision: dict | None,
    path: str,
) -> None:
    """Save the cross-channel summary PNG (``<base>_summary.png``).

    One figure consolidating what the per-channel PSF plots cannot show alone:
    the channels' axial intensity profiles overlaid (so the plane offsets /
    differential defocus that make multichannel 3D work are visible) and the
    joint all-channel z accuracy - the real localization pipeline, not a
    degenerate single plane (estimated-z-vs-stage, axial bias and precision).
    Never fatal (called inside a guarded block).
    """
    n_channels = len(per_channel)

    # z axis in the fitter's z = 0 convention (matches _save_diagnostic_plot)
    z_of_step = np.asarray(per_channel[0]["z_of_step"], dtype=float)
    n_steps = len(z_of_step)
    z_origin = float(calibration.get("z_center", 0.0))
    if n_steps > 1:
        dz = (float(z_of_step[-1]) - float(z_of_step[0])) / (n_steps - 1)
        z_ref = float(z_of_step[0]) + z_origin * dz
    else:
        z_ref = 0.0
    z_plot = z_of_step - z_ref

    have_joint = bool(joint_precision) and np.any(
        np.isfinite(joint_precision["precision_z"])
    )

    fig = Figure(figsize=(14, 3.8))
    FigureCanvasAgg(fig)
    fig.suptitle("Multichannel 3D calibration summary", fontsize=13)
    gs = fig.add_gridspec(
        1,
        4,
        wspace=0.38,
        left=0.055,
        right=0.975,
        top=0.82,
        bottom=0.18,
    )

    # (0) overlaid axial intensity profiles + per-channel focus markers
    axp = fig.add_subplot(gs[0, 0])
    for c in range(n_channels):
        col = _CHANNEL_COLORS[c % len(_CHANNEL_COLORS)]
        prof = per_channel[c]["template"].max(axis=(0, 1))
        zc = int(np.clip(int(per_channel[c]["z_center"]), 0, n_steps - 1))
        focus_nm = float(z_plot[zc])
        axp.plot(
            z_plot, prof, "-", color=col, label=f"ch{c} ({focus_nm:+.0f} nm)"
        )
        axp.axvline(focus_nm, color=col, ls=":", lw=1.0)
    axp.set_xlabel("Stage position (nm)")
    axp.set_ylabel("Peak pixel (norm.)")
    axp.set_title("Axial intensity per channel", fontsize=10)
    axp.legend(fontsize=8, loc="best", title="focus")

    if have_joint:
        lo, hi = float(np.min(z_plot)), float(np.max(z_plot))
        # (0, 1) joint estimated z vs stage
        axs = fig.add_subplot(gs[0, 1])
        st = np.asarray(joint_precision["scatter_stage"], float) - z_ref
        zf = np.asarray(joint_precision["scatter_fit"], float) - z_ref
        axs.plot(st, zf, ".k", alpha=0.1, markersize=2)
        axs.plot([lo, hi], [lo, hi], color="tab:red", lw=1.5, label="identity")
        axs.set_xlim(lo, hi)
        axs.set_ylim(lo, hi)
        axs.set_xlabel("Stage position (nm)")
        axs.set_ylabel("Estimated z (nm)")
        axs.set_title(f"Joint {n_channels}-ch fit", fontsize=10)
        axs.legend(fontsize=8)
        # (0, 2) joint axial bias
        axb = fig.add_subplot(gs[0, 2])
        axb.axhline(0.0, color="0.6", lw=1.0)
        axb.plot(z_plot, joint_precision["bias_z"], ".-", color="tab:red")
        axb.set_xlabel("Stage position (nm)")
        axb.set_ylabel("z bias (nm)")
        axb.set_title(f"Joint {n_channels}-ch fit", fontsize=10)
        # (0, 3) joint axial precision
        axpr = fig.add_subplot(gs[0, 3])
        axpr.plot(
            z_plot, joint_precision["precision_z"], ".-", color="tab:red"
        )
        axpr.set_ylim(bottom=0.0)
        axpr.set_xlabel("Stage position (nm)")
        axpr.set_ylabel("z precision (nm)")
        axpr.set_title(f"Joint {n_channels}-ch fit", fontsize=10)

    base, _ = os.path.splitext(path)
    fig.savefig(base + "_summary.png", format="png", dpi=200)


def _spot_roi_residuals(
    per_channel: list[dict], ref_xy: np.ndarray, transforms: list
) -> np.ndarray | None:
    """Per-spot, per-channel sub-pixel ROI residuals for the calibration's own
    bead spots, ``(n_spots, n_channels, 2)``, or None when they cannot be
    attributed.

    The bead spots are cut at ``rint`` of each bead's mapped position, exactly
    as at fit time, so they carry the same residual and the axial-precision
    refit has to be told about it - otherwise it measures a model mismatch that
    the real pipeline does not have. ``spot_bead_idx`` maps each spot back to
    its bead; it must agree across channels, since the caller stacks row *i* of
    every channel as one physical bead and frame."""
    idx = [p.get("spot_bead_idx") for p in per_channel]
    if any(i is None for i in idx):
        return None
    first = np.asarray(idx[0])
    if any(not np.array_equal(first, np.asarray(i)) for i in idx[1:]):
        # channels disagree on which bead each row is - the stacking the
        # caller does would be wrong too, so let it fall back
        return None
    per_bead = localize.channel_roi_residuals(
        pd.DataFrame({"x": ref_xy[:, 0], "y": ref_xy[:, 1]}), transforms
    )
    if first.size and int(first.max()) >= len(per_bead):
        return None
    return np.ascontiguousarray(per_bead[first])


def _save_multichannel_diagnostics(
    per_channel: list[dict],
    calibration: dict,
    coefficients: np.ndarray,
    reg_info: list[dict],
    path: str,
    spot_residuals: np.ndarray | None = None,
) -> None:
    """Write the multichannel calibration diagnostics next to ``path``.

    Three kinds of PNG are saved:

    * ``<base>_ch{c}.png`` - a per-channel PSF diagnostic (xy/xz/yz montages,
      axial intensity and per-channel model-vs-data agreement) via
      :func:`_save_diagnostic_plot`. The axial z-accuracy panels are *not* shown
      here: a single plane is z-degenerate, so that is a cross-channel property.
    * ``<base>_summary.png`` - the cross-channel summary
      (:func:`_save_multichannel_summary_plot`): overlaid axial intensity
      profiles with the per-channel focus (plane offsets) and the joint
      all-channel z accuracy.
    * ``<base>_registration.png`` - how well the non-reference channels align to
      the reference (residual field, RMS, and the affine decomposed into
      rotation / scale / mirror; see :func:`_save_registration_diagnostic_plot`).
    """
    n_channels = len(per_channel)
    base, _ = os.path.splitext(path)
    photon_scale = np.asarray(
        calibration.get("photon_scale", 1.0), dtype=float
    ).ravel()
    # Axial z-accuracy is a JOINT property: a single plane is z-degenerate, so it
    # is computed once from an all-channel fit (the real pipeline) and shown on
    # the summary figure, not repeated (misleadingly) on every channel's plot.
    try:
        joint_precision = _axial_precision_multichannel(
            per_channel, calibration, spot_residuals=spot_residuals
        )
    except Exception:
        joint_precision = None
    for c in range(n_channels):
        # a standalone single-channel spline-3d calibration for channel c
        calib_c = dict(calibration)
        calib_c["model"] = "spline-3d"
        calib_c["coefficients"] = np.ascontiguousarray(coefficients[..., c])
        for key in (
            "n_channels",
            "channel_transforms",
            "photon_ratios",
            "plane_offsets",
            "regions",
            "split_fov",
            "reference",
        ):
            calib_c.pop(key, None)
        if photon_scale.size == n_channels:
            calib_c["photon_scale"] = float(photon_scale[c])
        # The joint z accuracy is a cross-channel property shown once on the
        # summary figure, so the per-channel PSF plots drop the (duplicated,
        # single-plane-degenerate) axial panels and keep their own model-vs-data
        # agreement instead.
        label = "reference channel" if c == 0 else f"channel {c}"
        _save_diagnostic_plot(
            per_channel[c],
            calib_c,
            f"{base}_ch{c}.hdf5",
            precision=None,
            title_prefix=f"{label} - ",
        )
    # cross-channel summary: overlaid axial profiles and joint z accuracy
    # (see _save_multichannel_summary_plot)
    _save_multichannel_summary_plot(
        per_channel, calibration, joint_precision, path
    )
    if reg_info:
        _save_registration_diagnostic_plot(reg_info, calibration, path)


def _save_registration_diagnostic_plot(
    reg_info: list[dict], calibration: dict, path: str
) -> None:
    """Save a channel-registration diagnostic PNG (``<base>_registration.png``).

    For each non-reference channel the affine-fit residual (``transform(ref) -
    channel`` at the matched beads) is drawn as a vector field over the reference
    field of view (magnified for visibility) and summarized as an RMS bar chart.
    A footer table lists, per channel, the matched-bead count, RMS, the affine
    decomposed into rotation / principal scales / mirror (see
    :func:`_decompose_affine`) - so a reflected channel or an unexpected
    rotation/scale is obvious - plus the focus (plane) offset and photon scale. A
    tight, structure-free residual field with small RMS means the channels are
    well registered; a systematic pattern reveals a rotation/scale the affine
    could not absorb.
    """
    ps = float(calibration.get("pixelsize", 130)) or 130.0  # nm / camera px
    plane_offsets = np.asarray(
        calibration.get("plane_offsets", []), dtype=float
    ).ravel()
    photon_scale = np.asarray(
        calibration.get("photon_scale", 1.0), dtype=float
    ).ravel()
    colors = [
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
    ]

    # residuals per channel and overall bead-position bounds
    all_x, all_y = [], []
    for r in reg_info:
        ref = np.asarray(r["ref_xy"], dtype=float)
        cxy = np.asarray(r["c_xy"], dtype=float)
        M = np.asarray(r["transform"], dtype=float)
        pred = ref @ M[:, :2].T + M[:, 2]
        resid = cxy - pred  # (n, 2), camera px
        r["_resid"] = resid
        r["_ref"] = ref
        r["_rms_px"] = (
            float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
            if len(resid)
            else float("nan")
        )
        if len(ref):
            all_x.append(ref[:, 0])
            all_y.append(ref[:, 1])

    fig = Figure(figsize=(12.0, 5.4))
    FigureCanvasAgg(fig)
    fig.suptitle("Multichannel registration diagnostic", fontsize=12)

    ax = fig.add_axes([0.06, 0.26, 0.52, 0.60])
    max_resid = max(
        (
            float(np.sqrt(np.sum(r["_resid"] ** 2, axis=1)).max())
            for r in reg_info
            if len(r["_resid"])
        ),
        default=0.0,
    )
    if all_x:
        xs = np.concatenate(all_x)
        ys = np.concatenate(all_y)
        span = max(float(np.ptp(xs)), float(np.ptp(ys)), 1.0)
    else:
        span = 1.0
    # magnify residuals so the largest is ~12% of the FOV span, capped so a
    # near-perfect registration doesn't blow numerical noise up to full scale
    negligible = max_resid < 1e-3  # px: below this, residuals are just noise
    if negligible:
        mag = 1.0
    else:
        mag = float(np.clip((0.12 * span) / max_resid, 1.0, 200.0))
    for i, r in enumerate(reg_info):
        ref, resid = r["_ref"], r["_resid"]
        if not len(ref):
            continue
        color = colors[i % len(colors)]
        ax.plot(ref[:, 0], ref[:, 1], ".", color=color, markersize=3)
        if not negligible:
            ax.quiver(
                ref[:, 0],
                ref[:, 1],
                resid[:, 0] * mag,
                resid[:, 1] * mag,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                width=0.004,
                label=f"ch{r['channel']} (RMS {r['_rms_px'] * ps:.0f} nm)",
            )
    if negligible:
        ax.set_title(
            "Registration residuals (< 0.001 px, negligible)", fontsize=10
        )
    else:
        ax.set_title(
            f"Registration residuals (x{mag:.0f} magnified)", fontsize=10
        )
    ax.set_xlabel("x (camera px)")
    ax.set_ylabel("y (camera px)")
    ax.invert_yaxis()  # image convention: y increases downward
    ax.set_aspect("equal", adjustable="datalim")
    if not negligible and len(reg_info) > 1:
        ax.legend(loc="best", fontsize=8)

    # per-channel RMS bar chart
    ax2 = fig.add_axes([0.68, 0.34, 0.29, 0.52])
    chans = [f"ch{r['channel']}" for r in reg_info]
    rms_nm = [r["_rms_px"] * ps for r in reg_info]
    bar_colors = [colors[i % len(colors)] for i in range(len(reg_info))]
    ax2.bar(chans, rms_nm, color=bar_colors)
    ax2.set_ylabel("Registration RMS (nm)")
    ax2.set_title("Per-channel registration error", fontsize=10)
    ax2.set_ylim(bottom=0.0)

    # full-width text summary footer (so wide columns never clip). The affine is
    # decomposed into rotation / scale / mirror (see _decompose_affine) so a
    # reflected channel or an unexpected rotation/scale is obvious at a glance
    # rather than hidden in a raw 2x3 matrix.
    cols = (
        ("channel", 9),
        ("beads", 7),
        ("RMS", 10),
        ("rotation", 10),
        ("scale", 14),
        ("mirror", 10),
        ("focus off", 11),
        ("photon", 8),
    )
    widths = [w for _, w in cols]
    lines = ["".join(name.ljust(w) for name, w in cols)]
    for r in reg_info:
        c = r["channel"]
        po = plane_offsets[c] if c < plane_offsets.size else float("nan")
        pscale = photon_scale[c] if c < photon_scale.size else float("nan")
        dec = _decompose_affine(r["transform"], ps)
        mirror_s = f"yes ({dec['flip_axis']})" if dec["mirror"] else "no"
        fields = [
            f"ch{c}",
            str(r["n_matches"]),
            f"{r['_rms_px'] * ps:.1f} nm",
            f"{dec['rotation_deg']:+.2f}°",
            f"{dec['scale_major']:.3f}x{dec['scale_minor']:.3f}",
            mirror_s,
            f"{po:+.0f} nm",
            f"{pscale:.3f}",
        ]
        lines.append("".join(f.ljust(w) for f, w in zip(fields, widths)))
    fig.text(
        0.06,
        0.04,
        "\n".join(lines),
        fontsize=8.5,
        family="monospace",
        va="bottom",
    )

    base, _ = os.path.splitext(path)
    fig.savefig(base + "_registration.png", format="png", dpi=200)


def calibrate_spline_split_fov(
    movie,
    info,
    camera_info: dict,
    box: int,
    minimum_ng: float,
    d: float,
    regions: list,
    reference: int = 0,
    frames_per_step: int = 1,
    frame_bounds: tuple[int, int] | list | None = None,
    frame_order: Literal["fov", "z"] = "fov",
    magnification_factor: float = 0.79,
    correct_z_bias: bool = False,
    max_match_distance: float | None = None,
    photon_ratios: np.ndarray | list | None = None,
    link_photons: bool = True,
    path: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """Build a multichannel spline calibration from a *single* bead z-stack in
    which several rectangular field-of-view regions are the channels (split-FOV
    optics: spectral/ratiometric splitters, biplane relays).

    This is the acquisition geometry globLoc was designed around (Li et al.,
    Nat. Commun. 13, 3133, 2022).

    This is a thin wrapper over :func:`calibrate_spline_multichannel`: it repeats
    the one ``movie``/``info``/``camera_info`` once per region and forwards
    ``regions`` (and ``reference``) to the split-FOV path, which detects
    reference beads inside the reference region and estimates each region's
    affine from beads inside that region (pre-aligned by the known region-origin
    offset). All regions must have the same size; ``regions[reference]`` is the
    reference channel.

    Parameters
    ----------
    movie, info, camera_info
        The single bead z-stack movie, its info list and camera info dict.
    regions : list
        One ``[[y_min, x_min], [y_max, x_max]]`` rectangle per channel (as
        produced by the GUI ROI tool), all the same size.
    reference : int, optional
        Index into ``regions`` of the reference channel. Default 0.

    Remaining parameters are as in :func:`calibrate_spline_multichannel`.
    Returns a ``"spline-3d-multichannel"`` calibration with ``split_fov``,
    ``regions`` and ``reference`` stored for the fit path.
    """
    n_channels = len(regions)
    if n_channels < 2:
        raise ValueError(
            "Split-FOV calibration needs at least 2 regions; use "
            "calibrate_spline for a single channel."
        )
    return calibrate_spline_multichannel(
        movies=[movie] * n_channels,
        infos=[info] * n_channels,
        camera_infos=[camera_info] * n_channels,
        box=box,
        minimum_ng=minimum_ng,
        d=d,
        frames_per_step=frames_per_step,
        frame_bounds=frame_bounds,
        frame_order=frame_order,
        magnification_factor=magnification_factor,
        correct_z_bias=correct_z_bias,
        max_match_distance=max_match_distance,
        photon_ratios=photon_ratios,
        link_photons=link_photons,
        regions=regions,
        reference=reference,
        path=path,
        progress_callback=progress_callback,
    )


def _split_fov_local_affines(
    calibration: dict, region_rects: list
) -> list[np.ndarray]:
    """The stored region-local channel affines (reference first), decomposed
    from ``channel_transforms`` if ``channel_affines`` is absent."""
    affines = calibration.get("channel_affines")
    if affines is None:
        affines = [
            a.tolist()
            for a in localize.decompose_region_affines(
                calibration["regions"], calibration["channel_transforms"]
            )
        ]
    if len(affines) != len(region_rects):
        raise ValueError(
            f"Calibration has {len(affines)} channel affines but "
            f"{len(region_rects)} regions were given."
        )
    return [np.asarray(a, dtype=np.float64) for a in affines]


def _frames_in_bounds(
    n_frames: int, frame_bounds: tuple[int, int] | list | None
) -> np.ndarray:
    """Sorted array of the frame indices allowed by ``frame_bounds``.

    ``frame_bounds`` follows :func:`localize.identify`.
    """
    n_frames = int(n_frames)
    if frame_bounds is None:
        return np.arange(n_frames, dtype=int)
    segs = frame_bounds
    first = segs[0] if len(segs) else None
    if first is None or np.isscalar(first):
        segs = [frame_bounds]  # a single (min, max) range
    mask = np.zeros(n_frames, dtype=bool)
    for lo, hi in segs:
        lo = 0 if lo is None else max(0, int(lo))
        hi = n_frames - 1 if hi is None else min(n_frames - 1, int(hi))
        if hi >= lo:
            mask[lo : hi + 1] = True
    return np.nonzero(mask)[0]


# the four mirror orientations tried when nothing is known about the optical
# path (identity, flip-x, flip-y, flip-xy); (sx, sy) are the mirror signs
_FLIP_SIGNS = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))


def _flip_seed_transforms(
    channel: int,
    region_rects: list | None,
    frame_shape: tuple[int, int] | None,
    ref_xy: np.ndarray,
    chan_xy: np.ndarray,
) -> list[np.ndarray]:
    """Coarse reference->channel seed transforms, one per mirror orientation.

    Split-FOV (``region_rects`` given): the mirror is taken about the channel's
    region and the region origins supply the placement. Separate movies
    (``frame_shape`` given): the mirror is taken about the frame and the
    translation comes from aligning the pooled detection centroids.
    """
    seeds = []
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if region_rects is not None:
        (cy0, cx0), (cy1, cx1) = region_rects[channel]
        h, w = float(cy1 - cy0), float(cx1 - cx0)
        for sx, sy in _FLIP_SIGNS:
            local = np.array(
                [
                    [sx, 0.0, w if sx < 0 else 0.0],
                    [0.0, sy, h if sy < 0 else 0.0],
                ]
            )
            seeds.append(
                localize.compose_region_transforms(
                    [region_rects[0], region_rects[channel]],
                    [identity, local],
                )[1]
            )
        return seeds
    if len(ref_xy) == 0 or len(chan_xy) == 0:
        return [identity]
    h, w = (
        (float(frame_shape[0] - 1), float(frame_shape[1] - 1))
        if frame_shape is not None
        else (0.0, 0.0)
    )
    for sx, sy in _FLIP_SIGNS:
        seed = np.array(
            [
                [sx, 0.0, w if sx < 0 else 0.0],
                [0.0, sy, h if sy < 0 else 0.0],
            ]
        )
        pred = localize.apply_affine_transform(ref_xy, seed)
        seed[:, 2] += chan_xy.mean(axis=0) - pred.mean(axis=0)
        seeds.append(seed)
    return seeds


def estimate_transforms_from_identifications(
    identifications: list,
    box: int,
    regions: list | None = None,
    frame_shape: tuple[int, int] | None = None,
    max_frames: int = 50,
    n_iter: int = 4,
    min_pairs: int = 10,
) -> list | None:
    """Reference->channel affine transforms estimated from *identifications
    alone*, with no calibration and no prior knowledge of the optical path.

    This is the registration-free counterpart of
    :func:`refine_split_fov_transforms_from_signal` /
    :func:`refine_multichannel_transforms_from_signal`: it needs no seed
    transform and no access to the movies, only the per-channel detections that
    the Localize preview has already computed. Each mirror orientation
    (identity / flip-x / flip-y / flip-xy, see :data:`_FLIP_SIGNS`) is used as a
    coarse seed and refined by ICP - per-frame nearest-neighbour pairing with a
    shrinking radius, re-fitting the affine on the pooled correspondences - and
    the orientation that still holds the most pairs at the tightest radius wins.
    Mirrored channels (the common case for image splitters) are therefore picked
    up automatically, which an identity assumption cannot do.

    ``identifications`` is one detection table per channel in **absolute**
    coordinates, reference first; for split-FOV pass the detections of each
    region plus the ``regions`` themselves. Returns one ``(2, 3)`` transform per
    channel (the reference's is the identity) with ``None`` for channels that
    could not be registered, or ``None`` if none of them could.
    """
    n_channels = len(identifications)
    if n_channels < 2:
        return None

    # Only an evenly-spaced sample of frames is used (as the signal
    # re-registration does): the transform is global, so a few tens of frames
    # already give hundreds of correspondences, and nothing scales with the
    # length of a long movie. The frames are chosen once, on the reference, and
    # every channel is cut down to them before it is grouped per frame.
    ref_ids = identifications[0]
    if ref_ids is None or len(ref_ids) == 0:
        return None
    ref_frames = np.unique(np.asarray(ref_ids["frame"], dtype=np.int64))
    if ref_frames.size > max_frames:
        pick = np.unique(
            np.linspace(0, ref_frames.size - 1, int(max_frames)).astype(int)
        )
        ref_frames = ref_frames[pick]
    sample = set(int(f) for f in ref_frames)

    def by_frame(ids) -> dict:
        if ids is None or len(ids) == 0:
            return {}
        frame = np.asarray(ids["frame"], dtype=np.int64)
        keep = np.isin(frame, ref_frames)
        frame = frame[keep]
        xy = np.column_stack(
            [
                np.asarray(ids["x"], dtype=np.float64)[keep],
                np.asarray(ids["y"], dtype=np.float64)[keep],
            ]
        )
        return {int(f): xy[frame == f] for f in np.unique(frame)}

    per_channel = [by_frame(ids) for ids in identifications]
    ref_by_frame = per_channel[0]
    if not ref_by_frame:
        return None
    region_rects = None
    if regions is not None:
        if len(regions) != n_channels:
            return None
        region_rects = [_normalized_region(r) for r in regions]

    # radii shrink from generous (absorb the coarse seed's error) to sub-box
    tols = np.linspace(
        1.5 * float(box), max(2.0, 0.3 * float(box)), max(1, int(n_iter))
    )
    transforms: list = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
    for c in range(1, n_channels):
        chan_by_frame = per_channel[c]
        common = sorted(sample & set(chan_by_frame))
        if not common:
            transforms.append(None)
            continue
        ref_pool = np.vstack([ref_by_frame[f] for f in common])
        chan_pool = np.vstack([chan_by_frame[f] for f in common])
        best_pairs, best_transform = 0, None
        for seed in _flip_seed_transforms(
            c, region_rects, frame_shape, ref_pool, chan_pool
        ):
            transform = np.asarray(seed, dtype=np.float64)
            n_pairs = 0
            for tol in tols:
                acc_ref, acc_c = [], []
                for f in common:
                    rxy, cxy = ref_by_frame[f], chan_by_frame[f]
                    pred = localize.apply_affine_transform(rxy, transform)
                    ri, ci = _match_beads(pred, cxy, tol)
                    if len(ri):
                        acc_ref.append(rxy[ri])
                        acc_c.append(cxy[ci])
                if not acc_ref:
                    n_pairs = 0
                    break
                matched_ref = np.vstack(acc_ref)
                matched_c = np.vstack(acc_c)
                n_pairs = len(matched_ref)
                if n_pairs < 3:
                    break
                transform = localize.estimate_affine_transform(
                    matched_ref, matched_c
                )
            # a wrong orientation converges onto coincidental pairs, which are
            # few at the tightest radius and usually imply an absurd scale
            scale = abs(float(np.linalg.det(transform[:, :2])))
            if n_pairs > best_pairs and 0.5 <= scale <= 2.0:
                best_pairs, best_transform = n_pairs, transform
        transforms.append(best_transform if best_pairs >= min_pairs else None)
    if all(t is None for t in transforms[1:]):
        return None
    return transforms


def refine_split_fov_transforms_from_signal(
    movie,
    calibration: dict,
    regions: list,
    minimum_ng: float,
    box: int | None = None,
    reference: int = 0,
    frame_bounds: tuple[int, int] | list | None = None,
    max_frames: int = 50,
    max_pair_distance: float | None = None,
    n_iter: int = 4,
    min_pairs: int = 20,
    update: bool = True,
) -> tuple[dict, list]:
    """Re-register a split-FOV spline calibration from the experimental
    (blinking) data.

    The channels of a split-FOV calibration (biplane / ratiometric) share
    single-molecule signal: the same emitter fluoresces in the reference region
    and every other region in the same frame. This re-fits the inter-channel
    affine directly from that signal:

    1. Single-molecule positions are detected inside the drawn ``regions`` on a
       bounded, evenly-spaced sample of ``max_frames`` frames (several tens
       should be enough shared blinks to fit an affine without scanning the
       whole movie).
    2. Each channel is coarsely overlaid on the reference using only the flip
       the original calibration applied plus the drawn region offset - the
       stored fine rotation / scale / translation is discarded, so a stale or
       different-field registration cannot bias the result.
    3. Reference and channel detections are paired frame by frame at that
       seed and a fresh affine is fit on the pooled correspondences; a few ICP
       iterations with a shrinking radius tighten it and a final robust trim
       drops the coincidental (non-physical) pairs.

    ``regions`` are the channel ROIs in the current data (reference first, e.g.
    freshly drawn in the GUI). The PSF ``coefficients`` are untouched; only the
    registration (``channel_affines`` / ``channel_transforms`` / ``regions`` /
    ``region_size``) is updated. Requires channels that share signal.

    Returns ``(calibration, reg_info)`` where ``reg_info`` lists, per channel,
    the number of matched pairs, their coordinates, the fitted transform and the
    residual RMS (camera px). With ``update=True`` the passed calibration is
    modified in place; set it False to only compute the transforms.
    """
    if not calibration.get("split_fov"):
        raise ValueError(
            "refine_split_fov_transforms_from_signal requires a split-FOV "
            "calibration."
        )
    n_channels = int(calibration.get("n_channels", len(regions)))
    if len(regions) != n_channels:
        raise ValueError(
            f"Got {len(regions)} regions but the calibration has "
            f"{n_channels} channels."
        )
    if not (0 <= reference < n_channels):
        raise ValueError(f"reference={reference} out of range.")
    if box is None:
        box = int(calibration.get("box") or calibration["n_data"][0])
    if max_pair_distance is None:
        # the coarse seed is flip + ROI placement only, so allow a little more
        # slack than one box for an imperfectly drawn ROI (still well below the
        # typical single-molecule spacing, so per-frame matches stay unambiguous)
        max_pair_distance = 1.5 * float(box)

    # reference-first ordering for both the regions and the stored affines
    order = [reference] + [c for c in range(n_channels) if c != reference]
    region_rects = [_normalized_region(regions[c]) for c in order]
    sizes = {(r[1][0] - r[0][0], r[1][1] - r[0][1]) for r in region_rects}
    if len(sizes) != 1:
        raise ValueError("All regions must have the same size.")
    affines_stored = _split_fov_local_affines(calibration, regions)
    affines = [affines_stored[c] for c in order]

    # coarse seed = only the flip the calibration applied placed at the
    # drawn regions
    h = region_rects[0][1][0] - region_rects[0][0][0]
    w = region_rects[0][1][1] - region_rects[0][0][1]
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    seed_local_affines = [identity]
    for c in range(1, n_channels):
        dec = _decompose_affine(affines[c], pixelsize=1.0)
        if dec["mirror"] and dec["flip_axis"] == "x":
            seed_local_affines.append(
                np.array([[-1.0, 0.0, float(w)], [0.0, 1.0, 0.0]])
            )
        elif dec["mirror"] and dec["flip_axis"] == "y":
            seed_local_affines.append(
                np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(h)]])
            )
        else:
            seed_local_affines.append(identity.copy())
    seed_transforms = localize.compose_region_transforms(
        region_rects, seed_local_affines
    )

    # detect only on a bounded, evenly-spaced sample of frames (several tens):
    # the same subset is used for every region so per-frame pairing stays aligned
    n_frames = int(movie.shape[0])
    allowed = _frames_in_bounds(n_frames, frame_bounds)
    if allowed.size == 0:
        raise ValueError("No frames in the requested frame range.")
    pick = np.unique(
        np.linspace(
            0, allowed.size - 1, min(int(max_frames), allowed.size)
        ).astype(int)
    )
    sample_frames = allowed[pick]
    movie_sub = np.stack([np.asarray(movie[int(f)]) for f in sample_frames])

    # per-region, per-frame detections (absolute coords) on the sampled frames
    def _by_frame(rect):
        ids, _ = localize.identify(movie_sub, minimum_ng, box, roi=rect)
        if len(ids) == 0:
            return {}
        frame = np.asarray(ids["frame"], dtype=np.int64)
        xy = np.column_stack(
            [
                np.asarray(ids["x"], dtype=np.float64),
                np.asarray(ids["y"], dtype=np.float64),
            ]
        )
        out = {}
        for f in np.unique(frame):
            out[int(f)] = xy[frame == f]
        return out

    ref_by_frame = _by_frame(region_rects[0])
    if not ref_by_frame:
        raise ValueError(
            "No detections in the reference region; lower the minimum net "
            "gradient or check the drawn ROIs / frame range."
        )

    # match radii shrink from generous (absorb the seed's residual drift) to
    # tight (sub-box) over the ICP iterations
    tol_hi = float(max_pair_distance)
    tol_lo = max(2.0, 0.3 * box)
    tols = np.linspace(tol_hi, tol_lo, max(1, int(n_iter)))

    transforms = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
    reg_info = []
    for c in range(1, n_channels):
        chan_by_frame = _by_frame(region_rects[c])
        common = sorted(set(ref_by_frame) & set(chan_by_frame))
        if not common:
            raise ValueError(
                f"No frames with detections in both the reference and channel "
                f"{c} regions; the channels may not share signal (needs "
                "biplane / ratiometric data)."
            )
        transform = np.asarray(seed_transforms[c], dtype=np.float64)
        matched_ref = matched_c = np.empty((0, 2))
        for tol in tols:
            acc_ref, acc_c = [], []
            for f in common:
                rxy = ref_by_frame[f]
                cxy = chan_by_frame[f]
                pred = localize.apply_affine_transform(rxy, transform)
                ri, ci = _match_beads(pred, cxy, tol)
                if len(ri):
                    acc_ref.append(rxy[ri])
                    acc_c.append(cxy[ci])
            if not acc_ref:
                break
            matched_ref = np.vstack(acc_ref)
            matched_c = np.vstack(acc_c)
            if len(matched_ref) < 3:
                break
            transform = localize.estimate_affine_transform(
                matched_ref, matched_c
            )
        # robust trim: drop coincidental pairs far from the converged transform,
        # then re-fit once on the inliers
        if len(matched_ref) >= 3:
            resid = matched_c - localize.apply_affine_transform(
                matched_ref, transform
            )
            dist = np.sqrt(np.sum(resid**2, axis=1))
            keep = dist <= max(tol_lo, 3.0 * np.median(dist))
            if keep.sum() >= 3:
                matched_ref = matched_ref[keep]
                matched_c = matched_c[keep]
                transform = localize.estimate_affine_transform(
                    matched_ref, matched_c
                )
        n_pairs = int(len(matched_ref))
        if n_pairs < min_pairs:
            raise ValueError(
                f"Only {n_pairs} signal correspondences for channel {c} "
                f"(need >= {min_pairs}); use a longer / denser movie, lower the "
                "minimum net gradient, or re-register on beads instead."
            )
        resid = matched_c - localize.apply_affine_transform(
            matched_ref, transform
        )
        rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
        transforms.append(transform)
        reg_info.append(
            {
                "channel": c,
                "n_matches": n_pairs,
                "ref_xy": matched_ref,
                "c_xy": matched_c,
                "transform": transform,
                "rms": rms,
            }
        )

    new_affines = localize.decompose_region_affines(region_rects, transforms)
    if update:
        calibration["channel_affines"] = [a.tolist() for a in new_affines]
        calibration["channel_transforms"] = [t.tolist() for t in transforms]
        calibration["regions"] = [
            [[int(r[0][0]), int(r[0][1])], [int(r[1][0]), int(r[1][1])]]
            for r in region_rects
        ]
        h0 = region_rects[0][1][0] - region_rects[0][0][0]
        w0 = region_rects[0][1][1] - region_rects[0][0][1]
        calibration["region_size"] = [int(h0), int(w0)]
        calibration["reference"] = 0
    return calibration, reg_info


def refine_multichannel_transforms_from_signal(
    movies: list,
    calibration: dict,
    minimum_ng: float,
    box: int | None = None,
    reference: int = 0,
    frame_bounds: tuple[int, int] | list | None = None,
    max_frames: int = 50,
    max_pair_distance: float | None = None,
    n_iter: int = 4,
    min_pairs: int = 20,
    update: bool = True,
) -> tuple[dict, list]:
    """Re-register a separate-movie multichannel spline calibration from the
    experimental (blinking) data.

    The multi-movie analogue of :func:`refine_split_fov_transforms_from_signal`.
    In a multichannel acquisition the same emitter fluoresces in every channel's
    movie in the *same frame* (the channels are frame-synchronized, as the
    multichannel linking in :func:`localize.link_identifications_multichannel`
    already assumes), so the inter-channel affine can be re-fit directly from that
    shared signal:

    1. Single molecules are detected in each channel's movie on a bounded,
       evenly-spaced sample of ``max_frames`` frames.
    2. The calibration's **existing** ``channel_transforms`` seed the pairing -
       they are already close, so no flip/cross-correlation search is needed;
       this refines a registration that drifted (e.g. across days), it does not
       recover a grossly wrong one.
    3. Reference and channel detections are paired frame by frame at that seed and
       a fresh affine is fit on the pooled correspondences; a few ICP iterations
       with a shrinking radius tighten it and a final robust trim drops the
       coincidental (non-physical) pairs.

    Unlike the split-FOV refinement this needs no ``regions`` - the channels are
    whole separate movies - so only ``channel_transforms`` is updated (the PSF
    ``coefficients`` are untouched). ``movies`` are the loaded channel movies in
    calibration order (reference first).

    Returns ``(calibration, reg_info)`` where ``reg_info`` lists, per non-
    reference channel, the number of matched pairs, their coordinates, the fitted
    transform and the residual RMS (camera px). With ``update=True`` the passed
    calibration is modified in place.
    """
    if calibration.get("split_fov"):
        raise ValueError(
            "refine_multichannel_transforms_from_signal is for separate-movie "
            "multichannel calibrations; use "
            "refine_split_fov_transforms_from_signal for split-FOV."
        )
    stored = calibration.get("channel_transforms")
    if not stored:
        raise ValueError("Calibration has no channel_transforms to refine.")
    n_channels = int(calibration.get("n_channels", len(stored)))
    if len(movies) != n_channels:
        raise ValueError(
            f"Got {len(movies)} channel movies but the calibration has "
            f"{n_channels} channels."
        )
    if len(stored) != n_channels:
        raise ValueError(
            f"Calibration has {n_channels} channels but {len(stored)} "
            "channel transforms."
        )
    if not (0 <= reference < n_channels):
        raise ValueError(f"reference={reference} out of range.")
    if box is None:
        box = int(calibration.get("box") or calibration["n_data"][0])
    if max_pair_distance is None:
        # the seed is the stored (already close) transform, so a match radius of
        # about one box absorbs the residual drift without inviting coincidental
        # cross-molecule pairs
        max_pair_distance = float(box)

    seed_transforms = [np.asarray(t, dtype=np.float64) for t in stored]

    # detect on a bounded, evenly-spaced sample of frames shared by every movie,
    # so per-frame pairing across the synchronized movies stays aligned
    n_frames = min(int(m.shape[0]) for m in movies)
    allowed = _frames_in_bounds(n_frames, frame_bounds)
    if allowed.size == 0:
        raise ValueError("No frames in the requested frame range.")
    pick = np.unique(
        np.linspace(
            0, allowed.size - 1, min(int(max_frames), allowed.size)
        ).astype(int)
    )
    sample_frames = allowed[pick]

    def _by_frame(movie):
        """Per-frame detections (absolute coords) on the sampled frames."""
        movie_sub = np.stack(
            [np.asarray(movie[int(f)]) for f in sample_frames]
        )
        ids, _ = localize.identify(movie_sub, minimum_ng, box)
        if len(ids) == 0:
            return {}
        frame = np.asarray(ids["frame"], dtype=np.int64)
        xy = np.column_stack(
            [
                np.asarray(ids["x"], dtype=np.float64),
                np.asarray(ids["y"], dtype=np.float64),
            ]
        )
        return {int(f): xy[frame == f] for f in np.unique(frame)}

    ref_by_frame = _by_frame(movies[reference])
    if not ref_by_frame:
        raise ValueError(
            "No detections in the reference channel; lower the minimum net "
            "gradient or check the frame range."
        )

    # match radii shrink from generous (absorb the seed's residual drift) to
    # tight (sub-box) over the ICP iterations
    tol_hi = float(max_pair_distance)
    tol_lo = max(2.0, 0.3 * box)
    tols = np.linspace(tol_hi, tol_lo, max(1, int(n_iter)))

    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transforms = [
        identity if c == reference else None for c in range(n_channels)
    ]
    reg_info = []
    for c in range(n_channels):
        if c == reference:
            continue
        chan_by_frame = _by_frame(movies[c])
        common = sorted(set(ref_by_frame) & set(chan_by_frame))
        if not common:
            raise ValueError(
                f"No frames with detections in both the reference and channel "
                f"{c} movies; the channels may not share signal, or the movies "
                "are not frame-synchronized."
            )
        transform = np.asarray(seed_transforms[c], dtype=np.float64)
        matched_ref = matched_c = np.empty((0, 2))
        for tol in tols:
            acc_ref, acc_c = [], []
            for f in common:
                rxy = ref_by_frame[f]
                cxy = chan_by_frame[f]
                pred = localize.apply_affine_transform(rxy, transform)
                ri, ci = _match_beads(pred, cxy, tol)
                if len(ri):
                    acc_ref.append(rxy[ri])
                    acc_c.append(cxy[ci])
            if not acc_ref:
                break
            matched_ref = np.vstack(acc_ref)
            matched_c = np.vstack(acc_c)
            if len(matched_ref) < 3:
                break
            transform = localize.estimate_affine_transform(
                matched_ref, matched_c
            )
        # robust trim: drop coincidental pairs far from the converged transform,
        # then re-fit once on the inliers
        if len(matched_ref) >= 3:
            resid = matched_c - localize.apply_affine_transform(
                matched_ref, transform
            )
            dist = np.sqrt(np.sum(resid**2, axis=1))
            keep = dist <= max(tol_lo, 3.0 * np.median(dist))
            if keep.sum() >= 3:
                matched_ref = matched_ref[keep]
                matched_c = matched_c[keep]
                transform = localize.estimate_affine_transform(
                    matched_ref, matched_c
                )
        n_pairs = int(len(matched_ref))
        if n_pairs < min_pairs:
            raise ValueError(
                f"Only {n_pairs} signal correspondences for channel {c} "
                f"(need >= {min_pairs}); use a longer / denser movie, lower the "
                "minimum net gradient, or re-calibrate on beads instead."
            )
        resid = matched_c - localize.apply_affine_transform(
            matched_ref, transform
        )
        rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
        transforms[c] = transform
        reg_info.append(
            {
                "channel": c,
                "n_matches": n_pairs,
                "ref_xy": matched_ref,
                "c_xy": matched_c,
                "transform": transform,
                "rms": rms,
            }
        )

    if update:
        calibration["channel_transforms"] = [t.tolist() for t in transforms]
    return calibration, reg_info


def _even_slice_indices(n_total: int, n_want: int, forced: int | None = None):
    """Pick up to ``n_want`` evenly spaced indices over ``[0, n_total - 1]``.

    If ``forced`` is given, the nearest picked index is snapped to it so the
    slice is guaranteed to appear without inflating the count.
    """
    idx = np.rint(np.linspace(0, n_total - 1, min(n_want, n_total))).astype(
        int
    )
    if forced is not None and len(idx):
        idx[np.argmin(np.abs(idx - forced))] = forced
    return np.unique(idx)


def _place_row(fig, panels, top_in, fig_w_in, fig_h_in, scale, gap_in):
    """Draw one horizontally-centered row of image panels at a fixed scale.

    Each entry of ``panels`` is ``(image, title, imshow_kwargs, w_px, h_px,
    highlight, hline)``: the axes is sized to ``w_px`` x ``h_px`` *camera
    pixels* times ``scale`` (inches/pixel), so one camera pixel renders at the
    same physical size in every panel of the figure regardless of projection.
    ``highlight`` outlines the panel (in-focus slice); ``hline`` (or ``None``)
    draws a focal-plane marker in data coordinates. Panels are drawn with
    ``aspect="auto"`` so the image fills the correctly-proportioned axes.
    Returns the y (inches from the figure top) just below the row.
    """
    row_w_in = sum(p[3] * scale for p in panels) + gap_in * (len(panels) - 1)
    row_h_in = max(p[4] for p in panels) * scale
    x_in = (fig_w_in - row_w_in) / 2.0
    for img, ptitle, kw, w_px, h_px, highlight, hline in panels:
        w_in, h_in = w_px * scale, h_px * scale
        ax = fig.add_axes(
            [
                x_in / fig_w_in,
                (fig_h_in - top_in - h_in) / fig_h_in,
                w_in / fig_w_in,
                h_in / fig_h_in,
            ]
        )
        ax.imshow(img, aspect="auto", **kw)
        ax.set_title(ptitle, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if hline is not None:
            ax.axhline(hline, color="cyan", lw=0.8)
        if highlight:
            for spine in ax.spines.values():
                spine.set(color="cyan", linewidth=2.5, visible=True)
        x_in += w_in + gap_in
    return top_in + row_h_in


def _robust_bias_spread(deviation: np.ndarray) -> tuple[float, float]:
    """Robust ``(bias, spread)`` of a 1D deviation-to-truth array.

    ``bias`` is the mean deviation (the systematic z offset from the true stage
    position - e.g. a per-bead/registration offset) and ``spread`` is the
    standard deviation about that mean (the shot-noise-limited single-frame
    precision). Splitting the two keeps a consistent bias from masquerading as
    imprecision (the total RMSD is ``sqrt(bias**2 + spread**2)``). Robust to the
    occasional non-converged fit: deviations farther than 5 (scaled) MADs from
    the median are dropped first. Returns ``(nan, nan)`` if fewer than two
    finite values are present."""
    dev = np.asarray(deviation, dtype=np.float64)
    dev = dev[np.isfinite(dev)]
    if dev.size < 2:
        return float("nan"), float("nan")
    med = np.median(dev)
    mad = np.median(np.abs(dev - med))
    if mad > 0:
        keep = np.abs(dev - med) <= 5.0 * 1.4826 * mad
        if keep.sum() >= 2:
            dev = dev[keep]
    return float(np.mean(dev)), float(np.std(dev))


def _axial_precision(built: dict, calibration: dict) -> dict | None:
    """Empirical axial precision of the spline PSF model across z.

    Every individual single-frame bead spot (``built["spots"]``) is fitted with
    the calibration's own spline PSF model (the very GPU fitter used at
    localization time) and the recovered z - referenced to the scan center, the
    same zero as ``z_of_step`` - is compared against the known stage position of
    its frame. The per-z-step RMSD of that deviation is the calibration's axial
    precision at that z - the spline analog of the "mean z precision" curve in
    ``zfit.calibrate_z``, which likewise fits the calibration data back with its
    own model, per single-frame localization, and reports the RMSD to the true
    stage position. Fitting the raw per-frame
    spots (rather than the frame-averaged bead volumes) keeps the realistic,
    single-frame shot-noise regime and gives many samples per z-step. Only z
    has a ground truth (the known stage position), so no lateral precision is
    reported.

    Parameters
    ----------
    built : dict
        ``build_psf_template`` output built with ``return_spots=True``; uses
        ``spots`` (``(n_spots, box, box)``, photon units), ``spot_step_idx``
        (each spot's index into the template z-axis) and ``z_of_step`` (the
        known stage position, nm, of each z-step).
    calibration : dict
        The spline calibration (must already contain the fitted
        ``coefficients``; see ``calibrate_spline``).

    Returns
    -------
    precision : dict or None
        ``{"bias_z", "precision_z", "scatter_fit", "scatter_stage", "n_beads",
        "n_spots"}``. Per z-step and aligned to ``z_of_step``: the systematic z
        ``bias_z`` (mean deviation from the true stage position) and the
        shot-noise ``precision_z`` (std about that mean), both in nm.
        ``scatter_fit``/``scatter_stage`` are per-spot (subsampled) estimated z
        and stage position, in raw stage nm, for the fitted-vs-truth scatter.
        Returns ``None`` for a 2D calibration (no z), when the per-frame spots
        are absent, or when GPU spline fitting is unavailable or fails, so the
        caller can fall back to the model-vs-data RMSE panel.
    """
    if not localize.GPUFIT_INSTALLED:
        return None
    if calibration["model"] == "spline-2d":
        return None  # no axial coordinate to assess
    spots = built.get("spots")
    spot_step_idx = built.get("spot_step_idx")
    if spots is None or spot_step_idx is None or len(spots) < 2:
        return None
    z_of_step = np.asarray(built["z_of_step"], dtype=np.float64)
    spot_step_idx = np.asarray(spot_step_idx)
    # pixel layout stays (row=y, col=x), exactly as ``localize.get_spots``
    # returns - the fitter and the coefficient table already account for the
    # lateral axis order (see ``calibrate_spline``). This is the same spot
    # stack the normal fitting path consumes.
    spots = np.ascontiguousarray(spots)
    try:
        # MLE (Poisson) matches the shot-noise statistics of the single-frame
        # spots and stays closer to the CRLB than least squares, especially for
        # the dim, defocused spots in the tails.
        theta = localize.fit_spots_gpufit_spline(spots, calibration, mle=True)
    except Exception:
        return None
    return _axial_precision_from_theta(
        theta,
        spot_step_idx,
        z_of_step,
        calibration,
        int(built.get("n_beads", 0)),
    )


def _axial_precision_from_theta(
    theta: np.ndarray,
    spot_step_idx: np.ndarray,
    z_of_step: np.ndarray,
    calibration: dict,
    n_beads: int,
    z_col: int = 3,
) -> dict | None:
    """Per-z-step axial bias/precision from fitted spline parameters.

    Shared tail of :func:`_axial_precision` (single channel) and
    :func:`_axial_precision_multichannel` (joint fit): converts the fitted z
    (``theta[:, z_col]``) to stage nm, compares it to each spot's known stage
    position and reduces to a robust per-step bias and spread. ``z_col`` is the
    z_shift parameter column: 3 for the amplitude-shared models, 2 for the
    photon-decoupled (link-XYZ) model. Returns the same dict shape both callers
    emit, or ``None`` if nothing usable remains.
    """
    theta = np.asarray(theta)
    z_of_step = np.asarray(z_of_step, dtype=np.float64)
    spot_step_idx = np.asarray(spot_step_idx)
    z_step_nm = float(calibration.get("z_step_nm", 1.0))
    scan_center = _scan_center_index(z_of_step)
    z_fit = (theta[:, z_col] + scan_center) * z_step_nm  # (n_spots,)
    deviation = z_fit - z_of_step[spot_step_idx]  # (n_spots,)

    n_steps = len(z_of_step)
    bias_spread = [
        _robust_bias_spread(deviation[spot_step_idx == i])
        for i in range(n_steps)
    ]
    bias_z = np.array([b for b, _ in bias_spread])
    precision_z = np.array([s for _, s in bias_spread])
    if not np.any(np.isfinite(precision_z)):
        return None

    z_true = z_of_step[spot_step_idx]
    finite = np.isfinite(z_fit) & np.isfinite(z_true)
    scatter_fit = z_fit[finite]
    scatter_stage = z_true[finite]
    cap = 20000
    if scatter_fit.size > cap:
        stride = int(np.ceil(scatter_fit.size / cap))
        scatter_fit = scatter_fit[::stride]
        scatter_stage = scatter_stage[::stride]

    return {
        "bias_z": bias_z,
        "precision_z": precision_z,
        "scatter_fit": scatter_fit,
        "scatter_stage": scatter_stage,
        "n_beads": int(n_beads),
        "n_spots": int(np.isfinite(deviation).sum()),
    }


def _axial_precision_multichannel(
    per_channel: list[dict],
    calibration: dict,
    spot_residuals: np.ndarray | None = None,
) -> dict | None:
    """Joint (all-channel) axial precision of a multichannel spline
    calibration.

    ``spot_residuals`` (``(n_spots, n_channels, 2)``, see
    :func:`_spot_roi_residuals`) are the sub-pixel ROI offsets of the bead
    spots being refitted."""
    if not localize.GPUFIT_INSTALLED:
        return None
    # only the plain multichannel spline fitter is used here
    if calibration.get("model") != "spline-3d-multichannel":
        return None
    spots_c = [p.get("spots") for p in per_channel]
    step_idx = per_channel[0].get("spot_step_idx")
    if any(s is None for s in spots_c) or step_idx is None:
        return None
    n_spots = spots_c[0].shape[0]
    if any(s.shape[0] != n_spots for s in spots_c) or n_spots < 2:
        return None
    # (n_spots, box, box) per channel -> (n_spots, box, box, n_channels)
    spots = np.ascontiguousarray(np.stack(spots_c, axis=-1))
    # Refit with the model the user chose for this calibration
    n_channels = len(per_channel)
    link_photons = bool(calibration.get("link_photons", True))
    if not link_photons and (
        2 <= n_channels <= localize._LINK_XYZ_MAX_CHANNELS
    ):
        fit_cal = localize._as_link_xyz_calibration(calibration)
        z_col = 2
    else:
        fit_cal = calibration
        z_col = 3
    # z multi-start (like globLoc, Li et al., Nat. Commun. 13, 3133,
    # 2022): a single in-focus start leaves the biplane fit degenerate at
    # large |z|; several z seeds recover it.
    n_z = int(calibration["n_data"][2])
    n_z_starts = int(np.clip(n_z // 20, 5, 15))
    if spot_residuals is not None and (
        np.asarray(spot_residuals).shape != (n_spots, n_channels, 2)
    ):
        spot_residuals = None
    try:
        theta = localize.fit_spots_gpufit_spline(
            spots,
            fit_cal,
            mle=True,
            n_z_starts=n_z_starts,
            residuals=spot_residuals,
        )
    except Exception:
        return None
    result = _axial_precision_from_theta(
        theta,
        step_idx,
        per_channel[0]["z_of_step"],
        calibration,
        int(per_channel[0].get("n_beads", 0)),
        z_col=z_col,
    )
    if result is not None:
        result["joint"] = int(len(per_channel))
    return result


def _save_diagnostic_plot(
    built: dict,
    calibration: dict,
    path: str,
    n_slices: int = 10,
    precision: dict | None = None,
    title_prefix: str = "",
) -> None:
    """Save a PNG summarizing the calibration.

    Three montages of ``n_slices`` slices each - xy across z, xz across y, yz
    across x - plus, at the bottom, the axial intensity profile and, when
    ``precision`` is given (from re-fitting the beads, see ``_axial_precision``),
    three further panels: the estimated-z-vs-stage scatter (with the identity
    line), the systematic axial bias (z offset to the true stage position) and
    the shot-noise axial precision (z spread). Without ``precision`` the bottom
    row falls back to a single model-vs-data agreement (per-z RMSE) panel. The
    xz/yz cross-sections are oriented with z on the
    vertical axis (lateral on the horizontal). Every image panel shares one
    intensity scale, and one camera pixel renders at the same physical size in
    all panels: the z axis of the cross-sections is converted from nm to camera
    pixels via the calibration pixel size.
    """
    template = built["template"]
    z_center = int(built["z_center"])
    z_of_step = np.asarray(built["z_of_step"])
    gof = built.get("gof") or {}
    have_gof = bool(gof.get("n_used"))
    have_prec = bool(precision) and np.any(
        np.isfinite(precision["precision_z"])
    )
    box, _, n_steps = template.shape
    c = box // 2
    ps = float(calibration.get("pixelsize", 130)) or 130.0  # nm / camera px
    vmax = float(template.max()) or 1.0
    img_kw = dict(cmap="hot", vmin=0.0, vmax=vmax)

    # z = 0 reference the fitter (localize) uses
    z_origin = float(calibration.get("z_center", z_center))
    if n_steps > 1:
        dz_step = (float(z_of_step[-1]) - float(z_of_step[0])) / (n_steps - 1)
        z_ref_nm = (
            float(z_of_step[0]) + z_origin * dz_step
        )  # stage nm at origin
    else:
        z_ref_nm = 0.0
    z_plot = np.asarray(z_of_step, dtype=float) - z_ref_nm
    # After the shift the fitter's z = 0 reference sits at 0; the bottom profile
    # plots mark it with a vertical line. The in-focus (sharpest) slice is
    # outlined in the xy row and marked by the cyan line in the cross-sections
    z_origin_marker = 0.0
    outline_idx = int(np.clip(z_center, 0, n_steps - 1))
    z_outline_nm = float(z_plot[outline_idx])
    x_label = "Stage position (nm)"
    z_lo, z_hi = float(z_plot[-1]), float(z_plot[0])  # z decreases

    # slice indices for each projection
    z_idx = _even_slice_indices(n_steps, n_slices, forced=outline_idx)
    y_idx = _even_slice_indices(box, n_slices, forced=c)
    x_idx = _even_slice_indices(box, n_slices, forced=c)

    # cross-sections: z (converted to camera px) on the vertical axis, lateral
    # px on the horizontal axis, with z increasing upward. Extents are given at
    # pixel *edges* (half a pixel/step beyond the first and last sample
    # centers) so N pixels span exactly N units on every axis - the lateral and
    # z axes then render one camera pixel at the identical physical size, so xy,
    # xz and yz share one pixel scale. (Center-based extents would span only
    # box-1 laterally while the panel is box wide, stretching lateral pixels
    # ~box/(box-1) relative to z.)
    dz_px = (abs(z_hi - z_lo) / (n_steps - 1)) / ps if n_steps > 1 else 1.0
    z_top = z_hi / ps + dz_px / 2.0  # +z edge (top of the panel)
    z_bot = z_lo / ps - dz_px / 2.0  # -z edge (bottom of the panel)
    z_span_px = z_top - z_bot
    lat_lo, lat_hi = -c - 0.5, box - c - 0.5  # spans exactly `box` pixels
    cs_ext = [lat_lo, lat_hi, z_bot, z_top]
    cs_kw = dict(extent=cs_ext, origin="upper", **img_kw)
    xy_ext = [lat_lo, lat_hi, lat_hi, lat_lo]
    xy_kw = dict(extent=xy_ext, origin="upper", **img_kw)

    z_outline_px = (
        z_outline_nm / ps
    )  # cyan cross-section line at the sharpest slice
    # (image, title, imshow_kwargs, w_px, h_px, highlight, hline)
    xy_panels = [
        (
            template[:, :, k],
            f"z = {z_plot[k] + 0.0:.0f} nm",
            xy_kw,
            box,
            box,
            k == outline_idx,
            None,
        )
        for k in z_idx[::-1]
    ]
    # rotate 90 deg: transpose so z runs down the rows, lateral across columns
    xz_panels = [
        (
            template[y, :, :].T,
            f"y = {y - c:+d} px",
            cs_kw,
            box,
            z_span_px,
            False,
            z_outline_px,
        )
        for y in y_idx
    ]
    yz_panels = [
        (
            template[:, x, :].T,
            f"x = {x - c:+d} px",
            cs_kw,
            box,
            z_span_px,
            False,
            z_outline_px,
        )
        for x in x_idx
    ]

    # Manual inch-based layout so the pixel scale is identical across rows.
    scale = 0.09  # inches per camera pixel
    gap_in = 0.12  # gap between panels within a row
    margin = 0.5
    title_h = 0.5  # room for each row's heading + panel titles
    head_h = 0.16  # heading baseline offset from the top of its band
    row_gap = 0.35
    prof_h = 1.8

    def row_w(panels):
        return sum(p[3] * scale for p in panels) + gap_in * (len(panels) - 1)

    def row_h(panels):
        return max(p[4] for p in panels) * scale

    rows = [
        ("xy slices (across z, focus outlined)", xy_panels),
        ("xz cross-sections (across y)", xz_panels),
        ("yz cross-sections (across x)", yz_panels),
    ]
    fig_w = max(row_w(p) for _, p in rows) + 2 * margin
    fig_h = (
        margin  # top margin (also holds the suptitle)
        + sum(title_h + row_h(p) + row_gap for _, p in rows)
        + title_h
        + prof_h
        + margin
    )

    gof_txt = ""
    if have_gof:
        gof_txt = (
            f" | model vs data: R² = {gof['r2_median']:.3f}, "
            f"NRMSE = {gof['nrmse_pct']:.1f}% of peak"
        )

    # Object-oriented Agg figure (no pyplot): the calibration can run in a
    # worker thread, where the pyplot GUI backend warns/fails, and the layout
    # here is fully manual (fig.add_axes), so constrained_layout has nothing to
    # manage and would warn too.
    fig = Figure(figsize=(fig_w, fig_h))
    FigureCanvasAgg(fig)
    fig.suptitle(
        f"{title_prefix}{built['n_beads']} beads | z range {z_lo:.0f} to "
        f"{z_hi:.0f} nm | box {box} px | 1 px = 1 camera pixel ({ps:.0f} nm)"
        + gof_txt,
        fontsize=12,
        y=1.0 - 0.15 / fig_h,
    )

    top = margin
    for heading, panels in rows:
        fig.text(
            margin / fig_w,
            1.0 - (top + head_h) / fig_h,
            heading,
            fontsize=11,
            fontweight="bold",
            va="center",
        )
        top += title_h
        top = _place_row(fig, panels, top, fig_w, fig_h, scale, gap_in)
        top += row_gap

    # bottom row: the axial intensity profile plus, when available, the
    # empirical axial bias and precision (two panels); otherwise the single
    # model-vs-data agreement (per-z RMSE) curve.
    def _plot_intensity(ax):
        # axial intensity profile: brightest normalized pixel per slice, ~1 at
        # focus and decaying as the PSF spreads with defocus (a sharpness check)
        ax.plot(z_plot, template.max(axis=(0, 1)), ".-")
        ax.axvline(z_origin_marker, color="0.3", lw=1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Peak pixel value (norm.)")

    def _plot_bias(ax):
        # systematic z offset between the spline-refitted z and the known stage
        # position at each z-step (signed; ideally ~0). Kept separate from the
        # precision so a per-bead/registration z offset is not mistaken for
        # imprecision.
        ax.axhline(0.0, color="0.6", lw=1.0)
        ax.plot(z_plot, precision["bias_z"], ".-", color="tab:red")
        ax.axvline(z_origin_marker, color="0.3", lw=1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel("z bias (nm)")
        ax.set_title(f"{precision['n_beads']} beads", fontsize=9)

    def _plot_precision(ax):
        # shot-noise spread of the spline-refitted z about its per-step mean -
        # the single-frame axial precision the calibration delivers (lower is
        # better; rises with defocus). Analogous to the "mean z precision" panel
        # in ``zfit.calibrate_z``, but with the systematic bias removed.
        ax.plot(z_plot, precision["precision_z"], ".-", color="tab:red")
        ax.axvline(z_origin_marker, color="0.3", lw=1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel("z precision (nm)")
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{precision['n_beads']} beads", fontsize=9)

    def _plot_scatter(ax):
        # per-spot estimated z vs known stage position, with the identity line
        st = np.asarray(precision["scatter_stage"], dtype=float) - z_ref_nm
        zf = np.asarray(precision["scatter_fit"], dtype=float) - z_ref_nm
        lo, hi = float(np.min(z_plot)), float(np.max(z_plot))
        ax.plot(st, zf, ".k", alpha=0.1, markersize=2)
        ax.plot([lo, hi], [lo, hi], color="tab:red", lw=1.5, label="identity")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Estimated z (nm)")
        ax.legend(loc="best", fontsize=8)

    def _plot_gof(ax):
        # amplitude-normalized RMSE between the averaged PSF model and the
        # individual beads, per z-slice: how faithfully one PSF describes the
        # measured beads (lower is better; rises where beads disagree)
        ax.plot(z_plot, gof["residual_profile_pct"], ".-", color="tab:red")
        ax.axvline(z_origin_marker, color="0.3", lw=1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel("RMSE (% of peak)")
        ax.set_ylim(bottom=0.0)
        ax.set_title(
            f"median bead R² = {gof['r2_median']:.3f}  (n = {gof['n_used']})",
            fontsize=9,
        )

    bottom_panels = [("Axial intensity profile", _plot_intensity)]
    if have_prec:
        # a joint (multichannel) fit reflects the real pipeline; label it so the
        # panels aren't read as a per-channel (z-degenerate) single-plane fit
        n_joint = precision.get("joint") if precision else None
        sfx = f" (joint {n_joint}-ch fit)" if n_joint else ""
        bottom_panels.append((f"Estimated z vs stage{sfx}", _plot_scatter))
        bottom_panels.append((f"Axial bias{sfx}", _plot_bias))
        bottom_panels.append((f"Axial precision{sfx}", _plot_precision))
    elif have_gof:
        bottom_panels.append(("Model–data agreement (per-z RMSE)", _plot_gof))

    usable_w = fig_w - 2 * margin
    n_bp = len(bottom_panels)
    prof_gap = 0.95  # inter-panel gap; must fit a y-label + ticks
    max_panel_w = 2.6
    panel_w = min(max_panel_w, (usable_w - prof_gap * (n_bp - 1)) / n_bp)
    row_w = n_bp * panel_w + (n_bp - 1) * prof_gap
    x0 = margin + max(0.0, (usable_w - row_w) / 2.0)  # center the group
    head_y = 1.0 - (top + title_h * 0.35) / fig_h

    for i, (heading, _) in enumerate(bottom_panels):
        fig.text(
            (x0 + i * (panel_w + prof_gap)) / fig_w,
            head_y,
            heading,
            fontsize=11,
            fontweight="bold",
            va="center",
        )
    top += title_h
    for i, (_, plot_fn) in enumerate(bottom_panels):
        ax = fig.add_axes(
            [
                (x0 + i * (panel_w + prof_gap)) / fig_w,
                (fig_h - top - prof_h) / fig_h,
                panel_w / fig_w,
                (prof_h - 0.4) / fig_h,
            ]
        )
        plot_fn(ax)

    base, _ = os.path.splitext(path)
    fig.savefig(base + ".png", format="png", dpi=200)
