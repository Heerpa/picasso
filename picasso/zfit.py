"""
picasso.zfit
~~~~~~~~~~~~

Fitting z coordinates using astigmatism.

:authors: Joerg Schnitzbauer, Rafal Kowalewski, Thomas Kellerer
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import multiprocessing
import time
import warnings
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Literal

import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.ndimage import (
    affine_transform as _ndi_affine_transform,
    center_of_mass as _ndi_center_of_mass,
    gaussian_filter as _ndi_gaussian_filter,
    maximum_filter as _ndi_maximum_filter,
)
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist

from . import io, lib, gausslq, gaussmle, __version__


plt.style.use("ggplot")


# Default bead-detection / matching parameters for `calibrate_affine`.
_AFFINE_BEAD_MIN_DISTANCE = 15  # px between two peaks
_AFFINE_BEAD_THRESHOLD_REL = 0.25  # relative intensity threshold (0-1)
_AFFINE_BEAD_FIT_RADIUS = 6  # half-width of Gaussian fit patch
_AFFINE_MATCH_MAX_DIST_PX = 40.0  # max distance between matched pair
_AFFINE_XCORR_HALF_WIDTH = 18  # half-width of bead crop for xcorr


def _nan_index(y: lib.FloatArray1D) -> tuple[lib.BoolArray1D, Callable]:
    """Find indices of NaN values in an array."""
    return np.isnan(y), lambda z: z.nonzero()[0]


def _interpolate_nan(data: lib.FloatArray1D) -> lib.FloatArray1D:
    """Linear interpolattion of NaN values in an array ``data``."""
    nans, x = _nan_index(data)
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    return data


def calibrate_z(
    locs: pd.DataFrame,
    info: list[dict],
    d: float,
    magnification_factor: float,
    path: str | None = None,
    frame_bounds: tuple[int, int] | None = None,
    frames_per_step: int = 1,
    frame_order: Literal["fov", "z"] = "fov",
) -> dict:
    """Given localizations of a calibration sample (e.g., gold beads at
    different z positions), calibrate the z-axis by fitting a polynomial
    to the mean spot width/height of each z (stage) position. See Huang
    et al. Science, 2008. DOI: 10.1126/science.1153529.

    Several frames may be acquired at each z (stage) position to
    increase the number of localizations per position and thus the
    confidence of the calibration fit (see ``frames_per_step``).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations of a calibration sample.
    info : list of dicts
        Information about the calibration sample, including the number
        of frames.
    d : float
        Step size in nm, i.e., the distance between the z positions of
        the calibration sample.
    magnification_factor : float
        Magnification factor of the microscope, i.e., the ratio between
        the actual z position of the calibration sample and the
        estimated z position from the localization data.
    path : str, optional
        Path to save the calibration data as a YAML file. If None, the
        calibration data will not be saved. Default is None.
    frame_bounds : tuple, list of tuples, optional
        Frame numbers to consider for the calibration. A single
        ``(min, max)`` tuple restricts the calibration to one contiguous,
        inclusive range; a list of such tuples restricts it to several
        (disjoint) segments, where a frame is used if it falls in any
        segment. If None, all frames are used. If only min or max is to be
        specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    frames_per_step : int, optional
        Number of frames acquired at each z (stage) position. With the
        default of 1, every frame corresponds to a distinct z position
        (the classic behavior). Default is 1.
    frame_order : {"fov", "z"}, optional
        Acquisition order of the frames when ``frames_per_step`` > 1.
        ``"fov"`` means the z position is held constant while several
        fields of view are imaged, i.e., consecutive frames share the
        same z position (``[z0, z0, ..., z1, z1, ...]``). ``"z"`` means
        the full z stack is scanned sequentially and then repeated,
        i.e., frames cycle through all z positions
        (``[z0, z1, ..., z0, z1, ...]``). Ignored when
        ``frames_per_step`` is 1. Default is "fov".

    Returns
    -------
    calibration : dict
        Dictionary containing the calibration coefficients (i.e.,
        polynomial coefficients), number of frames, step size, and
        magnification factor.
    """
    n_frames = info[0]["Frames"]
    frames_per_step = max(1, int(frames_per_step))
    # number of distinct z (stage) positions; any trailing frames that do
    # not complete a step are ignored
    n_steps = n_frames // frames_per_step
    if n_steps < 1:
        raise ValueError(
            "Number of frames per step is larger than the number of "
            "frames in the movie."
        )

    # map each frame to the index of its z (stage) position
    all_frames = np.arange(n_frames)
    valid = all_frames < n_steps * frames_per_step
    if frame_order == "z":
        # full z stack scanned sequentially, then repeated
        step_of_frame = all_frames % n_steps
    else:  # "fov": consecutive frames share the same z position
        step_of_frame = all_frames // frames_per_step
    step_of_frame = np.where(valid, step_of_frame, -1)

    # z position of each step; negative so that the first frames of a
    # bottom-to-up scan are positive z coordinates
    z_span = (n_steps - 1) * d
    z_of_step = -(np.arange(n_steps) * d - z_span / 2)

    if frame_bounds is not None:
        # normalize to a list of inclusive (lo, hi) segments; a frame is
        # kept if it falls in any segment (frame bounds are inclusive,
        # like in picasso.localize)
        segments = lib.normalize_frame_bounds(frame_bounds, n_frames - 1)
        in_bounds = np.zeros(n_frames, dtype=bool)
        for frame_min, frame_max in segments:
            in_bounds |= (all_frames >= frame_min) & (all_frames <= frame_max)
        step_of_frame = np.where(in_bounds, step_of_frame, -1)

    # steps that still have at least one frame contributing to them
    step_range = np.unique(step_of_frame[step_of_frame >= 0])
    z_range = z_of_step[step_range]
    # position of each step within the (bounded) step_range
    step_to_pos = {int(s): i for i, s in enumerate(step_range)}

    # assign each localization to its z (stage) position
    locs = locs.copy()
    locs_step = step_of_frame[locs["frame"].to_numpy()]
    locs = locs[np.isin(locs_step, step_range)]
    locs_step = step_of_frame[locs["frame"].to_numpy()]

    mean_sx = np.array([locs["sx"][locs_step == _].mean() for _ in step_range])
    mean_sy = np.array([locs["sy"][locs_step == _].mean() for _ in step_range])
    var_sx = np.array([locs["sx"][locs_step == _].var() for _ in step_range])
    var_sy = np.array([locs["sy"][locs_step == _].var() for _ in step_range])

    # position of each localization's z (stage) position within step_range
    pos = np.array([step_to_pos[int(s)] for s in locs_step], dtype=int)
    keep_x = (locs["sx"] - mean_sx[pos]) ** 2 < var_sx[pos]
    keep_y = (locs["sy"] - mean_sy[pos]) ** 2 < var_sy[pos]
    keep = keep_x & keep_y
    locs = locs[keep]
    locs_step = step_of_frame[locs["frame"].to_numpy()]

    # Fits calibration curve to the mean of each z (stage) position
    mean_sx = np.array([locs["sx"][locs_step == _].mean() for _ in step_range])
    mean_sy = np.array([locs["sy"][locs_step == _].mean() for _ in step_range])

    # Fix nan
    mean_sx = _interpolate_nan(mean_sx)
    mean_sy = _interpolate_nan(mean_sy)

    cx = np.polyfit(z_range, mean_sx, 6, full=False)
    cy = np.polyfit(z_range, mean_sy, 6, full=False)

    # make sure that the calibration curves cross at z = 0
    z = np.linspace(z_range[0], z_range[-1], 10000)
    spot_width = np.poly1d(cx)
    spot_height = np.poly1d(cy)
    z_range -= z[np.argmin(np.abs(spot_width(z) - spot_height(z)))]
    cx = np.polyfit(z_range, mean_sx, 6, full=False)
    cy = np.polyfit(z_range, mean_sy, 6, full=False)

    calibration = {
        "X Coefficients": [float(_) for _ in cx],
        "Y Coefficients": [float(_) for _ in cy],
        "Number of frames": int(n_frames),
        "Step size in nm": float(d),
        "Magnification factor": float(magnification_factor),
        "Path": path if path is not None else "N/A",
        "Frame bounds": frame_bounds,
        "Frames per step": int(frames_per_step),
        "Frame order": frame_order,
    }
    if path is not None:
        io.save_calibration(path, calibration)

    # pixelsize does not matter here anyway
    locs = _fit_z(locs, info, calibration, magnification_factor, pixelsize=130)
    locs["z"] /= magnification_factor
    # position of each localization's z (stage) position within step_range
    locs_step = step_of_frame[locs["frame"].to_numpy()]
    pos = np.array([step_to_pos[int(s)] for s in locs_step], dtype=int)

    plt.figure(figsize=(18, 10))

    plt.subplot(231)
    plt.plot(z_range, mean_sx, ".-", label="x")
    plt.plot(z_range, mean_sy, ".-", label="y")
    plt.plot(z_range, np.polyval(cx, z_range), "0.3", lw=1.5, label="x fit")
    plt.plot(z_range, np.polyval(cy, z_range), "0.3", lw=1.5, label="y fit")
    plt.xlabel("Stage position")
    plt.ylabel("Mean spot width/height")
    plt.xlim(z_range.min(), z_range.max())
    plt.legend(loc="best")

    ax = plt.subplot(232)
    plt.scatter(locs["sx"], locs["sy"], c="k", lw=0, alpha=0.1)
    plt.plot(
        np.polyval(cx, z_range),
        np.polyval(cy, z_range),
        lw=1.5,
        label="calibration from fit of mean width/height",
    )
    plt.plot()
    ax.set_aspect("equal")
    plt.xlabel("Spot width")
    plt.ylabel("Spot height")
    plt.legend(loc="best")

    plt.subplot(233)
    plt.plot(locs["z"], locs["sx"], ".", label="x", alpha=0.2)
    plt.plot(locs["z"], locs["sy"], ".", label="y", alpha=0.2)
    plt.plot(
        z_range,
        np.polyval(cx, z_range),
        "0.3",
        lw=1.5,
        label="calibration",
    )
    plt.plot(z_range, np.polyval(cy, z_range), "0.3", lw=1.5)
    plt.xlim(z_range.min(), z_range.max())
    plt.xlabel("Estimated z")
    plt.ylabel("Spot width/height")
    plt.legend(loc="best")

    ax = plt.subplot(234)
    plt.plot(z_range[pos], locs["z"], ".k", alpha=0.1)
    plt.plot(
        [z_range.min(), z_range.max()],
        [z_range.min(), z_range.max()],
        lw=1.5,
        label="identity",
    )
    plt.xlim(z_range.min(), z_range.max())
    plt.ylim(z_range.min(), z_range.max())
    ax.set_aspect("equal")
    plt.xlabel("Stage position")
    plt.ylabel("Estimated z")
    plt.legend(loc="best")

    ax = plt.subplot(235)
    deviation = locs["z"] - z_range[pos]
    bins = lib.calculate_optimal_bins(deviation, max_n_bins=1000)
    plt.hist(deviation, bins)
    plt.xlabel("Deviation to true position")
    plt.ylabel("Occurence")

    ax = plt.subplot(236)
    square_deviation = deviation.to_numpy() ** 2
    mean_square_deviation_step = [
        np.mean(square_deviation[locs_step == _]) for _ in step_range
    ]
    rmsd_step = np.sqrt(mean_square_deviation_step)
    plt.plot(z_range, rmsd_step, ".-", color="0.3")
    plt.xlim(z_range.min(), z_range.max())
    plt.gca().set_ylim(bottom=0)
    plt.xlabel("Stage position")
    plt.ylabel("Mean z precision")

    plt.tight_layout(pad=2)

    if path is not None:
        path, ext = os.path.splitext(path)
        plt.savefig(path + ".png", format="png", dpi=300)

    plt.show()
    return calibration


def _movie_to_image(movie) -> np.ndarray:
    """Collapse a picasso movie to a single 2D float32 image normalised
    to [0, 1]. Multi-frame movies are averaged; single-frame movies are
    passed through. Frames are read one-at-a-time so the lazy-loading
    movie classes in ``picasso.io`` don't have to materialise the full
    stack at once."""
    n = len(movie)
    if n == 0:
        raise ValueError("Movie has zero frames.")
    if n == 1:
        img = np.asarray(movie[0], dtype=np.float32)
    else:
        acc = np.zeros(np.asarray(movie[0]).shape, dtype=np.float64)
        for i in range(n):
            acc += np.asarray(movie[i], dtype=np.float64)
        img = (acc / n).astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    return (img - mn) / (mx - mn + 1e-12)


def _affine_detect_beads(image: np.ndarray) -> np.ndarray:
    """Detect bead candidates via Gaussian blur + local-maximum filter.
    Returns (N, 2) array of [row, col] integer coordinates."""
    blurred = _ndi_gaussian_filter(image, sigma=1.5)
    size = 2 * _AFFINE_BEAD_MIN_DISTANCE + 1
    abs_thresh = (
        _AFFINE_BEAD_THRESHOLD_REL * (blurred.max() - blurred.min())
        + blurred.min()
    )
    is_peak = (_ndi_maximum_filter(blurred, size=size) == blurred) & (
        blurred > abs_thresh
    )
    border = max(5, _AFFINE_BEAD_MIN_DISTANCE // 2)
    is_peak[:border] = False
    is_peak[-border:] = False
    is_peak[:, :border] = False
    is_peak[:, -border:] = False
    return np.column_stack(np.where(is_peak))


def _affine_fit_gaussian_2d(patch: np.ndarray):
    """Fit an elliptical 2D Gaussian to a small image patch. Returns
    (x0, y0, sx, sy, amp, bg) or None on failure."""
    ny, nx = patch.shape
    y0g, x0g = np.unravel_index(np.argmax(patch), patch.shape)
    bg = np.percentile(patch, 15)
    amp = patch.max() - bg

    def model(xy, x0, y0, sx, sy, amp, bg):
        x, y = xy
        return bg + amp * np.exp(
            -((x - x0) ** 2 / (2 * sx**2) + (y - y0) ** 2 / (2 * sy**2))
        )

    xg, yg = np.meshgrid(np.arange(nx), np.arange(ny))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                model,
                (xg.ravel(), yg.ravel()),
                patch.ravel(),
                p0=[x0g, y0g, 2.0, 2.0, amp, bg],
                bounds=(
                    [0, 0, 0.3, 0.3, 0, -np.inf],
                    [nx, ny, nx, ny, np.inf, np.inf],
                ),
                maxfev=400,
            )
        return popt
    except Exception:
        return None


def _affine_refine_bead_positions(
    image: np.ndarray, coarse: np.ndarray
) -> np.ndarray:
    """Refine coarse bead positions to sub-pixel accuracy via 2D Gaussian
    fits. Returns (N, 2) array of refined [row, col] coordinates."""
    ny, nx = image.shape
    r = _AFFINE_BEAD_FIT_RADIUS
    refined = []
    for ry, rx in coarse:
        y0 = max(0, ry - r)
        y1 = min(ny, ry + r + 1)
        x0 = max(0, rx - r)
        x1 = min(nx, rx + r + 1)
        patch = image[y0:y1, x0:x1]
        popt = _affine_fit_gaussian_2d(patch) if patch.size > 0 else None
        if popt is not None:
            refined.append([popt[1] + y0, popt[0] + x0])
        else:
            cy, cx = _ndi_center_of_mass(patch) if patch.size > 0 else (ry, rx)
            refined.append([cy + y0, cx + x0])
    return np.array(refined)


def _affine_match_bead_pairs(
    coords_ref: np.ndarray, coords_mov: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match beads via mutual nearest-neighbour with a distance threshold.
    Returns (pairs_ref, pairs_mov), each (M, 2)."""
    if len(coords_ref) == 0 or len(coords_mov) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    D = cdist(coords_ref, coords_mov)
    nn_r2m = np.argmin(D, axis=1)
    nn_m2r = np.argmin(D, axis=0)
    pairs_r, pairs_m = [], []
    for i, j in enumerate(nn_r2m):
        if D[i, j] < _AFFINE_MATCH_MAX_DIST_PX and nn_m2r[j] == i:
            pairs_r.append(coords_ref[i])
            pairs_m.append(coords_mov[j])
    pairs_ref = np.array(pairs_r) if pairs_r else np.empty((0, 2))
    pairs_mov = np.array(pairs_m) if pairs_m else np.empty((0, 2))
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
    return _ndi_affine_transform(
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
    cyl_path: str = "",
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
    if ref_path or cyl_path:
        title += (
            f"\nref: {os.path.basename(ref_path)}   "
            f"cyl: {os.path.basename(cyl_path)}"
        )
    fig.suptitle(title, fontsize=10, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.25)

    ext = [0, img_ref.shape[1] * nm, img_ref.shape[0] * nm, 0]

    ax = fig.add_subplot(gs[0])
    ax.imshow(
        np.clip(np.stack([mov_n, ref_n, mov_n], axis=-1), 0, 1), extent=ext
    )
    ax.set_title(
        "Overlay  BEFORE\nRef (green) | Cylindrical (magenta)",
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


def calibrate_affine(
    movie_ref,
    movie_cyl,
    calibration: dict,
    pixelsize: float | None = None,
    ref_path: str = "",
    cyl_path: str = "",
    plot_path: str = "",
) -> dict:
    """Fit a 6-DOF affine transform that maps the cylindrical-lens bead
    image into the reference (no-lens) frame and append it to the given
    3D astigmatism calibration dict.

    The fit is performed in pixel coordinates on a per-pixel mean of
    each movie. Bead candidates are found by Gaussian-blur + local-max,
    refined to sub-pixel accuracy by a 2D Gaussian fit, then matched
    between the two images by mutual nearest neighbour. The affine
    matrix is solved by 6-DOF linear least squares and decomposed into
    rotation / anisotropic scale / shear via QR.  TODO: might correct this
    paragraph

    Parameters
    ----------
    movie_ref, movie_cyl : AbstractPicassoMovie
        In-focus bead movies without (reference) and with the
        cylindrical lens. If a movie has multiple frames they are
        averaged; a single-frame movie is used as-is.
    calibration : dict
        Existing 3D calibration; an "Affine transform" entry is appended.
    pixelsize : float, optional
        Camera pixel size in nm. If given, decomposition translations
        and the diagnostic plot are converted from pixels to nm. If
        None (default), values are reported in pixels. Default is None.
    ref_path, cyl_path : str, optional
        Paths to the source images, recorded in the calibration for
        traceability and shown in the diagnostic plot title. Default
        is "".
    plot_path : str, optional
        If given, the diagnostic figure is saved to this path. The
        figure is always shown interactively. Default is "".

    Returns
    -------
    calibration : dict
        The input calibration augmented with an "Affine transform" key.
        Use ``io.save_calibration`` to save the result.
    """
    img_ref = _movie_to_image(movie_ref)
    img_cyl = _movie_to_image(movie_cyl)

    coarse_ref = _affine_detect_beads(img_ref)
    coarse_cyl = _affine_detect_beads(img_cyl)
    refined_ref = _affine_refine_bead_positions(img_ref, coarse_ref)
    refined_cyl = _affine_refine_bead_positions(img_cyl, coarse_cyl)
    pairs_ref, pairs_cyl = _affine_match_bead_pairs(refined_ref, refined_cyl)

    if len(pairs_ref) < 3:
        raise ValueError(
            f"Only {len(pairs_ref)} matched bead pair(s) — need >= 3 to "
            "fit an affine transform. Check the input images / detection "
            "parameters."
        )

    M = _affine_estimate_2d(pairs_cyl, pairs_ref)
    decomp = _affine_decompose(M, pixelsize)

    img_cor = _affine_apply(img_cyl, M)
    _affine_plot_alignment(
        img_ref,
        img_cyl,
        img_cor,
        pairs_ref,
        decomp,
        n_pairs=len(pairs_ref),
        pixelsize=pixelsize,
        save_path=plot_path,
        ref_path=ref_path,
        cyl_path=cyl_path,
    )

    affine_entry = {
        "Matrix": [[float(v) for v in row] for row in M],
        "Direction": "cylindrical -> reference (x = col, y = row)",
        "Reference image": ref_path or "N/A",
        "Cylindrical image": cyl_path or "N/A",
        "Bead pairs": int(len(pairs_ref)),
        "Decomposition": decomp,
    }
    if pixelsize is not None:
        affine_entry["Pixelsize (nm)"] = float(pixelsize)
    calibration["Affine transform"] = affine_entry
    return calibration


@numba.jit(nopython=True, nogil=True)
def _fit_z_target(
    z: float,
    sx: float,
    sy: float,
    cx: lib.FloatArray1D,
    cy: lib.FloatArray1D,
) -> float:
    """Target function that's to be minimized for fitting the z
    coordinates given the single-emitter image width and height as well
    as the calibration curve coefficients. It calculates the difference
    between the square root of the spot width/height and the polynomial
    fit of the z-axis calibration curve. Based on Huang et al. Science,
    2008. DOI: 10.1126/science.1153529."""
    z2 = z * z
    z3 = z * z2
    z4 = z * z3
    z5 = z * z4
    z6 = z * z5
    wx = (
        cx[0] * z6
        + cx[1] * z5
        + cx[2] * z4
        + cx[3] * z3
        + cx[4] * z2
        + cx[5] * z
        + cx[6]
    )
    wy = (
        cy[0] * z6
        + cy[1] * z5
        + cy[2] * z4
        + cy[3] * z3
        + cy[4] * z2
        + cy[5] * z
        + cy[6]
    )
    return (sx**0.5 - wx**0.5) ** 2 + (sy**0.5 - wy**0.5) ** 2


def _fit_z(
    locs: pd.DataFrame,
    info: list[dict],
    calibration: dict,
    magnification_factor: float,
    pixelsize: float,
    fitting_method: Literal["gausslq", "gaussmle"] = "gausslq",
    filter: int = 2,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> pd.DataFrame:
    """Internal function for fitting z coordinates to the localizations.
    See `zfit` for details."""
    locs = locs.copy()
    cx = np.array(calibration["X Coefficients"])
    cy = np.array(calibration["Y Coefficients"])
    # in multiprocessing, pandas Series causes issues!
    sx = locs["sx"].to_numpy()
    sy = locs["sy"].to_numpy()
    z = np.zeros_like(locs["x"])
    square_d_zcalib = np.zeros_like(z)

    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(range(len(z)), desc="Fitting z...", unit="locs")
    else:
        iter_range = range(len(z))

    for i in iter_range:
        # set bounds to avoid potential gaps in the calibration curve,
        # credits to Loek Andriessen
        result = minimize_scalar(
            _fit_z_target,
            bounds=[-1000, 1000],
            args=(sx[i], sy[i], cx, cy),
        )
        z[i] = result.x
        square_d_zcalib[i] = result.fun

        if callable(progress_callback):
            progress_callback(i)

    locs["z"] = z * magnification_factor
    locs["d_zcalib"] = np.sqrt(square_d_zcalib)
    lpz = _axial_localization_precision_astig(
        locs,
        cx,
        cy,
        magnification_factor,
        pixelsize,
        fitting_method,
    )
    locs["lpz"] = lpz

    # Apply the cylindrical-lens -> reference affine transform to (x, y)
    # in camera-pixel coordinates, if one was calibrated.
    if "Affine transform" in calibration:
        M = np.asarray(calibration["Affine transform"]["Matrix"])
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        locs["x"] = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        locs["y"] = M[1, 0] * x + M[1, 1] * y + M[1, 2]

    locs = lib.ensure_sanity(locs, info)
    return filter_z_fits(locs, filter)


def _fit_z_parallel(
    locs: pd.DataFrame,
    info: list[dict],
    calibration: dict,
    magnification_factor: float,
    pixelsize: float,
    fitting_method: Literal["gausslq", "gaussmle"] = "gausslq",
    filter: int = 2,
    asynch: bool = False,
) -> pd.DataFrame | list[futures.Future]:
    """Internal function for fitting z coordinates to the localizations
    using multiprocessing. See `zfit` for details."""
    n_workers = min(
        60, max(1, int(0.75 * multiprocessing.cpu_count()))
    )  # Python crashes when using >64 cores
    n_locs = len(locs)
    n_tasks = 100 * n_workers
    spots_per_task = [
        (
            int(n_locs / n_tasks + 1)
            if _ < n_locs % n_tasks
            else int(n_locs / n_tasks)
        )
        for _ in range(n_tasks)
    ]
    start_indices = np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = ProcessPoolExecutor(n_workers)
    for i, n_locs_task in zip(start_indices, spots_per_task):
        fs.append(
            executor.submit(
                _fit_z,
                locs[i : i + n_locs_task],
                info,
                calibration,
                magnification_factor,
                pixelsize,
                fitting_method=fitting_method,
                filter=0,
            )
        )
    if asynch:
        return fs
    with tqdm(total=n_tasks, unit="task") as progress_bar:
        for f in futures.as_completed(fs):
            progress_bar.update()
    return locs_from_futures(fs, filter=filter)


def zfit(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    calibration: dict,
    magnification_factor: float | None = None,
    pixelsize: int | float | None = None,
    fitting_method: Literal["gausslq", "gaussmle"] = "gausslq",
    filter: int = 2,
    multiprocess: bool = False,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> tuple[pd.DataFrame, list[dict]] | tuple[None, None]:
    """Main function for fitting z coordinates to the localizations.

    Introduced in v0.10.0. Note that `fit_z` and `fit_z_parallel` were
    completely replaced in v0.11.0.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations (2D fitted).
    info : list of dicts
        Localizations metadata. Should include a "Pixelsize" key with
        the effective camera pixel size in nm. If not, must be provided
        as an argument (see below).
    calibration : dict
        Calibration dictionary containing the following keys:

        - "X Coefficients": list of 7 floats, polynomial coefficients
            for the x-axis calibration curve;
        - "Y Coefficients": list of 7 floats, polynomial coefficients
            for the y-axis calibration curve;
        - "Magnification factor": float, magnification factor of the
            microscope, i.e., the ratio between the actual z position of
            the calibration sample and the estimated z position from the
            localization data.

        Note that "Magnification factor" can be overwritten by the
        `magnification_factor` argument. See ``io.load_calibration``
        on how to open a calibration YAML file.
    magnification_factor : float, optional
        Magnification factor of the microscope, i.e., the ratio between
        the actual z position of the calibration sample and the
        estimated z position from the localization data. If None, the
        value must be given in `calibration`.
    pixelsize : float, optional
        Camera pixel size in nm. If given, the value in `info` will be
        ignored. If None, the value must be given in `info`.
    fitting_method : {"gausslq", "gaussmle"}, optional
        Fitting method used to obtain 2D localization parameters. Used
        to determine axial localization precision. Default is "gausslq".
    filter : int, optional
        Filter for the z fits. If set to 0, no filtering is applied.
        If set to 2, the z fits are filtered based on the root mean
        square deviation (RMSD) of the z calibration. Default is 2.
    multiprocess : bool, optional
        Whether to use multiprocessing for fitting the z coordinates.
        Default is False.
    progress_callback : callable, "console", or None, optional
        If a callable is provided, it will be called with the current
        progress (number of localizations processed) as an argument. If
        "console", a progress bar will be displayed in the console. If
        None, no progress will be reported. Default is None.
    abort_callback : callable or None, optional
        A callable for aborting multiprocessing in the GUI. If a
        callable provided, it must accept no input and return a boolean
        indicating whether the fitting should be aborted.

    Returns
    -------
    locs : pd.DataFrame
        Localizations with columns 'z', 'd_zcalib', and 'lpz' appended.
        If processing is aborted, returns None.
    info : list of dicts
        Updated info with the z fitting parameters attached. If
        processing is aborted, returns None.
    """
    assert fitting_method in ["gausslq", "gaussmle"], "Invalid fitting method."
    assert filter >= 0, "Filter must be non-negative."
    assert isinstance(
        calibration,
        dict,
    ), "Calibration must be a dict, see ``io.load_calibration``."
    if magnification_factor is not None:
        assert isinstance(
            magnification_factor, (int, float)
        ), "Magnification factor must be a number."
        calibration["Magnification factor"] = float(magnification_factor)
    else:
        assert (
            "Magnification factor" in calibration
        ), "Magnification factor is missing in calibration."
    if pixelsize is not None:
        assert isinstance(
            pixelsize, (int, float)
        ), "Pixelsize must be a number in nm."
        pixelsize = float(pixelsize)
        info.append({"Pixelsize": pixelsize})
    else:
        assert lib.get_from_metadata(info, "Pixelsize") is not None, (
            "Camera pixel size (nm) is missing. Enter it either in the "
            "info metadata, or as an argument."
        )

    return _zfit(
        locs,
        info,
        calibration,
        fitting_method,
        filter,
        multiprocess,
        progress_callback,
        abort_callback,
    )


def _zfit(
    locs: pd.DataFrame,
    info: list[dict],
    calibration: dict,
    fitting_method: Literal["gausslq", "gaussmle"],
    filter: int,
    multiprocess: bool,
    progress_callback: Callable[[int], None] | Literal["console"] | None,
    abort_callback: Callable[[], bool] | None,
) -> tuple[pd.DataFrame, list[dict]] | tuple[None, None]:
    """Internal function for fitting z coordinates to the localizations.
    See `zfit` for details."""
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    N = len(locs)
    if multiprocess:
        use_tqdm = progress_callback == "console"
        if use_tqdm:
            iter_range = tqdm(range(N), desc="Fitting z...", unit="locs")

        fs = _fit_z_parallel(
            locs=locs,
            info=info,
            calibration=calibration,
            magnification_factor=calibration["Magnification factor"],
            pixelsize=pixelsize,
            fitting_method=fitting_method,
            filter=0,  # will be applied later
            asynch=True,
        )
        n_tasks = len(fs)
        while lib.n_futures_done(fs) < n_tasks:
            # check for abort
            if abort_callback is not None and abort_callback():
                for f in fs:
                    f.cancel()
                return None, None

            n_finished = round(N * lib.n_futures_done(fs) / n_tasks)
            if use_tqdm:
                iter_range.update(n_finished - iter_range.n)
            elif callable(progress_callback):
                progress_callback(n_finished)
            time.sleep(0.2)
        locs = locs_from_futures(fs, filter=filter)
    else:
        locs = _fit_z(
            locs=locs,
            info=info,
            calibration=calibration,
            magnification_factor=calibration["Magnification factor"],
            pixelsize=pixelsize,
            fitting_method=fitting_method,
            filter=filter,
            progress_callback=progress_callback,
        )
    new_info = {
        "Generated by": f"Picasso v{__version__} Fit 3D",
        "Calibration path": calibration.get("Path", "N/A"),
        "Filter range": filter,
    }
    new_info = info + [new_info | calibration]
    return locs, new_info


def locs_from_futures(
    futures: list[futures.Future], filter: int = 2
) -> pd.DataFrame:
    """Combine the results from a list of futures (i.e.,
    multiprocessing results) into a single DataFrame of localizations
    with fitted z coordinates and their residuals (d_zcalib).

    Parameters
    ----------
    futures : list of futures.Future
        List of futures that contain the results of the z fits.
    filter : int, optional
        Filter for the z fits. If set to 0, no filtering is applied.
        If set to 2, the z fits are filtered based on the root mean
        square deviation (RMSD) of the z calibration. Default is 2.

    Returns
    -------
    locs : pd.DataFrame
        DataFrame of localizations with the fitted z coordinates and
        their residuals (d_zcalib).
    """
    locs = [_.result() for _ in futures]
    locs = pd.concat(locs, ignore_index=True)
    return filter_z_fits(locs, filter)


def filter_z_fits(locs: pd.DataFrame, range: int) -> pd.DataFrame:
    """Filter the z fits based on the root mean square deviation (RMSD)
    of the z calibration (d_zcalib residual). If `range` is set to 0, no
    filtering is applied. If `range` is greater than 0, the
    localizations with a RMSD greater than `range` are removed.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with fitted z coordinates and their residuals
        (d_zcalib).
    range : int
        Range for filtering the z fits. If set to 0, no filtering is
        applied. If set to a positive value, localizations with a
        RMSD greater than `range` times the RMSD of the z calibration
        are removed.

    Returns
    -------
    locs : pd.DataFrame
        DataFrame of localizations with the fitted z coordinates and
        their residuals (d_zcalib) after filtering.
    """
    if "d_zcalib" not in locs.columns:
        return locs
    if range > 0:
        rmsd = np.sqrt(np.nanmean(locs["d_zcalib"] ** 2))
        locs = locs[locs["d_zcalib"] <= range * rmsd]
    return locs


def axial_localization_precision(
    locs: np.recarray,
    info: list[dict],
    calibration: dict,
    fitting_method: Literal["gausslq", "gaussmle"] = "gausslq",
    modality: Literal["astigmatic"] = "astigmatic",
) -> lib.FloatArray1D:
    """Calculate axial localization precision for given localizations
    based on calibration.

    Parameters
    ----------
    locs : np.recarray
        Localizations.
    info : list of dicts
        Localizations metadata.
    calibration : dict
        Calibration dictionary with x and y coefficients and
        magnification factor.
    fitting_method : {"gausslq", "gaussmle"}, optional
        Fitting method used to obtain 2D localization parameters (x, y,
        sx, sy). Default is "gausslq".
    modality : {"astigmatic"}, optional
        3D imaging modality. Currently, only "astigmatic" is supported.
        Default is "astigmatic".

    Returns
    -------
    lpz: lib.FloatArray1D
        Calculated lpz values for the given localizations in nm.
    """
    if modality != "astigmatic":
        raise NotImplementedError(
            "Currently only 'astigmatic' modality is supported."
        )
    lpz = axial_localization_precision_astig(
        locs, info, calibration, fitting_method
    )
    return lpz


def axial_localization_precision_astig(
    locs: np.recarray,
    info: list[dict],
    calibration: dict,
    fitting_method: Literal["gausslq", "gaussmle"] = "gausslq",
) -> lib.FloatArray1D:
    """Calculate axial localization precision for astigmatic 3D imaging
    for given localizations based on calibration.

    Based on Kowalewski, Reinhardt, et al. Nature Comms, 2026.
    DOI: https://doi.org/10.1038/s41467-026-70198-5


    Parameters
    ----------
    locs : np.recarray
        Localizations.
    info : list of dicts
        Localizations metadata.
    calibration : dict
        Calibration dictionary with x and y coefficients, z step size
        and the number of frames.
    fitting_method : {"gausslq", "gaussmle"}, optional
        Fitting method used to obtain 2D localization parameters (x, y,
        sx, sy). Default is "gausslq".

    Returns
    -------
    lpz: lib.FloatArray1D
        Calculated lpz values for the given localizations in nm.
    """
    assert fitting_method in [
        "gausslq",
        "gaussmle",
    ], "fitting_method must be 'gausslq' or 'gaussmle'."
    assert (
        "X Coefficients" in calibration
        and "Y Coefficients" in calibration
        and "Magnification factor" in calibration
    ), (
        "Calibration dictionary must contain 'X Coefficients', "
        "'Y Coefficients', and 'Magnification factor'."
    )

    # get camera pixel size
    pixelsize = lib.get_from_metadata(info, "Pixelsize")
    if pixelsize is None:
        raise ValueError("Pixelsize not found in info.")

    mag_factor = calibration["Magnification factor"]
    cx = np.array(calibration["X Coefficients"])
    cy = np.array(calibration["Y Coefficients"])
    lpz = _axial_localization_precision_astig(
        locs, cx, cy, mag_factor, pixelsize, fitting_method
    )
    return lpz


def _axial_localization_precision_astig(
    locs: pd.DataFrame,
    cx: lib.FloatArray1D,
    cy: lib.FloatArray1D,
    magnification_factor: float,
    pixelsize: float,
    fitting_method: Literal["gausslq", "gaussmle"] = "gausslq",
) -> lib.FloatArray1D:
    """Calculate axial localization precision for astigmatic 3D imaging
    for given localizations based on calibration.

    Based on Kowalewski, Reinhardt, et al. Nature Comms, 2026.
    DOI: https://doi.org/10.1038/s41467-026-70198-5

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations. Must include columns 'photons', 'sx', 'sy', 'bg',
        and 'z', see https://picassosr.readthedocs.io/en/latest/files.html#localization-hdf5-files  # noqa: E501
    cx : lib.FloatArray1D
        3D calibration coefficients for x.
    cy : lib.FloatArray1D
        3D calibration coefficients for y.
    pixelsize : float
        Camera pixel size in nm.
    fitting_method : {"gausslq", "gaussmle"}, optional
        Fitting method used to obtain 2D localization parameters (x, y,
        sx, sy). Default is "gausslq".

    Returns
    -------
    lpz: lib.FloatArray1D
        Calculated lpz values for the given localizations in nm.
    """
    if fitting_method == "gausslq":
        se_sx = (
            gausslq.sigma_uncertainty(
                locs["sx"], locs["sy"], locs["photons"], locs["bg"]
            )
            * pixelsize
        )
        se_sy = (
            gausslq.sigma_uncertainty(
                locs["sy"], locs["sx"], locs["photons"], locs["bg"]
            )
            * pixelsize
        )
    elif fitting_method == "gaussmle":
        if "sx_unc" not in locs.columns or "sy_unc" not in locs.columns:
            se_sx = (
                gaussmle.sigma_uncertainty(
                    locs["sx"], locs["sy"], locs["photons"], locs["bg"]
                )
                * pixelsize
            )
            se_sy = (
                gaussmle.sigma_uncertainty(
                    locs["sy"], locs["sx"], locs["photons"], locs["bg"]
                )
                * pixelsize
            )
        else:
            se_sx = locs["sx_unc"] * pixelsize
            se_sy = locs["sy_unc"] * pixelsize
    else:
        raise ValueError("fitting_method must be 'gausslq' or 'gaussmle'.")

    # to pinpoint what was the actual spot size during measurement
    z = locs["z"] / magnification_factor
    wx_calib = _get_calib_size(cx, z) * pixelsize
    wy_calib = _get_calib_size(cy, z) * pixelsize
    wx_calib_prime = _get_prime_calib_size(cx, z) * pixelsize
    wy_calib_prime = _get_prime_calib_size(cy, z) * pixelsize
    sqrt_wx_calib = np.sqrt(wx_calib)
    sqrt_wx_calib_prime = wx_calib_prime / (2 * sqrt_wx_calib)
    sqrt_wy_calib = np.sqrt(wy_calib)
    sqrt_wy_calib_prime = wy_calib_prime / (2 * sqrt_wy_calib)
    delta_sqrt_wx = (1 / (2 * np.sqrt(locs["sx"] * pixelsize))) * se_sx
    delta_sqrt_wy = (1 / (2 * np.sqrt(locs["sy"] * pixelsize))) * se_sy
    swxc2 = sqrt_wx_calib_prime**2
    swyc2 = sqrt_wy_calib_prime**2
    swx2 = delta_sqrt_wx**2
    swy2 = delta_sqrt_wy**2
    lpz = np.sqrt((swxc2 * swx2 + swyc2 * swy2) / (swxc2 + swyc2) ** 2)
    return lpz * magnification_factor


def _get_calib_size(
    coeffs: lib.FloatArray1D, z: lib.FloatArray1D
) -> lib.FloatArray1D:
    """Calculate calibration spot size at the given z position given
    the calibration coefficients. Based on Huang et al., Science 2008."""
    size = (
        coeffs[0] * z**6
        + coeffs[1] * z**5
        + coeffs[2] * z**4
        + coeffs[3] * z**3
        + coeffs[4] * z**2
        + coeffs[5] * z
        + coeffs[6]
    )
    return size


def _get_prime_calib_size(
    coeffs: lib.FloatArray1D, z: lib.FloatArray1D
) -> lib.FloatArray1D:
    """Same as ``_get_calib_size`` but for the derivative of the size
    function."""
    size_prime = (
        6 * coeffs[0] * z**5
        + 5 * coeffs[1] * z**4
        + 4 * coeffs[2] * z**3
        + 3 * coeffs[3] * z**2
        + 2 * coeffs[4] * z
        + coeffs[5]
    )
    return size_prime
