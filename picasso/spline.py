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

References
----------
- Li, Y., Mund, M., Hoess, P., Deschamps, J., Matti, U., Nijmeijer, B.,
  Sabinina, V. J., Ellenberg, J., Schoen, I. & Ries, J. "Real-time 3D
  single-molecule localization using experimental point spread functions."
  Nature Methods 15, 367-369 (2018).
- Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
  "Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
  Scientific Reports 7, 15722 (2017).

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
from typing import Callable, Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import shift as _ndi_shift, zoom as _ndi_zoom

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
) -> pd.DataFrame:
    """Detect bead centers (integer pixel positions) from a set of reference
    frames (ideally the in-focus ones, where beads are brightest).

    Beads are static in x/y (only the stage moves in z), so we detect them
    once and reuse the positions across all z-steps. Detections are pooled
    across the reference frames, rounded to the pixel grid and de-duplicated
    spatially (detections within ``min_separation`` pixels - defaulting to the
    box size - are treated as the same bead); beads whose box would fall
    outside the frame are dropped.

    If ``roi`` is given, only detections inside the ROI(s) are kept; an empty
    list or None means the whole frame.

    Returns a data frame with integer ``x``/``y`` columns (one row per bead).
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
    # keep beads whose full box fits inside the frame
    height, width = movie.shape[1], movie.shape[2]
    half = box // 2
    inside = (
        (x - half >= 0)
        & (x + half < width)
        & (y - half >= 0)
        & (y + half < height)
    )
    x, y = x[inside], y[inside]

    # merge detections of the same physical bead pooled across reference frames
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
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and z-step-average a PSF volume for every bead.

    Returns an array of shape ``(n_beads, box, box, n_steps)`` in photon units,
    where each z-slice is the mean over all (multi-FOV) frames assigned to that
    step. If ``return_spots`` is True, also returns the individual (un-averaged)
    per-frame spots ``(n_valid_frames, n_beads, box, box)`` and the z-step of
    each valid frame ``(n_valid_frames,)``, so the axial precision can be
    measured by fitting every single-frame spot separately (the realistic,
    single-frame shot-noise regime) rather than the frame-averaged volumes.
    """
    n_beads = len(beads)
    valid_frames = np.where(step_of_frame >= 0)[0]
    n_valid = len(valid_frames)

    # identifications for every (frame, bead) pair, frame-major so the spot
    # stack reshapes cleanly to (n_valid, n_beads, box, box)
    frame_col = np.repeat(valid_frames, n_beads)
    x_col = np.tile(np.asarray(beads["x"]), n_valid)
    y_col = np.tile(np.asarray(beads["y"]), n_valid)
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
    n_steps = len(step_range)
    volumes = np.zeros((n_beads, box, box, n_steps), dtype=np.float32)
    for i, s in enumerate(step_range):
        mask = steps_of_valid == s
        # mean over the frames belonging to this step -> (n_beads, box, box)
        volumes[:, :, :, i] = spots[mask].mean(axis=0)
    if return_spots:
        return volumes, spots, steps_of_valid
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


def _reference_frame_bounds(
    step_of_frame: np.ndarray, step_range: np.ndarray
) -> tuple[int, int]:
    """Frame bounds of the middle third of the scan (near focus, brightest),
    used for bead detection."""
    n_steps = len(step_range)
    lo = step_range[n_steps // 3]
    hi = step_range[min(n_steps - 1, 2 * n_steps // 3)]
    ref_frames = np.where((step_of_frame >= lo) & (step_of_frame <= hi))[0]
    return int(ref_frames.min()), int(ref_frames.max())


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
    individual per-frame bead spot, ``(n_spots, box, box)``, photon units) and
    ``spot_step_idx`` (the index into the template z-axis of each spot's stage
    step), which ``_axial_precision`` fits one by one to measure the axial
    precision in the realistic single-frame regime.

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

    if beads is None:
        ref_bounds = _reference_frame_bounds(step_of_frame, step_range)
        beads = _detect_bead_positions(
            movie, minimum_ng, box, ref_bounds, roi=roi, threaded=threaded
        )
    if return_spots:
        volumes, spots, steps_of_valid = _bead_volumes(
            movie,
            camera_info,
            beads,
            box,
            step_of_frame,
            step_range,
            return_spots=True,
        )
    else:
        volumes = _bead_volumes(
            movie, camera_info, beads, box, step_of_frame, step_range
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
        # every individual per-frame bead spot flattened to (n_spots, box, box)
        # (frame-major, bead-minor) with, for each spot, the index into the
        # template z-axis (0..n_steps-1) of its stage step. z_of_step[step_idx]
        # is then the spot's known stage position (see _axial_precision).
        n_valid, n_beads_s = spots.shape[0], spots.shape[1]
        step_to_pos = {int(s): i for i, s in enumerate(step_range)}
        pos_of_frame = np.array(
            [step_to_pos[int(s)] for s in steps_of_valid], dtype=int
        )
        result["spots"] = spots.reshape(n_valid * n_beads_s, box, box)
        result["spot_step_idx"] = np.repeat(pos_of_frame, n_beads_s)
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


def _match_beads(
    ref_xy: np.ndarray, other_xy: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor match beads between two channels.

    Returns ``(ref_idx, other_idx)`` index arrays of matched pairs within
    ``max_distance``; each ``other`` bead is used at most once (closest match
    wins)."""
    from scipy.spatial import cKDTree

    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    other_xy = np.asarray(other_xy, dtype=np.float64)
    if len(ref_xy) == 0 or len(other_xy) == 0:
        empty = np.array([], dtype=int)
        return empty, empty
    tree = cKDTree(other_xy)
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
    ref_bounds: tuple[int, int],
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

    best_ref_idx, best_c_idx = np.array([], int), np.array([], int)
    for _label, aligned in candidates:
        ri, ci = _match_beads(ref_xy, aligned, max_distance)
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
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    regions: list | None = None,
    reference: int = 0,
    path: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """Generate a multichannel cubic-spline PSF calibration from registered
    bead z-stacks (one movie per channel).

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
    ``regions`` is a list of ``[[y_min, x_min], [y_max, x_max]]`` rectangles, one
    per channel, all the same size; ``regions[reference]`` is the reference
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
        (y0_ref, x0_ref) = region_rects[0][0]
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
    ref_bounds = _reference_frame_bounds(step_of_frame, step_range)
    mid_frame = (ref_bounds[0] + ref_bounds[1]) // 2

    beads_ref = _detect_bead_positions(
        movies[0], minimum_ng, box, ref_bounds, roi=ref_roi
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
        "z_center": float(z_origin),
        "z_init": float(z_init),
        "z_step_nm": float(d),
        "magnification_factor": float(magnification_factor),
        "correct_z_bias": bool(correct_z_bias),
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
                per_channel, calibration, coefficients, reg_info, path
            )
        except Exception:
            pass

    if callable(progress_callback):
        progress_callback(3)
    return calibration


def _save_multichannel_diagnostics(
    per_channel: list[dict],
    calibration: dict,
    coefficients: np.ndarray,
    reg_info: list[dict],
    path: str,
) -> None:
    """Write the multichannel calibration diagnostics next to ``path``.

    For every channel a single-channel-style PSF diagnostic PNG is saved
    (``<base>_ch{c}.png``) by re-using :func:`_save_diagnostic_plot` on that
    channel's template and coefficient block - including the GPU-refit axial
    bias/precision panels when a fitter is available (via :func:`_axial_precision`
    on the per-channel spots). A separate channel-registration PNG
    (``<base>_registration.png``) summarizes how well the non-reference channels
    align to the reference (per-channel residual field and RMS, matched-bead
    count, biplane focus offset and photon scale).
    """
    n_channels = len(per_channel)
    base, _ = os.path.splitext(path)
    photon_scale = np.asarray(
        calibration.get("photon_scale", 1.0), dtype=float
    ).ravel()
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
        try:
            precision = _axial_precision(per_channel[c], calib_c)
        except Exception:
            precision = None
        label = "reference channel" if c == 0 else f"channel {c}"
        _save_diagnostic_plot(
            per_channel[c],
            calib_c,
            f"{base}_ch{c}.hdf5",
            precision=precision,
            title_prefix=f"{label} - ",
        )
    if reg_info:
        _save_registration_diagnostic_plot(reg_info, calibration, path)


def _save_registration_diagnostic_plot(
    reg_info: list[dict], calibration: dict, path: str
) -> None:
    """Save a channel-registration diagnostic PNG (``<base>_registration.png``).

    For each non-reference channel the affine-fit residual (``transform(ref) -
    channel`` at the matched beads) is drawn as a vector field over the reference
    field of view (magnified for visibility) and summarized as an RMS bar chart,
    alongside the matched-bead count, the biplane focus offset and the per-channel
    photon scale. A tight, structure-free residual field with small RMS means the
    channels are well registered; a systematic pattern reveals a rotation/scale
    the affine could not absorb.
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

    fig = Figure(figsize=(10.5, 5.2))
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

    # full-width text summary footer (so wide columns never clip)
    lines = ["channel   beads    RMS         focus offset    photon scale"]
    for r in reg_info:
        c = r["channel"]
        po = plane_offsets[c] if c < plane_offsets.size else float("nan")
        pscale = photon_scale[c] if c < photon_scale.size else float("nan")
        lines.append(
            f"ch{c:<7d} {r['n_matches']:<6d}  "
            f"{r['_rms_px'] * ps:6.1f} nm    {po:+8.0f} nm     {pscale:7.3f}"
        )
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
    path: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """Build a multichannel spline calibration from a *single* bead z-stack in
    which several rectangular field-of-view regions are the channels (split-FOV
    optics: spectral/ratiometric splitters, biplane relays, 4Pi cameras).

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
        regions=regions,
        reference=reference,
        path=path,
        progress_callback=progress_callback,
    )


def reestimate_split_fov_transforms(
    movie,
    calibration: dict,
    regions: list,
    minimum_ng: float,
    box: int | None = None,
    reference: int = 0,
    frames_per_step: int = 1,
    frame_bounds: tuple[int, int] | list | None = None,
    frame_order: Literal["fov", "z"] = "fov",
    max_match_distance: float | None = None,
    update: bool = True,
) -> tuple[dict, list]:
    """Re-estimate the inter-channel registration of a split-FOV calibration on
    new bead data, keeping the PSF templates.

    Following globLoc, the channel registration can drift between the
    calibration and the experiment (e.g. calibrated on a different day). This
    detects beads in each region of ``movie``, matches reference<->channel and
    re-fits the per-channel affine, then updates the calibration's *region-local*
    ``channel_affines`` (and the derived ``channel_transforms``, ``regions``,
    ``region_size``) - the PSF ``coefficients`` are untouched. Run it on a bead /
    fiducial sample acquired with the current alignment; the drawn ``regions``
    (reference first) locate the channels in that movie.

    Returns ``(calibration, reg_info)`` where ``reg_info`` is the per-channel
    matched-bead diagnostic (as in :func:`calibrate_spline_multichannel`). With
    ``update=True`` the passed calibration dict is modified in place and also
    returned; set it False to only compute ``reg_info`` and the transforms.
    """
    if not calibration.get("split_fov"):
        raise ValueError(
            "reestimate_split_fov_transforms requires a split-FOV calibration."
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
    if max_match_distance is None:
        max_match_distance = float(box)

    region_rects = [_normalized_region(r) for r in regions]
    order = [reference] + [c for c in range(n_channels) if c != reference]
    region_rects = [region_rects[c] for c in order]
    sizes = {(r[1][0] - r[0][0], r[1][1] - r[0][1]) for r in region_rects}
    if len(sizes) != 1:
        raise ValueError("All regions must have the same size.")

    n_frames = int(movie.shape[0])
    d = float(calibration.get("z_step_nm", 1.0))
    step_of_frame, _, step_range = _step_of_frame(
        n_frames, d, frames_per_step, frame_order, frame_bounds
    )
    ref_bounds = _reference_frame_bounds(step_of_frame, step_range)
    mid_frame = (ref_bounds[0] + ref_bounds[1]) // 2

    beads_ref = _detect_bead_positions(
        movie, minimum_ng, box, ref_bounds, roi=region_rects[0]
    )
    (y0_ref, x0_ref) = region_rects[0][0]

    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transforms = [identity]
    reg_info = []
    for c in range(1, n_channels):
        r = region_rects[c]
        coarse_shift = (float(x0_ref - r[0][1]), float(y0_ref - r[0][0]))
        transform, n_matches, ref_m, c_m = _estimate_channel_transform(
            movie,
            movie,
            beads_ref,
            minimum_ng,
            box,
            ref_bounds,
            mid_frame,
            max_match_distance,
            channel_roi=r,
            coarse_shift=coarse_shift,
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

    affines = localize.decompose_region_affines(region_rects, transforms)
    if update:
        calibration["channel_affines"] = [a.tolist() for a in affines]
        calibration["channel_transforms"] = [t.tolist() for t in transforms]
        calibration["regions"] = [
            [[int(r[0][0]), int(r[0][1])], [int(r[1][0]), int(r[1][1])]]
            for r in region_rects
        ]
        h = region_rects[0][1][0] - region_rects[0][0][0]
        w = region_rects[0][1][1] - region_rects[0][0][1]
        calibration["region_size"] = [int(h), int(w)]
        calibration["reference"] = 0
    return calibration, reg_info


def refine_split_fov_registration(
    movie,
    calibration: dict,
    regions: list | None = None,
    box: int = 7,
    max_frames: int = 500,
    max_shift_frac: float = 0.25,
) -> tuple[dict, list]:
    """Refine a split-FOV channel registration from the current data itself, by
    cross-correlation - no beads required.

    For the common case where the split has drifted by a few pixels since the
    calibration, this re-measures each channel's region-local lateral offset
    from a projection of the loaded ``movie`` (reference region vs each channel
    region) and updates the calibration's ``channel_affines`` translation
    accordingly, keeping the linear part (rotation/scale) and the PSF templates.
    Reliable when the channels share structure (biplane / 4Pi, or fiducials in
    the field of view); for spectrally dissimilar channels the correlation is
    weak - a per-channel shift larger than ``max_shift_frac`` of the region is
    treated as unreliable and left unchanged.

    ``regions`` are the channel ROIs in the current data (reference first); when
    omitted the calibration's stored regions are used. Returns
    ``(calibration, refinements)`` with the calibration updated in place;
    ``refinements`` lists per channel the measured ``shift`` (x, y), the applied
    ``delta`` relative to the old offset, and whether it was ``applied``.
    """
    from . import imageprocess

    if not calibration.get("split_fov"):
        raise ValueError(
            "refine_split_fov_registration requires a split-FOV calibration."
        )
    rects_in = regions if regions is not None else calibration.get("regions")
    if not rects_in:
        raise ValueError(
            "No regions to refine (draw ROIs or load a split-FOV "
            "calibration)."
        )
    region_rects = [_normalized_region(r) for r in rects_in]
    n_channels = int(calibration.get("n_channels", len(region_rects)))
    if len(region_rects) != n_channels:
        raise ValueError(
            f"Got {len(region_rects)} regions but the calibration has "
            f"{n_channels} channels."
        )
    affines = calibration.get("channel_affines")
    if affines is None:
        affines = [
            a.tolist()
            for a in localize.decompose_region_affines(
                calibration["regions"], calibration["channel_transforms"]
            )
        ]

    # mean projection over a bounded, evenly-sampled set of frames
    n_frames = int(movie.shape[0])
    sample = np.unique(
        np.linspace(0, n_frames - 1, min(n_frames, max_frames)).astype(int)
    )
    proj = np.zeros(movie.shape[1:], dtype=np.float64)
    for f in sample:
        proj += np.asarray(movie[f], dtype=np.float64)
    proj /= len(sample)

    (ry0, rx0), (ry1, rx1) = region_rects[0]
    ref_img = proj[ry0:ry1, rx0:rx1]

    new_affines = [np.asarray(affines[0], dtype=np.float64)]
    refinements = []
    for c in range(1, n_channels):
        (cy0, cx0), (cy1, cx1) = region_rects[c]
        c_img = proj[cy0:cy1, cx0:cx1]
        h = min(ref_img.shape[0], c_img.shape[0])
        w = min(ref_img.shape[1], c_img.shape[1])
        yc, xc = imageprocess.get_image_shift(
            ref_img[:h, :w], c_img[:h, :w], box
        )
        a = np.asarray(affines[c], dtype=np.float64)
        linear, old_t = a[:, :2], a[:, 2]
        measured = np.array([float(xc), float(yc)])  # region-local (x, y)
        applied = max(abs(xc), abs(yc)) <= max_shift_frac * max(h, w)
        new_t = measured if applied else old_t
        new_affines.append(np.hstack([linear, new_t[:, None]]))
        refinements.append(
            {
                "channel": c,
                "shift": (float(measured[0]), float(measured[1])),
                "delta": (
                    float(new_t[0] - old_t[0]),
                    float(new_t[1] - old_t[1]),
                ),
                "applied": bool(applied),
            }
        )

    transforms = localize.compose_region_transforms(region_rects, new_affines)
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
    return calibration, refinements


def decompose_phase_volumes(
    volumes: np.ndarray, phases: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic (phase-stepping) decomposition of PSF volumes into the three
    spline components of the 4Pi phase model.

    Given ``P`` PSF volumes of the same emitter recorded at known interferometer
    phases ``phases`` (radians), least-squares fits, per voxel,

        ``I_p = mean + cos(phase_p) * modulation + sin(phase_p) * modulation_90deg``

    which is exactly the intensity the Gpufit phase model (model 12) reconstructs
    for emitter phase ``phase_p``. Returns ``(mean, modulation,
    modulation_90deg)``, each of shape ``volumes.shape[1:]``.

    ``P >= 3`` phases are required and they must not be degenerate on the unit
    circle (e.g. evenly-spaced steps over ``[0, 2*pi)`` are ideal).
    """
    volumes = np.asarray(volumes, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64).ravel()
    n_phases = len(volumes)
    if n_phases < 3 or len(phases) != n_phases:
        raise ValueError(
            "decompose_phase_volumes needs >= 3 volumes and one phase each."
        )
    design = np.stack(
        [np.ones(n_phases), np.cos(phases), np.sin(phases)], axis=1
    )  # (P, 3)
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError(
            "Degenerate phases: need >= 3 distinct interferometer phases "
            "spanning the unit circle."
        )
    coef, *_ = np.linalg.lstsq(
        design, volumes.reshape(n_phases, -1), rcond=None
    )
    shape = volumes.shape[1:]
    return (
        coef[0].reshape(shape),
        coef[1].reshape(shape),
        coef[2].reshape(shape),
    )


def calibrate_spline_phase(
    phase_templates: np.ndarray,
    phases: np.ndarray,
    d: float,
    magnification_factor: float = 0.79,
    channel_transforms: list | None = None,
    pixelsize: float = 130.0,
    z_center_index: int | None = None,
    path: str | None = None,
) -> dict:
    """Build a 4Pi phase (Gpufit model 12) spline calibration from phase-stepped
    PSF template volumes.

    This is the decomposition + spline half of a 4Pi calibration. The
    setup-specific front-end (detecting beads and building a background-free,
    registered PSF template volume from a bead z-stack at each interferometer
    phase step, and registering the spatial channels) is left to the caller,
    which can reuse :func:`build_psf_template` per phase-step movie - its
    ``template`` output is exactly the expected input here.

    Parameters
    ----------
    phase_templates : np.ndarray
        PSF template volumes at each interferometer phase step:
        ``(n_phase_steps, box, box, n_z)`` for a single spatial channel, or
        ``(n_phase_steps, box, box, n_z, n_channels)`` for several spatially
        registered channels. Axes are ``(row=y, col=x, z)`` as returned by
        ``build_psf_template`` / ``get_spots``.
    phases : np.ndarray
        The ``n_phase_steps`` interferometer phases (radians); see
        :func:`decompose_phase_volumes`.
    d : float
        Axial step size between z-slices (nm).
    magnification_factor, pixelsize, channel_transforms, z_center_index, path
        As in :func:`calibrate_spline_multichannel`. ``z_center_index`` is the
        in-focus z-slice (default: the central slice).

    Returns
    -------
    calibration : dict
        A ``"spline-3d-phase-multichannel"`` calibration with coefficients
        ``(64, n_int_x, n_int_y, n_int_z, n_channels, 3)`` (the trailing 3 are
        the mean / modulation / modulation-90deg spline sets), ready for
        ``localize.fit_spline_phase_multichannel``.
    """
    if not localize.GPUSPLINE_INSTALLED:
        raise ImportError(
            "Gpuspline is required to build a spline PSF calibration but "
            "could not be loaded. See picasso/ext/pygpuspline/README.txt."
        )
    templates = np.asarray(phase_templates, dtype=np.float64)
    if templates.ndim == 4:
        templates = templates[..., None]  # single spatial channel
    if templates.ndim != 5:
        raise ValueError(
            "phase_templates must be (n_phase_steps, box, box, n_z) or "
            "(n_phase_steps, box, box, n_z, n_channels)."
        )
    n_phase_steps, box, box2, n_z, n_channels = templates.shape
    if box != box2:
        raise ValueError("PSF templates must be square in the lateral axes.")
    # remove a common background so the model's separate offset is not absorbed
    templates = templates - float(templates.min())
    z_center = (
        (n_z - 1) // 2 if z_center_index is None else int(z_center_index)
    )

    n_intervals = [box - 1, box - 1, n_z - 1]
    coefficients = np.zeros(
        [64] + n_intervals + [n_channels, 3], dtype=np.float32
    )
    photon_scales = []
    gs = localize.gs
    for c in range(n_channels):
        mean, mod, mod90 = decompose_phase_volumes(templates[..., c], phases)
        peak = float(np.max(mean[:, :, z_center]))
        if peak <= 0:
            raise ValueError(
                "Non-positive mean PSF peak; the 4Pi calibration failed "
                "(check the bead brightness / phase sampling)."
            )
        # unit-peak normalize all three together (a common scale is absorbed by
        # the fitted amplitude); photon_scale from the mean in-focus integral
        mean, mod, mod90 = mean / peak, mod / peak, mod90 / peak
        photon_scales.append(
            float(np.clip(mean[:, :, z_center], 0, None).sum())
        )
        for s, vol in enumerate((mean, mod, mod90)):
            # swap lateral axes (row=y, col=x) -> (x, y) so the spline's fast
            # axis is x, matching the model's pixel index (see calibrate_spline)
            vol = np.ascontiguousarray(
                vol.transpose(1, 0, 2).astype(np.float32)
            )
            coeff = gs.spline_coefficients(vol)
            coefficients[..., c, s] = np.reshape(coeff, [64] + n_intervals)

    if channel_transforms is None:
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        channel_transforms = [identity for _ in range(n_channels)]

    calibration = {
        "model": "spline-3d-phase-multichannel",
        "coefficients": coefficients,
        "n_data": [int(box), int(box), int(n_z)],
        "n_intervals": n_intervals,
        "n_channels": int(n_channels),
        "channel_transforms": [
            np.asarray(t, dtype=float).tolist() for t in channel_transforms
        ],
        "oversampling": 1.0,
        "z_center": float(z_center),
        "z_init": float(z_center),
        "z_step_nm": float(d),
        "magnification_factor": float(magnification_factor),
        "photon_scale": photon_scales,
        "box": int(box),
        "pixelsize": float(pixelsize),
        "phases": np.asarray(phases, dtype=float).ravel().tolist(),
        "Generated by": (
            f"Picasso: v{__version__} Spline PSF calibration (4Pi phase)"
        ),
        "Path": path if path is not None else "N/A",
    }
    if path is not None:
        io.save_spline_calibration(path, calibration)
    return calibration


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
    theta = np.asarray(theta)

    z_step_nm = float(calibration.get("z_step_nm", 1.0))
    scan_center = _scan_center_index(z_of_step)
    z_fit = (theta[:, 3] + scan_center) * z_step_nm  # (n_spots,)
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
        "n_beads": int(built.get("n_beads", 0)),
        "n_spots": int(np.isfinite(deviation).sum()),
    }


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
        bottom_panels.append(("Estimated z vs stage", _plot_scatter))
        bottom_panels.append(("Axial bias", _plot_bias))
        bottom_panels.append(("Axial precision", _plot_precision))
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
