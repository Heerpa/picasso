"""
picasso.spline
~~~~~~~~~~~~~~

Generate cubic-spline PSF calibrations from a bead z-stack.

A calibration bead sample (e.g., fluorescent/gold beads) is imaged while the
stage is scanned through z. This module averages the beads into a clean,
laterally-centered PSF volume, normalizes it, and computes cubic-spline
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

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
from typing import Callable, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift as _ndi_shift

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

    Returns a data frame with integer ``x``/``y`` columns (one row per bead).
    """
    if min_separation is None:
        min_separation = box
    ids, _ = localize.identify(
        movie,
        minimum_ng,
        box,
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
) -> np.ndarray:
    """Extract and z-step-average a PSF volume for every bead.

    Returns an array of shape ``(n_beads, box, box, n_steps)`` in photon units,
    where each z-slice is the mean over all (multi-FOV) frames assigned to that
    step.
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


def _register_and_average(volumes: np.ndarray, z_center: int) -> np.ndarray:
    """Laterally center every bead volume on its in-focus slice and average.

    Each bead's sub-pixel center is measured from its ``z_center`` slice with
    a Gaussian fit (offset from the box center) and the whole bead volume is
    shifted to the center before averaging. Returns the mean PSF volume
    ``(box, box, n_steps)``.
    """
    n_beads, box, _, n_steps = volumes.shape
    accum = np.zeros((box, box, n_steps), dtype=np.float64)
    n_used = 0
    for b in range(n_beads):
        focus_slice = np.ascontiguousarray(volumes[b, :, :, z_center])
        theta = gausslq.fit_spot(focus_slice)
        dx, dy = theta[0], theta[1]  # offset from box center (col, row)
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue
        if abs(dx) > box / 2 or abs(dy) > box / 2:
            continue  # nonsense fit, skip this bead
        # shift so the bead center lands at the box center; axes (row, col, z)
        shifted = _ndi_shift(
            volumes[b], shift=(-dy, -dx, 0.0), order=3, mode="nearest"
        )
        accum += shifted
        n_used += 1
    if n_used == 0:
        raise ValueError(
            "No usable beads after centering; the calibration failed."
        )
    return (accum / n_used).astype(np.float32)


def _normalize_template(
    volume: np.ndarray, z_center: int
) -> tuple[np.ndarray, float, float, float]:
    """Normalize a PSF volume to a unit-peak template ``(psf - bg) / amp``.

    Background is the mean of the 1-pixel border across all slices; amplitude
    is the peak of the (background-subtracted) in-focus slice. Returns
    ``(template, background, amplitude, photon_scale)`` where ``photon_scale``
    is the integral of the in-focus normalized slice (used to convert a fitted
    amplitude into integrated photons)."""
    border = np.concatenate(
        [
            volume[0, :, :].ravel(),
            volume[-1, :, :].ravel(),
            volume[:, 0, :].ravel(),
            volume[:, -1, :].ravel(),
        ]
    )
    background = float(np.mean(border))
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
) -> dict:
    """Build a normalized PSF template volume from a bead z-stack.

    This is the GPU-independent part of the calibration (no Gpuspline needed),
    factored out so it can be unit-tested. Returns a dict with keys
    ``template`` (box, box, n_steps), ``z_center``, ``effective_sigma``,
    ``background``, ``amplitude``, ``photon_scale``, ``n_beads`` and
    ``z_of_step``.

    If ``beads`` (a data frame with integer ``x``/``y`` columns) is given, it
    is used instead of detecting beads on this movie - this lets the
    multichannel calibration reuse the same physical beads, mapped into each
    channel, across all channels.
    """
    n_frames = int(movie.shape[0])
    step_of_frame, z_of_step, step_range = _step_of_frame(
        n_frames, d, frames_per_step, frame_order, frame_bounds
    )

    if beads is None:
        ref_bounds = _reference_frame_bounds(step_of_frame, step_range)
        beads = _detect_bead_positions(
            movie, minimum_ng, box, ref_bounds, threaded=threaded
        )
    volumes = _bead_volumes(
        movie, camera_info, beads, box, step_of_frame, step_range
    )
    # first pass on the raw bead-average to locate focus, then register
    z_center, _ = _focus_step(volumes.mean(axis=0))
    mean_volume = _register_and_average(volumes, z_center)
    z_center, effective_sigma = _focus_step(mean_volume)
    template, background, amplitude, photon_scale = _normalize_template(
        mean_volume, z_center
    )
    return {
        "template": template,
        "z_center": z_center,
        "effective_sigma": effective_sigma,
        "background": background,
        "amplitude": amplitude,
        "photon_scale": photon_scale,
        "n_beads": int(len(beads)),
        "z_of_step": z_of_step[step_range],
    }


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
    path : str, optional
        Where to save the calibration (HDF5) and a diagnostic PNG. If None,
        nothing is written. Default None.
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
    )
    template = built["template"]  # (box, box, n_steps)
    z_center = built["z_center"]

    if callable(progress_callback):
        progress_callback(1)

    gs = localize.gs
    if model == "spline-2d":
        slab = np.ascontiguousarray(template[:, :, z_center])
        coefficients = gs.spline_coefficients(slab)
        n_intervals = [int(i) for i in (np.array(slab.shape) - 1)]
        coefficients = np.reshape(coefficients, [16] + n_intervals).astype(
            np.float32
        )
        n_data = [box, box]
    else:
        coefficients = gs.spline_coefficients(template)
        n_intervals = [int(i) for i in (np.array(template.shape) - 1)]
        coefficients = np.reshape(coefficients, [64] + n_intervals).astype(
            np.float32
        )
        n_data = [int(s) for s in template.shape]

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
        "z_center": float(z_center),
        "z_step_nm": float(d),
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
        _save_diagnostic_plot(built, calibration, path)

    if callable(progress_callback):
        progress_callback(3)
    return calibration


# ----------------------------------------------------------------------
# Multichannel calibration (SPLINE_3D_MULTICHANNEL)
# ----------------------------------------------------------------------


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
) -> tuple[np.ndarray, int]:
    """Estimate the affine transform mapping reference-channel coordinates to
    channel ``c``.

    Beads are detected in channel ``c``, coarsely aligned to the reference via
    image cross-correlation, matched to the reference beads and an affine
    transform is fitted to the correspondences. Returns
    ``(transform (2, 3), n_matches)``."""
    from . import imageprocess

    beads_c = _detect_bead_positions(movie_c, minimum_ng, box, ref_bounds)
    ref_xy = beads_ref[["x", "y"]].to_numpy(dtype=np.float64)
    c_xy = beads_c[["x", "y"]].to_numpy(dtype=np.float64)

    # coarse translational pre-alignment from a mid (in-focus) frame
    try:
        img_ref = np.asarray(movie_ref[mid_frame], dtype=np.float32)
        img_c = np.asarray(movie_c[mid_frame], dtype=np.float32)
        dy, dx = imageprocess.get_image_shift(img_ref, img_c, 5)
    except Exception:
        dx, dy = 0.0, 0.0

    # try both shift signs and keep whichever yields more correspondences
    best_ref_idx, best_c_idx = np.array([], int), np.array([], int)
    for sign in (1.0, -1.0):
        shifted = c_xy + sign * np.array([dx, dy])
        ri, ci = _match_beads(ref_xy, shifted, max_distance)
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
    return transform, int(len(best_ref_idx))


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
    max_match_distance: float | None = None,
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
    must share the same frame layout (z scan). Returns a
    ``"spline-3d-multichannel"`` calibration dict.
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

    if callable(progress_callback):
        progress_callback(0)

    n_frames = int(movies[0].shape[0])
    step_of_frame, _, step_range = _step_of_frame(
        n_frames, d, frames_per_step, frame_order, frame_bounds
    )
    ref_bounds = _reference_frame_bounds(step_of_frame, step_range)
    mid_frame = (ref_bounds[0] + ref_bounds[1]) // 2

    beads_ref = _detect_bead_positions(movies[0], minimum_ng, box, ref_bounds)

    # channel transforms (channel 0 is the identity reference)
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transforms = [identity]
    for c in range(1, n_channels):
        transform, _ = _estimate_channel_transform(
            movies[0],
            movies[c],
            beads_ref,
            minimum_ng,
            box,
            ref_bounds,
            mid_frame,
            max_match_distance,
        )
        transforms.append(transform)

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
        )
        per_channel.append(built)

    if callable(progress_callback):
        progress_callback(2)

    # assemble coefficients (64, n_int_x, n_int_y, n_int_z, n_channels)
    templates = [p["template"] for p in per_channel]
    n_intervals = [int(i) for i in (np.array(templates[0].shape) - 1)]
    coefficients = np.zeros(
        [64] + n_intervals + [n_channels], dtype=np.float32
    )
    for c, template in enumerate(templates):
        coeff_c = gs.spline_coefficients(template)
        coefficients[..., c] = np.reshape(coeff_c, [64] + n_intervals)

    ref = per_channel[0]
    pixelsize = camera_infos[0].get("Pixelsize", 130)
    calibration = {
        "model": "spline-3d-multichannel",
        "coefficients": coefficients,
        "n_data": [int(s) for s in templates[0].shape],
        "n_intervals": n_intervals,
        "n_channels": n_channels,
        "channel_transforms": [t.tolist() for t in transforms],
        "oversampling": 1.0,
        "z_center": float(ref["z_center"]),
        "z_step_nm": float(d),
        "effective_sigma": float(ref["effective_sigma"]),
        "photon_scale": float(ref["photon_scale"]),
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

    if path is not None:
        io.save_spline_calibration(path, calibration)

    if callable(progress_callback):
        progress_callback(3)
    return calibration


def _save_diagnostic_plot(built: dict, calibration: dict, path: str) -> None:
    """Save a PNG summarizing the calibration (PSF slices + focus curve)."""
    template = built["template"]
    z_center = built["z_center"]
    z_of_step = built["z_of_step"]
    n_steps = template.shape[2]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    slice_idx = [
        0,
        max(0, z_center // 2),
        z_center,
        n_steps - 1,
    ]
    for ax, k in zip(axes[:3], slice_idx[:3]):
        ax.imshow(template[:, :, k], cmap="hot")
        ax.set_title(f"z = {z_of_step[k]:.0f} nm (step {k})")
        ax.axis("off")

    # focus curve: peak intensity vs z step
    peak = template.max(axis=(0, 1))
    axes[3].plot(z_of_step, peak, ".-")
    axes[3].axvline(z_of_step[z_center], color="0.3", lw=1.0)
    axes[3].set_xlabel("Stage position (nm)")
    axes[3].set_ylabel("Normalized PSF peak")
    axes[3].set_title(f"{built['n_beads']} beads")
    plt.tight_layout()

    base, _ = os.path.splitext(path)
    plt.savefig(base + ".png", format="png", dpi=200)
    plt.close(fig)
