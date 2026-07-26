"""
picasso.fourpi
~~~~~~~~~~~~~~

4Pi / interferometric phase-to-z reconstruction.

A 4Pi (model 12) fit yields, per emitter, a coarse axial position ``z`` from the
astigmatic envelope and a fine interference ``phase`` that is precise
but wrap every ``pi / frequency`` in z. This module fuses the two into a
singl high-precision, non-degenerate axial coordinate, following the
SMAP pipelin (``z_from_phi_JR`` / ``getz0phase`` / ``cyclicaverage`` and
the ``Phase2z4Pi` plugin, Ries lab).

The interference term is ``cos(2 * frequency * z + phaseshift)``, so the phase
advances by ``2 * frequency`` per unit z and the fringe period (z per ``2*pi``
of phase) is ``zT = pi / frequency``. ``frequency`` must therefore be given in
radians per whatever unit ``z`` is expressed in (z-slice index, or nm).

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def z_from_phase(
    z_coarse: np.ndarray,
    phase: np.ndarray,
    frequency: float,
    z0: float = 0.0,
) -> np.ndarray:
    """Unwrap the interference phase into a precise z using the coarse z.

    Port of SMAP's ``z_from_phi_JR``. The coarse (astigmatic) z selects which
    interference fringe the wrapped ``phase`` belongs to; within that fringe the
    phase gives the precise sub-fringe position.

    Parameters
    ----------
    z_coarse : array_like
        Coarse astigmatic z (same units as the returned z).
    phase : array_like
        Interference phase (radians), typically wrapped to ``[0, 2*pi)``.
    frequency : float
        Interference frequency ``k`` such that the fringe term is
        ``cos(2*k*z + phaseshift)``; units of radians per unit z. The fringe
        period is ``pi / frequency``.
    z0 : float, optional
        Phase reference: the z at which the (unwrapped) interference phase is 0.
        See :func:`estimate_phase_reference`.

    Returns
    -------
    z : np.ndarray
        Precise z, same units as ``z_coarse``.
    """
    z_coarse = np.asarray(z_coarse, dtype=float)
    phase = np.asarray(phase, dtype=float)
    # expected continuous phase predicted from the coarse z
    phi_predicted = 2.0 * frequency * (z_coarse - z0)
    # nearest whole number of 2*pi fringes between prediction and measurement
    n_fringe = np.round((phi_predicted - phase) / (2.0 * np.pi))
    return n_fringe * (np.pi / frequency) + phase / (2.0 * frequency)


def cyclic_average(
    data: np.ndarray,
    period: float,
    weights: np.ndarray | None = None,
    n_candidates: int = 5,
    max_iter: int = 20,
) -> float:
    """Wrap-around (circular) robust mean over a periodic domain.

    Port of SMAP's ``cyclicaverage`` (the active branch). Picks the best of
    ``n_candidates`` evenly spaced starting centers (smallest wrapped spread),
    then iteratively refines a weighted mean of the data re-wrapped around the
    current center. Returns the mean in ``[0, period)``.

    Parameters
    ----------
    data : array_like
        Values on the periodic domain ``[0, period)`` (values outside are
        wrapped in).
    period : float
        The period.
    weights : array_like, optional
        Per-value weights (default: uniform).
    n_candidates, max_iter : int
        Number of candidate start centers and refinement iterations.

    Returns
    -------
    float
        The circular mean in ``[0, period)`` (``nan`` for empty input).
    """
    data = np.asarray(data, dtype=float).ravel()
    if weights is None:
        weights = np.ones_like(data)
    else:
        weights = np.asarray(weights, dtype=float).ravel()
    # Drop the unusable NaN values - a partly failed fit still has a
    # well-defined circular mean over the rest
    if not np.isfinite(weights).any():
        weights = np.ones_like(data)
    keep = np.isfinite(data) & np.isfinite(weights)
    if not keep.all():
        data, weights = data[keep], weights[keep]
    if data.size == 0 or np.sum(weights) <= 0:
        return float("nan")
    half = period / 2.0
    # coarse start: candidate center with the smallest wrapped spread
    candidates = np.arange(n_candidates) * (period / n_candidates)
    spread = [
        np.sqrt(np.sum((np.mod(data - c + half, period) - half) ** 2))
        for c in candidates
    ]
    center = float(candidates[int(np.nanargmin(spread))])
    # fixed-point refinement of the weighted circular mean
    max_err = 1e-9 * period
    w_sum = np.sum(weights)
    center_new = center
    for _ in range(max_iter):
        wrapped = np.mod(data - center + half, period)
        center_new = np.sum(wrapped * weights) / w_sum - half + center
        if abs(center_new - center) < max_err:
            break
        center = center_new
    return float(np.mod(center_new, period))


def estimate_phase_reference(
    z_coarse: np.ndarray,
    phase: np.ndarray,
    frequency: float,
    z0: float = 0.0,
    weights: np.ndarray | None = None,
) -> float:
    """Estimate the global phase reference ``z0`` (SMAP ``getz0phase``).

    Forms the residual between the phase-derived z (``phase/(2*frequency)``) and
    the coarse z, wraps it modulo one fringe period ``pi/frequency`` and takes a
    :func:`cyclic_average`. ``z0`` can be refined by passing the previous estimate
    back in.

    Parameters
    ----------
    z_coarse, phase, frequency : see :func:`z_from_phase`.
    z0 : float, optional
        Previous ``z0`` estimate to refine around (default 0).
    weights : array_like, optional
        Per-localization weights (e.g. inverse phase variance).

    Returns
    -------
    float
        The estimated phase reference ``z0`` (same units as ``z_coarse``).
    """
    z_coarse = np.asarray(z_coarse, dtype=float)
    phase = np.asarray(phase, dtype=float)
    period = np.pi / frequency
    z_phase = phase / (2.0 * frequency)
    residual = z_phase - z_coarse + period / 2.0 + z0
    residual = np.mod(residual, period)
    return (
        -cyclic_average(residual, period, weights=weights) + period / 2.0 + z0
    )


def frequency_from_calibration(
    calibration: dict, magnification_factor: float | None = None
) -> float | None:
    """Interference frequency in **rad per nm of localization z** from a phase
    calibration.

    Two z spaces meet here. The calibration measures the fringe against the
    **stage**, where its beads sit on the coverslip and move 1:1 with it, so the
    stored ``zt_nm`` (and the per-slice ``frequency`` together with
    ``z_step_nm``) are in raw stage nm. Localizations, by contrast, carry
    **sample-space** z: the fitter scales z by ``magnification_factor`` to
    correct the refractive-index focal shift. This converts the calibration's
    period into the localizations' units, so one stage nm becomes ``mag`` nm of
    localization z.

    Prefers the stored fringe period ``zt_nm``, falling back to the per-z-slice
    ``frequency`` scaled by the axial step. Returns ``None`` if the calibration
    carries neither. The sign is not meaningful here (the coarse-z / phase sign
    convention is resolved at reconstruction time); the magnitude is what
    matters.
    """
    mag = (
        magnification_factor
        if magnification_factor is not None
        else float(calibration.get("magnification_factor", 1.0))
    )
    mag = float(mag) or 1.0
    zt_nm = calibration.get("zt_nm")
    if zt_nm and np.isfinite(zt_nm) and zt_nm > 0:
        return float(np.pi / (zt_nm * mag))
    freq_slice = calibration.get("frequency")
    if freq_slice:
        d = float(calibration.get("z_step_nm", 1.0))
        denom = d * mag
        if abs(freq_slice) > 1e-12 and denom > 0:
            return float(abs(freq_slice) / denom)
    return None


def reconstruct_z_from_phase(
    locs: pd.DataFrame,
    calibration: dict,
    window_frames: int | None = 2000,
    frequency: float | None = None,
) -> pd.DataFrame:
    """Fuse the coarse (astigmatic) z and the fine interference phase of a 4Pi
    model-12 fit into one high-precision z (SMAP ``Phase2z4Pi`` port).

    Reads the coarse ``z`` (nm) and wrapped ``phase`` columns, unwraps the phase
    against the coarse z with :func:`z_from_phase`, and writes the precise z back
    into ``z`` (the coarse value is preserved as ``z_astig``). The global phase
    reference ``z0`` is estimated robustly (cyclic average); with
    ``window_frames`` set and enough windows, a smoothing spline ``z0(frame)``
    tracks slow cavity drift. The sign of the phase-vs-z relation is auto-
    detected (both signs tried; the one whose unwrapped z best matches the coarse
    z is kept). ``lpz`` is set from ``phase_unc / (2*|frequency|)``.

    Parameters
    ----------
    locs : pd.DataFrame
        Model-12 localizations with ``z``, ``phase`` (and ideally ``frame``,
        ``phase_unc``).
    calibration : dict
        The ``spline-3d-phase-multichannel`` calibration (for ``zt_nm`` /
        ``frequency``).
    window_frames : int or None
        Frames per window for the drift-tracking ``z0(frame)``. ``None`` (or too
        few frames) uses a single global ``z0``.
    frequency : float or None
        Signed interference frequency in rad per nm. If ``None`` it is taken from
        the calibration (magnitude) with the sign auto-detected.

    Returns
    -------
    pd.DataFrame
        A copy of ``locs`` with ``z`` replaced by the precise z, ``z_astig`` the
        coarse z, and (when available) ``lpz`` from the phase uncertainty.
    """
    out = locs.copy()
    if "z" not in out or "phase" not in out:
        raise ValueError("locs must have 'z' and 'phase' columns (model 12).")
    z_coarse = np.asarray(out["z"], dtype=float)
    phase = np.mod(np.asarray(out["phase"], dtype=float), 2.0 * np.pi)

    k = frequency
    if k is None:
        k = frequency_from_calibration(calibration)
    if not k:
        raise ValueError(
            "No interference frequency: pass `frequency` or a calibration with "
            "`zt_nm`/`frequency`."
        )

    # resolve the sign: pick the frequency sign whose unwrapped z best matches
    # the coarse z (smallest median |z_phi - z_coarse|)
    if frequency is None:
        best_k, best_res = k, np.inf
        for cand in (abs(k), -abs(k)):
            z0 = estimate_phase_reference(z_coarse, phase, cand)
            res = float(
                np.median(
                    np.abs(z_from_phase(z_coarse, phase, cand, z0) - z_coarse)
                )
            )
            if res < best_res:
                best_res, best_k = res, cand
        k = best_k

    weights = None
    if "phase_unc" in out:
        pu = np.asarray(out["phase_unc"], dtype=float)
        weights = 1.0 / np.clip(pu, 1e-6, None) ** 2
        # a non-finite CRLB (singular information matrix) must not weight a
        # localization out of existence, nor poison the circular statistics
        if not np.isfinite(weights).any():
            weights = None

    # global reference, refined once
    z0_global = estimate_phase_reference(z_coarse, phase, k, weights=weights)
    z0_global = estimate_phase_reference(
        z_coarse, phase, k, z0=z0_global, weights=weights
    )

    z0_per_loc = np.full(len(out), z0_global)
    frame = np.asarray(out["frame"], dtype=float) if "frame" in out else None
    if window_frames and frame is not None:
        f_lo, f_hi = float(frame.min()), float(frame.max())
        n_win = int(np.ceil((f_hi - f_lo + 1) / window_frames))
        if n_win >= 4:
            centers, z0s = [], []
            for w in range(n_win):
                a = f_lo + w * window_frames
                b = a + window_frames
                m = (frame >= a) & (frame < b)
                if m.sum() < 20:
                    continue
                ww = weights[m] if weights is not None else None
                z0w = estimate_phase_reference(
                    z_coarse[m], phase[m], k, z0=z0_global, weights=ww
                )
                centers.append((a + b) / 2.0)
                z0s.append(z0w)
            if len(centers) >= 4:
                from scipy.interpolate import make_smoothing_spline

                order = np.argsort(centers)
                centers = np.asarray(centers)[order]
                z0s = np.asarray(z0s)[order]
                spl = make_smoothing_spline(centers, z0s)
                z0_per_loc = spl(np.clip(frame, centers[0], centers[-1]))

    out["z_astig"] = z_coarse.astype(np.float32)
    out["z"] = z_from_phase(z_coarse, phase, k, z0_per_loc).astype(np.float32)
    if "phase_unc" in out:
        out["lpz"] = (
            np.asarray(out["phase_unc"], dtype=float) / (2.0 * abs(k))
        ).astype(np.float32)
    return out
