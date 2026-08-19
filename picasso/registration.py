"""
picasso.registration
~~~~~~~~~~~~~~~~~~~~

Registering the channels of a multichannel acquisition onto one reference
channel, and the standalone calibration file that stores the result.

A multichannel fit needs to know where a molecule seen at ``x`` in the
reference channel lands in every other channel. That mapping is a plain
geometric transform per channel, independent of the PSF model fitted
afterwards: :mod:`picasso.spline` bakes one into its multichannel PSF
calibration, and the 2D Gaussian multichannel fit reads one from the
standalone calibration built here.

Two ways to measure it, both producing the same file:

``calibrate_channel_registration_from_beads``
    From images of fiducial beads, which appear in every channel at once.

``calibrate_channel_registration_from_signal``
    From the experimental blinking data itself. The channels are
    frame-synchronized, so the same emitter fluoresces in every channel in the
    *same* frame; pairing those detections frame by frame registers the
    channels with no separate bead acquisition.

The matching machinery underneath is shared with :mod:`picasso.spline`, which
registers the channels of its own multichannel PSF calibration the same way.
That shared part is public API rather than module-private:

===============================  =============================================
:func:`match_points`             nearest-neighbour pairing of two point clouds
:func:`ransac_match`             robust pairing with no prior estimate
:func:`fit_registration`         one ICP iteration's transform
:func:`register_from_point_sets` the whole bootstrap + ICP + trim loop
:func:`resolve_model`            the transform model a calibration implies
:func:`frames_in_bounds`         the frame indices a bound allows
===============================  =============================================

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import datetime
import warnings
from itertools import combinations
from typing import Callable

import numpy as np
from scipy.spatial import KDTree

from . import io, localize, __version__

# aliased: `transforms` is used as a local name for lists of channel
# transforms throughout this module
from . import transforms as tform

#: ``model`` of the standalone channel-registration calibration, so it can be
#: told apart from a spline PSF calibration that also carries channel
#: transforms.
REGISTRATION_MODEL = "channel-registration"


def resolve_model(calibration: dict, model: str | None) -> str:
    """The transform model to register with.

    Parameters
    ----------
    calibration : dict
        A calibration carrying ``channel_transforms``; the model its stored
        transforms were fitted with is used when ``model`` is None. Read off
        the transforms themselves rather than a separate key, so it cannot
        disagree with them.
    model : str or None
        The model asked for, as in :mod:`picasso.transforms`. None falls back
        to the calibration's own.

    Returns
    -------
    model : str
        The model asked for, else the calibration's own, else ``"affine"``.
    """
    if model is not None:
        return model
    stored = calibration.get("channel_transforms")
    for entry in (stored or [])[1:] or (stored or [])[:1]:
        return tform.from_dict(entry).model
    return "affine"


def fit_registration(
    src: np.ndarray,
    dst: np.ndarray,
    model: str,
    final: bool = True,
) -> tuple[tform.Transform, str]:
    """Fit one ICP iteration's transform.

    Intermediate iterations (``final=False``) always fit an affine, however
    flexible ``model`` is: the early pairing is deliberately loose and contains
    cross-molecule mismatches, and a flexible model bends to accommodate them,
    locking in the wrong correspondence field on the next pass - a failure an
    affine cannot have. Only the final iteration, which pairs at the tightest
    radius, fits the model the user asked for.

    Parameters
    ----------
    src, dst : np.ndarray
        ``(n, 2)`` matched correspondences, in ``[x, y]``. The fitted transform
        maps ``src`` onto ``dst``.
    model : str
        Transform model to fit, as in :mod:`picasso.transforms`.
    final : bool, optional
        Whether this is the last ICP iteration, and therefore the one that
        fits ``model`` rather than an affine. Default True.

    Returns
    -------
    transform : picasso.transforms.Transform
        The fitted transform.
    model : str
        The model that was *actually* fitted. If too few correspondences
        survived for the one asked for, an affine is fitted instead and named
        here, so the caller can report the fallback rather than hide it.
    """
    if final and len(src) >= tform.min_points(model):
        return tform.estimate(src, dst, model), model
    with warnings.catch_warnings():
        # An intermediate ICP pass is an internal step towards the pairing,
        # not a registration anyone keeps, so its "thin data" warning would
        # only be noise; the final fit above still warns.
        if not final:
            warnings.simplefilter("ignore")
        return tform.estimate(src, dst, "affine"), "affine"


def _similarity_from_two(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> list[tform.AffineTransform]:
    """Candidate similarity transforms mapping ``a -> b`` from two point pairs.

    A similarity (translation + rotation + isotropic scale, optionally a
    reflection) is fixed by two correspondences up to the reflection ambiguity,
    so both the proper-rotation and the reflected solution are returned. Using a
    *similarity* (4 DOF) as the RANSAC minimal model - rather than a full 6-DOF
    affine, which three points always fit exactly - keeps
    a spare bead to validate the sample, so correct correspondences can be told
    from wrong ones even with only three beads. Empty if the two reference
    points coincide.

    This stays a similarity whatever model the registration is finally fitted
    with: matching only needs a hypothesis good enough to rank correspondences,
    and a higher-DOF minimal model would defeat the consensus vote - a degree-3
    polynomial fits *any* 10 points exactly, so every sample would score the
    maximum."""
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
        matrix = np.eye(3)
        matrix[:2, :2] = A
        matrix[:2, 2] = t
        out.append(tform.AffineTransform(matrix=matrix))
    return out


def _fov_groups(
    ref_fov: np.ndarray | None,
    c_fov: np.ndarray | None,
    n_ref: int,
    n_c: int,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Row blocks of the reference / channel bead clouds that share a field of
    view, as ``[(ref_idx, c_idx), ...]``, or None to treat them as one pooled
    cloud.

    Fields present in only one of the two clouds are dropped: a bead with no
    counterpart to pair against cannot contribute a correspondence, and letting
    it search the other fields is exactly the mis-pairing this prevents. None
    is returned when either label array is missing or does not describe its
    cloud, so callers without FOV information keep the pooled behaviour.
    """
    if ref_fov is None or c_fov is None:
        return None
    ref_fov = np.asarray(ref_fov)
    c_fov = np.asarray(c_fov)
    if len(ref_fov) != n_ref or len(c_fov) != n_c:
        return None
    groups = []
    for k in np.unique(ref_fov):
        ri = np.flatnonzero(ref_fov == k)
        ci = np.flatnonzero(c_fov == k)
        if len(ri) and len(ci):
            groups.append((ri, ci))
    return groups or None


def ransac_match(
    ref_xy: np.ndarray,
    c_xy: np.ndarray,
    aligned_c: np.ndarray,
    inlier_tol: float,
    radius: float,
    max_iter: int = 20000,
    ref_fov: np.ndarray | None = None,
    c_fov: np.ndarray | None = None,
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
    hypersensitive to ROI placement.

    Parameters
    ----------
    ref_xy : np.ndarray
        ``(n_ref, 2)`` reference-channel positions in ``[x, y]``.
    c_xy : np.ndarray
        ``(n_c, 2)`` the other channel's positions, in absolute coordinates.
        The transform is fitted on these, not on ``aligned_c``.
    aligned_c : np.ndarray
        ``c_xy`` coarsely overlaid onto the reference frame, used **only** to
        propose candidate pairs. Pass ``c_xy`` itself for an identity overlay.
    inlier_tol : float
        Distance (camera pixels) within which a mapped point counts as an
        inlier, both for the consensus vote and the final assignment.
    radius : float
        Radius (camera pixels) around each reference point in which the
        overlay proposes candidate partners. It only has to be generous enough
        to contain the true partner.
    max_iter : int, optional
        Cap on the number of sampled pair combinations. Above it the samples
        are drawn at random from a fixed seed, so a calibration stays
        reproducible. Default 20000.
    ref_fov, c_fov : np.ndarray, optional
        A group label per point. When both are given, a correspondence may only
        pair points of the **same** group, at every stage: candidate proposal,
        the consensus count and the final assignment. For a multi-FOV bead
        stack this is the field of view: every field images onto the same
        sensor coordinates, so pooling them packs the cloud far denser than any
        one field is, and a reference bead can then sit within ``inlier_tol``
        of an unrelated field's bead and be paired to it. For signal
        registration it is the *frame*: a molecule may only pair with one that
        blinked in the same frame (see :func:`register_from_point_sets`).
        Either way the transform itself stays global - one optical mapping -
        and is still fitted by the caller on the pooled inliers, so every
        group's points constrain it. Without the labels the pairing falls back
        to one pooled cloud.

    Returns
    -------
    ref_idx, c_idx : np.ndarray
        Index arrays of the winning consensus's inlier pairs, into ``ref_xy``
        and ``c_xy``. Both are empty when no consensus is found, or when either
        cloud holds fewer than three points.
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    c_xy = np.asarray(c_xy, dtype=np.float64)
    aligned_c = np.asarray(aligned_c, dtype=np.float64)
    empty = (np.array([], dtype=int), np.array([], dtype=int))
    if min(len(ref_xy), len(c_xy)) < 3:
        return empty

    # per-field index blocks (ref rows, channel rows), or None to pool
    groups = _fov_groups(ref_fov, c_fov, len(ref_xy), len(c_xy))

    # candidate (ref_i, c_j) pairs: c beads near ref_i in the coarse overlay
    if groups is None:
        overlay_tree = KDTree(aligned_c)
        pairs = [
            (i, j)
            for i in range(len(ref_xy))
            for j in overlay_tree.query_ball_point(ref_xy[i], radius)
        ]
    else:
        pairs = []
        for ri, ci in groups:
            overlay_tree = KDTree(aligned_c[ci])
            for i in ri:
                pairs.extend(
                    (int(i), int(ci[j]))
                    for j in overlay_tree.query_ball_point(ref_xy[i], radius)
                )
    if len(pairs) < 2:
        return empty
    pairs = np.asarray(pairs, dtype=int)

    if groups is None:
        c_tree = KDTree(c_xy)

        def consensus(pred: np.ndarray) -> int:
            dist, _ = c_tree.query(pred, k=1)
            return int(np.count_nonzero(dist <= inlier_tol))

    else:
        # one tree per field, so a bead can only find partners in its own
        group_trees = [(ri, KDTree(c_xy[ci])) for ri, ci in groups]

        def consensus(pred: np.ndarray) -> int:
            total = 0
            for ri, tree in group_trees:
                dist, _ = tree.query(pred[ri], k=1)
                total += int(np.count_nonzero(dist <= inlier_tol))
            return total

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
        # the two sampled pairs may come from different fields - that is
        # welcome, the transform is global and a longer baseline pins it down
        # better; only the correspondences themselves stay within a field
        for M in _similarity_from_two(
            ref_xy[i0], ref_xy[i1], c_xy[j0], c_xy[j1]
        ):
            count = consensus(M.apply(ref_xy))
            if count > best_count:
                best_count, best_M = count, M

    if best_M is None:
        return empty
    pred = best_M.apply(ref_xy)
    if groups is None:
        return match_points(pred, c_xy, inlier_tol)
    acc_ref, acc_c = [], []
    for ri, ci in groups:
        a, b = match_points(pred[ri], c_xy[ci], inlier_tol)
        if len(a):
            acc_ref.append(ri[a])
            acc_c.append(ci[b])
    if not acc_ref:
        return empty
    return np.concatenate(acc_ref), np.concatenate(acc_c)


def match_points(
    ref_xy: np.ndarray, other_xy: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour match two point clouds across channels.

    The points are fiducial beads when registering on beads and
    single-molecule detections when registering on signal; the matching is the
    same either way.

    Parameters
    ----------
    ref_xy : np.ndarray
        ``(n_ref, 2)`` reference positions in ``[x, y]``, already mapped into
        the other channel's frame by the current transform estimate.
    other_xy : np.ndarray
        ``(n_other, 2)`` the other channel's own positions.
    max_distance : float
        Pairing radius in camera pixels; a reference point with no partner
        inside it stays unmatched.

    Returns
    -------
    ref_idx, other_idx : np.ndarray
        Index arrays of the matched pairs, into ``ref_xy`` and ``other_xy``.
        Each ``other`` point is used at most once - conflicts are resolved in
        order of increasing distance, so the closest match wins. Both are empty
        if either cloud is.
    """
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


def frames_in_bounds(
    n_frames: int, frame_bounds: tuple[int, int] | list | None
) -> np.ndarray:
    """The frame indices a frame-range setting allows.

    Parameters
    ----------
    n_frames : int
        Total number of frames in the movie.
    frame_bounds : tuple, list or None
        The frames to allow, following :func:`picasso.localize.identify`:
        either a single ``(min, max)`` range or a list of such ranges, both
        inclusive and 0-indexed, with ``None`` for an open end. None allows
        every frame.

    Returns
    -------
    frames : np.ndarray
        Sorted, unique frame indices, clipped to ``[0, n_frames - 1]``. Empty
        when the bounds select nothing.
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


def _pooled(by_frame: dict, frames: list) -> tuple[np.ndarray, np.ndarray]:
    """One ``(xy, frame_label)`` cloud from a per-frame mapping."""
    if not frames:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    xy = np.vstack([by_frame[f] for f in frames])
    labels = np.concatenate(
        [np.full(len(by_frame[f]), f, dtype=int) for f in frames]
    )
    return xy, labels


def _bootstrap_transform(
    ref_by_frame: dict,
    chan_by_frame: dict,
    common: list,
    model: str,
    radius: float,
    inlier_tol: float,
) -> tform.Transform | None:
    """A first reference->channel transform with no prior estimate.

    Pairing normally starts from a seed that is already close. Without one,
    propose correspondences with :func:`ransac_match` on the pooled
    detections, labelling every point with its **frame** so a molecule can only
    pair with one that blinked in the same frame - the constraint that makes
    signal registration work at all, and exactly what the field-of-view
    grouping already implements. The overlay is the identity, which only has to
    be good enough to propose candidates: the winning transform is fitted on
    absolute coordinates, so a channel offset well inside ``radius`` is
    recovered even though the overlay ignores it.

    Returns None when no consensus is found.
    """
    ref_xy, ref_frames = _pooled(ref_by_frame, common)
    c_xy, c_frames = _pooled(chan_by_frame, common)
    ref_idx, c_idx = ransac_match(
        ref_xy,
        c_xy,
        c_xy,  # identity overlay: candidates only, the fit uses absolutes
        inlier_tol,
        radius,
        ref_fov=ref_frames,
        c_fov=c_frames,
    )
    if len(ref_idx) < 3:
        return None
    transform, _ = fit_registration(
        ref_xy[ref_idx], c_xy[c_idx], model, final=False
    )
    return transform


def register_from_point_sets(
    ref_by_frame: dict,
    chan_by_frame: dict,
    model: str,
    box: int,
    seed: tform.Transform | None = None,
    n_iter: int = 4,
    max_pair_distance: float | None = None,
    min_pairs: int = 20,
    bootstrap_radius: float | None = None,
) -> dict:
    """Fit the reference -> channel transform from per-frame point clouds.

    Correspondences are paired at the current transform, a fresh transform is
    fitted on them, and the pairing radius shrinks over ``n_iter`` ICP passes;
    a robust trim then drops the coincidental pairs and refits once on the
    inliers.

    Parameters
    ----------
    ref_by_frame, chan_by_frame : dict
        Frame index to that frame's ``(n, 2)`` ``[x, y]`` detections, for the
        reference channel and the channel being registered. Only frames present
        in both are used: the channels are frame-synchronized, so a molecule
        may only be paired with one that fluoresced in the *same* frame.
    model : str
        Transform model to fit, as in :mod:`picasso.transforms`.
    box : int
        Box side length (camera pixels). Sets the default pairing radii.
    seed : picasso.transforms.Transform, optional
        The transform the first pass pairs at. Pass a stored one to *refine* a
        registration that has drifted. None (the default) builds one from
        scratch, bootstrapping the pairing with a RANSAC consensus.
    n_iter : int, optional
        Number of ICP passes, over which the pairing radius shrinks from
        ``max_pair_distance`` to ``max(2, 0.3 * box)``. Default 4.
    max_pair_distance : float, optional
        Radius (camera pixels) the first pass pairs at. None (the default) uses
        one ``box``, which absorbs a seed's residual drift without inviting
        coincidental cross-molecule pairs.
    min_pairs : int, optional
        Fewest correspondences the result may rest on. Default 20.
    bootstrap_radius : float, optional
        Radius (camera pixels) the seedless bootstrap proposes candidate pairs
        within, so it bounds the inter-channel offset that can be recovered.
        None (the default) uses ``20 * box``. Ignored when ``seed`` is given.

    Returns
    -------
    info : dict
        ``transform`` (the fitted reference -> channel transform),
        ``n_matches`` (correspondences it was fitted on), ``ref_xy`` / ``c_xy``
        (those correspondences), ``rms`` (their residual, camera pixels),
        ``model`` (the model actually fitted) and ``model_requested`` (the one
        asked for; the two differ when too few pairs survived).

    Raises
    ------
    ValueError
        If no frame carries detections in both channels, if the seedless
        bootstrap finds no consistent set of correspondences, or if fewer than
        ``min_pairs`` survive.
    """
    common = sorted(set(ref_by_frame) & set(chan_by_frame))
    if not common:
        raise ValueError(
            "No frames with detections in both the reference and this "
            "channel; the channels may not share signal, or the movies are "
            "not frame-synchronized."
        )
    if max_pair_distance is None:
        # a seed is already close, so about one box absorbs the residual drift
        # without inviting coincidental cross-molecule pairs
        max_pair_distance = float(box)
    tol_hi = float(max_pair_distance)
    tol_lo = max(2.0, 0.3 * box)

    if seed is None:
        if bootstrap_radius is None:
            # nothing is known about the offset, so candidates are proposed
            # over a generous fraction of the frame
            bootstrap_radius = 20.0 * float(box)
        seed = _bootstrap_transform(
            ref_by_frame,
            chan_by_frame,
            common,
            model,
            bootstrap_radius,
            tol_hi,
        )
        if seed is None:
            raise ValueError(
                "Could not find a consistent set of correspondences between "
                "the reference and this channel. The channels may not share "
                "signal, or the offset between them may exceed the search "
                "radius."
            )

    transform = seed
    fitted_model = "affine"
    matched_ref = matched_c = np.empty((0, 2))
    tols = np.linspace(tol_hi, tol_lo, max(1, int(n_iter)))
    for k, tol in enumerate(tols):
        acc_ref, acc_c = [], []
        for f in common:
            rxy = ref_by_frame[f]
            cxy = chan_by_frame[f]
            pred = transform.apply(rxy)
            ri, ci = match_points(pred, cxy, tol)
            if len(ri):
                acc_ref.append(rxy[ri])
                acc_c.append(cxy[ci])
        if not acc_ref:
            break
        matched_ref = np.vstack(acc_ref)
        matched_c = np.vstack(acc_c)
        if len(matched_ref) < 3:
            break
        transform, fitted_model = fit_registration(
            matched_ref, matched_c, model, final=k == len(tols) - 1
        )

    # robust trim: drop coincidental pairs far from the converged transform,
    # then re-fit once on the inliers
    if len(matched_ref) >= 3:
        resid = matched_c - transform.apply(matched_ref)
        dist = np.sqrt(np.sum(resid**2, axis=1))
        keep = dist <= max(tol_lo, 3.0 * np.median(dist))
        if keep.sum() >= 3:
            matched_ref = matched_ref[keep]
            matched_c = matched_c[keep]
            transform, fitted_model = fit_registration(
                matched_ref, matched_c, model
            )
    n_pairs = int(len(matched_ref))
    if n_pairs < min_pairs:
        raise ValueError(
            f"Only {n_pairs} signal correspondences (need >= {min_pairs}); "
            "use a longer / denser movie, lower the minimum net gradient, or "
            "register on beads instead."
        )
    resid = matched_c - transform.apply(matched_ref)
    rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
    return {
        "transform": transform,
        "n_matches": n_pairs,
        "ref_xy": matched_ref,
        "c_xy": matched_c,
        "rms": rms,
        # what was actually fitted, and what was asked for: they differ when
        # too few pairs survived for the chosen model
        "model": fitted_model,
        "model_requested": model,
    }


def detections_by_frame(
    movie,
    minimum_ng: float,
    box: int,
    frames: np.ndarray,
) -> dict:
    """Detect spots on selected frames, grouped by frame.

    Parameters
    ----------
    movie : AbstractPicassoMovie
        The movie to detect in.
    minimum_ng : float
        Minimum net gradient for a spot to be kept.
    box : int
        Box side length (camera pixels) used for the detection.
    frames : np.ndarray
        Indices of the frames to detect on, e.g. from
        :func:`frames_in_bounds`.

    Returns
    -------
    by_frame : dict
        Frame index to that frame's ``(n, 2)`` ``[x, y]`` detections, in
        absolute camera pixels, in the form :func:`register_from_point_sets`
        takes. Frames with no detection are absent, and an empty dict is
        returned when nothing was detected at all. The frames are read into one
        stack and identified in a single pass, so the keys are the *original*
        frame indices rather than positions within that stack.
    """
    stack = np.stack([np.asarray(movie[int(f)]) for f in frames])
    ids, _ = localize.identify(stack, minimum_ng, box)
    if len(ids) == 0:
        return {}
    frame = np.asarray(ids["frame"], dtype=np.int64)
    xy = np.column_stack(
        [
            np.asarray(ids["x"], dtype=np.float64),
            np.asarray(ids["y"], dtype=np.float64),
        ]
    )
    frames = np.asarray(frames)
    return {int(frames[f]): xy[frame == f] for f in np.unique(frame)}


def _minimum_ng_for(minimum_ng: float | list, channel: int) -> float:
    """``minimum_ng`` for one channel, from a scalar or a per-channel list."""
    if isinstance(minimum_ng, (list, tuple, np.ndarray)):
        return float(minimum_ng[channel])
    return float(minimum_ng)


def _registration_calibration(
    transforms: list,
    reference: int,
    model: str,
    box: int,
    minimum_ng: float | list,
    source: str,
    infos: list,
    channel_paths: list[str] | None,
    extra: dict | None = None,
) -> dict:
    """Assemble the calibration dict both builders return.

    ``transforms`` is one entry per channel in channel order, the reference's
    being the identity, stored in the same wire format as a multichannel spline
    calibration's ``channel_transforms`` so every consumer of those works
    unchanged.
    """
    calibration = {
        "model": REGISTRATION_MODEL,
        "n_channels": len(transforms),
        "channel_transforms": [t.to_dict() for t in transforms],
        "registration_model": model,
        "reference": int(reference),
        "source": source,
        "box": int(box),
        "minimum_ng": minimum_ng,
        # per non-reference channel, in channel order
        "n_pairs": [int(i["n_matches"]) for i in infos],
        "rms": [float(i["rms"]) for i in infos],
        "fitted_model": [i["model"] for i in infos],
        "channel_paths": list(channel_paths or []),
        "date": datetime.datetime.now().isoformat(),
        "Generated by": f"Picasso v{__version__} Channel registration",
    }
    if extra:
        calibration.update(extra)
    return calibration


def calibrate_channel_registration_from_beads(
    movies: list,
    box: int,
    minimum_ng: float | list,
    model: str = "affine",
    reference: int = 0,
    channel_paths: list[str] | None = None,
    path: str | None = None,
) -> dict:
    """Register channels from images of fiducial beads.

    Each channel's movie is collapsed to one image, beads are detected and
    refined to sub-pixel accuracy, matched to the reference channel's beads by
    mutual nearest neighbour, and a transform is fitted per channel.

    Parameters
    ----------
    movies : list
        One bead movie per channel, in channel order. Multi-frame movies are
        averaged.
    box : int
        Box size used to detect and fit the beads.
    minimum_ng : float or list
        Minimum net gradient for a bead candidate, shared or per channel.
    model : str, optional
        Transform model, as in :mod:`picasso.transforms`. Default "affine".
    reference : int, optional
        Index of the reference channel. Default 0.
    channel_paths : list of str, optional
        Source paths, recorded in the calibration for traceability.
    path : str, optional
        If given, the calibration is saved there (YAML).

    Returns
    -------
    calibration : dict
        See :func:`_registration_calibration`. The transforms map **reference
        channel coordinates into each channel**, the direction
        ``picasso.localize.get_spots_multichannel`` expects.
    """
    n_channels = len(movies)
    if n_channels < 2:
        raise ValueError(
            f"Channel registration needs at least 2 channels, got {n_channels}."
        )
    if not (0 <= reference < n_channels):
        raise ValueError(f"reference={reference} out of range.")

    img_ref = localize._movie_to_image(movies[reference])
    coarse_ref = localize._lateral_detect_beads(
        img_ref, box, _minimum_ng_for(minimum_ng, reference)
    )
    refined_ref = localize._lateral_refine_bead_positions(
        img_ref, coarse_ref, box
    )

    needed = tform.min_points(model)
    transforms: list = [None] * n_channels
    transforms[reference] = tform.identity(model)
    infos = []
    for c in range(n_channels):
        if c == reference:
            continue
        img_c = localize._movie_to_image(movies[c])
        coarse_c = localize._lateral_detect_beads(
            img_c, box, _minimum_ng_for(minimum_ng, c)
        )
        refined_c = localize._lateral_refine_bead_positions(
            img_c, coarse_c, box
        )
        pairs_ref, pairs_c = localize._lateral_match_bead_pairs(
            refined_ref, refined_c
        )
        if len(pairs_ref) < needed:
            raise ValueError(
                f"Only {len(pairs_ref)} matched bead pair(s) for channel {c} "
                f"- a {model} transform needs at least {needed}. Check the "
                "bead images / detection parameters, or choose a simpler "
                "model."
            )
        # src = reference beads, dst = this channel's: the transform must map
        # the reference INTO the channel, the opposite direction from
        # localize.fit_lateral_transform, which corrects a channel back into
        # the reference frame.
        transform, keep = localize._estimate_lateral_transform(
            pairs_ref, pairs_c, model
        )
        # _estimate_lateral_transform works in [row, col]; the pairs are too.
        resid = pairs_c[keep][:, ::-1] - transform.apply(
            pairs_ref[keep][:, ::-1]
        )
        transforms[c] = transform
        infos.append(
            {
                "channel": c,
                "n_matches": int(keep.sum()),
                "rms": float(np.sqrt(np.mean(np.sum(resid**2, axis=1)))),
                "model": model,
            }
        )

    calibration = _registration_calibration(
        transforms,
        reference,
        model,
        box,
        minimum_ng,
        "beads",
        infos,
        channel_paths,
    )
    if path:
        io.save_any_calibration(path, calibration)
    return calibration


def calibrate_channel_registration_from_signal(
    movies: list,
    box: int,
    minimum_ng: float | list,
    model: str = "affine",
    reference: int = 0,
    frame_bounds: tuple[int, int] | list | None = None,
    max_frames: int = 50,
    seed_transforms: list | None = None,
    n_iter: int = 4,
    min_pairs: int = 20,
    channel_paths: list[str] | None = None,
    path: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """Register channels from the experimental (blinking) signal.

    The channels are frame-synchronized, so the same emitter fluoresces in
    every channel in the same frame. Single molecules are detected on an evenly
    spaced sample of frames and paired frame by frame, which registers the
    channels without a separate bead acquisition.

    Parameters
    ----------
    movies : list
        One movie per channel, in channel order.
    box, minimum_ng : int, float or list
        Detection settings, as used for localization. ``minimum_ng`` may be
        per channel.
    model : str, optional
        Transform model. Default "affine".
    reference : int, optional
        Index of the reference channel. Default 0.
    frame_bounds : optional
        Frames to draw the sample from, as in :func:`localize.identify`. Early
        frames are often too dense to pair unambiguously and late ones too
        sparse, so bounding this matters.
    max_frames : int, optional
        How many frames are evenly sampled from that range. Default 50.
    seed_transforms : list, optional
        One transform per channel to start the pairing from, e.g. an already
        loaded registration being re-aligned. None (the default) builds the
        registration from scratch, bootstrapping the pairing.
    n_iter, min_pairs : int, optional
        ICP passes, and the fewest correspondences a channel may end with.
    channel_paths : list of str, optional
        Source paths, recorded for traceability.
    path : str, optional
        If given, the calibration is saved there (YAML).
    progress_callback : callable, optional
        Called with the number of channels registered so far.

    Returns
    -------
    calibration : dict
        As :func:`calibrate_channel_registration_from_beads`.
    """
    n_channels = len(movies)
    if n_channels < 2:
        raise ValueError(
            f"Channel registration needs at least 2 channels, got {n_channels}."
        )
    if not (0 <= reference < n_channels):
        raise ValueError(f"reference={reference} out of range.")
    if seed_transforms is not None and len(seed_transforms) != n_channels:
        raise ValueError(
            f"Got {len(seed_transforms)} seed transforms but "
            f"{n_channels} channels."
        )

    # sample frames every movie has, so the per-frame pairing stays aligned
    n_frames = min(int(m.shape[0]) for m in movies)
    allowed = frames_in_bounds(n_frames, frame_bounds)
    if allowed.size == 0:
        raise ValueError("No frames in the requested frame range.")
    pick = np.unique(
        np.linspace(
            0, allowed.size - 1, min(int(max_frames), allowed.size)
        ).astype(int)
    )
    sample_frames = allowed[pick]

    ref_by_frame = detections_by_frame(
        movies[reference],
        _minimum_ng_for(minimum_ng, reference),
        box,
        sample_frames,
    )
    if not ref_by_frame:
        raise ValueError(
            "No detections in the reference channel; lower the minimum net "
            "gradient or check the frame range."
        )

    transforms: list = [None] * n_channels
    transforms[reference] = tform.identity(model)
    infos = []
    done = 0
    for c in range(n_channels):
        if c == reference:
            continue
        chan_by_frame = detections_by_frame(
            movies[c], _minimum_ng_for(minimum_ng, c), box, sample_frames
        )
        seed = None
        if seed_transforms is not None:
            entry = seed_transforms[c]
            seed = tform.from_dict(entry) if isinstance(entry, dict) else entry
        try:
            info = register_from_point_sets(
                ref_by_frame,
                chan_by_frame,
                model,
                box,
                seed=seed,
                n_iter=n_iter,
                min_pairs=min_pairs,
            )
        except ValueError as e:
            raise ValueError(f"Channel {c}: {e}") from e
        transforms[c] = info["transform"]
        info["channel"] = c
        infos.append(info)
        done += 1
        if callable(progress_callback):
            progress_callback(done)

    calibration = _registration_calibration(
        transforms,
        reference,
        model,
        box,
        minimum_ng,
        "signal",
        infos,
        channel_paths,
        extra={
            "frame_bounds": frame_bounds,
            "max_frames": int(max_frames),
            "n_sampled_frames": int(len(sample_frames)),
        },
    )
    if path:
        io.save_any_calibration(path, calibration)
    return calibration
