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

#: How far apart matched beads may start out when registering separate bead
#: movies of the same field, which overlap to begin with. Mirrors the radius
#: ``picasso.localize.calibrate_lateral_transform`` has always paired at.
_BEAD_MATCH_RADIUS_PX = 40.0


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


# The four mirror orientations tried when nothing is known about the optical
# path (identity, flip-x, flip-y, flip-xy); (sx, sy) are the mirror signs. A
# splitter that folds one channel about an axis is common, and no amount of
# ICP recovers from starting at the wrong orientation - the pairing has to be
# seeded at each in turn and the winner kept.
_FLIP_SIGNS = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))


def flip_affine(
    sx: float, sy: float, w: float, h: float
) -> tform.AffineTransform:
    """A pure mirror about a ``w`` x ``h`` box, per axis sign.

    Parameters
    ----------
    sx, sy : float
        ``-1`` to mirror that axis, ``+1`` to leave it.
    w, h : float
        Width and height of the box the mirror is taken about.

    Returns
    -------
    transform : picasso.transforms.AffineTransform
    """
    matrix = np.array(
        [
            [sx, 0.0, w if sx < 0 else 0.0],
            [0.0, sy, h if sy < 0 else 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return tform.AffineTransform(matrix=matrix)


def flip_seed_transforms(
    channel: int,
    region_rects: list | None,
    frame_shape: tuple[int, int] | None,
    ref_xy: np.ndarray,
    chan_xy: np.ndarray,
) -> list[tform.AffineTransform]:
    """Coarse reference->channel seed transforms, one per mirror orientation.

    Split-FOV (``region_rects`` given): the mirror is taken about the channel's
    region and the region origins supply the placement. Separate movies
    (``frame_shape`` given): the mirror is taken about the frame and the
    translation comes from aligning the pooled detection centroids.

    Seeds are always affine, whatever the registration is finally fitted with:
    they only have to get the pairing started.

    Parameters
    ----------
    channel : int
        Index of the channel being seeded.
    region_rects : list or None
        Split-FOV region rectangles, reference first. None for separate movies.
    frame_shape : tuple or None
        ``(height, width)`` of the frame, for the separate-movie case.
    ref_xy, chan_xy : np.ndarray
        Pooled detections of the reference and of this channel, used to line up
        the centroids in the separate-movie case.

    Returns
    -------
    seeds : list of picasso.transforms.AffineTransform
        One per mirror orientation, to be tried in turn.
    """
    seeds = []
    identity = tform.identity()
    if region_rects is not None:
        (cy0, cx0), (cy1, cx1) = region_rects[channel]
        h, w = float(cy1 - cy0), float(cx1 - cx0)
        for sx, sy in _FLIP_SIGNS:
            seeds.append(
                localize.compose_region_transforms(
                    [region_rects[0], region_rects[channel]],
                    [identity, flip_affine(sx, sy, w, h)],
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
        seed = flip_affine(sx, sy, w, h)
        pred = seed.apply(ref_xy)
        seeds.append(
            seed.compose_translations(
                post=chan_xy.mean(axis=0) - pred.mean(axis=0)
            )
        )
    return seeds


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


def _icp_from_seed(
    ref_by_frame: dict,
    chan_by_frame: dict,
    common: list,
    seed: tform.Transform,
    model: str,
    tols: np.ndarray,
    tol_lo: float,
) -> tuple:
    """One ICP run from one seed, plus the closing robust trim.

    Returns ``(transform, matched_ref, matched_c, fitted_model)``. Split out of
    :func:`register_from_point_sets` so several candidate seeds - the mirror
    orientations - can be run and compared."""
    transform = seed
    fitted_model = "affine"
    matched_ref = matched_c = np.empty((0, 2))
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
    return transform, matched_ref, matched_c, fitted_model


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
    seed : picasso.transforms.Transform or list, optional
        The transform the first pass pairs at. Pass a stored one to *refine* a
        registration that has drifted. None (the default) builds one from
        scratch, bootstrapping the pairing with a RANSAC consensus. A **list**
        of candidate seeds - e.g. the mirror orientations from
        :func:`flip_seed_transforms` - runs the pairing from each and keeps the
        one that ends with the most correspondences and a geometrically sane
        transform, which is how a mirrored channel is recovered.
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
    seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed]

    tols = np.linspace(tol_hi, tol_lo, max(1, int(n_iter)))
    with warnings.catch_warnings():
        if len(seeds) > 1:
            # Only one candidate is kept, so a losing orientation's "thin data"
            # warning would be noise about a registration nobody sees. The
            # winner is refitted below, outside this block, so a genuine
            # warning about the *kept* transform still reaches the user.
            warnings.simplefilter("ignore")
        best = None
        for candidate in seeds:
            result = _icp_from_seed(
                ref_by_frame,
                chan_by_frame,
                common,
                candidate,
                model,
                tols,
                tol_lo,
            )
            # A wrong mirror orientation converges onto coincidental pairs,
            # which are few at the tightest radius and usually imply an absurd
            # scale - so the most pairs wins, among the geometrically sane.
            if not tform.is_plausible(result[0]):
                continue
            if best is None or len(result[1]) > len(best[1]):
                best = result
        if best is None:
            # every orientation folded over; keep the first so the pair-count
            # check below reports the real problem rather than a bare
            # implausibility
            best = _icp_from_seed(
                ref_by_frame,
                chan_by_frame,
                common,
                seeds[0],
                model,
                tols,
                tol_lo,
            )
    transform, matched_ref, matched_c, fitted_model = best
    if len(seeds) > 1 and len(matched_ref) >= 3:
        # refit the winner audibly, so a thin-data warning is raised for the
        # registration that is actually kept
        transform, fitted_model = fit_registration(
            matched_ref, matched_c, model
        )
    n_pairs = int(len(matched_ref))
    if n_pairs < min_pairs:
        # Neutral wording: this is shared by the bead and the signal builders,
        # each of which appends the advice that applies to its own input.
        raise ValueError(
            f"Only {n_pairs} correspondences survived (need >= {min_pairs})."
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


def _region_mask(xy: np.ndarray, region: list) -> np.ndarray:
    """Which rows of ``xy`` (in ``[x, y]``) fall inside a
    ``[[y_min, x_min], [y_max, x_max]]`` rectangle."""
    (y0, x0), (y1, x1) = localize._normalize_rect(region)
    xy = np.asarray(xy, dtype=np.float64)
    if not len(xy):
        return np.zeros(0, dtype=bool)
    return (
        (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
    )


def _by_frame_in_region(by_frame: dict, region: list) -> dict:
    """Per-frame detections restricted to one region, frames with none
    dropped."""
    out = {}
    for f, xy in by_frame.items():
        kept = np.asarray(xy)[_region_mask(xy, region)]
        if len(kept):
            out[f] = kept
    return out


def _split_fov_flip_seeds(regions: list, reference: int, channel: int) -> list:
    """Candidate reference->channel seeds for a split-FOV channel, one per
    mirror orientation.

    The drawn ROIs fix where the channel sits, so no search over *placement* is
    needed - but not how it is oriented, and a splitter that folds one channel
    about an axis is common. :func:`flip_seed_transforms` reads the reference as
    ``region_rects[0]``, so the pair is handed over reference-first.
    """
    rects = [
        localize._normalize_rect(regions[reference]),
        localize._normalize_rect(regions[channel]),
    ]
    return flip_seed_transforms(
        1, rects, None, np.empty((0, 2)), np.empty((0, 2))
    )


def _split_fov_calibration_keys(
    regions: list, reference: int, transforms: list
) -> dict:
    """The split-FOV half of a registration calibration.

    The inter-channel registration is stored *region-local* (relative to the
    region origins), so the channels can be re-placed at fit time by re-drawing
    the ROIs; the absolute ``channel_transforms`` are rebuilt from these plus
    the ROIs actually in use. Same layout as a split-FOV spline calibration, so
    :func:`picasso.localize.split_fov_fit_geometry` reads either.
    """
    rects = [localize._normalize_rect(r) for r in regions]
    return {
        "split_fov": True,
        "reference": int(reference),
        "regions": rects,
        "channel_registration": [
            a.to_dict()
            for a in localize.decompose_region_transforms(rects, transforms)
        ],
    }


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
    regions: list | None = None,
    multi_fov: bool = False,
    min_pairs: int | None = None,
    channel_paths: list[str] | None = None,
    path: str | None = None,
) -> dict:
    """Register channels from images of fiducial beads.

    Beads are detected and refined to sub-pixel accuracy, matched to the
    reference channel's beads, and a transform is fitted per channel.

    Parameters
    ----------
    movies : list
        One bead movie per channel, in channel order. For split field of view
        (``regions`` given) the single bead movie whose regions are the
        channels. Multi-frame movies are averaged unless ``multi_fov``.
    box : int
        Box size used to detect and fit the beads.
    minimum_ng : float or list
        Minimum net gradient for a bead candidate, shared or per channel.
    model : str, optional
        Transform model, as in :mod:`picasso.transforms`. Default "affine".
    reference : int, optional
        Index of the reference channel. Default 0.
    regions : list, optional
        Split field of view: one ``[[y_min, x_min], [y_max, x_max]]`` rectangle
        per channel, reference first, marking where each channel sits on the
        single sensor. The beads are detected once and split by region, and
        every mirror orientation is tried so a folded channel is recovered.
        Default None (separate bead movies per channel).
    multi_fov : bool, optional
        Each **frame** of the bead movie images a different field of view. The
        beads are then detected frame by frame and a bead is only ever paired
        with one in the *same* frame - different fields land on the same sensor
        coordinates, so pooling them would pair beads that are nowhere near
        each other - while every field's pairs constrain the one global
        transform. Default False: the frames are repeats of a single field and
        are averaged into one image, which is what a plain bead acquisition is.
    min_pairs : int, optional
        Fewest correspondences a channel may end with. None (the default) uses
        the minimum the transform model needs.
    channel_paths : list of str, optional
        Source paths, recorded in the calibration for traceability.
    path : str, optional
        If given, the calibration is saved there (YAML).

    Returns
    -------
    calibration : dict
        See :func:`_registration_calibration`. The transforms map **reference
        channel coordinates into each channel**, the direction
        ``picasso.localize.get_spots_multichannel`` expects. A split-FOV
        registration additionally carries ``split_fov``, ``regions`` and the
        ROI-agnostic ``channel_registration``, so it can be re-placed at
        re-drawn ROIs.
    """
    split_fov = regions is not None
    n_channels = len(regions) if split_fov else len(movies)
    if n_channels < 2:
        raise ValueError(
            f"Channel registration needs at least 2 channels, got {n_channels}."
        )
    if split_fov and len(movies) != 1:
        raise ValueError(
            "Split-FOV registration takes the single bead movie whose regions "
            f"are the channels, got {len(movies)} movies."
        )
    if not (0 <= reference < n_channels):
        raise ValueError(f"reference={reference} out of range.")

    needed = tform.min_points(model)
    if min_pairs is None:
        min_pairs = needed

    def beads_by_frame(index: int) -> dict:
        """Sub-pixel bead positions of one channel, keyed by field of view.

        ``[x, y]``, in the per-frame form :func:`register_from_point_sets`
        takes. With ``multi_fov`` each frame is its own field and keeps its own
        key, so a bead can only ever be paired within it; otherwise the frames
        are averaged into one image under a single key."""
        movie = movies[0] if split_fov else movies[index]
        mng = _minimum_ng_for(minimum_ng, index)
        frames = (
            range(int(np.asarray(movie).shape[0])) if multi_fov else [None]
        )
        out = {}
        for f in frames:
            image = (
                np.asarray(np.asarray(movie)[f], dtype=np.float32)
                if multi_fov
                else localize._movie_to_image(movie)
            )
            coarse = localize._lateral_detect_beads(image, box, mng)
            refined = localize._lateral_refine_bead_positions(
                image, coarse, box
            )
            if split_fov and len(refined):
                # one image holds every channel, so keep this region's beads
                refined = refined[
                    _region_mask(refined[:, ::-1], regions[index])
                ]
            if len(refined):
                # the matching machinery works in [x, y]
                out[0 if f is None else int(f)] = refined[:, ::-1]
        return out

    ref_by_frame = beads_by_frame(reference)
    if not ref_by_frame:
        raise ValueError(
            "No beads detected in the reference channel; lower the minimum "
            "net gradient or check the bead image."
        )

    transforms: list = [None] * n_channels
    transforms[reference] = tform.identity(model)
    infos = []
    for c in range(n_channels):
        if c == reference:
            continue
        chan_by_frame = beads_by_frame(c)
        if split_fov:
            # The ROIs fix where the channel sits but not how it is oriented,
            # so every mirror orientation is tried and the best kept.
            seed, radius = _split_fov_flip_seeds(regions, reference, c), None
        else:
            # Separate bead movies of the same field: the channels overlap to
            # begin with, so the pairing starts from the identity and only has
            # to close the residual misalignment. This is the assumption the
            # single-pair bead calibration has always made - a mirrored or
            # far-displaced channel needs the split-FOV form or a signal
            # registration, which search for the orientation.
            seed, radius = tform.identity(model), _BEAD_MATCH_RADIUS_PX
        try:
            info = register_from_point_sets(
                ref_by_frame,
                chan_by_frame,
                model,
                box,
                seed=seed,
                max_pair_distance=radius,
                min_pairs=min_pairs,
            )
        except ValueError as e:
            raise ValueError(
                f"Channel {c}: {e} Too few matched bead pairs - check the "
                "bead images and the detection parameters, or choose a "
                "simpler transform model."
            ) from e
        transforms[c] = info["transform"]
        info["channel"] = c
        infos.append(info)

    calibration = _registration_calibration(
        transforms,
        reference,
        model,
        box,
        minimum_ng,
        "beads",
        infos,
        channel_paths,
        extra=(
            _split_fov_calibration_keys(regions, reference, transforms)
            if split_fov
            else None
        ),
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
    regions: list | None = None,
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
        One movie per channel, in channel order. For split field of view
        (``regions`` given) the single movie whose regions are the channels.
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
    regions : list, optional
        Split field of view: one ``[[y_min, x_min], [y_max, x_max]]`` rectangle
        per channel, reference first, marking where each channel sits on the
        single sensor. The movie is detected once and the detections split by
        region. The drawn regions also seed the pairing, so no search is
        needed to get started. Default None (separate movies per channel).
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
    split_fov = regions is not None
    n_channels = len(regions) if split_fov else len(movies)
    if n_channels < 2:
        raise ValueError(
            f"Channel registration needs at least 2 channels, got {n_channels}."
        )
    if split_fov and len(movies) != 1:
        raise ValueError(
            "Split-FOV registration takes the single movie whose regions are "
            f"the channels, got {len(movies)} movies."
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

    if split_fov:
        # One movie holds every channel, so it is detected once and the
        # detections are split by region afterwards.
        movie_by_frame = detections_by_frame(
            movies[0],
            _minimum_ng_for(minimum_ng, reference),
            box,
            sample_frames,
        )
        ref_by_frame = _by_frame_in_region(movie_by_frame, regions[reference])
    else:
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
        if split_fov:
            chan_by_frame = _by_frame_in_region(movie_by_frame, regions[c])
        else:
            chan_by_frame = detections_by_frame(
                movies[c], _minimum_ng_for(minimum_ng, c), box, sample_frames
            )
        seed = None
        if seed_transforms is not None:
            entry = seed_transforms[c]
            seed = tform.from_dict(entry) if isinstance(entry, dict) else entry
        elif split_fov:
            # The drawn ROIs say where the channel sits, but not how it is
            # oriented: a splitter commonly folds one channel about an axis.
            # Seed at every mirror orientation and let the best one win.
            seed = _split_fov_flip_seeds(regions, reference, c)
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
            raise ValueError(
                f"Channel {c}: {e} Use a longer / denser movie, lower the "
                "minimum net gradient, or register on beads instead."
            ) from e
        transforms[c] = info["transform"]
        info["channel"] = c
        infos.append(info)
        done += 1
        if callable(progress_callback):
            progress_callback(done)

    extra = {
        "frame_bounds": frame_bounds,
        "max_frames": int(max_frames),
        "n_sampled_frames": int(len(sample_frames)),
    }
    if split_fov:
        extra.update(
            _split_fov_calibration_keys(regions, reference, transforms)
        )
    calibration = _registration_calibration(
        transforms,
        reference,
        model,
        box,
        minimum_ng,
        "signal",
        infos,
        channel_paths,
        extra=extra,
    )
    if path:
        io.save_any_calibration(path, calibration)
    return calibration
