"""Test ``picasso.localize`` — spot identification, extraction, and the
high-level ``fit``/``fit_async`` MLE wrapper, plus the diagnostic
helpers.

Tests for ``gausslq``, ``gaussmle`` and ``zfit`` live in their own files
(``test_gausslq.py``, ``test_gaussmle.py``, ``test_zfit.py``).

:author: Rafal Kowalewski, 2025-2026
:copyright: Copyright (c) 2025-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import sys
import time

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline
from PyQt6 import QtWidgets

from picasso import gaussmle, gausslq, io, localize
from picasso.gui import localize as localize_gui

from tests.conftest import BOX, CALIB_3D, CAMERA_INFO, MIN_NG, PIXELSIZE

CAMERA_INFO_WITH_PIXELSIZE = {**CAMERA_INFO, "Pixelsize": PIXELSIZE}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gradient_meshgrid(box: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the normalised (uy, ux) direction vectors that
    ``identify_in_image`` constructs internally — needed to drive
    ``net_gradient`` directly. The center pixel is unused by
    ``net_gradient`` (it is skipped inside the loop) but its norm is 0,
    so we patch it to 1.0 to avoid emitting a divide-by-zero warning."""
    box_half = box // 2
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux**2 + uy**2)
    unorm[box_half, box_half] = 1.0
    ux /= unorm
    uy /= unorm
    return uy, ux


def _gaussian_frame(
    shape: tuple[int, int],
    center: tuple[int, int],
    sigma: float = 1.2,
    amplitude: float = 5000.0,
    background: float = 100.0,
) -> np.ndarray:
    """Build a single-Gaussian-peak frame on a flat background. Returned
    as float32 so it can be passed straight into the numba-jitted
    helpers without numba complaining about the dtype."""
    Y, X = shape
    cy, cx = center
    yy, xx = np.indices((Y, X), dtype=np.float32)
    g = amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    return (g + background).astype(np.float32)


# ---------------------------------------------------------------------------
# _local_maxima
# ---------------------------------------------------------------------------


class TestLocalMaxima:
    """Pure local-maxima search inside a sliding box."""

    def test_single_peak_detected(self):
        frame = np.zeros((20, 20), dtype=np.float32)
        frame[10, 12] = 100.0
        y, x = localize._local_maxima(frame, BOX)
        assert list(zip(y.tolist(), x.tolist())) == [(10, 12)]

    def test_multiple_peaks_far_apart_all_found(self):
        frame = np.zeros((30, 30), dtype=np.float32)
        peaks = [(8, 8), (8, 22), (22, 15)]
        for py, px in peaks:
            frame[py, px] = 50.0
        y, x = localize._local_maxima(frame, BOX)
        found = set(zip(y.tolist(), x.tolist()))
        assert found == set(peaks)

    def test_peaks_in_border_band_are_excluded(self):
        """``_local_maxima`` only scans i in [box_half, Y - box_half - 1)
        — peaks placed inside the border band must not be returned."""
        Y = X = 20
        box_half = BOX // 2
        frame = np.zeros((Y, X), dtype=np.float32)
        frame[1, 1] = 100.0  # top-left border
        frame[Y - 2, X - 2] = 100.0  # bottom-right border
        y, x = localize._local_maxima(frame, BOX)
        # All returned coordinates lie strictly inside the scan band
        assert ((y >= box_half) & (y < Y - box_half - 1)).all()
        assert ((x >= box_half) & (x < X - box_half - 1)).all()

    def test_flat_frame_returns_no_maxima(self):
        """A constant frame has no unique local max — the implementation's
        ``argmax`` returns the top-left pixel (index 0), so no local
        window has its max at the center."""
        frame = np.full((20, 20), 42.0, dtype=np.float32)
        y, x = localize._local_maxima(frame, BOX)
        assert len(y) == 0 and len(x) == 0


# ---------------------------------------------------------------------------
# _gradient_at
# ---------------------------------------------------------------------------


class TestGradientAt:
    """Two-point centered finite difference at (y, x)."""

    def test_horizontal_gradient(self):
        # frame[y, x+1] - frame[y, x-1]  along increasing x
        frame = np.tile(np.arange(10, dtype=np.float32), (10, 1))
        gy, gx = localize._gradient_at(frame, 5, 5, 0)
        assert gy == 0.0
        assert gx == 2.0  # 6 - 4

    def test_vertical_gradient(self):
        # frame[y+1, x] - frame[y-1, x]  along increasing y
        frame = np.tile(
            np.arange(10, dtype=np.float32).reshape(-1, 1), (1, 10)
        )
        gy, gx = localize._gradient_at(frame, 5, 5, 0)
        assert gy == 2.0  # 6 - 4
        assert gx == 0.0

    def test_zero_gradient_in_flat_region(self):
        frame = np.full((10, 10), 7.0, dtype=np.float32)
        gy, gx = localize._gradient_at(frame, 5, 5, 0)
        assert gy == 0.0 and gx == 0.0

    def test_i_argument_is_ignored(self):
        """``i`` is documented as unused — different values must not
        affect the returned gradient."""
        frame = _gaussian_frame((15, 15), (7, 7))
        a = localize._gradient_at(frame, 7, 8, 0)
        b = localize._gradient_at(frame, 7, 8, 999)
        assert a == b


# ---------------------------------------------------------------------------
# net_gradient
# ---------------------------------------------------------------------------


class TestNetGradient:
    """Inner-product of the local gradient field with the radial
    direction vectors — peaks point outward, so the dot product is large
    and positive at a true peak."""

    def test_gaussian_peak_has_positive_net_gradient(self):
        frame = _gaussian_frame((15, 15), (7, 7))
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([7], dtype=np.int64)
        x = np.array([7], dtype=np.int64)
        ng = localize._net_gradient(frame, y, x, BOX, uy, ux)
        assert ng.shape == (1,)
        assert ng[0] > 0

    def test_flat_frame_yields_zero(self):
        frame = np.full((15, 15), 50.0, dtype=np.float32)
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([7], dtype=np.int64)
        x = np.array([7], dtype=np.int64)
        ng = localize._net_gradient(frame, y, x, BOX, uy, ux)
        np.testing.assert_allclose(ng, [0.0], atol=1e-6)

    def test_inverted_peak_yields_negative(self):
        """A dip (gradients pointing inward) gives a negative net
        gradient — the sign is the discriminator between peaks and
        troughs."""
        frame = -_gaussian_frame((15, 15), (7, 7), background=0.0)
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([7], dtype=np.int64)
        x = np.array([7], dtype=np.int64)
        ng = localize._net_gradient(frame, y, x, BOX, uy, ux)
        assert ng[0] < 0

    def test_output_length_matches_input(self):
        frame = _gaussian_frame((30, 30), (10, 10))
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([10, 10, 10], dtype=np.int64)
        x = np.array([8, 10, 12], dtype=np.int64)
        ng = localize._net_gradient(frame, y, x, BOX, uy, ux)
        assert ng.shape == (3,)


# ---------------------------------------------------------------------------
# identify_in_image
# ---------------------------------------------------------------------------


class TestIdentifyInImage:
    """``_local_maxima`` + net-gradient threshold, in one shot."""

    def test_single_gaussian_is_identified(self):
        frame = _gaussian_frame((20, 20), (10, 10), amplitude=5000.0)
        y, x, ng = localize.identify_in_image(frame, 1.0, BOX)
        # One detection at the seeded peak
        assert len(y) == 1 == len(x) == len(ng)
        assert y[0] == 10 and x[0] == 10
        assert ng[0] > 1.0

    def test_high_threshold_rejects_all(self):
        frame = _gaussian_frame((20, 20), (10, 10), amplitude=5000.0)
        y, x, ng = localize.identify_in_image(frame, 1e12, BOX)
        assert len(y) == 0 and len(x) == 0 and len(ng) == 0

    def test_arrays_have_consistent_length(self):
        frame = _gaussian_frame((30, 30), (10, 10))
        # Add a second well-separated peak
        frame2 = _gaussian_frame((30, 30), (20, 22))
        combined = np.maximum(frame, frame2)
        y, x, ng = localize.identify_in_image(combined, 1.0, BOX)
        assert len(y) == len(x) == len(ng)
        assert len(y) >= 2

    def test_flat_frame_returns_empty(self):
        frame = np.full((20, 20), 100.0, dtype=np.float32)
        y, x, ng = localize.identify_in_image(frame, 0.0, BOX)
        assert len(y) == 0


# ---------------------------------------------------------------------------
# identify_in_frame
# ---------------------------------------------------------------------------


class TestIdentifyInFrame:
    """Wrapper that casts to float32 and applies an ROI offset."""

    def test_no_roi_matches_identify_in_image(self):
        frame = _gaussian_frame((20, 20), (10, 10)).astype(np.int32)
        y_a, x_a, ng_a = localize.identify_in_frame(frame, 1.0, BOX)
        y_b, x_b, ng_b = localize.identify_in_image(
            np.float32(frame), 1.0, BOX
        )
        np.testing.assert_array_equal(y_a, y_b)
        np.testing.assert_array_equal(x_a, x_b)
        np.testing.assert_allclose(ng_a, ng_b)

    def test_roi_offsets_coordinates_back_to_global(self):
        """When ROI = ((y0, x0), (y1, x1)) is supplied, returned (y, x)
        are in the *original* frame's coordinate system, not the ROI's."""
        # Peak at global (15, 17); ROI starts at (10, 12)
        frame = _gaussian_frame((30, 30), (15, 17)).astype(np.int32)
        roi = ((10, 12), (25, 28))
        y, x, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=roi)
        assert len(y) == 1
        assert (int(y[0]), int(x[0])) == (15, 17)

    def test_roi_excludes_peaks_outside(self):
        """A peak outside the ROI window is not seen at all."""
        frame = _gaussian_frame((30, 30), (5, 5)).astype(np.int32)
        roi = ((15, 15), (28, 28))
        y, x, ng = localize.identify_in_frame(frame, 1.0, BOX, roi=roi)
        assert len(y) == 0 and len(x) == 0 and len(ng) == 0

    def test_multiple_rois_find_all_peaks(self):
        """A list of disjoint ROIs collects peaks from every region and
        ignores peaks outside all of them."""
        frame = _gaussian_frame((40, 40), (8, 8)).astype(np.int32)
        frame += (_gaussian_frame((40, 40), (30, 30)) - 100).astype(np.int32)
        frame += (_gaussian_frame((40, 40), (8, 30)) - 100).astype(np.int32)
        rois = [((0, 0), (16, 16)), ((24, 24), (38, 38))]
        y, x, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=rois)
        found = {(int(yi), int(xi)) for yi, xi in zip(y, x)}
        assert (8, 8) in found  # first ROI
        assert (30, 30) in found  # second ROI
        assert (8, 30) not in found  # outside both ROIs

    def test_multiple_rois_no_double_counting(self):
        """Disjoint ROIs never report the same peak twice."""
        frame = _gaussian_frame((40, 40), (8, 8)).astype(np.int32)
        frame += (_gaussian_frame((40, 40), (30, 30)) - 100).astype(np.int32)
        rois = [((0, 0), (16, 16)), ((24, 24), (38, 38))]
        y, x, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=rois)
        coords = list(zip([int(v) for v in y], [int(v) for v in x]))
        assert len(coords) == len(set(coords))

    def test_peak_near_roi_border_is_found(self):
        """A peak close to the ROI border is detected: the slice is
        padded internally so the gradient box still sees real pixels."""
        # Peak two pixels inside the bottom-right corner of the ROI.
        frame = _gaussian_frame((40, 40), (18, 18)).astype(np.int32)
        roi = ((5, 5), (20, 20))
        y, x, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=roi)
        found = {(int(yi), int(xi)) for yi, xi in zip(y, x)}
        assert (18, 18) in found

    def test_adjacent_rois_have_no_seam_gap(self):
        """Splitting a region into two ROIs that share an edge must find
        the same peaks as the single, undivided region - no gap of
        ~``box`` pixels along the seam (regression test)."""
        # Two peaks straddling the y = 20 seam, each only two pixels away
        # from it (i.e. within ``box`` pixels) and far apart in x so they
        # do not suppress one another.
        centers = [(18, 12), (22, 28)]
        frame = np.full((40, 40), 100, dtype=np.int32)
        for cy, cx in centers:
            frame += (_gaussian_frame((40, 40), (cy, cx)) - 100).astype(
                np.int32
            )
        whole = ((8, 8), (32, 32))
        split = [((8, 8), (20, 32)), ((20, 8), (32, 32))]
        y_w, x_w, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=whole)
        y_s, x_s, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=split)
        found_whole = {(int(a), int(b)) for a, b in zip(y_w, x_w)}
        found_split = {(int(a), int(b)) for a, b in zip(y_s, x_s)}
        # every seeded peak is found in both cases ...
        for c in centers:
            assert c in found_whole
            assert c in found_split
        # ... and the two ROIs together reproduce the single-region result
        assert found_whole == found_split


# ---------------------------------------------------------------------------
# clip_rois
# ---------------------------------------------------------------------------


def _area(rects) -> int:
    """Total area of a list of [[y0, x0], [y1, x1]] rectangles."""
    return sum((y1 - y0) * (x1 - x0) for (y0, x0), (y1, x1) in rects)


def _overlap(a, b) -> int:
    """Area of the overlap between two [[y0, x0], [y1, x1]] rectangles."""
    (ay0, ax0), (ay1, ax1) = a
    (by0, bx0), (by1, bx1) = b
    dy = max(0, min(ay1, by1) - max(ay0, by0))
    dx = max(0, min(ax1, bx1) - max(ax0, bx0))
    return dy * dx


class TestClipRois:
    """Geometric clipping of (possibly overlapping) ROIs into disjoint
    rectangles."""

    def test_disjoint_unchanged(self):
        rois = [((0, 0), (5, 5)), ((10, 10), (15, 15))]
        out = localize.clip_rois(rois)
        assert out == [[[0, 0], [5, 5]], [[10, 10], [15, 15]]]

    def test_overlap_is_disjoint_and_preserves_union(self):
        rois = [((0, 0), (10, 10)), ((5, 5), (15, 15))]
        out = localize.clip_rois(rois)
        # pairwise disjoint
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                assert _overlap(out[i], out[j]) == 0
        # union area = 100 + 100 - 25 (overlap)
        assert _area(out) == 175

    def test_full_containment_drops_inner(self):
        rois = [((0, 0), (20, 20)), ((5, 5), (10, 10))]
        out = localize.clip_rois(rois)
        assert out == [[[0, 0], [20, 20]]]

    def test_corner_overlap(self):
        rois = [((0, 0), (10, 10)), ((8, 8), (18, 18))]
        out = localize.clip_rois(rois)
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                assert _overlap(out[i], out[j]) == 0
        assert _area(out) == 100 + 100 - 4

    def test_min_size_drops_slivers(self):
        # second ROI overlaps the first leaving a 1-pixel-tall sliver
        rois = [((0, 0), (10, 10)), ((9, 0), (20, 20))]
        out = localize.clip_rois(rois, min_size=3)
        assert [[10, 0], [20, 20]] in out
        # the 1-pixel band (y in 9..10) is discarded
        assert all(piece[1][0] - piece[0][0] >= 3 for piece in out)

    def test_normalizes_corner_order(self):
        out = localize.clip_rois([((25, 28), (10, 12))])
        assert out == [[[10, 12], [25, 28]]]


# ---------------------------------------------------------------------------
# _to_photons
# ---------------------------------------------------------------------------


class TestToPhotons:
    """Camera-signal -> photon-count conversion: (s - baseline) * sens / gain."""

    def test_identity_camera_returns_input(self):
        spots = np.arange(2 * BOX * BOX, dtype=np.float32).reshape(2, BOX, BOX)
        out = localize._to_photons(spots, CAMERA_INFO)
        np.testing.assert_allclose(out, spots)

    def test_baseline_subtracts(self):
        spots = np.full((2, BOX, BOX), 500.0, dtype=np.float32)
        cam = {"Baseline": 100, "Sensitivity": 1, "Gain": 1}
        out = localize._to_photons(spots, cam)
        np.testing.assert_allclose(out, 400.0)

    def test_sensitivity_multiplies(self):
        spots = np.full((2, BOX, BOX), 50.0, dtype=np.float32)
        cam = {"Baseline": 0, "Sensitivity": 3, "Gain": 1}
        out = localize._to_photons(spots, cam)
        np.testing.assert_allclose(out, 150.0)

    def test_gain_divides(self):
        spots = np.full((2, BOX, BOX), 60.0, dtype=np.float32)
        cam = {"Baseline": 0, "Sensitivity": 1, "Gain": 3}
        out = localize._to_photons(spots, cam)
        np.testing.assert_allclose(out, 20.0)

    def test_combined_transform(self):
        spots = np.full((1, BOX, BOX), 1000.0, dtype=np.float32)
        cam = {"Baseline": 100, "Sensitivity": 2, "Gain": 4}
        out = localize._to_photons(spots, cam)
        # (1000 - 100) * 2 / 4 = 450
        np.testing.assert_allclose(out, 450.0)

    def test_output_is_float32(self):
        spots = np.ones((1, BOX, BOX), dtype=np.uint16) * 100
        out = localize._to_photons(spots, CAMERA_INFO)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# identify
# ---------------------------------------------------------------------------


class TestIdentify:
    """Spot identification on the bundled .raw movie."""

    def test_required_columns_and_finite(self, real_identifications, movie):
        ids = real_identifications
        assert not ids.empty
        for col in ["frame", "x", "y", "net_gradient"]:
            assert col in ids.columns
        assert (ids["net_gradient"] >= MIN_NG).all()
        assert ids["frame"].min() >= 0
        assert ids["frame"].max() < len(movie)

    def test_x_y_inside_movie_bounds(self, real_identifications, movie):
        _, height, width = movie.shape
        ids = real_identifications
        assert (ids["x"] >= 0).all() and (ids["x"] < width).all()
        assert (ids["y"] >= 0).all() and (ids["y"] < height).all()

    def test_roi_is_strict_subset(self, movie, real_identifications):
        """ROI restricts identifications to that pixel window only."""
        roi = ((0, 0), (16, 16))  # ((y_start, x_start), (y_end, x_end))
        ids_roi = localize.identify(
            movie, MIN_NG, BOX, roi=roi, return_info=False
        )
        if len(ids_roi):
            assert (ids_roi["x"] < 16).all()
            assert (ids_roi["y"] < 16).all()
        # subset relationship — ROI cannot find more spots than full image
        assert len(ids_roi) <= len(real_identifications)

    def test_threaded_matches_serial_on_record_set(self, movie):
        """The (frame, y, x) sets identified threaded vs. serial must
        match exactly (order-independent)."""
        ids_t = localize.identify(
            movie, MIN_NG, BOX, threaded=True, return_info=False
        )
        ids_s = localize.identify(
            movie, MIN_NG, BOX, threaded=False, return_info=False
        )
        # Compare as set of (frame, y, x) tuples — same spots, possibly
        # different row order
        set_t = set(zip(ids_t["frame"], ids_t["y"], ids_t["x"]))
        set_s = set(zip(ids_s["frame"], ids_s["y"], ids_s["x"]))
        assert set_t == set_s

    def test_frame_bounds_excludes_outside(self, movie):
        """Setting ``frame_bounds`` confines identifications to that
        range of frame indices."""
        ids = localize.identify(
            movie, MIN_NG, BOX, frame_bounds=(20, 50), return_info=False
        )
        if len(ids):
            assert (ids["frame"] >= 20).all()
            assert (ids["frame"] <= 50).all()

    def test_frame_bounds_multiple_segments(self, movie):
        """A list of ``(min, max)`` segments confines identifications to
        the union of those (disjoint) frame ranges."""
        segments = [(10, 20), (40, 50)]
        ids = localize.identify(
            movie, MIN_NG, BOX, frame_bounds=segments, return_info=False
        )
        if len(ids):
            in_any = ((ids["frame"] >= 10) & (ids["frame"] <= 20)) | (
                (ids["frame"] >= 40) & (ids["frame"] <= 50)
            )
            assert in_any.all()
            # nothing in the gap between segments
            assert not ((ids["frame"] > 20) & (ids["frame"] < 40)).any()

    def test_frame_bounds_single_segment_matches_flat_tuple(self, movie):
        """A single-segment list behaves identically to the flat
        ``(min, max)`` tuple form."""
        flat = localize.identify(
            movie, MIN_NG, BOX, frame_bounds=(20, 50), return_info=False
        )
        listed = localize.identify(
            movie, MIN_NG, BOX, frame_bounds=[(20, 50)], return_info=False
        )
        flat_set = set(zip(flat["frame"], flat["y"], flat["x"]))
        listed_set = set(zip(listed["frame"], listed["y"], listed["x"]))
        assert flat_set == listed_set

    def test_return_info_returns_metadata_dict(self, movie):
        ids, info = localize.identify(
            movie, MIN_NG, BOX, return_info=True, threaded=False
        )
        assert isinstance(info, dict)
        for key in [
            "Generated by",
            "Min. Net Gradient",
            "Box Size",
            "ROI",
            "Frame Bounds",
        ]:
            assert key in info
        assert info["Min. Net Gradient"] == MIN_NG
        assert info["Box Size"] == BOX
        # ids itself is still a DataFrame
        assert isinstance(ids, pd.DataFrame)


class TestIdentifyAsync:
    """The thread-pool identification path."""

    @pytest.mark.slow
    def test_async_finishes_and_matches_serial(self, movie):
        current, fs = localize.identify_async(movie, MIN_NG, BOX)
        n_frames = len(movie)
        # Wait for completion
        t0 = time.time()
        while current[0] < n_frames:
            assert time.time() - t0 < 30, "identify_async timed out"
            time.sleep(0.05)
        ids_async = localize.identifications_from_futures(fs)
        ids_serial = localize.identify(
            movie, MIN_NG, BOX, threaded=False, return_info=False
        )
        set_a = set(zip(ids_async["frame"], ids_async["y"], ids_async["x"]))
        set_s = set(zip(ids_serial["frame"], ids_serial["y"], ids_serial["x"]))
        assert set_a == set_s


class TestIdentifyByFrameNumber:
    """Per-frame identification helper."""

    def test_subset_of_full_identify(self, movie, real_identifications):
        """Per-frame call returns the rows of ``identify`` for that
        frame, no more, no less."""
        for frame in [10, 30, 60]:
            single = localize.identify_by_frame_number(
                movie, MIN_NG, BOX, frame
            )
            full_subset = real_identifications[
                real_identifications["frame"] == frame
            ]
            single_set = set(zip(single["y"], single["x"]))
            full_set = set(zip(full_subset["y"], full_subset["x"]))
            assert (
                single_set == full_set
            ), f"frame={frame}: per-frame {single_set} != full {full_set}"

    def test_frame_outside_bounds_returns_empty(self, movie):
        out = localize.identify_by_frame_number(
            movie, MIN_NG, BOX, 5, frame_bounds=(20, 30)
        )
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

    def test_frame_in_one_of_several_segments(self, movie):
        """A frame inside any of several segments is processed; a frame
        in the gap between segments returns empty."""
        segments = [(0, 4), (20, 30)]
        inside = localize.identify_by_frame_number(
            movie, MIN_NG, BOX, 25, frame_bounds=segments
        )
        gap = localize.identify_by_frame_number(
            movie, MIN_NG, BOX, 10, frame_bounds=segments
        )
        # frame 25 matches the full per-frame identification...
        full = localize.identify_by_frame_number(movie, MIN_NG, BOX, 25)
        assert set(zip(inside["y"], inside["x"])) == set(
            zip(full["y"], full["x"])
        )
        # ...while frame 10 (in the gap) yields nothing
        assert len(gap) == 0


# ---------------------------------------------------------------------------
# picks_to_identifications / locs_to_identifications
# ---------------------------------------------------------------------------


class TestPicksToIdentifications:
    """Convert circular picks into identification rows."""

    def test_basic(self):
        picks = [(5.5, 5.5), (15.5, 15.5), (25.5, 25.5)]
        n_frames = 10
        ids = localize.picks_to_identifications(picks, n_frames=n_frames)
        # 3 picks * 10 frames = 30 rows
        assert len(ids) == len(picks) * n_frames
        for col in ["frame", "x", "y", "net_gradient", "n_id"]:
            assert col in ids.columns

    def test_each_pick_present_in_all_frames(self):
        picks = [(5.5, 5.5), (15.5, 15.5)]
        n_frames = 4
        ids = localize.picks_to_identifications(picks, n_frames=n_frames)
        # Every frame must contain one row per pick
        for f in range(n_frames):
            assert (ids["frame"] == f).sum() == len(picks)

    def test_drift_applied_to_positions(self):
        picks = [(5.5, 5.5)]
        drift = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, -1.0, -2.0]})
        ids = localize.picks_to_identifications(picks, drift=drift)
        # Per-frame x/y reflects pick + drift
        ids_sorted = ids.sort_values("frame").reset_index(drop=True)
        np.testing.assert_allclose(
            ids_sorted["x"], [5.5 + 0.0, 5.5 + 1.0, 5.5 + 2.0]
        )
        np.testing.assert_allclose(
            ids_sorted["y"], [5.5 + 0.0, 5.5 - 1.0, 5.5 - 2.0]
        )

    def test_no_n_frames_no_drift_raises(self):
        with pytest.raises(ValueError):
            localize.picks_to_identifications([(1.0, 2.0)])

    def test_non_circular_picks_rejected(self):
        # Each pick must contain exactly two coordinates (circular pick);
        # 3-element picks are rejected.
        with pytest.raises(AssertionError):
            localize.picks_to_identifications([(1.0, 2.0, 3.0)], n_frames=5)

    def test_non_list_input_rejected(self):
        with pytest.raises(AssertionError):
            localize.picks_to_identifications("not a list", n_frames=5)


class TestLocsToIdentifications:
    """Round-trip locs back into identifications spanning a window of
    frames around each loc."""

    def test_columns_and_window_size(self, locs, info):
        n_frames = 2  # ±2 around each loc -> 5 rows per kept loc
        ids = localize.locs_to_identifications(
            locs.iloc[:5], info, n_frames=n_frames
        )
        for col in ["frame", "x", "y", "net_gradient", "n_id"]:
            assert col in ids.columns
        # Each kept loc contributes 2*n_frames + 1 rows
        unique_n_id = ids["n_id"].nunique()
        assert len(ids) == unique_n_id * (2 * n_frames + 1)

    def test_locs_near_movie_edges_excluded(self, locs, info):
        """Locs whose frame is within ``n_frames`` of the movie edges
        are skipped."""
        n_frames = 2
        # Build a locs frame with one "near edge" loc and one in the middle
        movie_frames = info[0]["Frames"]
        edge_locs = pd.DataFrame(
            {
                "frame": [0, 1, movie_frames // 2, movie_frames - 1],
                "x": [10.0, 10.0, 10.0, 10.0],
                "y": [10.0, 10.0, 10.0, 10.0],
            }
        )
        ids = localize.locs_to_identifications(
            edge_locs, info, n_frames=n_frames
        )
        # Only the middle loc passes the edge check
        assert ids["n_id"].nunique() == 1


# ---------------------------------------------------------------------------
# save_identifications / load_identifications (picasso.io)
# ---------------------------------------------------------------------------


class TestSaveLoadIdentifications:
    """HDF5 round-trip for the identifications DataFrame.

    The two functions are defined in ``picasso.io`` but exist solely to
    persist what the Localize GUI calls ``identifications`` (a DataFrame
    with columns ``frame``, ``x``, ``y``, ``net_gradient`` and optionally
    ``n_id``). They mirror the ``save_locs`` / ``load_locs`` pattern: an
    HDF5 file containing an ``"identifications"`` dataset and an
    accompanying ``.yaml`` sidecar with metadata.
    """

    def _info(self) -> list[dict]:
        return [
            {"Width": 32, "Height": 32, "Frames": 100},
            {
                "Generated by": "test_localize",
                "Box Size": BOX,
                "Min. Net Gradient": MIN_NG,
            },
        ]

    def test_roundtrip_real_identifications(
        self, tmp_path, real_identifications
    ):
        """Save identifications coming out of ``localize.identify``, then
        load them back — columns and content survive intact."""
        path = tmp_path / "test_identifications.hdf5"
        info = self._info()
        io.save_identifications(str(path), real_identifications, info)
        loaded, loaded_info = io.load_identifications(str(path))
        assert len(loaded) == len(real_identifications)
        assert set(loaded.columns) == set(real_identifications.columns)
        for col in ["frame", "x", "y", "net_gradient"]:
            np.testing.assert_array_equal(
                loaded[col].to_numpy(),
                real_identifications[col].to_numpy(),
            )
        assert loaded_info == info

    def test_roundtrip_picks_identifications(self, tmp_path):
        """``picks_to_identifications`` produces a DataFrame that also
        carries an ``n_id`` column — verify that round-trips too."""
        ids = localize.picks_to_identifications(
            [(5.5, 5.5), (15.5, 15.5)], n_frames=4
        )
        path = tmp_path / "picks_identifications.hdf5"
        io.save_identifications(str(path), ids, self._info())
        loaded, _ = io.load_identifications(str(path))
        assert "n_id" in loaded.columns
        assert len(loaded) == len(ids)
        np.testing.assert_array_equal(
            np.sort(loaded["n_id"].to_numpy()),
            np.sort(ids["n_id"].to_numpy()),
        )

    def test_yaml_sidecar_written(self, tmp_path, real_identifications):
        """``save_identifications`` must drop a YAML next to the HDF5 so
        ``load_identifications`` can recover the metadata."""
        path = tmp_path / "ids.hdf5"
        io.save_identifications(str(path), real_identifications, self._info())
        assert path.exists()
        assert (tmp_path / "ids.yaml").exists()

    def test_hdf5_dataset_key_is_identifications(
        self, tmp_path, real_identifications
    ):
        """The on-disk dataset key must be ``"identifications"`` — that's
        what ``load_identifications`` reads, and that's how a file is
        distinguished from a ``_locs.hdf5``."""
        path = tmp_path / "ids.hdf5"
        io.save_identifications(str(path), real_identifications, self._info())
        with h5py.File(path, "r") as f:
            assert "identifications" in f
            assert f["identifications"].shape == (len(real_identifications),)

    def test_load_missing_dataset_raises_keyerror(self, tmp_path):
        """Loading an HDF5 that has no ``identifications`` dataset (e.g.
        a ``_locs.hdf5``) must raise — silent fallback would let bad
        files masquerade as identifications."""
        path = tmp_path / "wrong.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "locs",
                data=pd.DataFrame({"x": [1.0]}).to_records(index=False),
            )
        # accompanying yaml so we hit the KeyError path, not NoMetadataFileError
        io.save_info(str(tmp_path / "wrong.yaml"), self._info())
        with pytest.raises(KeyError):
            io.load_identifications(str(path))

    def test_load_missing_yaml_falls_back_to_embedded(
        self, tmp_path, real_identifications
    ):
        """If the YAML sidecar is removed, ``load_identifications`` falls
        back to the metadata embedded in the HDF5 ``/metadata`` dataset
        (same contract as ``load_locs``)."""
        path = tmp_path / "ids.hdf5"
        info = self._info()
        io.save_identifications(str(path), real_identifications, info)
        (tmp_path / "ids.yaml").unlink()
        _, loaded_info = io.load_identifications(str(path))
        assert loaded_info == info

    def test_save_uses_yaml_sidecar_path(self, tmp_path, real_identifications):
        """The YAML path is derived from the HDF5 path's base name —
        verify that even when the HDF5 has a non-standard suffix, the
        YAML lands at ``<base>.yaml``."""
        path = tmp_path / "custom_name.hdf5"
        io.save_identifications(str(path), real_identifications, self._info())
        assert (tmp_path / "custom_name.yaml").exists()


# ---------------------------------------------------------------------------
# get_spots
# ---------------------------------------------------------------------------


class TestGetSpots:
    """Pixel patches around identified spots."""

    def test_shape_dtype(self, real_spots, real_identifications):
        n = len(real_identifications)
        assert real_spots.shape == (n, BOX, BOX)
        assert real_spots.dtype == np.float32

    def test_baseline_subtraction_via_camera_info(
        self, movie, real_identifications
    ):
        """Increasing the baseline subtracts from every spot pixel."""
        cam_a = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
        cam_b = {"Baseline": 100, "Sensitivity": 1, "Gain": 1}
        spots_a = localize.get_spots(movie, real_identifications, BOX, cam_a)
        spots_b = localize.get_spots(movie, real_identifications, BOX, cam_b)
        np.testing.assert_allclose(spots_a - spots_b, 100.0)

    def test_sensitivity_scales_signal(self, movie, real_identifications):
        cam_x1 = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
        cam_x2 = {"Baseline": 0, "Sensitivity": 2, "Gain": 1}
        spots_x1 = localize.get_spots(movie, real_identifications, BOX, cam_x1)
        spots_x2 = localize.get_spots(movie, real_identifications, BOX, cam_x2)
        np.testing.assert_allclose(spots_x2, spots_x1 * 2)

    def test_gain_divides_signal(self, movie, real_identifications):
        cam_x1 = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
        cam_x2 = {"Baseline": 0, "Sensitivity": 1, "Gain": 2}
        spots_x1 = localize.get_spots(movie, real_identifications, BOX, cam_x1)
        spots_x2 = localize.get_spots(movie, real_identifications, BOX, cam_x2)
        np.testing.assert_allclose(spots_x2, spots_x1 / 2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    """``check_nena``, ``check_kinetics``, ``check_drift``."""

    def test_check_nena_returns_float(self, locs, info):
        # ``check_nena`` currently hard-codes ``info=None`` when calling
        # ``postprocess.nena`` and catches the resulting failure, returning
        # NaN. The contract observable from the outside is "return a float".
        nena = localize.check_nena(locs, info)
        assert isinstance(nena, float)
        assert nena > 0

    def test_check_kinetics_returns_positive_scalar(self, locs, info):
        len_mean = localize.check_kinetics(locs, info)
        assert np.isfinite(len_mean)
        assert len_mean > 0

    def test_check_drift_returns_two_floats(self, locs, info):
        result = localize.check_drift(locs, info)
        assert isinstance(result, tuple)
        assert len(result) == 2
        drift_x, drift_y = result
        assert np.isfinite(drift_x)
        assert np.isfinite(drift_y)


# ---------------------------------------------------------------------------
# identifications_from_futures (low-level helper used by the GUI)
# ---------------------------------------------------------------------------


class TestIdentificationsFromFutures:
    """Stitch the per-thread results back into a sorted DataFrame."""

    def test_concatenates_and_sorts_by_frame(self):
        # Build two fake futures whose .result() returns lists of DFs.
        df_a = pd.DataFrame(
            {
                "frame": [3, 1],
                "x": [10, 20],
                "y": [11, 21],
                "net_gradient": [5000.0, 6000.0],
            }
        )
        df_b = pd.DataFrame(
            {
                "frame": [2, 0],
                "x": [30, 40],
                "y": [31, 41],
                "net_gradient": [7000.0, 8000.0],
            }
        )

        class _DummyFuture:
            def __init__(self, lst):
                self._lst = lst

            def result(self):
                return self._lst

        out = localize.identifications_from_futures(
            [_DummyFuture([df_a]), _DummyFuture([df_b])]
        )
        # All four rows preserved
        assert len(out) == 4
        # Sorted ascending by frame
        assert list(out["frame"]) == sorted(out["frame"])


# ---------------------------------------------------------------------------
# fit2D — high-level wrapper that supports gausslq / gaussmle / avg
# ---------------------------------------------------------------------------
#
# ``fit2D`` and ``localize`` both assert ``isinstance(movie,
# AbstractPicassoMovie)``. The bundled .raw movie loads as a plain
# ``np.memmap`` so we feed in the ``picasso_movie`` fixture from conftest
# (a thin AbstractPicassoMovie wrapper around the same memmap).


class TestFit2D:
    """The 2D fitting dispatcher used by Picasso: Localize."""

    def test_gausslq_returns_locs_and_metadata(
        self, picasso_movie, real_identifications, movie_info
    ):
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gausslq",
            multiprocess=False,
        )
        assert len(locs) == len(real_identifications)
        for col in ["x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy"]:
            assert col in locs.columns
        # metadata reflects the chosen fitting method
        assert new_info["Fit method"] == "gausslq"
        # camera_info keys merged into new_info
        assert new_info["Pixelsize"] == 130

    def test_gaussmle_returns_locs(
        self, picasso_movie, real_identifications, movie_info
    ):
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gaussmle",
            multiprocess=False,
        )
        assert len(locs) == len(real_identifications)
        # MLE-specific metadata
        assert new_info["Fit method"] == "gaussmle"
        assert new_info["Convergence criterion"] == 0.001
        assert new_info["Max iterations"] == 100

    def test_avg_returns_locs(
        self, picasso_movie, real_identifications, movie_info
    ):
        """The ``avg`` method takes per-pixel averages — produces a locs
        DataFrame even though it doesn't fit a Gaussian."""
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="avg",
            multiprocess=False,
        )
        assert len(locs) == len(real_identifications)
        assert new_info["Fit method"] == "avg"

    def test_invalid_fitting_method_raises(
        self, picasso_movie, real_identifications, movie_info
    ):
        with pytest.raises(AssertionError):
            localize.fit2D(
                picasso_movie,
                movie_info,
                CAMERA_INFO_WITH_PIXELSIZE,
                real_identifications,
                BOX,
                fitting_method="bogus",
                multiprocess=False,
            )

    def test_negative_eps_rejected(
        self, picasso_movie, real_identifications, movie_info
    ):
        with pytest.raises(AssertionError):
            localize.fit2D(
                picasso_movie,
                movie_info,
                CAMERA_INFO_WITH_PIXELSIZE,
                real_identifications,
                BOX,
                fitting_method="gaussmle",
                eps=-1.0,
                multiprocess=False,
            )

    def test_missing_pixelsize_warns_and_defaults(
        self, picasso_movie, real_identifications, movie_info
    ):
        """If ``Pixelsize`` is absent from camera_info, fit2D emits a
        warning and defaults to 130 nm."""
        cam = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
        with pytest.warns(UserWarning, match="Pixelsize"):
            _, new_info = localize.fit2D(
                picasso_movie,
                movie_info,
                cam,
                real_identifications,
                BOX,
                fitting_method="gausslq",
                multiprocess=False,
            )
        assert new_info["Pixelsize"] == 130


# ---------------------------------------------------------------------------
# localize — monolithic identify + fit2D entry point
# ---------------------------------------------------------------------------


class TestLocalize:
    """The top-level ``localize`` pipeline (identify -> get_spots -> fit)."""

    def test_basic_pipeline_returns_locs(self, picasso_movie, movie_info):
        locs = localize.localize(
            picasso_movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
            movie_info=movie_info,
            fitting_method="gausslq",
            threaded=False,
            return_info=False,
        )
        assert isinstance(locs, pd.DataFrame)
        assert len(locs) > 0
        for col in ["frame", "x", "y", "photons", "sx", "sy", "bg"]:
            assert col in locs.columns

    def test_return_info_returns_full_info_chain(
        self, picasso_movie, movie_info
    ):
        """With ``return_info=True``, returns ``(locs, info)`` where info
        contains the original movie info, the identify metadata, and the
        fit metadata."""
        locs, info = localize.localize(
            picasso_movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
            movie_info=movie_info,
            fitting_method="gausslq",
            threaded=False,
            return_info=True,
        )
        assert isinstance(locs, pd.DataFrame)
        assert isinstance(info, list)
        assert len(info) == len(movie_info) + 2
        # Identify info appears second-to-last; fit info last.
        assert "Min. Net Gradient" in info[-2]
        assert "Fit method" in info[-1]

    def test_localize_matches_identify_plus_fit2d(
        self, picasso_movie, real_identifications, movie_info
    ):
        """Calling ``localize`` should produce the same result (up to
        ordering) as calling ``identify`` + ``fit2D`` separately, since
        ``localize`` is just glue."""
        # Direct path
        locs_direct, _ = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gausslq",
            multiprocess=False,
        )
        # Through the high-level entry point
        locs_high = localize.localize(
            picasso_movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
            movie_info=movie_info,
            fitting_method="gausslq",
            threaded=False,
            return_info=False,
        )
        assert len(locs_direct) == len(locs_high)
        # photons sums match
        np.testing.assert_allclose(
            np.sort(locs_direct["photons"].to_numpy()),
            np.sort(locs_high["photons"].to_numpy()),
            rtol=1e-3,
        )

    def test_roi_is_applied_at_identification(self, picasso_movie, movie_info):
        """Passing an ROI confines the localizations to that pixel
        window."""
        roi = ((0, 0), (16, 16))
        locs = localize.localize(
            picasso_movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
            movie_info=movie_info,
            roi=roi,
            fitting_method="gausslq",
            threaded=False,
            return_info=False,
        )
        # No localization outside the ROI window
        if len(locs) > 0:
            assert (locs["x"] < 16).all()
            assert (locs["y"] < 16).all()


# ---------------------------------------------------------------------------
# localize_3D — identify + 2D fit + z fitting
# ---------------------------------------------------------------------------


class TestLocalize3D:
    """End-to-end 3D localization pipeline.

    Note: the public ``localize_3D`` validates its movie argument with
    ``isinstance(movie, (np.ndarray, ND2Movie))`` — but the inner
    ``fit2D`` then asserts ``isinstance(movie, AbstractPicassoMovie)``,
    which conflicts. So the public ``localize_3D`` is unusable for
    AbstractPicassoMovie inputs; we exercise the internal
    ``_localize_3D`` (which has no such guard) to verify that the actual
    pipeline produces sensible 3D locs.
    """

    def test_public_localize_3d_rejects_wrapper(
        self, picasso_movie, movie_info
    ):
        """The public function's input check excludes AbstractPicassoMovie."""
        with pytest.raises(AssertionError, match="numpy array or ND2Movie"):
            localize.localize_3D(
                picasso_movie,
                movie_info=movie_info,
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                box=BOX,
                minimum_ng=MIN_NG,
                calibration_3d=dict(CALIB_3D),
                fitting_method="gausslq",
                multiprocess=False,
            )

    def test_public_localize_3d_invalid_calibration_type(
        self, movie, movie_info
    ):
        with pytest.raises(AssertionError, match="calibration_3d"):
            localize.localize_3D(
                movie,
                movie_info=movie_info,
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                box=BOX,
                minimum_ng=MIN_NG,
                calibration_3d=12345,  # neither dict nor str
                fitting_method="gausslq",
                multiprocess=False,
            )

    def test_underlying_pipeline_produces_z_locs(
        self, picasso_movie, movie_info
    ):
        """Drive the full identify->fit->zfit pipeline through
        ``_localize_3D`` and verify the output has the expected 3D
        columns and finite z values."""
        locs, _ = localize._localize_3D(
            picasso_movie,
            movie_info=movie_info,
            camera_info=CAMERA_INFO_WITH_PIXELSIZE,
            box=BOX,
            minimum_ng=MIN_NG,
            calibration_3d=dict(CALIB_3D),
            fitting_method="gausslq",
            multiprocess=False,
        )
        assert isinstance(locs, pd.DataFrame)
        assert len(locs) > 0
        for col in ["x", "y", "z", "d_zcalib", "lpz", "sx", "sy"]:
            assert col in locs.columns
        assert np.all(np.isfinite(locs["z"].to_numpy()))
        assert (locs["lpz"] > 0).all()


# ---------------------------------------------------------------------------
# MovieLoadWorker — background loader for Picasso: Localize
#
# The GUI opens movies on a background QThread so the window stays
# responsive (see picasso.gui.localize.MovieLoadWorker). These tests drive
# the worker's run() directly with monkeypatched io loaders — no QThread is
# started, so signals delivered to same-thread receivers fire synchronously
# (Qt DirectConnection), which is exactly what we rely on to observe them.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _qt_app():
    """A QApplication must exist before any QObject (the worker) is built."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    yield app


class _Collector:
    """Records the worker's terminal signals for assertions."""

    def __init__(self, worker):
        self.finished = None
        self.failed = None
        self.progress = []
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.progress.connect(
            lambda i, name: self.progress.append((i, name))
        )

    def _on_finished(self, movies, infos, paths):
        self.finished = (movies, infos, paths)

    def _on_failed(self, message):
        self.failed = message


def _info(name="Channel 0"):
    return [{"Frames": 1, "Height": 4, "Width": 4, "Channel": name}]


class TestMovieLoadWorker:
    def test_load_movie_per_file(self, monkeypatch):
        """load_all=False loads one channel per path via io.load_movie."""
        loaded = {}

        def fake_load_movie(path, prompt_info=None, progress=None):
            loaded[path] = prompt_info
            return f"movie::{path}", _info(path)

        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        worker = localize_gui.MovieLoadWorker(
            ["a.tif", "b.tif"], prompt_for_path=lambda p: None
        )
        out = _Collector(worker)
        worker.run()

        movies, infos, paths = out.finished
        assert movies == ["movie::a.tif", "movie::b.tif"]
        assert paths == ["a.tif", "b.tif"]
        assert infos[1][0]["Channel"] == "b.tif"
        assert out.failed is None
        # One progress tick per file, in order.
        assert [i for i, _ in out.progress] == [0, 1]

    def test_load_all_expands_channels(self, monkeypatch):
        """load_all=True reads every channel of each file via
        io.load_movie_all; each returned movie maps back to its path."""

        def fake_load_movie_all(path, prompt_info=None, progress=None):
            return ["m0", "m1", "m2"], [_info("c0"), _info("c1"), _info("c2")]

        monkeypatch.setattr(io, "load_movie_all", fake_load_movie_all)
        worker = localize_gui.MovieLoadWorker(
            ["multi.lif"], prompt_for_path=lambda p: None, load_all=True
        )
        out = _Collector(worker)
        worker.run()

        movies, infos, paths = out.finished
        assert movies == ["m0", "m1", "m2"]
        # Every channel is attributed to the one source file.
        assert paths == ["multi.lif", "multi.lif", "multi.lif"]
        assert len(infos) == 3

    def test_none_result_is_skipped(self, monkeypatch):
        """A loader returning None (e.g. a cancelled prompt) drops that
        file without aborting the rest of the batch."""

        def fake_load_movie(path, prompt_info=None, progress=None):
            if path == "skip.tif":
                return None
            return f"movie::{path}", _info(path)

        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        worker = localize_gui.MovieLoadWorker(
            ["skip.tif", "keep.tif"], prompt_for_path=lambda p: None
        )
        out = _Collector(worker)
        worker.run()

        movies, _, paths = out.finished
        assert movies == ["movie::keep.tif"]
        assert paths == ["keep.tif"]

    def test_prompt_is_proxied_and_result_returned(self, monkeypatch):
        """A loader that calls prompt_info gets the value the GUI handler
        supplies: the worker emits prompt_requested, the (same-thread)
        handler fills the holder and releases the worker's event."""

        def fake_load_movie(path, prompt_info=None, progress=None):
            chosen = prompt_info(["DAPI", "GFP"])
            return f"movie::{chosen}", _info(chosen)

        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        worker = localize_gui.MovieLoadWorker(
            ["m.czi"], prompt_for_path=lambda p: (lambda chans: "GFP")
        )

        seen = {}

        def handle_prompt(callback, args_kwargs, holder):
            args, kwargs = args_kwargs
            seen["channels"] = args[0]
            holder["result"] = callback(*args, **kwargs)
            worker._prompt_event.set()

        worker.prompt_requested.connect(handle_prompt)
        out = _Collector(worker)
        worker.run()

        assert seen["channels"] == ["DAPI", "GFP"]
        assert out.finished[0] == ["movie::GFP"]

    def test_exception_emits_failed(self, monkeypatch):
        """A loader error is reported through the failed signal rather than
        propagating out of run() (which runs on a worker thread)."""

        def boom(path, prompt_info=None, progress=None):
            raise RuntimeError("bad file")

        monkeypatch.setattr(io, "load_movie", boom)
        worker = localize_gui.MovieLoadWorker(
            ["x.tif"], prompt_for_path=lambda p: None
        )
        out = _Collector(worker)
        worker.run()

        assert out.finished is None
        assert "bad file" in out.failed

    def test_cancel_stops_before_next_file(self, monkeypatch):
        """cancel() set during a load stops the loop before the next
        file, so the batch ends with only the files read so far."""
        seen = []

        def fake_load_movie(path, prompt_info=None, progress=None):
            seen.append(path)
            worker.cancel()  # cancel while the first file is "loading"
            return f"movie::{path}", _info(path)

        worker = localize_gui.MovieLoadWorker(
            ["first.tif", "second.tif"], prompt_for_path=lambda p: None
        )
        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        out = _Collector(worker)
        worker.run()

        # Only the first file was loaded; the second was never attempted.
        assert seen == ["first.tif"]
        assert out.finished[2] == ["first.tif"]


# ---------------------------------------------------------------------------
# Optional GPU backend (Gpufit) — skipped when no CUDA GPU is available
# ---------------------------------------------------------------------------


# Gpufit reports a per-spot termination code; 0 is a converged fit. The MLE
# (Poisson) estimator additionally emits code 3 (NEG_CURVATURE_MLE) when the
# likelihood Hessian loses positive-definiteness - the returned parameters are
# then the last (unconverged, unreliable) iterate, so tests that assert
# numerical recovery must restrict to converged spots.
_GPUFIT_CONVERGED = 0


def _gpufit_gauss_with_states(spots, rotated=False, mle=False):
    """Run the low-level Gpufit Gaussian fit while keeping the per-spot fit
    states that :func:`localize.fit_spots_gpufit` drops. Mirrors that function
    exactly (same initial parameters, model, estimator, tolerance and iteration
    cap) and applies the same ``photons = amplitude * 2*pi*sx*sy`` conversion,
    returning ``(theta, states, n_iterations)``."""
    gf = localize.gf
    data = np.maximum(spots, 0) if mle else spots
    size = data.shape[1]
    init = localize._initial_parameters_gpufit(data, size, rotated=rotated)
    model_id = (
        gf.ModelID.GAUSS_2D_ROTATED
        if rotated
        else gf.ModelID.GAUSS_2D_ELLIPTIC
    )
    estimator_id = gf.EstimatorID.MLE if mle else gf.EstimatorID.LSE
    params, states, chi_squares, n_iter, _ = gf.fit(
        data.reshape((len(data), size * size)),
        None,
        model_id,
        init,
        tolerance=1e-2,
        max_number_iterations=20,
        estimator_id=estimator_id,
    )
    params = params.copy()
    params[:, 0] *= 2.0 * np.pi * params[:, 3] * params[:, 4]
    return params, states, n_iter


def _make_rotated_spot(box, x0, y0, sx, sy, photons, bg, angle):
    """Point-sampled rotated elliptical Gaussian, matching the model Gpufit's
    GAUSS_2D_ROTATED optimizes (and the reference ``_gauss_model`` used by the
    CRLB tests): ``mu = photons/(2 pi sx sy) * exp(...) + bg``. ``x0``/``y0``
    are offsets from the box center."""
    half = box // 2
    g = np.arange(-half, half + 1, dtype=np.float64)
    X, Y = np.meshgrid(g, g)  # X varies along columns (x), Y along rows (y)
    dx, dy = X - x0, Y - y0
    ct, st = np.cos(angle), np.sin(angle)
    u = dx * ct - dy * st
    w = dx * st + dy * ct
    e = np.exp(-0.5 * (u**2 / sx**2 + w**2 / sy**2))
    return (photons / (2 * np.pi * sx * sy) * e + bg).astype(np.float32)


@pytest.mark.skipif(
    not localize.GPUFIT_INSTALLED, reason="GPUfit/CUDA not available"
)
class TestGpufit:
    """Thorough tests for the Gpufit Gaussian codepath (``fit_spots_gpufit``).
    Requires a CUDA-capable GPU, so skipped in the typical test environment.

    Parameter order returned by Gpufit is ``[photons, x, y, sx, sy, bg]`` and,
    for the rotated model, ``[photons, x, y, sx, sy, bg, angle]``. ``x``/``y``
    are box-pixel coordinates, so the ground-truth center offset is
    ``x - box // 2``."""

    # -- least squares: the deterministic, model-exact path ----------------

    def test_lse_recovers_all_parameters_noiseless(self, synthetic_spots):
        """On noiseless spots the model matches the data exactly, so LSE must
        recover every parameter - not just photons - to tight tolerance."""
        spots, gt = synthetic_spots
        theta = localize.fit_spots_gpufit(spots, mle=False)
        assert theta.shape == (len(spots), 6)
        half = spots.shape[1] // 2
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=1e-3)
        np.testing.assert_allclose(theta[:, 1] - half, gt.x.values, atol=2e-3)
        np.testing.assert_allclose(theta[:, 2] - half, gt.y.values, atol=2e-3)
        np.testing.assert_allclose(theta[:, 3], gt.sx.values, atol=2e-3)
        np.testing.assert_allclose(theta[:, 4], gt.sy.values, atol=2e-3)
        np.testing.assert_allclose(theta[:, 5], gt.bg.values, atol=1e-2)

    def test_lse_recovers_parameters_noisy(self, synthetic_spots_noisy):
        """With Poisson noise LSE still recovers ground truth, at looser
        (noise-limited) tolerance."""
        spots, gt = synthetic_spots_noisy
        theta = localize.fit_spots_gpufit(spots, mle=False)
        half = spots.shape[1] // 2
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=0.05)
        np.testing.assert_allclose(theta[:, 1] - half, gt.x.values, atol=0.1)
        np.testing.assert_allclose(theta[:, 2] - half, gt.y.values, atol=0.1)

    def test_photons_are_amplitude_times_2pi_sxsy(self, synthetic_spots):
        """The reported photon count is the raw Gpufit Gaussian amplitude
        scaled by its integral ``2*pi*sx*sy`` - the conversion
        fit_spots_gpufit applies to Gpufit's peak-height parameter."""
        gf = localize.gf
        spots, _ = synthetic_spots
        size = spots.shape[1]
        init = localize._initial_parameters_gpufit(spots, size)
        raw, _, _, _, _ = gf.fit(
            spots.reshape((len(spots), size * size)),
            None,
            gf.ModelID.GAUSS_2D_ELLIPTIC,
            init,
            tolerance=1e-2,
            max_number_iterations=20,
            estimator_id=gf.EstimatorID.LSE,
        )
        theta = localize.fit_spots_gpufit(spots, mle=False)
        expected = raw[:, 0] * 2.0 * np.pi * raw[:, 3] * raw[:, 4]
        np.testing.assert_allclose(theta[:, 0], expected, rtol=1e-5)

    # -- rotated elliptical Gaussian --------------------------------------

    def test_rotated_recovers_angle(self):
        """The rotated model recovers the ground-truth rotation angle (radians)
        for a range of angles."""
        box = 9
        angles = np.array([0.2, 0.6, -0.5, 1.0])
        spots = np.stack(
            [
                _make_rotated_spot(box, 0.1, -0.15, 1.6, 0.9, 5000.0, 10.0, a)
                for a in angles
            ]
        )
        theta = localize.fit_spots_gpufit(spots, rotated=True, mle=False)
        assert theta.shape == (len(angles), 7)
        np.testing.assert_allclose(theta[:, 6], angles, atol=1e-3)
        # widths recovered along the rotated axes
        np.testing.assert_allclose(theta[:, 3], 1.6, atol=5e-3)
        np.testing.assert_allclose(theta[:, 4], 0.9, atol=5e-3)

    # -- maximum likelihood (Poisson) -------------------------------------

    def test_mle_converged_spots_recover(self, synthetic_spots_noisy):
        """Gpufit's MLE terminates a fraction of fits with NEG_CURVATURE_MLE
        (state 3) and returns their last, unreliable iterate. The fits that DO
        converge (state 0) must recover ground truth tightly. This documents
        the real contract of the vendored MLE estimator."""
        spots, gt = synthetic_spots_noisy
        theta, states, _ = _gpufit_gauss_with_states(spots, mle=True)
        converged = states == _GPUFIT_CONVERGED
        assert converged.any(), "expected at least some MLE fits to converge"
        half = spots.shape[1] // 2
        idx = np.where(converged)[0]
        np.testing.assert_allclose(
            theta[idx, 0], gt.photons.values[idx], rtol=0.08
        )
        np.testing.assert_allclose(
            theta[idx, 1] - half, gt.x.values[idx], atol=0.15
        )
        np.testing.assert_allclose(
            theta[idx, 2] - half, gt.y.values[idx], atol=0.15
        )

    def test_mle_matches_lse_on_converged_spots(self, synthetic_spots_noisy):
        """Where the MLE fit converges, its photon estimate agrees with the
        (always-converging) LSE fit on the same data."""
        spots, _ = synthetic_spots_noisy
        lse = localize.fit_spots_gpufit(spots, mle=False)
        mle, states, _ = _gpufit_gauss_with_states(spots, mle=True)
        idx = np.where(states == _GPUFIT_CONVERGED)[0]
        np.testing.assert_allclose(mle[idx, 0], lse[idx, 0], rtol=0.1)

    def test_mle_clamps_negative_pixels(self, synthetic_spots_noisy):
        """The MLE path clamps negative pixel values (Poisson counts cannot be
        negative) rather than crashing; the public function must accept spots
        with negatives and return finite parameters for converged fits."""
        spots, _ = synthetic_spots_noisy
        spots = spots.copy()
        spots[:, 0, 0] -= 50.0  # inject a negative pixel per spot
        theta, states, _ = _gpufit_gauss_with_states(spots, mle=True)
        idx = states == _GPUFIT_CONVERGED
        assert np.all(np.isfinite(theta[idx]))

    # -- return_stats semantics -------------------------------------------

    def test_return_stats_mle(self, synthetic_spots_noisy):
        spots, _ = synthetic_spots_noisy
        theta, ll, n_iter = localize.fit_spots_gpufit(
            spots, mle=True, return_stats=True
        )
        assert theta.shape == (len(spots), 6)
        # MLE: log-likelihood is -0.5 * chi-square, finite, one per spot
        assert ll is not None and ll.shape == (len(spots),)
        assert np.all(np.isfinite(ll))
        assert n_iter.shape == (len(spots),)

    def test_return_stats_lse_has_no_likelihood(self, synthetic_spots):
        spots, _ = synthetic_spots
        theta, ll, n_iter = localize.fit_spots_gpufit(
            spots, mle=False, return_stats=True
        )
        # LSE reports a residual sum of squares, not a likelihood -> None
        assert ll is None
        assert n_iter.shape == (len(spots),)

    # -- end-to-end: fit -> localizations ---------------------------------

    def test_end_to_end_locs_absolute_position(self, synthetic_spots):
        """fit_spots_gpufit + locs_from_fits_gpufit place each spot at its true
        absolute position (identification pixel + sub-pixel fit offset)."""
        spots, gt = synthetic_spots
        n = len(spots)
        box = spots.shape[1]
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 50, dtype=np.int64),
                "y": np.full(n, 70, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )
        theta = localize.fit_spots_gpufit(spots, mle=False)
        locs = localize.locs_from_fits_gpufit(ids, theta, box, em=False)
        box_offset = int(box / 2)
        # x_abs = x_id + (x_fit - box_offset); x_fit - box//2 == gt.x
        np.testing.assert_allclose(locs["x"], 50 + gt.x.values, atol=3e-3)
        np.testing.assert_allclose(locs["y"], 70 + gt.y.values, atol=3e-3)
        np.testing.assert_allclose(
            locs["photons"], gt.photons.values, rtol=1e-3
        )
        assert np.all(np.isfinite(locs["lpx"])) and (locs["lpx"] > 0).all()


# ---------------------------------------------------------------------------
# GPU-free gpufit helpers — the initial-parameter seed and the fit ->
# localization converter are pure NumPy/pandas and run without a CUDA GPU.
# ---------------------------------------------------------------------------


class TestInitialParametersGpufit:
    """``localize._initial_parameters_gpufit`` seeds the Levenberg-Marquardt
    fit; it is pure NumPy and needs no GPU."""

    def test_elliptic_layout_and_values(self):
        # Two spots with known per-spot max/min so amplitude (max - min) and
        # background (min) are predictable.
        box = 7
        spots = np.zeros((2, box, box), dtype=np.float32)
        spots[0] = 3.0  # flat -> max == min
        spots[0, 3, 3] = 103.0  # peak
        spots[1] = 7.0
        spots[1, 2, 4] = 57.0
        init = localize._initial_parameters_gpufit(spots, box)

        assert init.shape == (2, 6)
        assert init.dtype == np.float32
        center = box / 2.0 - 0.5  # 3.0
        width = max(box / 5.0, 1.0)  # 1.4
        # amplitude = max - min
        np.testing.assert_allclose(init[:, 0], [100.0, 50.0])
        # x, y seeded at the geometric box center
        np.testing.assert_allclose(init[:, 1], center)
        np.testing.assert_allclose(init[:, 2], center)
        # both widths seeded equal
        np.testing.assert_allclose(init[:, 3], width)
        np.testing.assert_allclose(init[:, 4], width)
        # background = per-spot minimum
        np.testing.assert_allclose(init[:, 5], [3.0, 7.0])

    def test_width_floor_for_small_box(self):
        # box / 5 < 1 -> the width floor of 1.0 kicks in.
        box = 4
        spots = np.ones((1, box, box), dtype=np.float32)
        init = localize._initial_parameters_gpufit(spots, box)
        np.testing.assert_allclose(init[:, 3], 1.0)
        np.testing.assert_allclose(init[:, 4], 1.0)

    def test_rotated_breaks_width_symmetry(self):
        # The rotated model gets a 7th (angle) parameter, and the two widths
        # are deliberately made unequal so the angle derivative is non-zero
        # (an isotropic seed makes the first LM Hessian singular).
        box = 7
        spots = np.zeros((3, box, box), dtype=np.float32)
        spots[:, 3, 3] = 100.0
        init = localize._initial_parameters_gpufit(spots, box, rotated=True)
        assert init.shape == (3, 7)
        width = max(box / 5.0, 1.0)
        np.testing.assert_allclose(init[:, 3], width * 1.1)
        np.testing.assert_allclose(init[:, 4], width * 0.9)
        assert (init[:, 3] != init[:, 4]).all()
        np.testing.assert_allclose(init[:, 6], 0.0)


class TestLocsFromFitsGpufit:
    """``localize.locs_from_fits_gpufit`` maps gpufit theta
    ``[photons, x, y, sx, sy, bg, (angle)]`` to a localizations frame. Pure
    pandas/NumPy - no GPU needed."""

    def _ids(self, n, frames=None):
        return pd.DataFrame(
            {
                "frame": (
                    np.arange(n, dtype=np.uint32)
                    if frames is None
                    else np.asarray(frames, dtype=np.uint32)
                ),
                "x": np.arange(n, dtype=np.int64) + 10,
                "y": np.arange(n, dtype=np.int64) + 20,
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )

    def test_xy_offset_and_passthrough_columns(self):
        # x/y are the sub-pixel fit offset plus the integer identification
        # position minus the box half-offset; photons/sx/sy/bg pass through.
        theta = np.array(
            [
                [500.0, 3.2, 3.7, 1.3, 1.1, 5.0],
                [800.0, 3.4, 3.1, 1.2, 1.4, 4.0],
            ],
            dtype=np.float32,
        )
        ids = self._ids(2)
        locs = localize.locs_from_fits_gpufit(ids, theta, BOX, em=False)
        box_offset = int(BOX / 2)
        np.testing.assert_allclose(
            locs["x"], theta[:, 1] + ids["x"].to_numpy() - box_offset
        )
        np.testing.assert_allclose(
            locs["y"], theta[:, 2] + ids["y"].to_numpy() - box_offset
        )
        np.testing.assert_allclose(locs["photons"], theta[:, 0])
        np.testing.assert_allclose(locs["sx"], theta[:, 3])
        np.testing.assert_allclose(locs["sy"], theta[:, 4])
        np.testing.assert_allclose(locs["bg"], theta[:, 5])

    def test_ellipticity_formula(self):
        theta = np.array([[500.0, 3.0, 3.0, 1.4, 1.0, 5.0]], dtype=np.float32)
        locs = localize.locs_from_fits_gpufit(
            theta=theta, box=BOX, em=False, identifications=self._ids(1)
        )
        # (max - min) / max = (1.4 - 1.0) / 1.4
        np.testing.assert_allclose(
            locs["ellipticity"], (1.4 - 1.0) / 1.4, rtol=1e-6
        )

    def test_lse_precision_is_mortensen_no_unc_columns(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.3, 1.1, 5.0]], dtype=np.float32)
        locs = localize.locs_from_fits_gpufit(
            self._ids(1), theta, BOX, em=False, mle=False
        )
        expected_lpx = gausslq.localization_precision(
            theta[:, 0], theta[:, 3], theta[:, 4], theta[:, 5], em=False
        )
        expected_lpy = gausslq.localization_precision(
            theta[:, 0], theta[:, 4], theta[:, 3], theta[:, 5], em=False
        )
        np.testing.assert_allclose(locs["lpx"], expected_lpx, rtol=1e-6)
        np.testing.assert_allclose(locs["lpy"], expected_lpy, rtol=1e-6)
        # least squares does not emit per-parameter uncertainties
        for col in ("photons_unc", "bg_unc", "sx_unc", "sy_unc"):
            assert col not in locs.columns

    def test_mle_precision_is_crlb_with_unc_columns(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.3, 1.1, 5.0]], dtype=np.float32)
        locs = localize.locs_from_fits_gpufit(
            self._ids(1), theta, BOX, em=False, mle=True
        )
        crlb = localize._gauss_crlb(theta, BOX, em=False)
        np.testing.assert_allclose(locs["lpx"], np.sqrt(crlb[:, 1]), rtol=1e-6)
        np.testing.assert_allclose(locs["lpy"], np.sqrt(crlb[:, 2]), rtol=1e-6)
        np.testing.assert_allclose(
            locs["photons_unc"], np.sqrt(crlb[:, 0]), rtol=1e-6
        )
        np.testing.assert_allclose(
            locs["bg_unc"], np.sqrt(crlb[:, 5]), rtol=1e-6
        )
        np.testing.assert_allclose(
            locs["sx_unc"], np.sqrt(crlb[:, 3]), rtol=1e-6
        )
        np.testing.assert_allclose(
            locs["sy_unc"], np.sqrt(crlb[:, 4]), rtol=1e-6
        )

    def test_rotated_angle_column_normalized(self):
        # angle is stored in degrees, sign-flipped from the radians theta and
        # wrapped to [-90, 90) since an ellipse repeats every half turn.
        # 100 deg -> -100 deg after the sign flip -> wraps to +80.
        theta = np.array(
            [[500.0, 3.0, 3.0, 1.4, 1.0, 5.0, np.deg2rad(100.0)]],
            dtype=np.float32,
        )
        locs = localize.locs_from_fits_gpufit(
            self._ids(1), theta, BOX, em=False, mle=True
        )
        assert "angle" in locs.columns
        np.testing.assert_allclose(locs["angle"], 80.0, atol=1e-4)
        assert -90.0 <= locs["angle"].iloc[0] < 90.0
        assert "angle_unc" in locs.columns

    def test_sorted_by_frame(self):
        theta = np.tile(
            np.array([500.0, 3.0, 3.0, 1.2, 1.2, 5.0], dtype=np.float32),
            (3, 1),
        )
        ids = self._ids(3, frames=[2, 0, 1])
        locs = localize.locs_from_fits_gpufit(ids, theta, BOX, em=False)
        assert list(locs["frame"]) == [0, 1, 2]

    def test_stats_columns_optional(self):
        theta = np.array([[500.0, 3.0, 3.0, 1.2, 1.2, 5.0]], dtype=np.float32)
        # without stats, no log_likelihood / iterations
        locs = localize.locs_from_fits_gpufit(
            self._ids(1), theta, BOX, em=False
        )
        assert "log_likelihood" not in locs.columns
        assert "iterations" not in locs.columns
        # with stats they appear, correctly typed
        locs = localize.locs_from_fits_gpufit(
            self._ids(1),
            theta,
            BOX,
            em=False,
            mle=True,
            log_likelihood=np.array([-12.0], dtype=np.float32),
            iterations=np.array([7], dtype=np.int32),
        )
        assert locs["log_likelihood"].dtype == np.float32
        assert locs["iterations"].dtype == np.int32
        assert locs["iterations"].iloc[0] == 7

    def test_em_scales_lse_precision_by_sqrt2(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.3, 1.1, 5.0]], dtype=np.float32)
        no_em = localize.locs_from_fits_gpufit(
            self._ids(1), theta, BOX, em=False, mle=False
        )
        em = localize.locs_from_fits_gpufit(
            self._ids(1), theta, BOX, em=True, mle=False
        )
        np.testing.assert_allclose(
            em["lpx"] / no_em["lpx"], np.sqrt(2.0), rtol=1e-5
        )


# ---------------------------------------------------------------------------
# Cubic-spline PSF fitting (Gpufit SPLINE_2D / SPLINE_3D)
# ---------------------------------------------------------------------------


def _fake_spline_calibration(model="spline-3d", box=BOX, n_channels=2):
    """Build a small, structurally valid spline calibration dict. The
    coefficient values are arbitrary - these tests exercise the packing,
    parameter mapping and I/O, not a real fit."""
    if model == "spline-2d":
        n_intervals = [box - 1, box - 1]
        n_data = [box, box]
        n_coef = 16
        coefficients = np.arange(
            n_coef * np.prod(n_intervals), dtype=np.float32
        ).reshape([n_coef] + n_intervals)
    elif model == "spline-3d-multichannel":
        nz = 21
        n_intervals = [box - 1, box - 1, nz - 1]
        n_data = [box, box, nz]
        coefficients = np.arange(
            64 * np.prod(n_intervals) * n_channels, dtype=np.float32
        ).reshape([64] + n_intervals + [n_channels])
    else:
        nz = 21
        n_intervals = [box - 1, box - 1, nz - 1]
        n_data = [box, box, nz]
        n_coef = 64
        coefficients = np.arange(
            n_coef * np.prod(n_intervals), dtype=np.float32
        ).reshape([n_coef] + n_intervals)
    calib = {
        "model": model,
        "coefficients": coefficients,
        "n_data": n_data,
        "n_intervals": n_intervals,
        "oversampling": 1.0,
        "z_center": 10.0,
        "z_step_nm": 20.0,
        "effective_sigma": 1.2,
        "photon_scale": 1.0,
        "box": box,
        "pixelsize": PIXELSIZE,
        "Path": "test_calibration",
    }
    if model == "spline-3d-multichannel":
        calib["n_channels"] = n_channels
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        calib["channel_transforms"] = [identity for _ in range(n_channels)]
    return calib


# ---------------------------------------------------------------------------
# Known separable Gaussian spline for CRLB tests. Built with scipy CubicSpline
# so the exact model Phi = gx(x) gy(y) [gz(z)] and its analytic derivatives are
# known, giving a closed-form reference for _spline_crlb / _spline_model_and_grad
# WITHOUT the compiled Gpuspline library. sx != sy makes it astigmatic
# (lpx != lpy); gz encodes axial information (finite lpz). The coefficient table
# is written in the raw Gpuspline-binding layout the evaluator/kernels expect.
# ---------------------------------------------------------------------------
def _gauss_spline_1d(sigma, center, n):
    x = np.arange(n, dtype=np.float64)
    return CubicSpline(x, np.exp(-0.5 * ((x - center) / sigma) ** 2))


def _gauss_spline_calibration(
    model="spline-3d",
    box=BOX,
    nz=21,
    sx=1.0,
    sy=1.4,
    sz=3.0,
    n_channels=2,
    photon_scale=1.0,
):
    """Calibration dict for a separable Gaussian spline, plus the reference 1D
    splines ``(gx, gy, gz)`` (``gz`` None for 2D) for the closed-form CRLB."""
    cxy = (box - 1) / 2.0
    gx = _gauss_spline_1d(sx, cxy, box)
    gy = _gauss_spline_1d(sy, cxy, box)
    nix = niy = box - 1
    # per-interval coeffs, ascending powers: c[i, p] = spline.c[3 - p, i]
    cx = gx.c[::-1].T
    cy = gy.c[::-1].T
    calib = {
        "model": model,
        "oversampling": 1.0,
        "photon_scale": photon_scale,
        "box": box,
        "pixelsize": PIXELSIZE,
    }
    if model == "spline-2d":
        # (niy, nix, yp, xp) -> raw (16, nix, niy)
        c = np.einsum("yY,xX->yxYX", cy, cx).reshape(16, nix, niy)
        calib.update(coefficients=c.astype(np.float32), n_data=[box, box])
        return calib, (gx, gy, None)
    gz = _gauss_spline_1d(sz, (nz - 1) / 2.0, nz)
    niz = nz - 1
    cz = gz.c[::-1].T
    # (niz, niy, nix, zp, yp, xp) -> raw (64, nix, niy, niz)
    c = np.einsum("zZ,yY,xX->zyxZYX", cz, cy, cx).reshape(64, nix, niy, niz)
    if model == "spline-3d-multichannel":
        c = np.repeat(c[..., None], n_channels, axis=-1)
        calib["n_channels"] = n_channels
    calib.update(
        coefficients=c.astype(np.float32),
        n_data=[box, box, nz],
        z_center=(nz - 1) / 2.0,
        z_step_nm=20.0,
        magnification_factor=1.0,
    )
    return calib, (gx, gy, gz)


def _ref_model_grad(splines, box, x_shift, y_shift, z_eval):
    """Reference Phi and native x/y/z derivatives on the box grid, ``(M, box,
    box)`` indexed ``[loc, x-pixel, y-pixel]`` (``z_eval`` None for 2D)."""
    gx, gy, gz = splines
    gi = np.arange(box)
    xc = gi[None, :] - np.asarray(x_shift, float)[:, None]
    yc = gi[None, :] - np.asarray(y_shift, float)[:, None]
    Gx = gx(xc)[:, :, None]
    Gy = gy(yc)[:, None, :]
    dGx = gx.derivative()(xc)[:, :, None]
    dGy = gy.derivative()(yc)[:, None, :]
    if gz is None:
        return Gx * Gy, dGx * Gy, Gx * dGy, None
    z = np.asarray(z_eval, float)[:, None, None]
    Gz = gz(z)
    dGz = gz.derivative()(z)
    return Gx * Gy * Gz, dGx * Gy * Gz, Gx * dGy * Gz, Gx * Gy * dGz


def _ref_crlb(splines, box, amp, xs, ys, ze, off):
    """Closed-form Poisson CRLB variances ``[x, y, (z,) amp, off]`` for one
    localization of the separable Gaussian spline."""
    gz = splines[2]
    phi, dx, dy, dz = _ref_model_grad(
        splines, box, [xs], [ys], None if gz is None else [ze]
    )
    cols = [amp * dx, amp * dy]
    if gz is not None:
        cols.append(amp * dz)
    cols += [phi, np.ones_like(phi)]
    deriv = np.stack([c.reshape(-1) for c in cols], axis=1)
    mu = (off + amp * phi).reshape(-1)
    weight = 1.0 / np.maximum(mu, localize._SPLINE_CRLB_MU_FLOOR)
    return np.diag(np.linalg.pinv((deriv * weight[:, None]).T @ deriv))


def _ref_crlb_lsq(splines, box, amp, xs, ys, ze, off):
    """Closed-form unweighted-least-squares sandwich covariance ``J⁻¹ M J⁻¹``
    (Poisson pixel variance ``σ² = μ``) ``[x, y, (z,) amp, off]`` for one
    localization of the separable Gaussian spline."""
    gz = splines[2]
    phi, dx, dy, dz = _ref_model_grad(
        splines, box, [xs], [ys], None if gz is None else [ze]
    )
    cols = [amp * dx, amp * dy]
    if gz is not None:
        cols.append(amp * dz)
    cols += [phi, np.ones_like(phi)]
    deriv = np.stack([c.reshape(-1) for c in cols], axis=1)
    mu = np.maximum(
        (off + amp * phi).reshape(-1), localize._SPLINE_CRLB_MU_FLOOR
    )
    j = deriv.T @ deriv  # Σ g gᵀ (Gauss-Newton normal matrix)
    m = (deriv * mu[:, None]).T @ deriv  # Σ μ g gᵀ (sandwich meat)
    j_inv = np.linalg.pinv(j)
    return np.diag(j_inv @ m @ j_inv)


class TestCropSplineCalibration:
    """``localize.crop_spline_calibration`` — fitting a smaller, PSF-centered
    box against a larger calibration (no GPU needed).

    The core check evaluates the coefficients through ``_spline_coeff_reshaped``,
    the exact physical layout both real consumers read (Gpufit's ``user_info``
    reorder and the CRLB kernel), and asserts the crop equals the *central*
    lateral intervals of the full calibration — i.e. the crop selects the true
    piecewise-cubic pieces of the calibrated PSF (a faithful central slice), not
    a re-interpolation."""

    @staticmethod
    def _central(reshaped, off, ni, is_2d):
        # _spline_coeff_reshaped -> (n_ch, niy, nix, 4, 4) (2D) or
        # (n_ch, niz, niy, nix, 4, 4, 4) (3D); crop the two lateral axes.
        if is_2d:
            return reshaped[:, off : off + ni, off : off + ni]
        return reshaped[:, :, off : off + ni, off : off + ni]

    @pytest.mark.parametrize(
        "model", ["spline-2d", "spline-3d", "spline-3d-multichannel"]
    )
    def test_crop_matches_central_intervals(self, model):
        cal_box, box = 13, 7
        calib, _ = _gauss_spline_calibration(model=model, box=cal_box)
        cropped = localize.crop_spline_calibration(calib, box)

        # metadata is updated consistently with the smaller grid
        assert cropped["box"] == box
        assert cropped["n_data"][0] == box and cropped["n_data"][1] == box
        assert cropped["n_intervals"][0] == box - 1
        assert cropped["n_intervals"][1] == box - 1

        # the coefficients both consumers read == the full calibration's central
        # lateral intervals (exact selection: no numerical error)
        off, ni = (cal_box - box) // 2, box - 1
        full = localize._spline_coeff_reshaped(calib)
        crop = localize._spline_coeff_reshaped(cropped)
        expected = self._central(full, off, ni, model == "spline-2d")
        assert crop.shape == expected.shape
        np.testing.assert_array_equal(crop, expected)

    def test_z_grid_and_photon_scale_preserved(self):
        calib, _ = _gauss_spline_calibration(
            model="spline-3d", box=13, nz=21, photon_scale=3.7
        )
        cropped = localize.crop_spline_calibration(calib, 7)
        # photon_scale is the full-PSF integral (total photons), not rescaled
        assert cropped["photon_scale"] == 3.7
        # the axial grid is untouched (only the lateral box shrinks): the z
        # data size is preserved and its interval count follows from it
        assert cropped["n_data"][2] == calib["n_data"][2]
        assert cropped["n_intervals"][2] == calib["n_data"][2] - 1

    def test_equal_box_returns_calibration_unchanged(self):
        calib, _ = _gauss_spline_calibration(model="spline-3d", box=BOX)
        assert localize.crop_spline_calibration(calib, BOX) is calib

    def test_larger_box_rejected(self):
        calib, _ = _gauss_spline_calibration(model="spline-3d", box=7)
        with pytest.raises(ValueError, match="no larger than"):
            localize.crop_spline_calibration(calib, 9)

    def test_smaller_box_any_parity_is_cropped(self):
        """A smaller box of the opposite parity is allowed - the crop is then
        off-center by at most half a pixel (a harmless constant shift) - and
        still yields a valid calibration of the requested box."""
        calib, _ = _gauss_spline_calibration(model="spline-3d", box=13)
        cropped = localize.crop_spline_calibration(calib, 8)
        assert cropped["box"] == 8
        assert cropped["n_data"][0] == 8 and cropped["n_data"][1] == 8
        assert (
            cropped["n_intervals"][0] == 7 and cropped["n_intervals"][1] == 7
        )
        # the coefficient grid matches the requested box, and stays consumable
        # by the shared reshape both the fit and the CRLB use
        assert cropped["coefficients"].shape[1:3] == (7, 7)
        reshaped = localize._spline_coeff_reshaped(cropped)
        assert reshaped.shape[2:4] == (7, 7)

    def test_locs_from_fits_auto_crops_smaller_box(self):
        """locs_from_fits_spline handles a smaller box than the calibration by
        cropping internally: passing the full calibration with a small box gives
        exactly the same localizations as passing a pre-cropped one (CPU only,
        no GPU). This confirms the fit/CRLB/reconstruction stay consistent."""
        cal_box, box, n = 13, 7, 4
        full, _ = _gauss_spline_calibration(model="spline-3d", box=cal_box)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 20.0),
                "y": np.full(n, 30.0),
                "net_gradient": np.full(n, 1000.0),
            }
        )
        theta = np.zeros((n, 5), dtype=np.float32)
        theta[:, 0] = 5000.0  # amplitude
        theta[:, 1] = 0.3  # x_shift
        theta[:, 2] = -0.4  # y_shift
        theta[:, 3] = -full["z_center"] + 2.0  # z_shift
        theta[:, 4] = 12.0  # offset

        auto = localize.locs_from_fits_spline(ids, theta, box, False, full)
        pre_cal = localize.crop_spline_calibration(full, box)
        pre = localize.locs_from_fits_spline(ids, theta, box, False, pre_cal)

        for col in ("x", "y", "z", "photons", "bg", "lpx", "lpy", "lpz"):
            np.testing.assert_array_equal(
                auto[col].to_numpy(), pre[col].to_numpy()
            )
        # the CRLB columns are real (finite, positive) at the smaller box
        for col in ("lpx", "lpy", "lpz"):
            assert np.all(np.isfinite(auto[col])) and np.all(auto[col] > 0)


class TestSplineCalibrationIO:
    """Round-trip of the spline PSF calibration through HDF5 (no GPU)."""

    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_save_load_roundtrip(self, tmp_path, model):
        calib = _fake_spline_calibration(model=model)
        path = str(tmp_path / "psf_spline_calib.hdf5")
        io.save_spline_calibration(path, calib)
        loaded = io.load_spline_calibration(path)

        assert loaded["model"] == model
        np.testing.assert_array_equal(
            loaded["coefficients"], calib["coefficients"]
        )
        assert loaded["coefficients"].dtype == np.float32
        assert list(loaded["n_data"]) == list(calib["n_data"])
        assert list(loaded["n_intervals"]) == list(calib["n_intervals"])
        assert loaded["z_step_nm"] == calib["z_step_nm"]
        assert loaded["Path"] == "test_calibration"

    def test_save_requires_coefficients(self, tmp_path):
        with pytest.raises(ValueError):
            io.save_spline_calibration(
                str(tmp_path / "bad.hdf5"), {"model": "spline-3d"}
            )

    def test_load_rejects_non_spline_file(self, tmp_path):
        path = str(tmp_path / "not_a_spline.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset("locs", data=np.zeros(3))
        with pytest.raises(ValueError):
            io.load_spline_calibration(path)


class TestSplineHelpers:
    """Pure-logic tests for the spline backend (run without a GPU)."""

    def test_model_id_mapping(self):
        if not localize.GPUFIT_INSTALLED:
            pytest.skip("ModelID enum needs the Gpufit binding")
        assert localize._spline_model_id("spline-2d") == (
            localize.gf.ModelID.SPLINE_2D
        )
        assert localize._spline_model_id("spline-3d") == (
            localize.gf.ModelID.SPLINE_3D
        )
        with pytest.raises(ValueError):
            localize._spline_model_id("nonsense")

    def test_pack_user_info_3d_layout(self):
        calib = _fake_spline_calibration(model="spline-3d")
        user_info = localize._pack_spline_user_info(calib)
        # dtype must be float32 (matches single-precision Gpufit build)
        assert user_info.dtype == np.float32
        nx, ny, _ = calib["n_data"]
        ix, iy, iz = calib["n_intervals"]
        # header: [n_data_x, n_data_y, n_data_z=1, n_int_x, n_int_y, n_int_z]
        np.testing.assert_array_equal(
            user_info[:6], np.array([nx, ny, 1, ix, iy, iz], np.float32)
        )
        expected_len = 6 + calib["coefficients"].size
        assert user_info.size == expected_len
        # The coefficient block is REORDERED into Gpufit's forward axis order
        # (see _reorder_spline_coefficients_for_gpufit) - not the raw
        # Gpuspline-binding C-order ravel, which is the layout the axis-packing
        # bug shipped. It must be a permutation of the same values...
        coeff = calib["coefficients"]
        np.testing.assert_array_equal(
            np.sort(user_info[6:]), np.sort(coeff.ravel(order="C"))
        )
        # ...matching the reorder helper exactly...
        np.testing.assert_array_equal(
            user_info[6:],
            localize._reorder_spline_coefficients_for_gpufit(
                coeff, "spline-3d"
            ),
        )
        # ...and NOT the raw forward ravel (guards against a regression to the
        # un-reordered packing that made Gpufit read scrambled coefficients).
        assert not np.array_equal(user_info[6:], coeff.ravel(order="C"))

    def test_pack_user_info_2d_layout(self):
        calib = _fake_spline_calibration(model="spline-2d")
        user_info = localize._pack_spline_user_info(calib)
        nx, ny = calib["n_data"]
        ix, iy = calib["n_intervals"]
        np.testing.assert_array_equal(
            user_info[:4], np.array([nx, ny, ix, iy], np.float32)
        )
        assert user_info.size == 4 + calib["coefficients"].size
        # coefficient block is reordered into Gpufit's forward axis order
        np.testing.assert_array_equal(
            user_info[4:],
            localize._reorder_spline_coefficients_for_gpufit(
                calib["coefficients"], "spline-2d"
            ),
        )

    def test_initial_parameters_shape(self, synthetic_spots):
        spots, _ = synthetic_spots
        calib_3d = _fake_spline_calibration(model="spline-3d")
        init_3d = localize._initial_parameters_spline(spots, calib_3d)
        assert init_3d.shape == (len(spots), 5)
        assert init_3d.dtype == np.float32
        # z_shift is initialised to -z_init (the in-focus slice), so that the
        # Gpufit model's native z = -z_shift starts at the focus (see
        # _initial_parameters_spline / spline.calibrate_spline). z_init defaults
        # to z_center for this calibration.
        np.testing.assert_allclose(init_3d[:, 3], -calib_3d["z_center"])

        calib_2d = _fake_spline_calibration(model="spline-2d")
        init_2d = localize._initial_parameters_spline(spots, calib_2d)
        assert init_2d.shape == (len(spots), 4)

    def test_locs_from_fits_spline_3d(self):
        # a realistic (Gaussian) calibration so the CRLB columns are meaningful
        calib, _ = _gauss_spline_calibration(model="spline-3d")
        n = 5
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 20.0),
                "y": np.full(n, 30.0),
                "net_gradient": np.full(n, 1000.0),
            }
        )
        # theta: [amplitude, x_shift, y_shift, z_shift, offset]
        theta = np.zeros((n, 5), dtype=np.float32)
        theta[:, 0] = 5000.0  # amplitude
        theta[:, 1] = 0.5  # x_shift
        theta[:, 2] = -0.5  # y_shift
        # z_shift is initialized to -z_center (focus); 2 slices past focus
        theta[:, 3] = -calib["z_center"] + 2.0
        theta[:, 4] = 12.0  # offset (bg)

        locs = localize.locs_from_fits_spline(ids, theta, BOX, False, calib)

        box_offset = int(BOX / 2)
        center = (BOX - 1) / 2.0
        np.testing.assert_allclose(locs["x"], 0.5 + center + 20.0 - box_offset)
        np.testing.assert_allclose(
            locs["y"], -0.5 + center + 30.0 - box_offset
        )
        np.testing.assert_allclose(locs["photons"], 5000.0)
        np.testing.assert_allclose(locs["bg"], 12.0)
        # z follows the astigmatism (zfit / z_of_step) convention - it rises
        # with stage z - and is scaled by the magnification factor (default 1.0
        # when absent):
        # z = (z_shift + z_center) * z_step_nm = (2) * 20 = 40 nm
        np.testing.assert_allclose(locs["z"], 40.0)
        # lpx/lpy/lpz are now real Cramer-Rao bounds (finite, positive), not
        # the former 0.01 placeholder, and the CRLB uncertainty columns exist.
        for col in ("lpx", "lpy", "lpz", "photons_unc", "bg_unc"):
            assert col in locs.columns
            assert np.all(np.isfinite(locs[col])) and np.all(locs[col] > 0)

    def test_locs_from_fits_spline_2d_has_no_z(self):
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        n = 3
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 10.0),
                "y": np.full(n, 10.0),
                "net_gradient": np.full(n, 500.0),
            }
        )
        theta = np.zeros((n, 4), dtype=np.float32)
        theta[:, 0] = 3000.0
        theta[:, 3] = 8.0
        locs = localize.locs_from_fits_spline(ids, theta, BOX, False, calib)
        assert "z" not in locs.columns
        assert "lpz" not in locs.columns
        np.testing.assert_allclose(locs["photons"], 3000.0)
        # 2D still reports lateral + photon/bg CRLB uncertainties
        for col in ("lpx", "lpy", "photons_unc", "bg_unc"):
            assert col in locs.columns
            assert np.all(np.isfinite(locs[col])) and np.all(locs[col] > 0)

    @pytest.mark.skipif(
        localize.GPUFIT_INSTALLED,
        reason="ImportError only raised when Gpufit is unavailable",
    )
    def test_fit_spots_spline_without_gpu_raises(self, synthetic_spots):
        spots, _ = synthetic_spots
        calib = _fake_spline_calibration(model="spline-3d")
        with pytest.raises(ImportError):
            localize.fit_spots_gpufit_spline(spots, calib)

    def test_affine_transform_roundtrip(self):
        src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [5.0, 7.0]])
        m_true = np.array([[1.02, -0.03, 3.0], [0.01, 0.98, -2.0]])
        dst = localize.apply_affine_transform(src, m_true)
        m_est = localize.estimate_affine_transform(src, dst)
        np.testing.assert_allclose(m_est, m_true, atol=1e-9)

    def test_estimate_affine_needs_three_points(self):
        with pytest.raises(ValueError):
            localize.estimate_affine_transform(
                np.zeros((2, 2)), np.zeros((2, 2))
            )

    def test_pack_user_info_multichannel_layout(self):
        n_channels = 3
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        user_info = localize._pack_spline_user_info(calib)
        assert user_info.dtype == np.float32
        nx, ny, _ = calib["n_data"]
        ix, iy, iz = calib["n_intervals"]
        # header: [n_channels, nx, ny, nz=1, ix, iy, iz]
        np.testing.assert_array_equal(
            user_info[:7],
            np.array([n_channels, nx, ny, 1, ix, iy, iz], np.float32),
        )
        assert user_info.size == 7 + calib["coefficients"].size
        # each channel's block is reordered to forward axis order and the
        # blocks are concatenated channel-major (outermost axis)
        np.testing.assert_array_equal(
            user_info[7:],
            localize._reorder_spline_coefficients_for_gpufit(
                calib["coefficients"], "spline-3d-multichannel"
            ),
        )

    def test_initial_parameters_multichannel_stacked(self):
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        # channel-stacked spots (n, box, box, n_channels)
        spots = (
            np.random.default_rng(0)
            .random((4, BOX, BOX, 2), dtype=np.float64)
            .astype(np.float32)
        )
        init = localize._initial_parameters_spline(spots, calib)
        assert init.shape == (4, 5)
        # z_shift initialised to -z_init (= -z_center here); see the
        # single-channel test above.
        np.testing.assert_allclose(init[:, 3], -calib["z_center"])

    def test_get_spots_multichannel_identity(
        self, movie, real_identifications
    ):
        single = localize.get_spots(
            movie, real_identifications, BOX, CAMERA_INFO
        )
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        stacked = localize.get_spots_multichannel(
            [movie, movie],
            real_identifications,
            BOX,
            [CAMERA_INFO, CAMERA_INFO],
            [identity, identity],
        )
        assert stacked.shape == (len(real_identifications), BOX, BOX, 2)
        # identity transform => each channel equals the single-movie extraction
        np.testing.assert_array_equal(stacked[..., 0], single)
        np.testing.assert_array_equal(stacked[..., 1], single)


def _synthetic_spline_3d_calibration(box=BOX, nz=41):
    """Build a 3D spline calibration from a synthetic astigmatic PSF,
    mirroring the reference pyGpufit splinefit_3d example. Requires Gpuspline
    (CPU) to compute the coefficients; returns (calibration, template,
    amplitude, offset)."""
    gs = localize.gs
    x = np.arange(box, dtype=np.float32)
    y = np.arange(box, dtype=np.float32)
    amplitude, offset = 100.0, 10.0
    s0 = box / 6.0
    template = np.zeros((box, box, nz), dtype=np.float32)
    for k in range(nz):
        sx = s0 * (1.0 + 0.4 * (k - nz / 2) / nz)
        sy = s0 * (1.0 - 0.4 * (k - nz / 2) / nz)
        gx = np.exp(-0.5 * ((x - (box - 1) / 2) / sx) ** 2)
        gy = np.exp(-0.5 * ((y - (box - 1) / 2) / sy) ** 2)
        template[:, :, k] = np.outer(gx, gy)
    coefficients = gs.spline_coefficients(template)
    n_intervals = np.array(template.shape) - 1
    coefficients = np.reshape(
        coefficients,
        (64, n_intervals[0], n_intervals[1], n_intervals[2]),
    )
    calib = {
        "model": "spline-3d",
        "coefficients": coefficients.astype(np.float32),
        "n_data": [box, box, nz],
        "n_intervals": [int(i) for i in n_intervals],
        "oversampling": 1.0,
        "z_center": (nz - 1) / 2,
        "z_step_nm": 20.0,
        "effective_sigma": s0,
        "photon_scale": 1.0,
        "box": box,
        "pixelsize": PIXELSIZE,
        "Path": "synthetic",
    }
    return calib, template, amplitude, offset


@pytest.mark.skipif(
    not localize.GPUSPLINE_INSTALLED,
    reason="Gpuspline not available",
)
class TestSplineCoefficients:
    """Coefficient computation/evaluation with Gpuspline. This is a plain CPU
    library (no GPU/CUDA), so these run wherever the compiled splines library
    is present - independently of Gpufit."""

    def test_spline_coefficients_roundtrip(self):
        """spline_values on the computed coefficients must reproduce the
        source template (validates coefficient layout / flatten order)."""
        gs = localize.gs
        calib, template, _, _ = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        x = np.arange(box, dtype=np.float32)
        y = np.arange(box, dtype=np.float32)
        z = np.arange(nz, dtype=np.float32)
        # spline_values reads coefficients.shape[3], [2], [1]; it needs the full
        # (64, nix, niy, niz) table - a reshape to (64, -1) raises IndexError.
        # This is exactly what _spline_crlb passes.
        values = gs.spline_values(calib["coefficients"], x, y, z)
        np.testing.assert_allclose(values, template, atol=1e-3)

    def test_spline_crlb_real_coefficients(self):
        """_spline_crlb runs on a real Gpuspline calibration and yields finite,
        positive precisions (exercises the actual spline_values path, not the
        analytic fake used by TestSplineCRLB)."""
        calib, _, amplitude, offset = _synthetic_spline_3d_calibration()
        box = calib["n_data"][0]
        z_focus = calib["z_center"]
        # a centred molecule a few slices below focus
        theta = np.array(
            [[amplitude, 0.0, 0.0, -(z_focus - 5.0), offset]], np.float64
        )
        crlb = localize._spline_crlb(theta, calib, box)[0]
        assert np.all(np.isfinite(crlb)) and np.all(crlb > 0)

    def test_model_and_grad_matches_gpuspline(self):
        """Authoritative layout check: _spline_model_and_grad's value (which the
        numba kernel mirrors) must equal gpuspline.spline_values at sub-pixel
        shifts on a real calibration. The local tests use a self-built spline,
        so they cannot catch a coefficient-layout mismatch - this one can."""
        calib, _, _, _ = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        rng = np.random.default_rng(1)
        m = 12
        xs = rng.uniform(-0.5, 0.5, m)
        ys = rng.uniform(-0.5, 0.5, m)
        ze = rng.uniform(5, nz - 6, m)
        phi, _, _, _ = localize._spline_model_and_grad(
            calib["coefficients"], box, xs, ys, ze
        )
        grid = np.arange(box, dtype=np.float32)
        for k in range(m):
            # gs.spline_values is indexed [x-pixel, y-pixel] like phi[k]
            gsv = localize.gs.spline_values(
                calib["coefficients"],
                grid - np.float32(xs[k]),
                grid - np.float32(ys[k]),
                np.array([ze[k]], np.float32),
            )[:, :, 0]
            np.testing.assert_allclose(phi[k], gsv, atol=1e-3)


def _synthetic_spline_2d_calibration(box=13, sigma=1.4):
    """Build a 2D (16-coefficient) spline calibration from a single isotropic
    Gaussian slice, using Gpuspline (CPU). Isotropic -> swap-invariant in x/y,
    so recovery assertions are convention-agnostic. Returns
    ``(calibration, amplitude, offset)``."""
    gs = localize.gs
    x = np.arange(box, dtype=np.float32)
    g = np.exp(-0.5 * ((x - (box - 1) / 2) / sigma) ** 2)
    template = np.outer(g, g).astype(np.float32)
    n_intervals = np.array(template.shape) - 1
    coefficients = np.reshape(
        gs.spline_coefficients(template),
        (16, n_intervals[0], n_intervals[1]),
    ).astype(np.float32)
    calib = {
        "model": "spline-2d",
        "coefficients": coefficients,
        "n_data": [box, box],
        "n_intervals": [int(i) for i in n_intervals],
        "oversampling": 1.0,
        "z_center": 0.0,
        "z_step_nm": 20.0,
        "effective_sigma": sigma,
        "photon_scale": 1.0,
        "box": box,
        "pixelsize": PIXELSIZE,
        "Path": "synthetic-2d",
    }
    return calib, 100.0, 10.0


@pytest.mark.skipif(
    not (localize.GPUFIT_INSTALLED and localize.GPUSPLINE_INSTALLED),
    reason="Gpufit (CUDA GPU) + Gpuspline not available",
)
class TestSplineGpufit:
    """End-to-end spline fitting. The fit itself runs on Gpufit (CUDA GPU);
    the calibration is built with Gpuspline (CPU). Skipped in the typical
    (GPU-less) test environment.

    Spline theta is ``[amplitude, x_shift, y_shift, z_shift, offset]`` for 3D
    and ``[amplitude, x_shift, y_shift, offset]`` for 2D. The exact x/y and z
    conventions of a manually built calibration are subtle (the astigmatic
    PSF couples an x<->y swap with a z mirror away from focus), so recovery
    assertions here stay convention-agnostic: amplitude/offset, symmetric
    (diagonal) sub-pixel shifts, monotonic z, LSE/MLE agreement, and model
    round-trip residuals."""

    def test_fit_spots_spline_3d(self):
        calib, template, amplitude, offset = _synthetic_spline_3d_calibration()
        z_slice = int(calib["z_center"])
        spot = (amplitude * template[:, :, z_slice] + offset).astype(
            np.float32
        )
        spots = np.stack([spot] * 4)
        theta = localize.fit_spots_gpufit_spline(spots, calib)
        assert theta.shape == (len(spots), 5)
        # The fitted z_shift maps to the taken native slice with magnitude
        # ~= z_slice; the sign encodes the native_z = -z_shift convention
        # (see test_spline_crlb_native_z_reconstruction), so we check the
        # magnitude here to stay convention-agnostic.
        np.testing.assert_allclose(np.abs(theta[:, 3]), z_slice, atol=2.0)

    def test_spline_crlb_native_z_reconstruction(self):
        """Definitive check of the native_z = -z_shift convention that
        ``_spline_crlb`` relies on for lpz: re-evaluating the spline at
        ``-z_shift`` must reproduce the fitted spot better than at ``+z_shift``.
        """
        calib, template, amplitude, offset = _synthetic_spline_3d_calibration()
        z_slice = int(calib["z_center"])
        spot = (amplitude * template[:, :, z_slice] + offset).astype(
            np.float32
        )
        amp, xs, ys, zs, off = localize.fit_spots_gpufit_spline(
            np.stack([spot]), calib
        )[0]
        box, _, nz = calib["n_data"]
        grid = np.arange(box, dtype=np.float32)

        def model(native_z):
            zc = np.array([np.clip(native_z, 0.0, nz - 1)], np.float32)
            phi = localize.gs.spline_values(
                calib["coefficients"],
                grid - np.float32(xs),
                grid - np.float32(ys),
                zc,
            )[:, :, 0]
            return off + amp * phi

        res_minus = float(np.mean((model(-zs) - spot) ** 2))
        res_plus = float(np.mean((model(+zs) - spot) ** 2))
        assert res_minus < res_plus

    def test_fit_spots_spline_box_mismatch(self):
        calib, _, _, _ = _synthetic_spline_3d_calibration(box=BOX)
        wrong_box = BOX + 2
        spots = np.zeros((2, wrong_box, wrong_box), dtype=np.float32)
        with pytest.raises(ValueError):
            localize.fit_spots_gpufit_spline(spots, calib)

    def test_spline_3d_recovers_amplitude_offset_at_focus(self):
        """A centered in-focus spot recovers its amplitude, offset and (near)
        zero lateral shift. At focus the PSF is ~isotropic, so the lateral
        recovery is convention-agnostic."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        z_slice = int(calib["z_center"])
        spot = (amp * template[:, :, z_slice] + off).astype(np.float32)
        theta = localize.fit_spots_gpufit_spline(np.stack([spot] * 3), calib)
        np.testing.assert_allclose(theta[:, 0], amp, rtol=1e-3)  # amplitude
        np.testing.assert_allclose(theta[:, 4], off, atol=1e-2)  # offset
        np.testing.assert_allclose(theta[:, 1], 0.0, atol=1e-2)  # x_shift
        np.testing.assert_allclose(theta[:, 2], 0.0, atol=1e-2)  # y_shift

    def test_spline_3d_recovers_diagonal_subpixel_shift(self):
        """Sub-pixel shifts recover exactly. Uses symmetric (dx == dy) shifts
        so the result is invariant to the x<->y axis convention."""
        calib, _, amp, off = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        z_focus = np.float32(calib["z_center"])
        grid = np.arange(box, dtype=np.float32)
        shifts = [0.0, 0.3, -0.4, 0.45]
        spots = []
        for d in shifts:
            phi = localize.gs.spline_values(
                calib["coefficients"],
                grid - np.float32(d),
                grid - np.float32(d),
                np.array([z_focus], np.float32),
            )[:, :, 0]
            spots.append((off + amp * phi).astype(np.float32))
        theta = localize.fit_spots_gpufit_spline(np.stack(spots), calib)
        np.testing.assert_allclose(theta[:, 1], shifts, atol=5e-3)
        np.testing.assert_allclose(theta[:, 2], shifts, atol=5e-3)

    def test_spline_3d_native_z_monotonic(self):
        """The recovered native z (= -z_shift) advances monotonically as the
        input is taken from successive slices of the calibration stack -
        convention-agnostic evidence that the axial fit tracks defocus."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        slices = [12, 16, 20, 24, 28]
        spots = np.stack(
            [
                (amp * template[:, :, k] + off).astype(np.float32)
                for k in slices
            ]
        )
        theta = localize.fit_spots_gpufit_spline(spots, calib)
        native_z = -theta[:, 3]
        diffs = np.diff(native_z)
        # strictly monotonic (all steps share one sign)
        assert np.all(diffs > 0) or np.all(diffs < 0)

    def test_spline_3d_mle_agrees_with_lse(self):
        """Unlike the Gaussian MLE, the spline MLE estimator converges cleanly
        here; its amplitude/offset must agree with the least-squares fit."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        slices = [14, 20, 26]
        spots = np.stack(
            [
                (amp * template[:, :, k] + off).astype(np.float32)
                for k in slices
            ]
        )
        lse = localize.fit_spots_gpufit_spline(spots, calib, mle=False)
        mle = localize.fit_spots_gpufit_spline(spots, calib, mle=True)
        np.testing.assert_allclose(
            mle[:, 0], lse[:, 0], rtol=1e-3
        )  # amplitude
        np.testing.assert_allclose(mle[:, 4], lse[:, 4], atol=1e-2)  # offset
        np.testing.assert_allclose(mle[:, 3], lse[:, 3], atol=0.2)  # z_shift

    def test_spline_3d_roundtrip_reproduces_focus_spot(self):
        """Convention-independent correctness check: re-evaluating the spline
        at the fitted parameters reproduces the input spot to the noise
        floor."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        z_slice = int(calib["z_center"])
        spot = (amp * template[:, :, z_slice] + off).astype(np.float32)
        a, xs, ys, zs, o = localize.fit_spots_gpufit_spline(
            np.stack([spot]), calib
        )[0]
        grid = np.arange(box, dtype=np.float32)
        phi = localize.gs.spline_values(
            calib["coefficients"],
            grid - np.float32(xs),
            grid - np.float32(ys),
            np.array([np.clip(-zs, 0, nz - 1)], np.float32),
        )[:, :, 0]
        model = o + a * phi
        rms = np.sqrt(np.mean((model - spot) ** 2))
        assert rms < 0.5  # amplitude is 100 -> < 0.5% of peak

    def test_spline_2d_recovers_amplitude_offset_shift(self):
        """The 2D spline model (16 coefficients, no z) recovers amplitude,
        offset and symmetric sub-pixel shift."""
        calib, amp, off = _synthetic_spline_2d_calibration()
        box = calib["n_data"][0]
        grid = np.arange(box, dtype=np.float32)
        shifts = [0.0, 0.3, -0.35]
        spots = []
        for d in shifts:
            phi = localize.gs.spline_values(
                calib["coefficients"],
                grid - np.float32(d),
                grid - np.float32(d),
            )
            phi = np.asarray(phi)
            if phi.ndim == 3:
                phi = phi[:, :, 0]
            spots.append((off + amp * phi).astype(np.float32))
        theta = localize.fit_spots_gpufit_spline(np.stack(spots), calib)
        assert theta.shape == (len(shifts), 4)  # [amp, x_shift, y_shift, off]
        np.testing.assert_allclose(theta[:, 0], amp, rtol=1e-3)
        np.testing.assert_allclose(theta[:, 3], off, atol=1e-2)
        np.testing.assert_allclose(theta[:, 1], shifts, atol=5e-3)
        np.testing.assert_allclose(theta[:, 2], shifts, atol=5e-3)

    def test_spline_3d_locs_end_to_end(self):
        """fit_spots_gpufit_spline + locs_from_fits_spline yields a valid
        localizations frame with finite, positive CRLB precisions and a z
        column for the 3D model."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        n = 4
        z_slice = int(calib["z_center"])
        spot = (amp * template[:, :, z_slice] + off).astype(np.float32)
        spots = np.stack([spot] * n)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 40.0),
                "y": np.full(n, 60.0),
                "net_gradient": np.full(n, 1000.0),
            }
        )
        theta = localize.fit_spots_gpufit_spline(spots, calib)
        box = calib["n_data"][0]
        locs = localize.locs_from_fits_spline(ids, theta, box, False, calib)
        assert len(locs) == n
        for col in ("x", "y", "z", "photons", "bg", "lpx", "lpy", "lpz"):
            assert col in locs.columns
        for col in ("lpx", "lpy", "lpz"):
            assert np.all(np.isfinite(locs[col])) and (locs[col] > 0).all()
        np.testing.assert_allclose(locs["photons"], amp, rtol=1e-3)


@pytest.mark.skipif(
    not localize.GPUFIT_INSTALLED, reason="GPUfit/CUDA not available"
)
class TestFit2DGpu:
    """End-to-end ``localize.fit2D`` through every GPU fitting method, driven by
    the bundled movie and its real identifications. Verifies the high-level
    dispatch, spot extraction, GPU fit and localization assembly hang together
    and produce a saveable localizations frame."""

    CAMERA_INFO = {**CAMERA_INFO, "Pixelsize": PIXELSIZE}

    @pytest.mark.parametrize(
        "method,has_angle",
        [
            ("gausslq-gpu", False),
            ("gaussmle-gpu", False),
            ("gausslq-rotated-gpu", True),
            ("gaussmle-rotated-gpu", True),
        ],
    )
    def test_gauss_gpu_methods(
        self,
        picasso_movie,
        movie_info,
        real_identifications,
        method,
        has_angle,
    ):
        locs, info = localize.fit2D(
            picasso_movie,
            movie_info,
            self.CAMERA_INFO,
            real_identifications,
            BOX,
            fitting_method=method,
        )
        assert len(locs) == len(real_identifications)
        for col in (
            "frame",
            "x",
            "y",
            "photons",
            "sx",
            "sy",
            "bg",
            "lpx",
            "lpy",
        ):
            assert col in locs.columns
        assert ("angle" in locs.columns) == has_angle
        assert info["Fit method"] == method
        # MLE methods attach per-parameter uncertainties + fit diagnostics
        if method.startswith("gaussmle"):
            for col in (
                "photons_unc",
                "bg_unc",
                "log_likelihood",
                "iterations",
            ):
                assert col in locs.columns

    @pytest.mark.parametrize("method", ["spline-gpu", "spline-mle-gpu"])
    def test_spline_gpu_methods(
        self, picasso_movie, movie_info, real_identifications, method
    ):
        if not localize.GPUSPLINE_INSTALLED:
            pytest.skip("Gpuspline needed to build the calibration")
        calib, _, _, _ = _synthetic_spline_3d_calibration(box=BOX)
        locs, info = localize.fit2D(
            picasso_movie,
            movie_info,
            self.CAMERA_INFO,
            real_identifications,
            BOX,
            fitting_method=method,
            spline_calibration=calib,
        )
        assert len(locs) == len(real_identifications)
        for col in (
            "frame",
            "x",
            "y",
            "z",
            "photons",
            "bg",
            "lpx",
            "lpy",
            "lpz",
        ):
            assert col in locs.columns
        assert info["Spline calibration model"] == "spline-3d"

    def test_gpu_matches_direct_fit_path(
        self, picasso_movie, movie_info, real_identifications
    ):
        """fit2D('gausslq-gpu') equals calling the spot extraction + GPU fit +
        localization assembly directly - i.e. the wrapper adds no drift."""
        locs, _ = localize.fit2D(
            picasso_movie,
            movie_info,
            self.CAMERA_INFO,
            real_identifications,
            BOX,
            fitting_method="gausslq-gpu",
        )
        spots = localize.get_spots(
            picasso_movie, real_identifications, BOX, self.CAMERA_INFO
        )
        theta = localize.fit_spots_gpufit(spots, mle=False)
        direct = localize.locs_from_fits_gpufit(
            real_identifications, theta, BOX, em=False
        )
        np.testing.assert_allclose(
            locs["x"].to_numpy(), direct["x"].to_numpy(), rtol=1e-5
        )
        np.testing.assert_allclose(
            locs["photons"].to_numpy(), direct["photons"].to_numpy(), rtol=1e-5
        )


@pytest.mark.skipif(
    not localize.GPUSPLINE_INSTALLED,
    reason="Gpuspline not available",
)
class TestSplineCRLBReal:
    """The CRLB path on a real Gpuspline-built calibration (CPU, no GPU fit)."""

    def test_crlb_finite_across_z(self):
        calib, _, amplitude, offset = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        z_focus = calib["z_center"]
        thetas = np.array(
            [
                [amplitude, 0.1, -0.1, -(z_focus - dz), offset]
                for dz in (-6.0, 0.0, 6.0)
            ],
            np.float64,
        )
        crlb = localize._spline_crlb(thetas, calib, box)
        assert np.all(np.isfinite(crlb)) and np.all(crlb > 0)


class TestSplineCRLB:
    """Cramer-Rao lower bounds for spline-fitted localizations. Uses a known
    separable Gaussian spline built with scipy (see _gauss_spline_calibration),
    so the exact CRLB reference is available without the compiled Gpuspline
    library or a GPU. The numba kernel is validated against the closed-form
    reference; the layout-vs-Gpuspline check is in TestSplineCoefficients."""

    def test_evaluator_matches_scipy_3d(self):
        # _spline_model_and_grad (the NumPy reference the numba kernel mirrors)
        # must reproduce the scipy spline value and its x/y/z derivatives.
        calib, splines = _gauss_spline_calibration(model="spline-3d")
        nz = calib["n_data"][2]
        rng = np.random.default_rng(0)
        m = 50
        xs = rng.uniform(-0.7, 0.7, m)
        ys = rng.uniform(-0.7, 0.7, m)
        ze = rng.uniform(5, nz - 6, m)
        phi, dx, dy, dz = localize._spline_model_and_grad(
            calib["coefficients"], BOX, xs, ys, ze
        )
        rphi, rdx, rdy, rdz = _ref_model_grad(splines, BOX, xs, ys, ze)
        assert np.abs(phi - rphi).max() < 1e-4
        assert np.abs(dx - rdx).max() < 1e-3
        assert np.abs(dy - rdy).max() < 1e-3
        assert np.abs(dz - rdz).max() < 1e-3

    def test_crlb_matches_reference_3d(self):
        # sx < sy (astigmatic) so lpx < lpy - also guards the x/y association.
        calib, splines = _gauss_spline_calibration(
            model="spline-3d", sx=1.0, sy=1.4
        )
        # native_z = -z_shift = 6, off the gz focus (=10) so the separable test
        # PSF carries real axial information (dPhi/dz != 0) and lpz is finite.
        amp, off, z_shift = 4000.0, 20.0, -6.0
        theta = np.array([[amp, 0.2, -0.15, z_shift, off]])
        crlb = localize._spline_crlb(theta, calib, BOX)[0]
        ref = _ref_crlb(splines, BOX, amp, 0.2, -0.15, -z_shift, off)
        np.testing.assert_allclose(crlb, ref, rtol=1e-2)
        assert crlb[0] < crlb[1]  # lpx < lpy
        assert np.isfinite(crlb[2]) and crlb[2] > 0  # finite lpz

    def test_crlb_matches_reference_2d(self):
        calib, splines = _gauss_spline_calibration(
            model="spline-2d", sx=1.0, sy=1.3
        )
        amp, off = 3000.0, 15.0
        theta = np.array([[amp, 0.1, -0.1, off]])
        crlb = localize._spline_crlb(theta, calib, BOX)[0]
        ref = _ref_crlb(splines, BOX, amp, 0.1, -0.1, None, off)
        np.testing.assert_allclose(crlb, ref, rtol=1e-2)
        assert crlb[0] < crlb[1]

    def test_lsq_sandwich_matches_reference_3d(self):
        # mle=False must return the unweighted-least-squares sandwich covariance.
        calib, splines = _gauss_spline_calibration(
            model="spline-3d", sx=1.0, sy=1.4
        )
        amp, off, z_shift = 4000.0, 20.0, -6.0
        theta = np.array([[amp, 0.2, -0.15, z_shift, off]])
        var = localize._spline_crlb(theta, calib, BOX, mle=False)[0]
        ref = _ref_crlb_lsq(splines, BOX, amp, 0.2, -0.15, -z_shift, off)
        np.testing.assert_allclose(var, ref, rtol=1e-2)
        assert var[0] < var[1]  # lpx < lpy
        assert np.isfinite(var[2]) and var[2] > 0  # finite lpz

    def test_lsq_sandwich_matches_reference_2d(self):
        calib, splines = _gauss_spline_calibration(
            model="spline-2d", sx=1.0, sy=1.3
        )
        amp, off = 3000.0, 15.0
        theta = np.array([[amp, 0.1, -0.1, off]])
        var = localize._spline_crlb(theta, calib, BOX, mle=False)[0]
        ref = _ref_crlb_lsq(splines, BOX, amp, 0.1, -0.1, None, off)
        np.testing.assert_allclose(var, ref, rtol=1e-2)
        assert var[0] < var[1]

    def test_lsq_variance_geq_crlb(self):
        # Least squares is not efficient for Poisson data: with background the
        # sandwich covariance is strictly above the Cramer-Rao (MLE) bound.
        calib, _ = _gauss_spline_calibration(model="spline-2d", sx=1.0, sy=1.3)
        theta = np.array([[3000.0, 0.1, -0.1, 40.0]])
        crlb = localize._spline_crlb(theta, calib, BOX, mle=True)[0]
        lsq = localize._spline_crlb(theta, calib, BOX, mle=False)[0]
        assert np.all(np.isfinite(lsq))
        # allow tiny numerical slack, then require the x/y positions to be worse
        assert np.all(lsq >= crlb * (1 - 1e-6))
        assert lsq[0] > crlb[0] and lsq[1] > crlb[1]

    def test_lsq_nan_theta_row_isolated(self):
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[3000.0, 0.1, 0.0, 15.0], [np.nan, 0.0, 0.0, 10.0]])
        var = localize._spline_crlb(theta, calib, BOX, mle=False)
        assert np.all(np.isfinite(var[0]))
        assert np.all(np.isnan(var[1]))

    def test_multichannel_sums_fisher(self):
        # Two identical channels double the Fisher information -> half variance.
        calib_1, _ = _gauss_spline_calibration(model="spline-3d")
        calib_2, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        theta = np.array([[5000.0, 0.1, -0.1, -8.0, 20.0]])
        crlb_1 = localize._spline_crlb(theta, calib_1, BOX)[0]
        crlb_2 = localize._spline_crlb(theta, calib_2, BOX)[0]
        np.testing.assert_allclose(crlb_2, crlb_1 / 2.0, rtol=1e-3)

    def test_nan_theta_row_isolated(self):
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[3000.0, 0.1, 0.0, 15.0], [np.nan, 0.0, 0.0, 10.0]])
        crlb = localize._spline_crlb(theta, calib, BOX)
        assert np.all(np.isfinite(crlb[0]))
        assert np.all(np.isnan(crlb[1]))

    def test_low_signal_stays_finite(self):
        # offset = 0 drives some model pixels to ~0; the MU_FLOOR guard keeps
        # the Fisher weight (1 / mu) finite.
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[500.0, 0.0, 0.0, 0.0]])
        assert np.all(np.isfinite(localize._spline_crlb(theta, calib, BOX)[0]))

    def test_progress_callback_and_console(self):
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        n = 250
        rng = np.random.default_rng(0)
        theta = np.zeros((n, 4))
        theta[:, 0] = 3000.0
        theta[:, 1:3] = rng.uniform(-0.5, 0.5, (n, 2))
        theta[:, 3] = 15.0
        seen = []
        localize._spline_crlb(theta, calib, BOX, progress_callback=seen.append)
        assert seen and seen[-1] == n and seen == sorted(seen)
        # the tqdm ("console") path must not raise
        localize._spline_crlb(theta, calib, BOX, progress_callback="console")


def _gauss_model(theta, box, rotated):
    """Point-sampled Gaussian gpufit optimizes, parametrized by total photons N:
    mu(i, j) = N / (2 pi sx sy) * E + bg, i = x (column), j = y (row)."""
    N, x, y, sx, sy, bg = theta[:6]
    grid = np.arange(box, dtype=np.float64)
    dx = grid[:, None] - x
    dy = grid[None, :] - y
    if rotated:
        ct, st = np.cos(theta[6]), np.sin(theta[6])
        u = dx * ct - dy * st
        w = dx * st + dy * ct
        E = np.exp(-0.5 * (u**2 / sx**2 + w**2 / sy**2))
    else:
        E = np.exp(-0.5 * (dx**2 / sx**2 + dy**2 / sy**2))
    return N / (2 * np.pi * sx * sy) * E + bg


def _ref_gauss_crlb(theta, box, rotated, floor=1e-3):
    """Finite-difference Poisson Fisher CRLB reference for _gauss_crlb: builds
    I = sum g gᵀ / mu with numerical gradients g = d mu / d theta and inverts.
    """
    n_params = len(theta)
    mu = np.maximum(_gauss_model(theta, box, rotated), floor)
    g = np.zeros((n_params,) + mu.shape)
    for k in range(n_params):
        h = 1e-6 * max(abs(theta[k]), 1e-3)
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        g[k] = (
            _gauss_model(tp, box, rotated) - _gauss_model(tm, box, rotated)
        ) / (2 * h)
    return np.diag(np.linalg.pinv(np.einsum("pij,qij->pq", g / mu, g)))


class TestGaussCRLB:
    """Poisson Cramer-Rao lower bounds for gpufit MLE Gaussian fits
    (localize._gauss_crlb). The analytic Fisher matrix of the point-sampled
    Gaussian is validated against a finite-difference reference; no GPU is
    needed since the CRLB is evaluated at given parameters."""

    def test_crlb_matches_reference_elliptic(self):
        # sx > sy so var_x > var_y - also guards the x/y association.
        theta = np.array([[500.0, 3.2, 3.7, 1.4, 1.0, 5.0]])
        crlb = localize._gauss_crlb(theta, BOX, em=False)[0]
        ref = _ref_gauss_crlb(theta[0], BOX, rotated=False)
        np.testing.assert_allclose(crlb, ref, rtol=1e-4)
        assert crlb[1] > crlb[2]  # var_x > var_y

    def test_crlb_matches_reference_rotated(self):
        theta = np.array([[800.0, 3.4, 3.1, 1.5, 0.9, 4.0, 0.6]])
        crlb = localize._gauss_crlb(theta, BOX, em=False, rotated=True)[0]
        ref = _ref_gauss_crlb(theta[0], BOX, rotated=True)
        np.testing.assert_allclose(crlb, ref, rtol=1e-4)
        assert np.isfinite(crlb[6]) and crlb[6] > 0  # finite angle variance

    def test_em_doubles_variance(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.3, 1.1, 5.0]])
        crlb = localize._gauss_crlb(theta, BOX, em=False)[0]
        crlb_em = localize._gauss_crlb(theta, BOX, em=True)[0]
        np.testing.assert_allclose(crlb_em, 2.0 * crlb, rtol=1e-10)

    def test_nan_theta_row_isolated(self):
        theta = np.array(
            [
                [500.0, 3.2, 3.7, 1.3, 1.1, 5.0],
                [np.nan, 3.0, 3.0, 1.0, 1.0, 5.0],
            ]
        )
        crlb = localize._gauss_crlb(theta, BOX, em=False)
        assert np.all(np.isfinite(crlb[0]))
        assert np.all(np.isnan(crlb[1]))

    def test_low_signal_stays_finite(self):
        # bg = 0 drives outer model pixels to ~0; the MU_FLOOR guard keeps the
        # Fisher weight (1 / mu) finite.
        theta = np.array([[300.0, 3.0, 3.0, 1.2, 1.2, 0.0]])
        assert np.all(
            np.isfinite(localize._gauss_crlb(theta, BOX, em=False)[0])
        )

    def test_empty_input(self):
        assert localize._gauss_crlb(np.zeros((0, 6)), BOX, em=False).shape == (
            0,
            6,
        )

    def test_more_photons_tightens_bound(self):
        # More photons tighten the position (x, y) and width (sx, sy) bounds.
        # var(N) itself grows with N (absolute photon-count noise increases).
        base = [3.2, 3.7, 1.3, 1.1, 5.0]
        dim = localize._gauss_crlb(np.array([[200.0, *base]]), BOX, em=False)[
            0
        ]
        bright = localize._gauss_crlb(
            np.array([[4000.0, *base]]), BOX, em=False
        )[0]
        assert np.all(bright[1:5] < dim[1:5])


class TestSaveableColumns:
    """Guard the save whitelist: every column a fit path can emit must be
    registered in ``localize.LOCALIZATION_COLUMNS``. The GUI's
    ``Window.select_locs_columns`` keeps only whitelisted columns, so any
    unregistered column (e.g. the MLE ``*_unc`` uncertainties) is silently
    dropped before the .hdf5 is written. This test builds a representative
    localizations frame from every ``locs_from_fits_*`` constructor and fails
    if it produces a column the whitelist does not cover."""

    IDS = pd.DataFrame(
        {
            "frame": [0, 1],
            "x": [10, 20],
            "y": [15, 25],
            "net_gradient": [100.0, 120.0],
        }
    )

    def _fit_frames(self):
        """One output frame per fit path Picasso can save (GPU-free: the
        constructors are pure functions of the fitted parameters)."""
        ids = self.IDS
        stats = dict(
            log_likelihood=np.array([-10.0, -11.0], dtype=np.float32),
            iterations=np.array([5, 6], dtype=np.int32),
        )
        frames = {}

        # gpufit Gaussian, [photons, x, y, sx, sy, bg] (+ angle if rotated)
        theta_e = np.array(
            [
                [500.0, 3.2, 3.7, 1.3, 1.1, 5.0],
                [800.0, 3.4, 3.1, 1.2, 1.2, 4.0],
            ],
            dtype=np.float32,
        )
        theta_r = np.array(
            [
                [800.0, 3.4, 3.1, 1.5, 0.9, 4.0, 0.6],
                [900.0, 3.0, 3.5, 1.4, 1.0, 3.0, -0.3],
            ],
            dtype=np.float32,
        )
        frames["gpufit-mle"] = localize.locs_from_fits_gpufit(
            ids, theta_e, BOX, em=False, mle=True, **stats
        )
        frames["gpufit-mle-rotated"] = localize.locs_from_fits_gpufit(
            ids, theta_r, BOX, em=False, mle=True, **stats
        )
        frames["gpufit-lse"] = localize.locs_from_fits_gpufit(
            ids, theta_e, BOX, em=False, mle=False
        )
        frames["gpufit-lse-rotated"] = localize.locs_from_fits_gpufit(
            ids, theta_r, BOX, em=False, mle=False
        )

        # spline PSF, [amplitude, x_shift, y_shift, (z_shift,) offset]
        calib_2d, _ = _gauss_spline_calibration(model="spline-2d")
        calib_3d, _ = _gauss_spline_calibration(model="spline-3d")
        theta_s2 = np.array(
            [[3000.0, 0.1, -0.1, 15.0], [2800.0, 0.05, -0.05, 14.0]]
        )
        theta_s3 = np.array(
            [[4000.0, 0.2, -0.15, -6.0, 20.0], [3800.0, 0.1, -0.1, -5.0, 18.0]]
        )
        frames["spline-2d"] = localize.locs_from_fits_spline(
            ids,
            theta_s2,
            BOX,
            em=False,
            calibration=calib_2d,
            mle=True,
            **stats,
        )
        frames["spline-3d"] = localize.locs_from_fits_spline(
            ids,
            theta_s3,
            BOX,
            em=False,
            calibration=calib_3d,
            mle=True,
            **stats,
        )

        # CPU MLE Gaussian, [x, y, photons, bg, sx, sy] with its 6-col CRLB
        theta_cpu = np.array(
            [
                [3.0, 3.0, 500.0, 5.0, 1.2, 1.1],
                [3.1, 2.9, 600.0, 4.0, 1.0, 1.3],
            ]
        )
        crlbs = np.full((2, 6), 0.01)
        frames["gaussmle-cpu"] = gaussmle.locs_from_fits(
            ids,
            theta_cpu,
            crlbs,
            stats["log_likelihood"],
            stats["iterations"],
            BOX,
        )
        return frames

    def test_all_fit_columns_are_whitelisted(self):
        saveable = {
            col
            for cols in localize.LOCALIZATION_COLUMNS.values()
            for col in cols
        }
        offenders = {
            name: sorted(set(locs.columns) - saveable)
            for name, locs in self._fit_frames().items()
            if set(locs.columns) - saveable
        }
        assert not offenders, (
            "fit paths emit columns missing from LOCALIZATION_COLUMNS (they "
            f"would be dropped on save): {offenders}"
        )

    def test_uncertainty_columns_survive_select_locs_columns(self):
        # Mirror Window.select_locs_columns: keep only whitelisted columns (all
        # checkboxes default checked). The MLE uncertainties must survive.
        saveable = {
            col
            for cols in localize.LOCALIZATION_COLUMNS.values()
            for col in cols
        }
        frames = self._fit_frames()
        for expected in ("photons_unc", "bg_unc", "sx_unc", "sy_unc"):
            locs = frames["gpufit-mle"]
            kept = [c for c in locs.columns if c in saveable]
            assert expected in kept, f"{expected} dropped for gpufit-mle"
        assert "angle_unc" in [
            c for c in frames["gpufit-mle-rotated"].columns if c in saveable
        ]


# ---------------------------------------------------------------------------
# Ratiometric multichannel spline fitting (globLoc-style color assignment)
# ---------------------------------------------------------------------------


def _stack_multichannel(
    calib1, n_channels=2, photon_scale=None, transforms=None
):
    """Turn a single-channel 3D spline calibration into a multichannel one by
    replicating its (real, fit-grade) coefficient block across channels."""
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    calib = dict(calib1)
    calib["model"] = "spline-3d-multichannel"
    calib["coefficients"] = np.repeat(
        np.asarray(calib1["coefficients"])[..., None], n_channels, axis=-1
    ).astype(np.float32)
    calib["n_channels"] = n_channels
    calib["channel_transforms"] = transforms or [identity] * n_channels
    if photon_scale is not None:
        calib["photon_scale"] = photon_scale
    return calib


class TestScaleChannelBlocks:
    """Pure-Python coefficient scaling / photon_scale helpers (no GPU)."""

    def test_scales_each_channel(self):
        coeff = np.arange(64 * 2 * 2 * 3 * 2, dtype=np.float32).reshape(
            64, 2, 2, 3, 2
        )
        scaled = localize.scale_channel_blocks(coeff, [0.25, 4.0])
        np.testing.assert_allclose(scaled[..., 0], coeff[..., 0] * 0.25)
        np.testing.assert_allclose(scaled[..., 1], coeff[..., 1] * 4.0)

    def test_input_not_modified(self):
        coeff = np.ones((64, 2, 2, 3, 2), dtype=np.float32)
        before = coeff.copy()
        localize.scale_channel_blocks(coeff, [2.0, 3.0])
        np.testing.assert_array_equal(coeff, before)

    def test_ratio_length_mismatch(self):
        coeff = np.ones((64, 2, 2, 3, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            localize.scale_channel_blocks(coeff, [1.0, 2.0, 3.0])

    def test_requires_multichannel_table(self):
        with pytest.raises(ValueError):
            localize.scale_channel_blocks(
                np.ones((64, 2, 2, 3), np.float32), [1.0]
            )

    def test_photon_scales_scalar_broadcast(self):
        ps = localize._photon_scales({"photon_scale": 2.5}, 3)
        np.testing.assert_array_equal(ps, [2.5, 2.5, 2.5])

    def test_photon_scales_array_passthrough(self):
        ps = localize._photon_scales({"photon_scale": [1.0, 2.0, 3.0]}, 3)
        np.testing.assert_array_equal(ps, [1.0, 2.0, 3.0])


class TestSplinePerChannelPhotonScale:
    """locs_from_fits_spline maps the shared amplitude to TOTAL photons when
    photon_scale is a per-channel array (sum). CPU-only (numba CRLB)."""

    def test_total_photons_sum_per_channel(self):
        calib, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel",
            n_channels=2,
            photon_scale=[2.0, 3.0],
        )
        box = calib["n_data"][0]  # box == cal box -> crop is a no-op
        amp = 100.0
        theta = np.array(
            [[amp, 0.0, 0.0, -calib["z_center"], 5.0]], np.float64
        )
        ids = pd.DataFrame(
            {
                "frame": [0],
                "x": [box // 2],
                "y": [box // 2],
                "net_gradient": [1.0],
            }
        )
        locs = localize.locs_from_fits_spline(
            ids, theta, box, em=False, calibration=calib, mle=False
        )
        # photons = amplitude * (2 + 3)
        np.testing.assert_allclose(
            locs["photons"].iloc[0], amp * 5.0, rtol=1e-4
        )


@pytest.mark.skipif(
    not (localize.GPUFIT_INSTALLED and localize.GPUSPLINE_INSTALLED),
    reason="Gpufit (CUDA GPU) + Gpuspline not available",
)
class TestSplineRatiometric:
    """End-to-end ratiometric color assignment on a real (Gpuspline) PSF,
    stacked into two identical channels. The two channels differ only by a
    known photon split; the fitter must recover it as the winning ratio."""

    CANDS = np.array([[0.9, 0.1], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75]])
    TRUE_IDX = 1  # [0.75, 0.25]

    def _calib_and_base(self):
        calib1, template, _, _ = _synthetic_spline_3d_calibration()
        z_slice = int(calib1["z_center"])
        base = template[:, :, z_slice].astype(np.float32)
        calib = _stack_multichannel(calib1, 2, photon_scale=[1.0, 1.0])
        return calib, base

    def test_core_selection_recovers_true_ratio(self):
        calib, base = self._calib_and_base()
        box = calib["n_data"][0]
        r_true = self.CANDS[self.TRUE_IDX]
        rng = np.random.default_rng(0)
        n = 150
        spots = np.empty((n, box, box, 2), np.float32)
        for c in range(2):
            mu = np.clip(20.0 + 3000.0 * r_true[c] * base, 0, None)
            spots[..., c] = rng.poisson(mu, size=(n, box, box)).astype(
                np.float32
            )
        rn = self.CANDS / self.CANDS.sum(1, keepdims=True)
        scores = np.full((len(rn), n), np.inf)
        for k, r in enumerate(rn):
            ck = dict(calib)
            ck["coefficients"] = localize.scale_channel_blocks(
                calib["coefficients"], r
            )
            params, states, chi, *_ = localize._run_gpufit_spline(
                spots, ck, mle=False
            )
            fin = np.isfinite(params).all(1) & np.isfinite(chi)
            scores[k] = np.where(fin, chi, np.inf)
        best = np.argmin(scores, axis=0)
        assert (best == self.TRUE_IDX).mean() > 0.95

    def test_entry_point_colors_and_photons(self):
        calib, base = self._calib_and_base()
        box = calib["n_data"][0]
        cam = {"Baseline": 0, "Sensitivity": 1.0, "Gain": 1, "Qe": 1.0}
        r_true = self.CANDS[self.TRUE_IDX]
        rng = np.random.default_rng(1)
        n = 80
        movies = []
        for c in range(2):
            mu = np.clip(20.0 + 3000.0 * r_true[c] * base, 0, None)
            mov = np.stack(
                [rng.poisson(mu).astype(np.float32) for _ in range(n)]
            )
            movies.append(mov)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n),
                "x": np.full(n, box // 2),
                "y": np.full(n, box // 2),
                "net_gradient": np.ones(n),
            }
        )
        locs = localize.fit_spline_multichannel_ratiometric(
            movies,
            [cam, cam],
            ids,
            box,
            calib,
            photon_ratios=self.CANDS,
        )
        assert len(locs) == n
        assert (locs["color"] == self.TRUE_IDX).mean() > 0.9
        # per-channel photons track the 0.75/0.25 split; total ~= 3000
        frac0 = locs["photons_ch0"].median() / (
            locs["photons_ch0"].median() + locs["photons_ch1"].median()
        )
        assert abs(frac0 - 0.75) < 0.06
        for col in ("z", "lpx", "lpy", "lpz", "photons_ch0", "photons_ch1"):
            assert col in locs.columns

    def test_ratios_from_calibration_when_omitted(self):
        calib, base = self._calib_and_base()
        calib["photon_ratios"] = self.CANDS.tolist()
        box = calib["n_data"][0]
        cam = {"Baseline": 0, "Sensitivity": 1.0, "Gain": 1, "Qe": 1.0}
        r_true = self.CANDS[self.TRUE_IDX]
        rng = np.random.default_rng(2)
        n = 40
        movies = []
        for c in range(2):
            mu = np.clip(20.0 + 3000.0 * r_true[c] * base, 0, None)
            movies.append(
                np.stack(
                    [rng.poisson(mu).astype(np.float32) for _ in range(n)]
                )
            )
        ids = pd.DataFrame(
            {
                "frame": np.arange(n),
                "x": np.full(n, box // 2),
                "y": np.full(n, box // 2),
                "net_gradient": np.ones(n),
            }
        )
        # no photon_ratios argument -> taken from the calibration
        locs = localize.fit_spline_multichannel_ratiometric(
            movies, [cam, cam], ids, box, calib
        )
        assert (locs["color"] == self.TRUE_IDX).mean() > 0.9

    def test_missing_ratios_raises(self):
        calib, _ = self._calib_and_base()
        cam = {"Baseline": 0, "Sensitivity": 1.0, "Gain": 1}
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        box = calib["n_data"][0]
        movie = np.zeros((1, box, box), np.float32)
        with pytest.raises(ValueError):
            localize.fit_spline_multichannel_ratiometric(
                [movie, movie], [cam, cam], ids, box, calib
            )


# ---------------------------------------------------------------------------
# 4Pi phase model (Gpufit SPLINE_3D_PHASE_MULTICHANNEL = 12)
# ---------------------------------------------------------------------------


def _fake_phase_calibration(box=BOX, n_channels=2, nz=21):
    """Structurally valid 4Pi phase calibration with arbitrary coefficients:
    ``(64, nix, niy, niz, n_channels, 3)`` (the trailing 3 are the mean /
    modulation / modulation-90deg spline sets). For layout/packing tests."""
    nix = niy = box - 1
    niz = nz - 1
    coeff = np.arange(
        64 * nix * niy * niz * n_channels * 3, dtype=np.float32
    ).reshape(64, nix, niy, niz, n_channels, 3)
    return {
        "model": "spline-3d-phase-multichannel",
        "coefficients": coeff,
        "n_data": [box, box, nz],
        "n_intervals": [nix, niy, niz],
        "n_channels": n_channels,
        "oversampling": 1.0,
        "z_center": (nz - 1) / 2.0,
        "z_step_nm": 20.0,
        "photon_scale": 1.0,
        "box": box,
    }


class TestSplinePhaseModelLayout:
    """Pure (no-GPU) packing/parameter layout for model 12."""

    def test_n_channels_from_second_to_last_axis(self):
        assert (
            localize._spline_n_channels(_fake_phase_calibration(n_channels=3))
            == 3
        )

    def test_pack_user_info_three_sets(self):
        nc = 2
        calib = _fake_phase_calibration(n_channels=nc)
        ui = localize._pack_spline_user_info(calib)
        nx, ny, _ = calib["n_data"]
        ix, iy, iz = calib["n_intervals"]
        # same 7-value header as the (non-phase) multichannel model
        np.testing.assert_array_equal(
            ui[:7], np.array([nc, nx, ny, 1, ix, iy, iz], np.float32)
        )
        ni = ix * iy * iz
        # coefficient block = 3 sets x nc channels x ni intervals x 64
        assert ui.size == 7 + 3 * nc * ni * 64
        # each of the 3 concatenated blocks is that set reordered exactly like
        # the multichannel model (mean, modulation, modulation_90deg order)
        block = ui[7:]
        per_set = nc * ni * 64
        for s in range(3):
            expected = localize._reorder_spline_coefficients_for_gpufit(
                np.ascontiguousarray(calib["coefficients"][..., s]),
                "spline-3d-multichannel",
            )
            np.testing.assert_array_equal(
                block[s * per_set : (s + 1) * per_set], expected
            )

    def test_initial_parameters_six_with_phase_zero(self):
        calib = _fake_phase_calibration(n_channels=2)
        spots = (
            np.random.default_rng(0)
            .random((5, BOX, BOX, 2))
            .astype(np.float32)
        )
        init = localize._initial_parameters_spline(spots, calib)
        assert init.shape == (5, 6)
        np.testing.assert_allclose(init[:, 5], 0.0)  # phase starts at 0

    def test_model_id_maps_to_twelve(self):
        assert (
            localize._spline_model_id("spline-3d-phase-multichannel")
            == localize.gf.ModelID.SPLINE_3D_PHASE_MULTICHANNEL
        )

    def test_locs_conversion_offset_index_and_phase_column(self):
        # minimal sane phase calib (flat mean, no modulation) so the CRLB stays
        # finite; the point is the OUTPUT layout: offset is theta[:, 4] (not the
        # last column), photons = amp*photon_scale, and a wrapped phase column.
        box, nz = BOX, 21
        coeff = np.zeros((64, box - 1, box - 1, nz - 1, 1, 3), np.float32)
        coeff[0, :, :, :, :, 0] = 1.0  # constant mean spline, zero modulation
        calib = {
            "model": "spline-3d-phase-multichannel",
            "coefficients": coeff,
            "n_data": [box, box, nz],
            "n_intervals": [box - 1, box - 1, nz - 1],
            "n_channels": 1,
            "oversampling": 1.0,
            "z_center": 10.0,
            "z_step_nm": 20.0,
            "photon_scale": [1.0],
            "box": box,
        }
        theta = np.zeros((2, 6), np.float32)
        theta[:, 0] = 100.0  # amplitude
        theta[:, 4] = 5.0  # offset (column 4, NOT the last)
        theta[:, 5] = 1.3  # phase
        ids = pd.DataFrame(
            {
                "frame": [0, 1],
                "x": [6, 6],
                "y": [6, 6],
                "net_gradient": [1.0, 1.0],
            }
        )
        locs = localize.locs_from_fits_spline(ids, theta, box, False, calib)
        assert "phase" in locs.columns and "phase_unc" in locs.columns
        np.testing.assert_allclose(locs["phase"].to_numpy(), 1.3, atol=1e-4)
        np.testing.assert_allclose(locs["bg"].to_numpy(), 5.0, atol=1e-4)
        np.testing.assert_allclose(
            locs["photons"].to_numpy(), 100.0, rtol=1e-4
        )


def _phase_spline_calibration(
    box=BOX, nz=41, n_channels=1, widths=(1.1, 1.9, 0.8)
):
    """4Pi phase calibration from three DISTINCT real Gpuspline sets (mean,
    modulation, modulation_90deg). Returns (calibration, [coeff_sets])."""
    gs = localize.gs
    zc = (nz - 1) / 2.0

    def stack(s0):
        x = np.arange(box, dtype=np.float32)
        c = (box - 1) / 2.0
        t = np.zeros((box, box, nz), np.float32)
        for k in range(nz):
            s = s0 * (1.0 + 0.4 * abs(k - zc) / nz)
            g = np.exp(-0.5 * ((x - c) / s) ** 2)
            t[:, :, k] = np.outer(g, g)  # isotropic -> swap-invariant
        return t

    coeffs = []
    for s0 in widths:
        t = stack(s0)
        ni = np.array(t.shape) - 1
        coeffs.append(
            np.reshape(gs.spline_coefficients(t), (64, ni[0], ni[1], ni[2]))
        )
    nix, niy, niz = coeffs[0].shape[1:]
    per_set = [np.repeat(c[..., None], n_channels, axis=-1) for c in coeffs]
    coeff_arr = np.stack(per_set, axis=-1).astype(np.float32)
    calib = {
        "model": "spline-3d-phase-multichannel",
        "coefficients": coeff_arr,
        "n_data": [box, box, nz],
        "n_intervals": [int(nix), int(niy), int(niz)],
        "n_channels": n_channels,
        "oversampling": 1.0,
        "z_center": zc,
        "z_step_nm": 20.0,
        "photon_scale": 1.0,
        "box": box,
    }
    return calib, coeffs


@pytest.mark.skipif(
    not (localize.GPUFIT_INSTALLED and localize.GPUSPLINE_INSTALLED),
    reason="Gpufit (CUDA GPU) + Gpuspline not available",
)
class TestSplinePhaseModelGpu:
    """Model 12 wiring proof: with correctly packed 3-set coefficients, the
    ground-truth parameters are a zero-residual fixed point of the compiled
    Gpufit phase model (value = amp*(mean + cos p*mod + sin p*mod90) + off)."""

    def test_truth_is_zero_residual_fixed_point(self):
        calib, coeffs = _phase_spline_calibration(n_channels=1)
        box = calib["n_data"][0]
        z0 = calib["z_center"]
        phis = [
            localize._spline_model_and_grad(
                c, box, np.array([0.0]), np.array([0.0]), np.array([float(z0)])
            )[0][0]
            for c in coeffs
        ]
        mean_phi, mod_phi, mod90_phi = phis
        A, O, phase = 3000.0, 30.0, 0.7
        img = (
            A
            * (mean_phi + np.cos(phase) * mod_phi + np.sin(phase) * mod90_phi)
            + O
        )
        ui = localize._pack_spline_user_info(calib)
        n = 8
        data = np.tile(img.reshape(1, -1), (n, 1)).astype(np.float32)
        truth = np.array([A, 0.0, 0.0, -z0, O, phase], np.float32)
        init = np.tile(truth, (n, 1)).astype(np.float32)
        params, states, chi, _, _ = localize.gf.fit(
            data,
            None,
            localize.gf.ModelID.SPLINE_3D_PHASE_MULTICHANNEL,
            init,
            tolerance=1e-6,
            max_number_iterations=30,
            estimator_id=localize.gf.EstimatorID.LSE,
            user_info=ui,
        )
        # truth-init on noise-free data must stay put with ~zero residual
        np.testing.assert_allclose(params[0], truth, atol=1e-2, rtol=1e-3)
        signal = float(np.abs(img - O).sum())
        assert float(np.median(chi)) < 1e-2 * signal


class TestMultichannelWorkerRouting:
    """MultichannelSplineFitWorker routes to the ratiometric fitter iff the
    calibration carries photon_ratios (no GPU; the fit fns are monkeypatched).
    """

    def _run(self, calibration, monkeypatch):
        calls = []
        df = pd.DataFrame({"frame": [0], "x": [0.0], "y": [0.0]})
        monkeypatch.setattr(
            localize,
            "fit_spline_multichannel_ratiometric",
            lambda *a, **k: (calls.append("ratio"), df)[1],
        )
        monkeypatch.setattr(
            localize,
            "fit_spline_multichannel",
            lambda *a, **k: (calls.append("plain"), df)[1],
        )
        monkeypatch.setattr(
            localize,
            "fit_spline_phase_multichannel",
            lambda *a, **k: (calls.append("phase"), df)[1],
        )
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        worker = localize_gui.MultichannelSplineFitWorker(
            [None, None], [{}, {}], ids, BOX, calibration, mle=False
        )
        result = {}
        worker.finished.connect(
            lambda locs, dt, a, b: result.update(locs=locs)
        )
        worker.run()
        return calls, result

    def test_routes_to_ratiometric_with_ratios(self, monkeypatch):
        calls, result = self._run(
            {"model": "spline-3d-multichannel", "photon_ratios": [[0.7, 0.3]]},
            monkeypatch,
        )
        assert calls == ["ratio"]
        assert "locs" in result

    def test_routes_to_plain_without_ratios(self, monkeypatch):
        calls, _ = self._run({"model": "spline-3d-multichannel"}, monkeypatch)
        assert calls == ["plain"]

    def test_routes_to_phase_with_phase_model(self, monkeypatch):
        calls, _ = self._run(
            {"model": "spline-3d-phase-multichannel"}, monkeypatch
        )
        assert calls == ["phase"]


# ---------------------------------------------------------------------------
# 4Pi phase fitting: multi-start, CRLB, and the movie-level entry point
# ---------------------------------------------------------------------------


def _phase_4pi_calibration(box=BOX, nz=41, n_channels=4):
    """Well-conditioned 4Pi phase calibration (needs Gpuspline): n_channels
    phase channels at psi_c = c*2pi/n_channels, each with mean = env,
    modulation = cos(psi_c)*env, modulation_90deg = sin(psi_c)*env, where env is
    an isotropic Gaussian whose width grows with |z - z_center| (encodes z)."""
    gs = localize.gs
    zc = (nz - 1) / 2.0
    xg = np.arange(box, dtype=np.float32)
    c0 = (box - 1) / 2.0
    env = np.zeros((box, box, nz), np.float32)
    for k in range(nz):
        s = 1.3 * (1.0 + 0.6 * abs(k - zc) / nz)
        g = np.exp(-0.5 * ((xg - c0) / s) ** 2)
        env[:, :, k] = np.outer(g, g)
    coeff_env = np.reshape(
        gs.spline_coefficients(env), (64, box - 1, box - 1, nz - 1)
    )
    nix, niy, niz = coeff_env.shape[1:]
    psis = np.arange(n_channels) * (2 * np.pi / n_channels)
    coeff = np.zeros((64, nix, niy, niz, n_channels, 3), np.float32)
    for ch in range(n_channels):
        coeff[..., ch, 0] = coeff_env
        coeff[..., ch, 1] = np.cos(psis[ch]) * coeff_env
        coeff[..., ch, 2] = np.sin(psis[ch]) * coeff_env
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    calib = {
        "model": "spline-3d-phase-multichannel",
        "coefficients": coeff,
        "n_data": [box, box, nz],
        "n_intervals": [int(nix), int(niy), int(niz)],
        "n_channels": n_channels,
        "channel_transforms": [identity] * n_channels,
        "oversampling": 1.0,
        "z_center": zc,
        "z_step_nm": 20.0,
        "photon_scale": [1.0] * n_channels,
        "box": box,
    }
    return calib, psis


def _phase_channel_image(calib, ch, z0, phase, A, O):
    box = calib["n_data"][0]
    sets = [
        localize._spline_model_and_grad(
            calib["coefficients"][..., ch, s],
            box,
            np.array([0.0]),
            np.array([0.0]),
            np.array([float(z0)]),
        )[0][0]
        for s in range(3)
    ]
    return (
        A * (sets[0] + np.cos(phase) * sets[1] + np.sin(phase) * sets[2]) + O
    )


def test_initial_parameters_phase_uses_channel_average(monkeypatch):
    """The phase model seeds amplitude/offset from the channel-averaged spot
    (interference cancels), not the brightest channel's 1+cos peak. Pure."""
    box, nch = BOX, 4
    calib = {
        "model": "spline-3d-phase-multichannel",
        "coefficients": np.zeros(
            (64, box - 1, box - 1, 20, nch, 3), np.float32
        ),
        "z_center": 10.0,
    }
    # channel 0 twice as bright as the average; average max-min = 100
    spots = np.zeros((1, box, box, nch), np.float32)
    spots[0, box // 2, box // 2, 0] = 200.0  # bright channel
    spots[0, box // 2, box // 2, 1] = 100.0
    spots[0, :, :, :] += 10.0  # baseline
    spots[0, box // 2, box // 2, 0] += 200.0
    init = localize._initial_parameters_spline(spots, calib)
    assert init.shape == (1, 6)
    avg = spots[0].mean(axis=-1)
    assert init[0, 0] == pytest.approx(avg.max() - avg.min(), rel=1e-5)
    assert init[0, 4] == pytest.approx(avg.min(), rel=1e-5)


@pytest.mark.skipif(
    not (localize.GPUFIT_INSTALLED and localize.GPUSPLINE_INSTALLED),
    reason="Gpufit (CUDA GPU) + Gpuspline not available",
)
class TestSplinePhaseFit:
    def _movies(self, calib, z0, phase_true, seed=0, n=60, A=3000.0, O=30.0):
        box = calib["n_data"][0]
        rng = np.random.default_rng(seed)
        movies = []
        for c in range(calib["n_channels"]):
            img = _phase_channel_image(calib, c, z0, phase_true, A, O)
            movies.append(
                np.stack(
                    [
                        rng.poisson(np.clip(img, 0, None)).astype(np.float32)
                        for _ in range(n)
                    ]
                )
            )
        ids = pd.DataFrame(
            {
                "frame": np.arange(n),
                "x": np.full(n, box // 2),
                "y": np.full(n, box // 2),
                "net_gradient": np.ones(n),
            }
        )
        return movies, ids

    def test_multistart_recovers_phase_where_single_start_fails(self):
        calib, _ = _phase_4pi_calibration()
        box = calib["n_data"][0]
        z0 = calib["z_center"] + 5.0
        phase_true = 3.1  # near pi: a single phase=0 start cannot reach it
        rng = np.random.default_rng(0)
        n = 60
        spots = np.empty((n, box, box, calib["n_channels"]), np.float32)
        for c in range(calib["n_channels"]):
            img = _phase_channel_image(calib, c, z0, phase_true, 3000.0, 30.0)
            for i in range(n):
                spots[i, :, :, c] = rng.poisson(np.clip(img, 0, None))
        multi = localize.fit_spots_gpufit_spline_phase(
            spots, calib, n_phase_starts=8
        )
        single = localize.fit_spots_gpufit_spline_phase(
            spots, calib, n_phase_starts=1
        )

        def dphi(theta):
            ok = np.isfinite(theta).all(1)
            ph = np.median(theta[ok, 5]) % (2 * np.pi)
            return abs((ph - phase_true + np.pi) % (2 * np.pi) - np.pi)

        assert dphi(multi) < 0.1  # multi-start reaches it
        assert dphi(single) > 0.3  # single start does not

    def test_phase_entry_point_outputs(self):
        calib, _ = _phase_4pi_calibration()
        z0 = calib["z_center"] + 5.0
        phase_true = 2.0
        movies, ids = self._movies(calib, z0, phase_true, seed=1)
        cam = {"Baseline": 0, "Sensitivity": 1.0, "Gain": 1}
        locs = localize.fit_spline_phase_multichannel(
            movies,
            [cam] * calib["n_channels"],
            ids,
            calib["n_data"][0],
            calib,
            mle=False,
            n_phase_starts=8,
        )
        med = locs.median(numeric_only=True)
        d = abs(
            (float(med["phase"]) - phase_true + np.pi) % (2 * np.pi) - np.pi
        )
        assert d < 0.06
        for col in (
            "z",
            "phase",
            "phase_unc",
            "photons",
            "lpx",
            "lpy",
            "lpz",
            "photons_unc",
            "bg_unc",
        ):
            assert col in locs.columns
            assert np.isfinite(float(med[col]))
        # z reconstruction: (z_shift + z_init)*step*mag, z_init=z_center here
        exp_z = (-z0 + calib["z_center"]) * calib["z_step_nm"] * 1.0
        assert abs(float(med["z"]) - exp_z) < 15.0

    def test_phase_crlb_matches_finite_difference(self):
        calib, _ = _phase_4pi_calibration()
        box = calib["n_data"][0]
        zc = calib["z_center"]
        theta = np.array(
            [
                [3000.0, 0.1, -0.2, -(zc + 5), 30.0, 0.6],
                [2000.0, 0.0, 0.0, -(zc + 7), 25.0, 2.4],
            ],
            np.float64,
        )
        crlb = localize._spline_phase_crlb(theta, calib, box, mle=True)
        assert np.all(np.isfinite(crlb)) and np.all(crlb > 0)

        # independent finite-difference Fisher (MLE) for row 0
        coeff = calib["coefficients"]
        nch = calib["n_channels"]

        def mu(th):
            out = np.zeros((box, box, nch))
            for c in range(nch):
                s = [
                    localize._spline_model_and_grad(
                        coeff[..., c, k], box, th[1:2], th[2:3], -th[3:4]
                    )[0][0]
                    for k in range(3)
                ]
                out[..., c] = (
                    th[0]
                    * (s[0] + np.cos(th[5]) * s[1] + np.sin(th[5]) * s[2])
                    + th[4]
                )
            return out

        th0 = theta[0]
        eps = np.array([th0[0] * 1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4])
        base = np.maximum(mu(th0), localize._SPLINE_CRLB_MU_FLOOR)
        grads = []
        for j in range(6):
            tp = th0.copy()
            tp[j] += eps[j]
            tm = th0.copy()
            tm[j] -= eps[j]
            grads.append((mu(tp) - mu(tm)) / (2 * eps[j]))
        order = [1, 2, 3, 0, 4, 5]  # -> [x, y, z, amp, off, phase]
        G = np.stack([grads[o] for o in order], axis=-1).reshape(-1, 6)
        fisher = (G / base.reshape(-1, 1)).T @ G
        fd_crlb = np.diag(np.linalg.inv(fisher))
        np.testing.assert_allclose(crlb[0], fd_crlb, rtol=0.02)
