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
from PyQt6 import QtWidgets

from picasso import io, localize
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


@pytest.mark.skipif(
    not localize.GPUFIT_INSTALLED, reason="GPUfit/CUDA not available"
)
class TestGpufit:
    """Tests for the optional GPU codepath. Skipped when the Gpufit
    library or a CUDA-capable GPU is not available (which is true for
    the typical test environment)."""

    def test_fit_spots_gpufit(self, synthetic_spots):
        spots, gt = synthetic_spots
        theta = localize.fit_spots_gpufit(spots)
        assert theta.shape == (len(spots), 6)
        # GPU returns parameters as [photons, x, y, sx, sy, bg]
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=0.05)

    def test_fit_spots_gpufit_mle(self, synthetic_spots):
        spots, gt = synthetic_spots
        theta = localize.fit_spots_gpufit(spots, mle=True)
        assert theta.shape == (len(spots), 6)
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=0.05)

    def test_fit_spots_gpufit_rotated(self, synthetic_spots):
        spots, gt = synthetic_spots
        theta = localize.fit_spots_gpufit(spots, rotated=True)
        assert theta.shape == (len(spots), 7)
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=0.05)


# ---------------------------------------------------------------------------
# Cubic-spline PSF fitting (Gpufit SPLINE_2D / SPLINE_3D)
# ---------------------------------------------------------------------------


def _fake_spline_calibration(model="spline-3d", box=BOX):
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
    else:
        nz = 21
        n_intervals = [box - 1, box - 1, nz - 1]
        n_data = [box, box, nz]
        n_coef = 64
        coefficients = np.arange(
            n_coef * np.prod(n_intervals), dtype=np.float32
        ).reshape([n_coef] + n_intervals)
    return {
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
        np.testing.assert_array_equal(
            user_info[6:], calib["coefficients"].ravel(order="C")
        )

    def test_pack_user_info_2d_layout(self):
        calib = _fake_spline_calibration(model="spline-2d")
        user_info = localize._pack_spline_user_info(calib)
        nx, ny = calib["n_data"]
        ix, iy = calib["n_intervals"]
        np.testing.assert_array_equal(
            user_info[:4], np.array([nx, ny, ix, iy], np.float32)
        )
        assert user_info.size == 4 + calib["coefficients"].size

    def test_initial_parameters_shape(self, synthetic_spots):
        spots, _ = synthetic_spots
        calib_3d = _fake_spline_calibration(model="spline-3d")
        init_3d = localize._initial_parameters_spline(spots, calib_3d)
        assert init_3d.shape == (len(spots), 5)
        assert init_3d.dtype == np.float32
        # z_shift initialised to the calibration's in-focus slice
        np.testing.assert_allclose(init_3d[:, 3], calib_3d["z_center"])

        calib_2d = _fake_spline_calibration(model="spline-2d")
        init_2d = localize._initial_parameters_spline(spots, calib_2d)
        assert init_2d.shape == (len(spots), 4)

    def test_locs_from_fits_spline_3d(self):
        calib = _fake_spline_calibration(model="spline-3d")
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
        theta[:, 3] = calib["z_center"] + 2.0  # z_shift (2 slices past focus)
        theta[:, 4] = 12.0  # offset (bg)

        locs = localize.locs_from_fits_spline(ids, theta, BOX, False, calib)

        box_offset = int(BOX / 2)
        np.testing.assert_allclose(locs["x"], 0.5 + 20.0 - box_offset)
        np.testing.assert_allclose(locs["y"], -0.5 + 30.0 - box_offset)
        np.testing.assert_allclose(locs["photons"], 5000.0)
        np.testing.assert_allclose(locs["bg"], 12.0)
        # z = (z_shift - z_center) * z_step_nm = 2 * 20 = 40 nm
        np.testing.assert_allclose(locs["z"], 40.0)
        # d_zcalib / lpz are not available for the spline model
        assert np.all(np.isnan(locs["d_zcalib"]))
        assert np.all(np.isnan(locs["lpz"]))

    def test_locs_from_fits_spline_2d_has_no_z(self):
        calib = _fake_spline_calibration(model="spline-2d")
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
        np.testing.assert_allclose(locs["photons"], 3000.0)

    @pytest.mark.skipif(
        localize.GPUFIT_INSTALLED,
        reason="ImportError only raised when Gpufit is unavailable",
    )
    def test_fit_spots_spline_without_gpu_raises(self, synthetic_spots):
        spots, _ = synthetic_spots
        calib = _fake_spline_calibration(model="spline-3d")
        with pytest.raises(ImportError):
            localize.fit_spots_gpufit_spline(spots, calib)


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
        # spline_values wants the flat (per-interval) coefficient array
        coeffs_flat = calib["coefficients"].reshape(64, -1)
        values = gs.spline_values(coeffs_flat, x, y, z)
        np.testing.assert_allclose(values, template, atol=1e-3)


@pytest.mark.skipif(
    not (localize.GPUFIT_INSTALLED and localize.GPUSPLINE_INSTALLED),
    reason="Gpufit (CUDA GPU) + Gpuspline not available",
)
class TestSplineGpufit:
    """End-to-end spline fitting. The fit itself runs on Gpufit (CUDA GPU);
    the calibration is built with Gpuspline (CPU). Skipped in the typical
    (GPU-less) test environment."""

    def test_fit_spots_spline_3d(self):
        calib, template, amplitude, offset = _synthetic_spline_3d_calibration()
        z_slice = int(calib["z_center"])
        spot = (amplitude * template[:, :, z_slice] + offset).astype(
            np.float32
        )
        spots = np.stack([spot] * 4)
        theta = localize.fit_spots_gpufit_spline(spots, calib)
        assert theta.shape == (len(spots), 5)
        # recovered z shift should sit near the taken slice
        np.testing.assert_allclose(theta[:, 3], z_slice, atol=2.0)

    def test_fit_spots_spline_box_mismatch(self):
        calib, _, _, _ = _synthetic_spline_3d_calibration(box=BOX)
        wrong_box = BOX + 2
        spots = np.zeros((2, wrong_box, wrong_box), dtype=np.float32)
        with pytest.raises(ValueError):
            localize.fit_spots_gpufit_spline(spots, calib)
