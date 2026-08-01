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
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline
from PyQt6 import QtWidgets

from picasso import gaussmle, gausslq, io, localize, spline
from picasso.gui import localize as localize_gui

from tests.conftest import BOX, CALIB_3D, CAMERA_INFO, MIN_NG, PIXELSIZE

CAMERA_INFO_WITH_PIXELSIZE = {**CAMERA_INFO, "Pixelsize": PIXELSIZE}

# Devices a spline-CRLB test can be pinned to (see ``_crlb``). The GPU variant
# needs a real CUDA device; ``NUMBA_ENABLE_CUDASIM=1`` also satisfies it, which
# is how the kernels can be exercised on a machine without one.
SPLINE_CRLB_DEVICES = [
    False,
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            not localize.SPLINE_CRLB_CUDA_AVAILABLE,
            reason="no CUDA device for the spline CRLB kernels",
        ),
    ),
]


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

    @pytest.mark.parametrize("multiprocess", [False, True])
    def test_gausslq_saves_chi_square_not_likelihood(
        self, picasso_movie, real_identifications, movie_info, multiprocess
    ):
        """The CPU least-squares methods report their goodness of fit as
        ``chi_square`` (the residual sum of squares at the optimum), the
        least-squares counterpart of the MLE fits' ``log_likelihood``. Both
        the serial and the multiprocessing path must carry it, since the
        multiprocessing path ferries it as an extra ``theta`` column."""
        locs, _ = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gausslq",
            multiprocess=multiprocess,
        )
        assert "chi_square" in locs.columns
        assert "log_likelihood" not in locs.columns
        chi = locs["chi_square"].to_numpy()
        assert np.all(chi >= 0) and np.all(np.isfinite(chi))
        # the chi-square column must not have displaced a parameter column
        for col in ["x", "y", "photons", "sx", "sy", "bg", "ellipticity"]:
            assert col in locs.columns

    @pytest.mark.parametrize(
        "method", ["gausslq", "gausslq-spherical", "gausslq-rotated"]
    )
    def test_all_cpu_lq_variants_save_chi_square(
        self, picasso_movie, real_identifications, movie_info, method
    ):
        """Every CPU least-squares model variant carries the column, and the
        rotated one still recovers its ``angle`` (whose detection keys off the
        parameter count, so the extra column must be split off first)."""
        locs, _ = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method=method,
            multiprocess=False,
        )
        assert "chi_square" in locs.columns
        assert np.all(locs["chi_square"].to_numpy() >= 0)
        assert ("angle" in locs.columns) == (method == "gausslq-rotated")

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

    @pytest.mark.parametrize(
        "method", ["gausslq-spherical", "gaussmle-spherical"]
    )
    def test_spherical_methods_drop_ellipticity(
        self, picasso_movie, real_identifications, movie_info, method
    ):
        """The spherical CPU methods (LQ and MLE) fit sx == sy and omit the
        always-zero ellipticity column, while keeping every other column."""
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method=method,
            multiprocess=False,
        )
        assert new_info["Fit method"] == method
        assert len(locs) == len(real_identifications)
        assert "ellipticity" not in locs.columns
        assert (locs["sx"].to_numpy() == locs["sy"].to_numpy()).all()
        for col in ["x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy"]:
            assert col in locs.columns

    def test_rotated_lq_keeps_ellipticity_and_adds_angle(
        self, picasso_movie, real_identifications, movie_info
    ):
        """The rotated CPU LQ method keeps ellipticity (widths differ) and
        adds an ``angle`` column wrapped to [-90, 90)."""
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gausslq-rotated",
            multiprocess=False,
        )
        assert new_info["Fit method"] == "gausslq-rotated"
        assert "ellipticity" in locs.columns
        assert "angle" in locs.columns
        assert ((locs["angle"] >= -90.0) & (locs["angle"] < 90.0)).all()

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
        file, and the whole batch is discarded: the file that was being
        read cannot be interrupted mid-way, so it is dropped rather than
        delivered as a half-loaded channel."""
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
        # finished still fires (so the GUI can tear the dialog down), but
        # with an empty batch.
        assert out.finished == ([], [], [])


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

    def test_rotated_and_spherical_mutually_exclusive(self, synthetic_spots):
        spots, _ = synthetic_spots
        with pytest.raises(ValueError):
            localize.fit_spots_gpufit(spots, rotated=True, spherical=True)

    # -- spherical (isotropic, single-width GAUSS_2D model) ---------------

    def test_spherical_lse_recovers_isotropic(self, synthetic_spots_isotropic):
        """The GAUSS_2D model recovers the shared width and returns the
        expanded elliptical layout with sx == sy."""
        spots, gt = synthetic_spots_isotropic
        theta = localize.fit_spots_gpufit(spots, spherical=True, mle=False)
        assert theta.shape == (len(spots), 6)
        np.testing.assert_array_equal(theta[:, 3], theta[:, 4])
        half = spots.shape[1] // 2
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=2e-3)
        np.testing.assert_allclose(theta[:, 1] - half, gt.x.values, atol=3e-3)
        np.testing.assert_allclose(theta[:, 2] - half, gt.y.values, atol=3e-3)
        np.testing.assert_allclose(theta[:, 3], gt.sx.values, atol=3e-3)

    def test_spherical_mle_converged_recover(self, synthetic_spots_isotropic):
        spots, gt = synthetic_spots_isotropic
        theta, ll, n_iter, chi2 = localize.fit_spots_gpufit(
            spots, spherical=True, mle=True, return_stats=True
        )
        assert theta.shape == (len(spots), 6)
        np.testing.assert_array_equal(theta[:, 3], theta[:, 4])
        assert ll is not None and ll.shape == (len(spots),)
        assert chi2 is None

    def test_spherical_end_to_end_omits_ellipticity(
        self, synthetic_spots_isotropic
    ):
        """The full GPU spherical path (via ``_fit2d_gauss_gpu``) drops the
        ellipticity column."""
        spots, _ = synthetic_spots_isotropic
        n = len(spots)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 50, dtype=np.int64),
                "y": np.full(n, 70, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )
        locs = localize._fit2d_gauss_gpu(
            spots, ids, spots.shape[1], em=False, spherical=True
        )
        assert "ellipticity" not in locs.columns
        assert (locs["sx"].to_numpy() == locs["sy"].to_numpy()).all()

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
        theta, ll, n_iter, chi2 = localize.fit_spots_gpufit(
            spots, mle=True, return_stats=True
        )
        assert theta.shape == (len(spots), 6)
        # MLE: log-likelihood is -0.5 * chi-square, finite, one per spot
        assert ll is not None and ll.shape == (len(spots),)
        assert np.all(np.isfinite(ll))
        assert n_iter.shape == (len(spots),)
        # the chi-square IS the likelihood here, so it is not reported twice
        assert chi2 is None

    def test_return_stats_lse_has_chi_square_not_likelihood(
        self, synthetic_spots
    ):
        spots, _ = synthetic_spots
        theta, ll, n_iter, chi2 = localize.fit_spots_gpufit(
            spots, mle=False, return_stats=True
        )
        # LSE assumes no noise model, so there is no likelihood; what it does
        # report is the residual sum of squares at the optimum.
        assert ll is None
        assert n_iter.shape == (len(spots),)
        assert chi2 is not None and chi2.shape == (len(spots),)
        assert np.all(chi2 >= 0)
        assert np.all(np.isfinite(chi2))

    def test_lse_end_to_end_saves_chi_square(self, synthetic_spots):
        """The full GPU least-squares path emits a chi_square column and no
        log_likelihood."""
        spots, _ = synthetic_spots
        n = len(spots)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 50, dtype=np.int64),
                "y": np.full(n, 70, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )
        locs = localize._fit2d_gauss_gpu(
            spots, ids, spots.shape[1], em=False, mle=False
        )
        assert "chi_square" in locs.columns
        assert "log_likelihood" not in locs.columns
        assert np.all(locs["chi_square"].to_numpy() >= 0)

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

    def test_spherical_single_width_layout(self):
        # Gpufit's isotropic GAUSS_2D model takes 5 parameters:
        # [amplitude, x, y, s, bg] — a single width, unlike the elliptic
        # 6-parameter layout.
        box = 7
        spots = np.zeros((2, box, box), dtype=np.float32)
        spots[0] = 3.0
        spots[0, 3, 3] = 103.0
        spots[1] = 7.0
        spots[1, 2, 4] = 57.0
        init = localize._initial_parameters_gpufit(spots, box, spherical=True)
        assert init.shape == (2, 5)
        assert init.dtype == np.float32
        center = box / 2.0 - 0.5
        width = max(box / 5.0, 1.0)
        np.testing.assert_allclose(init[:, 0], [100.0, 50.0])  # amplitude
        np.testing.assert_allclose(init[:, 1], center)  # x
        np.testing.assert_allclose(init[:, 2], center)  # y
        np.testing.assert_allclose(init[:, 3], width)  # single width
        np.testing.assert_allclose(init[:, 4], [3.0, 7.0])  # background


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

    def test_spherical_omits_ellipticity_column(self):
        # A spherical fit has sx == sy, so ellipticity is always 0 and is
        # dropped entirely. The rest of the columns are unaffected.
        theta = np.array([[500.0, 3.0, 3.0, 1.2, 1.2, 5.0]], dtype=np.float32)
        locs = localize.locs_from_fits_gpufit(
            self._ids(1), theta, BOX, em=False, spherical=True
        )
        assert "ellipticity" not in locs.columns
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
            "net_gradient",
        ):
            assert col in locs.columns

    def test_spherical_flag_only_drops_ellipticity(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.2, 1.2, 5.0]], dtype=np.float32)
        full = localize.locs_from_fits_gpufit(
            self._ids(1), theta, BOX, em=False, mle=True, spherical=False
        )
        sph = localize.locs_from_fits_gpufit(
            self._ids(1), theta, BOX, em=False, mle=True, spherical=True
        )
        assert set(full.columns) - set(sph.columns) == {"ellipticity"}
        for col in sph.columns:
            np.testing.assert_array_equal(
                sph[col].to_numpy(), full[col].to_numpy()
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

    def test_spline_kind_covers_every_model(self):
        from picasso import splinefit

        assert localize._spline_kind("spline-2d") == splinefit.KIND_2D
        assert localize._spline_kind("spline-3d") == splinefit.KIND_3D
        assert (
            localize._spline_kind("spline-3d-multichannel")
            == splinefit.KIND_3D
        )
        assert (
            localize._spline_kind(localize._LINK_XYZ_MODEL)
            == splinefit.KIND_LINK_XYZ
        )
        with pytest.raises(ValueError, match="Unknown spline"):
            localize._spline_kind("nonsense")

    def test_single_channel_spots_are_not_copied(self):
        """The CPU kernels want channel-major spots. For a single channel that
        must stay a view - transposing instead would copy the whole spot stack
        (hundreds of MB for a real movie) for no reason."""
        spots = np.zeros((32, BOX, BOX), np.float32)
        reshaped = localize._spline_channel_major(spots, 1)
        assert reshaped.shape == (32, 1, BOX, BOX)
        assert np.shares_memory(reshaped, spots)

    def test_multichannel_spots_become_channel_major(self):
        spots = np.arange(2 * BOX * BOX * 3, dtype=np.float32).reshape(
            2, BOX, BOX, 3
        )
        reshaped = localize._spline_channel_major(spots, 3)
        assert reshaped.shape == (2, 3, BOX, BOX)
        np.testing.assert_array_equal(reshaped[0, 1], spots[0, :, :, 1])
        with pytest.raises(ValueError, match="channels"):
            localize._spline_channel_major(spots, 2)

    def test_cpu_z_seeds_match_the_gpu_grid(self):
        """CPU and GPU must explore the same axial minima, or the two devices
        disagree for reasons that have nothing to do with the fit."""
        calib = _fake_spline_calibration(model="spline-3d")
        n_starts = localize._default_n_z_starts(calib)
        seeds, apply_seeds = localize._spline_z_seeds(calib, n_starts)
        assert apply_seeds
        # the grid _fit_spline_z_multistart builds for the GPU
        n_z = int(calib["n_data"][2])
        np.testing.assert_allclose(
            seeds, np.linspace(-(n_z - 1), 0.0, n_starts)
        )
        assert localize._spline_z_seeds(calib, 1)[1] is False
        calib_2d = _fake_spline_calibration(model="spline-2d")
        assert localize._spline_z_seeds(calib_2d, 9)[1] is False

    def test_cpu_schedule_defaults_and_overrides(self):
        from picasso import splinefit

        assert localize._spline_cpu_schedule(True, None, None) == (
            splinefit.TOLERANCE_MULTI_START,
            splinefit.MAX_ITERATIONS_MULTI_START,
        )
        assert localize._spline_cpu_schedule(False, None, None) == (
            splinefit.TOLERANCE_SINGLE_START,
            splinefit.MAX_ITERATIONS_SINGLE_START,
        )
        assert localize._spline_cpu_schedule(True, 1e-7, 3) == (1e-7, 3)

    @pytest.mark.parametrize("apply_seeds", [False, True])
    def test_cpu_and_gpu_use_the_same_schedule(self, apply_seeds):
        """Both devices must stop in the same place, or a CPU and a GPU fit of
        the same spots differ for reasons unrelated to the fit. The convergence
        test is relative, so a different tolerance is a real difference - and
        the multi-start ranks its axial seeds on that chi-square."""
        import inspect
        from picasso import splinefit

        shared = splinefit.convergence_schedule(apply_seeds)
        assert localize._spline_cpu_schedule(apply_seeds, None, None) == shared
        assert splinefit.resolve_schedule(apply_seeds) == shared
        # ...and the Gpufit call sites read the same function rather than
        # repeating the numbers.
        gpu_single = inspect.getsource(localize._run_gpufit_spline)
        gpu_multi = inspect.getsource(localize._fit_spline_z_multistart)
        assert "splinefit.resolve_schedule" in gpu_single
        assert "splinefit.convergence_schedule(True)" in gpu_multi
        for source in (gpu_single, gpu_multi):
            assert "TOLERANCE_MULTI_START" not in source
            assert "TOLERANCE_SINGLE_START" not in source

    def test_spline_use_gpu_resolution(self):
        # no GPU on this machine unless Gpufit is installed
        assert localize._spline_use_gpu(None) is localize.GPUFIT_INSTALLED
        assert localize._spline_use_gpu(False) is False
        if not localize.GPUFIT_INSTALLED:
            with pytest.raises(ImportError, match="use_gpu=False"):
                localize._spline_use_gpu(True)

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

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    @pytest.mark.parametrize("link_xyz", [False, True])
    def test_pack_user_info_multichannel_layout(self, link_xyz, n_channels):
        n_channels = 3
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        # the photon-decoupled model reuses the multichannel blob verbatim -
        # only the parameter handling differs
        if link_xyz:
            calib = localize._as_link_xyz_calibration(calib)
        model = calib["model"]
        user_info = localize._pack_spline_user_info(calib)
        assert user_info.dtype == np.float32
        nx, ny, _ = calib["n_data"]
        ix, iy, iz = calib["n_intervals"]
        # header: [n_channels, nx, ny, nz=1, ix, iy, iz]
        np.testing.assert_array_equal(
            user_info[:7],
            np.array([n_channels, nx, ny, 1, ix, iy, iz], np.float32),
        )
        n_coeff = calib["coefficients"].size
        # coefficients, then the per-channel lateral affine channel-major (4
        # reals each). The CUDA models detect that trailing block purely by the
        # total size, so its length is part of the contract.
        assert user_info.size == 7 + n_coeff + 4 * n_channels
        # each channel's block is reordered to forward axis order and the
        # blocks are concatenated channel-major (outermost axis)
        np.testing.assert_array_equal(
            user_info[7 : 7 + n_coeff],
            localize._reorder_spline_coefficients_for_gpufit(
                calib["coefficients"], model
            ),
        )
        np.testing.assert_array_equal(
            user_info[7 + n_coeff :],
            np.tile(np.array([1.0, 0.0, 0.0, 1.0], np.float32), n_channels),
        )
        # without transforms the blob stops after the coefficients and the
        # CUDA models fall back to an identity per channel
        calib = dict(calib)
        calib.pop("channel_transforms")
        assert localize._pack_spline_user_info(calib).size == 7 + n_coeff

    def test_channel_roi_residuals_pure_translation_is_constant(self):
        # Detections sit on integer pixels, so a pure translation shifts every
        # box by the same fractional amount - the case the calibration absorbs.
        ids = pd.DataFrame(
            {"x": np.arange(40, 340, 7), "y": np.arange(60, 360, 7)}
        )
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        shifted = np.array([[1.0, 0.0, 12.3], [0.0, 1.0, -4.8]])
        res = localize.channel_roi_residuals(ids, [identity, shifted])
        assert res.shape == (len(ids), 2, 2)
        # channel 0 is the reference: its box is the detection itself
        np.testing.assert_array_equal(res[:, 0, :], 0.0)
        # frac(12.3) = .3 -> residual .3; frac(-4.8) -> .2
        np.testing.assert_allclose(res[:, 1, 0], 0.3, atol=1e-5)
        np.testing.assert_allclose(res[:, 1, 1], 0.2, atol=1e-5)
        assert np.ptp(res[:, 1, :], axis=0).max() < 1e-5

    def test_channel_roi_residuals_rotation_varies_across_field(self):
        # With any linear part the residual is no longer constant: it sweeps
        # the full +-0.5 px as soon as E@x moves by a pixel across the field.
        ids = pd.DataFrame(
            {"x": np.arange(0, 512, 3), "y": np.arange(0, 512, 3)}
        )
        theta = np.deg2rad(0.5)
        rotated = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
            ]
        )
        res = localize.channel_roi_residuals(ids, [np.eye(2, 3), rotated])
        assert np.all(np.abs(res) <= 0.5 + 1e-6)
        # spans most of the available range rather than sitting at one value
        assert np.ptp(res[:, 1, 0]) > 0.8
        # and it is not noise: it tracks the mapped position deterministically
        mapped = localize.apply_affine_transform(
            ids[["x", "y"]].to_numpy(float), rotated
        )
        np.testing.assert_allclose(
            res[:, 1, :], mapped - np.rint(mapped), atol=1e-6
        )

    def test_get_spots_multichannel_residuals_match_helper(
        self, movie, real_identifications
    ):
        # the extractor's by-product and the standalone helper must agree,
        # otherwise the model is told about a shift the ROIs do not have
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        skewed = np.array([[1.002, 0.004, 1.4], [-0.004, 1.001, -0.6]])
        transforms = [identity, skewed]
        ids = localize.multichannel_inbounds_ids(
            real_identifications, BOX, [movie, movie], transforms
        )
        spots, res = localize.get_spots_multichannel(
            [movie, movie],
            ids,
            BOX,
            [CAMERA_INFO, CAMERA_INFO],
            transforms,
            return_residuals=True,
        )
        assert spots.shape == (len(ids), BOX, BOX, 2)
        np.testing.assert_allclose(
            res, localize.channel_roi_residuals(ids, transforms), atol=1e-6
        )
        # default stays a bare array, so existing callers are unaffected
        assert isinstance(
            localize.get_spots_multichannel(
                [movie, movie],
                ids,
                BOX,
                [CAMERA_INFO, CAMERA_INFO],
                transforms,
            ),
            np.ndarray,
        )

    def test_pack_user_info_multichannel_residual_block(self):
        n_channels = 2
        n_fits = 5
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        n_coeff = calib["coefficients"].size
        rng = np.random.default_rng(0)
        residuals = rng.uniform(-0.5, 0.5, (n_fits, n_channels, 2)).astype(
            np.float32
        )
        user_info = localize._pack_spline_user_info(calib, residuals)
        assert user_info.dtype == np.float32
        # the CUDA model locates the block by total size, so the length and the
        # [fit][channel][x, y] order are both part of the wire contract
        base = 7 + n_coeff + 4 * n_channels
        assert user_info.size == base + n_fits * n_channels * 2
        np.testing.assert_array_equal(
            user_info[base:], residuals.ravel(order="C")
        )
        # omitting them leaves the previous blob byte for byte
        np.testing.assert_array_equal(
            localize._pack_spline_user_info(calib),
            user_info[:base],
        )

    def test_pack_user_info_rejects_unusable_residuals(self):
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        good = np.zeros((3, 2, 2), np.float32)
        # wrong channel count / rank
        with pytest.raises(ValueError, match="n_channels, 2"):
            localize._pack_spline_user_info(calib, np.zeros((3, 3, 2)))
        with pytest.raises(ValueError, match="n_channels, 2"):
            localize._pack_spline_user_info(calib, np.zeros((3, 2)))
        # the model finds the block at a fixed offset past the affine, so
        # without the affine the two would be confused for each other
        no_affine = dict(calib)
        no_affine.pop("channel_transforms")
        with pytest.raises(ValueError, match="affine"):
            localize._pack_spline_user_info(no_affine, good)
        # single-channel models have nowhere to put them
        with pytest.raises(ValueError, match="multichannel"):
            localize._pack_spline_user_info(
                _fake_spline_calibration(model="spline-3d"), good
            )

    def test_gpufit_batch_size_stays_within_bounds(self):
        # batching is what keeps Gpufit's chunk_index at 0, which the model's
        # residual indexing relies on; both bounds must hold for any input
        for n_points in (1, 169, 338, 2646, 10**7, 10**9):
            batch = localize._gpufit_batch_size(n_points)
            assert 1 <= batch <= localize._GPUFIT_MAX_FITS_PER_CALL
            assert (
                batch == 1
                or batch * n_points <= localize._GPUFIT_MAX_POINTS_PER_CALL
            )

    @pytest.mark.parametrize("gpu", SPLINE_CRLB_DEVICES)
    def test_spline_crlb_residual_equals_a_lateral_shift(self, gpu):
        # A ROI residual just moves where the spline is evaluated, so it must
        # be indistinguishable from moving the lateral parameter by the same
        # amount. This pins the sign and the placement of the residual term -
        # on whichever device computes it.
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        calib = dict(calib)
        calib["n_channels"] = 1
        calib["coefficients"] = calib["coefficients"][..., 1:2]
        calib["channel_transforms"] = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        box = calib["box"]
        rng = np.random.default_rng(1)
        theta = np.zeros((4, 5))  # [amplitude, x, y, z, offset]
        theta[:, 0], theta[:, 3], theta[:, 4] = 500.0, -8.0, 5.0
        theta[:, 1] = rng.uniform(-0.3, 0.3, 4)
        theta[:, 2] = rng.uniform(-0.3, 0.3, 4)
        dx, dy = 0.31, -0.24
        res = np.zeros((4, 1, 2))
        res[:, 0, 0], res[:, 0, 1] = dx, dy
        with_residual = _crlb(
            theta, calib, box, gpu=gpu, mle=True, residuals=res
        )
        moved = theta.copy()
        moved[:, 1] += dx
        moved[:, 2] += dy
        np.testing.assert_allclose(
            with_residual,
            _crlb(moved, calib, box, gpu=gpu, mle=True),
            rtol=1e-10,
        )

    @pytest.mark.parametrize("gpu", SPLINE_CRLB_DEVICES)
    def test_spline_crlb_affine_scales_lateral_variance(self, gpu):
        # The shared shift reaches a channel through its affine, so d(mu)/dp
        # carries A^T. For an isotropic A = s*I the x/y variances must come out
        # exactly s^-2 times the identity case evaluated at the same position,
        # and z must be untouched - which is what the chain rule buys.
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        calib = dict(calib)
        calib["n_channels"] = 1
        calib["coefficients"] = calib["coefficients"][..., 1:2]
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        calib["channel_transforms"] = [identity]
        box = calib["box"]
        rng = np.random.default_rng(2)
        theta = np.zeros((4, 5))
        theta[:, 0], theta[:, 3], theta[:, 4] = 500.0, -8.0, 5.0
        theta[:, 1] = rng.uniform(-0.3, 0.3, 4)
        theta[:, 2] = rng.uniform(-0.3, 0.3, 4)

        s = 1.05
        scaled = dict(calib)
        scaled["channel_transforms"] = [[[s, 0.0, 0.0], [0.0, s, 0.0]]]
        crlb_scaled = _crlb(theta, scaled, box, gpu=gpu, mle=True)
        # identity affine reaches the same evaluation point at s * p
        moved = theta.copy()
        moved[:, 1] *= s
        moved[:, 2] *= s
        crlb_identity = _crlb(moved, calib, box, gpu=gpu, mle=True)
        np.testing.assert_allclose(
            crlb_scaled[:, 0], crlb_identity[:, 0] / s**2, rtol=1e-8
        )
        np.testing.assert_allclose(
            crlb_scaled[:, 1], crlb_identity[:, 1] / s**2, rtol=1e-8
        )
        np.testing.assert_allclose(
            crlb_scaled[:, 2], crlb_identity[:, 2], rtol=1e-8
        )

    @pytest.mark.parametrize("gpu", SPLINE_CRLB_DEVICES)
    @pytest.mark.parametrize("link_xyz", [False, True])
    def test_spline_crlb_default_geometry_is_unchanged(self, link_xyz, gpu):
        # identity affine + zero residual must reproduce the pre-correction
        # result exactly, so single-channel output cannot drift
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        if link_xyz:
            calib = localize._as_link_xyz_calibration(calib)
            theta = np.zeros((3, 7))  # [x, y, z, N0, N1, bg0, bg1]
            theta[:, 2] = -8.0
            theta[:, 3:5] = 400.0
            theta[:, 5:] = 5.0
        else:
            theta = np.zeros((3, 5))  # [amplitude, x, y, z, offset]
            theta[:, 0], theta[:, 3], theta[:, 4] = 500.0, -8.0, 5.0
        box = calib["box"]
        n_channels = localize._spline_n_channels(calib)
        explicit = _crlb(
            theta,
            calib,
            box,
            gpu=gpu,
            mle=True,
            residuals=np.zeros((len(theta), n_channels, 2)),
        )
        implied = _crlb(theta, calib, box, gpu=gpu, mle=True)
        assert np.isfinite(implied).all()
        np.testing.assert_array_equal(explicit, implied)

    def test_spline_channel_affines_defaults_to_identity(self):
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=3
        )
        aff = localize._spline_channel_affines(calib, 3)
        np.testing.assert_array_equal(
            aff, np.tile([1.0, 0.0, 0.0, 1.0], (3, 1))
        )
        # a calibration without usable transforms falls back to the identity,
        # matching what the CUDA models do with no affine block
        stripped = dict(calib)
        stripped.pop("channel_transforms")
        np.testing.assert_array_equal(
            localize._spline_channel_affines(stripped, 3), aff
        )

    def test_default_n_z_starts_from_calibration_depth(self):
        # one seed per ~20 calibration planes, bounded; 2D has no z to be
        # degenerate in and must stay on a single start
        calib = _fake_spline_calibration(model="spline-3d")
        assert localize._default_n_z_starts(calib) == 5
        assert (
            localize._default_n_z_starts(
                _fake_spline_calibration(model="spline-2d")
            )
            == 1
        )
        deep = dict(calib)
        deep["n_data"] = [7, 7, 600]  # far past the upper bound
        assert localize._default_n_z_starts(deep) == localize._Z_STARTS_MAX
        shallow = dict(calib)
        shallow["n_data"] = [7, 7, 4]
        assert localize._default_n_z_starts(shallow) == localize._Z_STARTS_MIN
        # a calibration that does not describe a z axis cannot be multi-started
        assert localize._default_n_z_starts({"n_data": [7, 7]}) == 1

    @pytest.mark.parametrize("model", ["spline-3d", "spline-3d-multichannel"])
    def test_fit_spots_multistarts_by_default(self, monkeypatch, model):
        # the whole fit runs once per seed, so "did it multi-start" is exactly
        # "how many times was the fitter called"
        calib = _fake_spline_calibration(model=model)
        expected = localize._default_n_z_starts(calib)
        assert expected > 1
        n_channels = localize._spline_n_channels(calib)
        n_params = 5
        box = calib["box"]
        spots = np.zeros((3, box, box), np.float32)
        if model == "spline-3d-multichannel":
            spots = np.zeros((3, box, box, n_channels), np.float32)

        calls = []

        def fake_run(spots_, calibration, **kw):
            calls.append(kw)
            n = len(spots_)
            return (
                np.zeros((n, n_params), np.float32),
                np.zeros(n, np.int32),
                np.full(n, 1.0),
                np.ones(n, np.int32),
                0.0,
            )

        monkeypatch.setattr(localize, "_run_gpufit_spline", fake_run)
        localize.fit_spots_gpufit_spline(spots, calib, mle=True)
        assert len(calls) == expected
        # seeded runs must use the tight convergence, else neighbouring axial
        # minima are indistinguishable and the multi-start is pointless
        assert all(c["tolerance"] == 1e-4 for c in calls)
        assert all(c["max_number_iterations"] == 100 for c in calls)
        # and the seeds must actually differ, spanning the stack
        z_col = 3
        seeds = [float(c["initial_parameters"][0, z_col]) for c in calls]
        assert len(set(seeds)) == expected
        assert min(seeds) == pytest.approx(-(calib["n_data"][2] - 1))
        assert max(seeds) == pytest.approx(0.0)

    def test_ratiometric_multistarts_every_hypothesis(self, monkeypatch):
        # colour is decided by comparing hypotheses' residuals, so they all
        # need the same axial search - otherwise a hypothesis can win merely by
        # having stumbled into a better z minimum
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        n_starts = localize._default_n_z_starts(calib)
        n_hyp = 3
        calls = []

        def fake_multistart(spots_, calibration, **kw):
            calls.append(kw["n_z_starts"])
            n = len(spots_)
            return (
                np.zeros((n, 5), np.float32),
                np.full(n, 1.0),
                np.ones(n, bool),
                np.ones(n, np.int32),
            )

        monkeypatch.setattr(
            # the device-agnostic dispatcher, which is what the ratiometric
            # fitter calls (it routes to the GPU or CPU multi-start)
            localize,
            "_fit_spline_multistart",
            fake_multistart,
        )
        # exercise only the hypothesis loop; locs building needs a real fit
        monkeypatch.setattr(
            localize,
            "locs_from_fits_spline",
            lambda ids, theta, box, em, cal, **kw: pd.DataFrame(
                {"frame": np.asarray(ids["frame"])}
            ),
        )
        box = calib["box"]
        n_spots = 4
        spots = np.zeros((n_spots, box, box, 2), np.float32)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n_spots),
                "x": np.full(n_spots, 20),
                "y": np.full(n_spots, 20),
                "net_gradient": np.ones(n_spots),
            }
        )
        monkeypatch.setattr(
            localize,
            "get_spots_multichannel",
            lambda *a, **kw: (
                (spots, np.zeros((n_spots, 2, 2), np.float32))
                if kw.get("return_residuals")
                else spots
            ),
        )
        monkeypatch.setattr(
            localize, "multichannel_inbounds_ids", lambda i, *a, **kw: i
        )
        localize.fit_spline_multichannel_ratiometric(
            [None, None],
            [CAMERA_INFO, CAMERA_INFO],
            ids,
            box,
            calib,
            photon_ratios=np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
        )
        assert calls == [n_starts] * n_hyp

    def test_fit_spots_single_start_is_still_reachable(self, monkeypatch):
        calib = _fake_spline_calibration(model="spline-3d")
        box = calib["box"]
        calls = []

        def fake_run(spots_, calibration, **kw):
            calls.append(kw)
            n = len(spots_)
            return (
                np.zeros((n, 5), np.float32),
                np.zeros(n, np.int32),
                np.full(n, 1.0),
                np.ones(n, np.int32),
                0.0,
            )

        monkeypatch.setattr(localize, "_run_gpufit_spline", fake_run)
        localize.fit_spots_gpufit_spline(
            np.zeros((3, box, box), np.float32), calib, n_z_starts=1
        )
        assert len(calls) == 1
        # the single-start path keeps the loose defaults
        assert "tolerance" not in calls[0]

    def test_link_xyz_model_ids_cover_the_supported_range(self):
        # the map and the advertised maximum must not drift apart
        assert sorted(localize._LINK_XYZ_MODEL_ID_NAMES) == list(
            range(2, localize._LINK_XYZ_MAX_CHANNELS + 1)
        )

    def test_model_id_mapping_link_xyz(self):
        if not localize.GPUFIT_INSTALLED:
            pytest.skip("ModelID enum needs the Gpufit binding")
        model = localize._LINK_XYZ_MODEL
        # Gpufit fixes a model's parameter count at compile time, and link-XYZ
        # needs 3 + 2*n_channels, so the channel count is encoded in the id.
        # These literals are the wire contract with Gpufit.dll.
        for n_channels, model_id in {
            2: 15,
            3: 16,
            4: 17,
            5: 18,
            6: 19,
        }.items():
            assert localize._spline_model_id(model, n_channels) == model_id
        # a bare call keeps meaning the original 2-channel model
        assert localize._spline_model_id(model) == 15
        for bad in (0, 1, localize._LINK_XYZ_MAX_CHANNELS + 1):
            with pytest.raises(ValueError):
                localize._spline_model_id(model, bad)
        # the shared-amplitude model has 5 parameters whatever the channel
        # count, so it ignores the argument
        assert localize._spline_model_id("spline-3d-multichannel", 6) == (
            localize.gf.ModelID.SPLINE_3D_MULTICHANNEL
        )

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_as_link_xyz_calibration_accepts_supported_channels(
        self, n_channels
    ):
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        link = localize._as_link_xyz_calibration(calib)
        assert link["model"] == localize._LINK_XYZ_MODEL
        # shallow copy: the original is untouched and the (large) coefficient
        # table is shared, not duplicated
        assert calib["model"] == "spline-3d-multichannel"
        assert link["coefficients"] is calib["coefficients"]

    @pytest.mark.parametrize("n_channels", [1, 7, 12])
    def test_as_link_xyz_calibration_rejects_unsupported_channels(
        self, n_channels
    ):
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        with pytest.raises(ValueError, match="2 to 6 channels"):
            localize._as_link_xyz_calibration(calib)

    def test_as_link_xyz_calibration_requires_multichannel(self):
        with pytest.raises(ValueError):
            localize._as_link_xyz_calibration(
                _fake_spline_calibration(model="spline-3d")
            )

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_initial_parameters_link_xyz_per_channel(self, n_channels):
        calib = localize._as_link_xyz_calibration(
            _fake_spline_calibration(
                model="spline-3d-multichannel", n_channels=n_channels
            )
        )
        rng = np.random.default_rng(0)
        spots = rng.random((4, BOX, BOX, n_channels)).astype(np.float32)
        # scale the channels apart so a transposed or channel-major/minor
        # mix-up cannot pass
        spots = spots * (1.0 + np.arange(n_channels, dtype=np.float32))
        init = localize._initial_parameters_spline(spots, calib)
        # [x, y, z, N_0..N_{c-1}, bg_0..bg_{c-1}]
        assert init.shape == (4, 3 + 2 * n_channels)
        np.testing.assert_allclose(init[:, :2], 0.0)
        np.testing.assert_allclose(init[:, 2], -calib["z_center"])
        ch_max = spots.max(axis=(1, 2))
        ch_min = spots.min(axis=(1, 2))
        np.testing.assert_allclose(
            init[:, 3 : 3 + n_channels], ch_max - ch_min, rtol=1e-5
        )
        np.testing.assert_allclose(
            init[:, 3 + n_channels :], ch_min, rtol=1e-5
        )

    def test_run_gpufit_spline_rejects_mismatched_parameter_width(self):
        # Gpufit sizes its output from initial_parameters but strides the
        # device buffers by the ModelID's parameter count, so a too-narrow
        # array would corrupt memory instead of raising. The guard runs before
        # anything reaches the GPU.
        if not localize.GPUFIT_INSTALLED:
            pytest.skip("needs the Gpufit binding")
        n_channels = 3
        calib = localize._as_link_xyz_calibration(
            _fake_spline_calibration(
                model="spline-3d-multichannel", n_channels=n_channels
            )
        )
        spots = np.zeros((2, BOX, BOX, n_channels), dtype=np.float32)
        too_narrow = np.zeros((2, 3 + 2 * (n_channels - 1)), dtype=np.float32)
        with pytest.raises(ValueError, match="initial parameters"):
            localize._run_gpufit_spline(
                spots, calib, initial_parameters=too_narrow
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

    def test_multichannel_inbounds_ids_drops_edge_spots(self, movie):
        """A detection whose box falls outside the frame in any channel is
        dropped, so the joint extractor never reads an out-of-bounds box."""
        height, width = movie.shape[1], movie.shape[2]
        r = BOX // 2
        ids = pd.DataFrame(
            {
                "frame": [0, 0, 0, 0],
                # centred (ok), top edge (box off the top), left edge, and a
                # spot that only leaves the frame once mapped in channel 1.
                "x": [width // 2, width // 2, 0, width - r - 1],
                "y": [height // 2, 0, height // 2, height // 2],
                "net_gradient": [1.0, 1.0, 1.0, 1.0],
            }
        )
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # channel 1 shifts +r in x, pushing the last (near-right-edge) spot out
        shift = np.array([[1.0, 0.0, float(r + 1)], [0.0, 1.0, 0.0]])
        kept = localize.multichannel_inbounds_ids(
            ids, BOX, [movie, movie], [identity, shift]
        )
        # only the centred spot survives in both channels
        assert len(kept) == 1
        assert int(kept["x"].iloc[0]) == width // 2
        assert int(kept["y"].iloc[0]) == height // 2
        # extraction on the filtered ids must not raise
        stacked = localize.get_spots_multichannel(
            [movie, movie],
            kept,
            BOX,
            [CAMERA_INFO, CAMERA_INFO],
            [identity, shift],
        )
        assert stacked.shape == (1, BOX, BOX, 2)


class TestCrossChannelLinking:
    """Reference detections must be reduced to those found in every channel:
    the joint multichannel model shares x, y, z across channels, so a spot
    missing from one channel would be fitted there against background only."""

    IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    SHIFT_X = [[1.0, 0.0, 100.0], [0.0, 1.0, 0.0]]

    @staticmethod
    def _ids(frames, xs, ys):
        return pd.DataFrame(
            {
                "frame": np.asarray(frames, dtype=int),
                "x": np.asarray(xs, dtype=float),
                "y": np.asarray(ys, dtype=float),
                "net_gradient": np.ones(len(xs), dtype=float),
            }
        )

    def test_requires_a_match_in_every_channel(self):
        ref = self._ids([0, 0, 0, 1], [10, 20, 30, 40], [10, 20, 30, 40])
        # channel 1 lives 100 px to the right; it sees ref spots 0, 2 and 3
        ch1 = self._ids([0, 0, 1], [110.4, 130.2, 140], [10.2, 30, 40])
        # channel 2 is aligned with the reference; it sees ref spots 0 and 1
        ch2 = self._ids([0, 0], [10.1, 20], [9.9, 20])
        linked = localize.link_identifications_multichannel(
            [ref, ch1, ch2], [self.IDENTITY, self.SHIFT_X, self.IDENTITY], 3.0
        )
        # only ref spot 0 is present in all three channels
        assert list(linked) == [True, False, False, False]

    def test_pairing_is_one_to_one(self):
        """Two reference spots near one detection: only the closer one links."""
        ref = self._ids([0, 0], [10, 11], [10, 10])
        target = self._ids([0], [10.2], [10])
        linked = localize.link_identifications_multichannel(
            [ref, target], [self.IDENTITY, self.IDENTITY], 3.0
        )
        assert list(linked) == [True, False]

    def test_tolerance_and_frame_order(self):
        ref = self._ids([1, 0, 0], [40, 10, 20], [40, 10, 20])
        ch1 = self._ids([0, 1], [10.1, 40], [10, 40])
        args = ([ref, ch1], [self.IDENTITY, self.IDENTITY])
        # unsorted reference frames are handled (matching is per frame)
        assert list(
            localize.link_identifications_multichannel(*args, 3.0)
        ) == [
            True,
            True,
            False,
        ]
        # a tight tolerance rejects the 0.1 px mismatch but keeps the exact one
        assert list(
            localize.link_identifications_multichannel(*args, 0.05)
        ) == [True, False, False]

    def test_unidentified_channels_are_skipped(self):
        ref = self._ids([0, 0], [10, 20], [10, 20])
        ch1 = self._ids([0], [110], [10])
        transforms = [self.IDENTITY, self.SHIFT_X, self.IDENTITY]
        # channel 2 was never identified -> link against channel 1 only
        linked = localize.link_identifications_multichannel(
            [ref, ch1, None], transforms, 3.0
        )
        assert list(linked) == [True, False]
        # no other channel identified at all -> nothing is linked
        linked = localize.link_identifications_multichannel(
            [ref, None, None], transforms, 3.0
        )
        assert not linked.any()

    def test_filter_wrapper_counts_and_passthrough(self):
        ref = self._ids([0, 0], [10, 20], [10, 20])
        ch1 = self._ids([0], [110], [10])
        kept, n_kept, n_total = localize.filter_linked_identifications(
            [ref, ch1], [self.IDENTITY, self.SHIFT_X], box=2
        )
        assert (n_kept, n_total) == (1, 2)
        assert len(kept) == 1 and kept["x"].iloc[0] == 10
        # without identifications in any other channel the table is unchanged,
        # so an un-identified set degrades to the unfiltered behaviour
        same, n_kept, n_total = localize.filter_linked_identifications(
            [ref, None], [self.IDENTITY, self.IDENTITY], box=2
        )
        assert same is ref and (n_kept, n_total) == (2, 2)

    def test_progress_is_monotone_across_channels(self):
        ref = self._ids([0, 0, 1], [10, 20, 30], [10, 20, 30])
        ch1 = self._ids([0, 1], [110, 130], [10, 30])
        ch2 = self._ids([0, 1], [10, 30], [10, 30])
        seen = []
        localize.link_identifications_multichannel(
            [ref, ch1, ch2],
            [self.IDENTITY, self.SHIFT_X, self.IDENTITY],
            3.0,
            progress_callback=seen.append,
        )
        # one 0 -> n_ref progression for the whole linking pass, not one per
        # channel (the GUI shows this as a single status count)
        assert seen and seen[-1] == len(ref)
        assert all(b >= a for a, b in zip(seen, seen[1:]))


class TestCrossRegionLinkingSplitFov:
    """Split-FOV: one movie is identified as a whole, so a single table holds
    every region's detections. They must be split by region and linked across
    regions, leaving one fitted spot per molecule - not one per detection."""

    # two side-by-side 32x32 regions of a 32x64 movie; channel 1 sits half a
    # pixel to the right and a quarter pixel up of the perfect split
    REGIONS = [[[0, 0], [32, 32]], [[0, 32], [32, 64]]]
    AFFINES = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]],
    ]

    def _calibration(self, reference=0):
        transforms = localize.compose_region_transforms(
            [localize._normalize_rect(r) for r in self.REGIONS],
            [np.asarray(a, dtype=float) for a in self.AFFINES],
        )
        return {
            "model": "spline-3d-multichannel",
            "split_fov": True,
            "n_channels": 2,
            "reference": reference,
            "regions": self.REGIONS,
            "channel_affines": self.AFFINES,
            "channel_transforms": [t.tolist() for t in transforms],
        }, transforms

    @staticmethod
    def _ids(xy):
        xy = np.asarray(xy, dtype=float)
        return pd.DataFrame(
            {
                "frame": np.zeros(len(xy), dtype=int),
                "x": xy[:, 0],
                "y": xy[:, 1],
                "net_gradient": np.ones(len(xy), dtype=float),
            }
        )

    def _detections(self, transforms):
        """Three reference-region spots, two of which have a counterpart in
        region 1, plus one unpaired detection in region 1."""
        ref = np.array([[5.0, 5.0], [10.0, 20.0], [25.0, 8.0]])
        partners = (
            localize.apply_affine_transform(ref[:2], transforms[1]) + 0.3
        )
        return self._ids(np.vstack([ref, partners, [[50.0, 30.0]]])), ref

    def test_keeps_reference_spots_found_in_every_region(self):
        calib, transforms = self._calibration()
        ids, ref = self._detections(transforms)
        linked, n_kept, n_total = (
            localize.filter_linked_identifications_split_fov(ids, calib, box=7)
        )
        # 6 detections over both regions -> 3 in the reference region -> the 2
        # molecules seen in both. The count the fit progress must show.
        assert len(ids) == 6
        assert (n_kept, n_total) == (2, 3)
        assert np.allclose(linked[["x", "y"]].to_numpy(), ref[:2])

    def test_regions_may_be_re_placed(self):
        """Drawn ROIs shifted off the calibration positions link the same."""
        calib, transforms = self._calibration()
        ids, _ = self._detections(transforms)
        shifted = ids.copy()
        shifted["x"] += 3.0
        shifted["y"] += 2.0
        regions = [
            [[y0 + 2, x0 + 3], [y1 + 2, x1 + 3]]
            for (y0, x0), (y1, x1) in self.REGIONS
        ]
        _, n_kept, n_total = localize.filter_linked_identifications_split_fov(
            shifted, calib, box=7, regions=regions
        )
        assert (n_kept, n_total) == (2, 3)

    def test_non_zero_reference_region(self):
        """A calibration whose reference is not region 0 links against it."""
        calib, transforms = self._calibration(reference=1)
        ids, _ = self._detections(transforms)
        # transforms must map the reference region (1) into every region
        inverse = np.array([[1.0, 0.0, -32.5], [0.0, 1.0, 0.25]])
        calib["channel_transforms"] = [
            inverse.tolist(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
        linked, n_kept, n_total = (
            localize.filter_linked_identifications_split_fov(ids, calib, box=7)
        )
        # region 1 holds the 2 counterparts plus one unpaired detection
        assert (n_kept, n_total) == (2, 3)
        assert (linked["x"] > 32).all()

    def test_misregistration_links_nothing(self):
        calib, transforms = self._calibration()
        ids, _ = self._detections(transforms)
        # the stored transform puts channel 1 40 px off its true position
        calib["channel_transforms"] = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 32.5], [0.0, 1.0, 39.75]],
        ]
        with pytest.warns(UserWarning, match="registration"):
            _, n_kept, n_total = (
                localize.filter_linked_identifications_split_fov(
                    ids, calib, box=7
                )
            )
        assert (n_kept, n_total) == (0, 3)

    def test_other_regions_unidentified_passes_through(self):
        """Identifying only the reference region (e.g. a restricted ROI) keeps
        every reference detection, as for separately loaded channels."""
        calib, _ = self._calibration()
        ids = self._ids([[5.0, 5.0], [10.0, 20.0]])
        kept, n_kept, n_total = (
            localize.filter_linked_identifications_split_fov(ids, calib, box=7)
        )
        assert (n_kept, n_total) == (2, 2)
        assert len(kept) == 2

    def test_confine_to_region(self):
        ids = self._ids([[5.0, 5.0], [40.0, 5.0], [31.9, 31.9]])
        inside = localize.confine_to_region(ids, self.REGIONS[0])
        # the region is half-open: [y_min, y_max) x [x_min, x_max)
        assert list(inside["x"]) == [5.0, 31.9]

    def test_geometry_requires_a_split_fov_calibration(self):
        with pytest.raises(ValueError, match="split-FOV"):
            localize.split_fov_fit_geometry({"model": "spline-3d"})


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

    @pytest.mark.skipif(
        not localize.SPLINE_CRLB_CUDA_AVAILABLE,
        reason="requires a CUDA-capable GPU (numba-cuda)",
    )
    def test_gpu_matches_cpu_on_real_coefficients(self):
        """Parity on a genuinely non-separable, Gpuspline-generated coefficient
        table - the analytic test spline cannot exercise that layout."""
        calib, _, amplitude, offset = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        rng = np.random.default_rng(0)
        n = 500
        thetas = np.zeros((n, 5))
        thetas[:, 0] = amplitude * rng.uniform(0.5, 2.0, n)
        thetas[:, 1] = rng.uniform(-0.7, 0.7, n)
        thetas[:, 2] = rng.uniform(-0.7, 0.7, n)
        thetas[:, 3] = -(calib["z_center"] + rng.uniform(-6.0, 6.0, n))
        thetas[:, 4] = offset * rng.uniform(0.5, 2.0, n)
        for mle in (True, False):
            np.testing.assert_allclose(
                _crlb(thetas, calib, box, mle=mle, gpu=True),
                _crlb(thetas, calib, box, mle=mle, gpu=False),
                rtol=1e-6,
            )


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


def _crlb(theta, calibration, box, *, gpu, **kwargs):
    """``localize._spline_crlb`` pinned to one device.

    Production dispatch is automatic (GPU when present, CPU otherwise), so
    hiding the GPU is the only way to exercise the CPU path deliberately - and
    the only way to compare the two against each other."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(localize, "SPLINE_CRLB_CUDA_AVAILABLE", gpu)
        return localize._spline_crlb(theta, calibration, box, **kwargs)


def _shared_theta(n, rng, n_params=5, nz=21):
    """Well-spread fitted parameters for the shared-amplitude spline models,
    ``[amplitude, x, y, (z,) offset]``."""
    theta = np.zeros((n, n_params))
    theta[:, 0] = rng.uniform(500.0, 8000.0, n)
    theta[:, 1] = rng.uniform(-0.7, 0.7, n)
    theta[:, 2] = rng.uniform(-0.7, 0.7, n)
    if n_params == 5:
        # native z = -z_shift, kept clear of the ends of the calibration stack
        theta[:, 3] = -rng.uniform(4.0, nz - 5.0, n)
    theta[:, -1] = rng.uniform(1.0, 50.0, n)
    return theta


def _with_channel_geometry(calib, n_locs, rng):
    """Give ``calib`` a non-identity per-channel affine and return matching
    sub-pixel ROI residuals.

    Without these, every channel is evaluated at the same place and the
    geometry terms in the CRLB kernels are dead code - so a GPU/CPU comparison
    on a plain calibration cannot tell whether either side applies them. The
    reference channel keeps the identity and a zero residual, as the real
    pipeline does (see :func:`localize.channel_roi_residuals`)."""
    n_channels = localize._spline_n_channels(calib)
    calib = dict(calib)
    calib["channel_transforms"] = [
        (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            if c == 0
            else [
                [1.0 + 0.03 * c, 0.02 * c, 0.0],
                [-0.015 * c, 1.0 - 0.01 * c, 0.0],
            ]
        )
        for c in range(n_channels)
    ]
    residuals = rng.uniform(-0.5, 0.5, (n_locs, n_channels, 2))
    residuals[:, 0, :] = 0.0
    return calib, residuals


def _link_xyz_calib_and_theta(n_channels, n_locs, rng, nz=21):
    """Link-XYZ calibration on the separable Gaussian spline, plus a batch of
    fitted parameters ``[x, y, z, N_0.., bg_0..]`` with unequal per-channel
    photons so a channel mix-up cannot pass unnoticed."""
    calib, _ = _gauss_spline_calibration(
        model="spline-3d-multichannel", n_channels=n_channels, nz=nz
    )
    calib = localize._as_link_xyz_calibration(calib)
    theta = np.zeros((n_locs, 3 + 2 * n_channels))
    theta[:, 0] = rng.uniform(-0.7, 0.7, n_locs)
    theta[:, 1] = rng.uniform(-0.7, 0.7, n_locs)
    theta[:, 2] = -rng.uniform(4.0, nz - 5.0, n_locs)
    theta[:, 3 : 3 + n_channels] = rng.uniform(
        200.0, 3000.0, (n_locs, n_channels)
    )
    theta[:, 3 + n_channels :] = rng.uniform(2.0, 40.0, (n_locs, n_channels))
    return calib, theta


class TestSplineLinkXyzCRLB:
    """Variances of the photon-decoupled (link-XYZ) spline model, the
    ``(3 + 2*n_channels)``-parameter block-sparse kernel. CPU only; the GPU
    parity tests live in ``TestSplineCRLBGPU``."""

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    @pytest.mark.parametrize("mle", [True, False])
    def test_shape_and_positivity(self, n_channels, mle):
        rng = np.random.default_rng(0)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 5, rng)
        var = _crlb(theta, calib, BOX, mle=mle, gpu=False)
        assert var.shape == (5, 3 + 2 * n_channels)
        assert np.all(np.isfinite(var)) and np.all(var > 0)

    def test_channel_order_is_carried_through(self):
        """Reordering the channels permutes the per-channel variances and
        leaves the shared x/y/z ones alone. Guards the block-sparse indexing
        (``3 + ch`` for photons, ``3 + n_channels + ch`` for background), which
        is where a channel mix-up would hide."""
        n_channels = 4
        calib, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        # Make the channels distinguishable: without this a permutation is a
        # no-op and the test proves nothing.
        coeff = np.asarray(calib["coefficients"], dtype=np.float32).copy()
        for c in range(n_channels):
            coeff[..., c] *= 0.5 + 0.4 * c
        calib["coefficients"] = coeff
        calib = localize._as_link_xyz_calibration(calib)

        rng = np.random.default_rng(1)
        _, theta = _link_xyz_calib_and_theta(n_channels, 6, rng)
        var = _crlb(theta, calib, BOX, gpu=False)

        perm = np.array([2, 0, 3, 1])
        permuted = dict(calib)
        permuted["coefficients"] = coeff[..., perm]
        theta_p = theta.copy()
        theta_p[:, 3 : 3 + n_channels] = theta[:, 3 + perm]
        theta_p[:, 3 + n_channels :] = theta[:, 3 + n_channels + perm]
        var_p = _crlb(theta_p, permuted, BOX, gpu=False)

        np.testing.assert_allclose(var_p[:, :3], var[:, :3], rtol=1e-9)
        np.testing.assert_allclose(
            var_p[:, 3 : 3 + n_channels],
            var[:, 3 + perm],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            var_p[:, 3 + n_channels :],
            var[:, 3 + n_channels + perm],
            rtol=1e-9,
        )

    @pytest.mark.parametrize("n_channels", [2, 4])
    def test_lsq_variance_geq_crlb(self, n_channels):
        # Least squares is not efficient for Poisson data.
        rng = np.random.default_rng(2)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 4, rng)
        crlb = _crlb(theta, calib, BOX, mle=True, gpu=False)
        lsq = _crlb(theta, calib, BOX, mle=False, gpu=False)
        assert np.all(lsq >= crlb * (1 - 1e-6))

    def test_nan_theta_row_isolated(self):
        rng = np.random.default_rng(3)
        calib, theta = _link_xyz_calib_and_theta(2, 2, rng)
        theta[1, 3] = np.nan
        var = _crlb(theta, calib, BOX, gpu=False)
        assert np.all(np.isfinite(var[0]))
        assert np.all(np.isnan(var[1]))

    def test_progress_callback_and_console(self):
        rng = np.random.default_rng(4)
        calib, theta = _link_xyz_calib_and_theta(2, 120, rng)
        seen = []
        _crlb(theta, calib, BOX, progress_callback=seen.append, gpu=False)
        assert seen and seen[-1] == len(theta) and seen == sorted(seen)
        _crlb(theta, calib, BOX, progress_callback="console", gpu=False)


class TestSplineCRLBEMCCD:
    """EMCCD excess noise in the spline uncertainties. Stochastic electron
    multiplication doubles every pixel's variance on top of the Poisson term,
    so every reported variance has to double with it - for both estimators,
    since that is a property of the detector and not of the fit. Matches
    ``gausslq.localization_precision`` and ``_gauss_crlb``."""

    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize(
        "model", ["spline-2d", "spline-3d", "spline-3d-multichannel"]
    )
    def test_em_doubles_the_variance(self, model, mle):
        calib, _ = _gauss_spline_calibration(model=model, n_channels=2)
        rng = np.random.default_rng(0)
        theta = _shared_theta(
            32, rng, n_params=4 if model == "spline-2d" else 5
        )
        plain = localize._spline_crlb(theta, calib, BOX, mle=mle)
        em = localize._spline_crlb(theta, calib, BOX, mle=mle, em=True)
        np.testing.assert_allclose(em, 2.0 * plain, rtol=1e-12)

    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize("n_channels", [2, 4])
    def test_em_doubles_the_variance_link_xyz(self, n_channels, mle):
        rng = np.random.default_rng(1)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 16, rng)
        plain = localize._spline_crlb(theta, calib, BOX, mle=mle)
        em = localize._spline_crlb(theta, calib, BOX, mle=mle, em=True)
        np.testing.assert_allclose(em, 2.0 * plain, rtol=1e-12)

    @pytest.mark.parametrize("mle", [True, False])
    def test_em_reaches_the_reported_precisions(self, mle):
        """The end-to-end check: ``em`` must survive the trip through
        ``locs_from_fits_spline`` into lpx/lpy/lpz/photons_unc/bg_unc. It used
        to be accepted there and silently dropped, leaving EMCCD precisions a
        factor sqrt(2) too optimistic."""
        calib, _ = _gauss_spline_calibration(model="spline-3d")
        rng = np.random.default_rng(2)
        theta = _shared_theta(24, rng)
        ids = pd.DataFrame(
            {
                "frame": np.arange(len(theta), dtype=np.uint32),
                "x": np.full(len(theta), 20.0),
                "y": np.full(len(theta), 30.0),
                "net_gradient": np.full(len(theta), 100.0),
            }
        )
        cols = ["lpx", "lpy", "lpz", "photons_unc", "bg_unc"]
        plain = localize.locs_from_fits_spline(
            ids, theta, BOX, False, calib, mle=mle
        )
        em = localize.locs_from_fits_spline(
            ids, theta, BOX, True, calib, mle=mle
        )
        for col in cols:
            np.testing.assert_allclose(
                em[col].to_numpy(),
                np.sqrt(2.0) * plain[col].to_numpy(),
                rtol=1e-5,
            )


requires_crlb_gpu = pytest.mark.skipif(
    not localize.SPLINE_CRLB_CUDA_AVAILABLE,
    reason="requires a CUDA-capable GPU (numba-cuda)",
)


class TestSplineCRLBGPU:
    """The numba.cuda spline CRLB path.

    The GPU reproduces the CPU kernels plus ``numpy.linalg.pinv`` rather than
    approximating them, so these are parity tests, run through :func:`_crlb` to
    pin each side to one device. The separable Gaussian test PSF makes that a
    demanding comparison: ``dPhi/dz`` is proportional to ``Phi``, so the z and
    amplitude columns are collinear and the information matrix is exactly rank
    deficient - both paths have to truncate the same mode for the answers to
    agree at all.
    """

    # Headroom for the pseudo-inverse of that rank-deficient matrix; on a real
    # (well-conditioned) calibration the two agree to ~1e-8 relative.
    RTOL = 1e-5

    def test_no_cuda_uses_the_cpu_silently(self, monkeypatch):
        """Without a CUDA device the CPU kernels are used, with no warning and
        no way (or need) for the caller to ask for anything else."""
        monkeypatch.setattr(localize, "SPLINE_CRLB_CUDA_AVAILABLE", False)
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[3000.0, 0.1, -0.1, 15.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = localize._spline_crlb(theta, calib, BOX)
        np.testing.assert_array_equal(got, _crlb(theta, calib, BOX, gpu=False))

    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_unsolvable_rows_are_recomputed_on_the_cpu(
        self, monkeypatch, model
    ):
        """Rows the device reports as unsolvable must come back with the CPU's
        pinv numbers, not whatever the kernel left behind. Driven through a stub
        device driver so it runs without a GPU - this fallback is the only
        structural difference between the two paths, so it needs a test."""
        monkeypatch.setattr(localize, "SPLINE_CRLB_CUDA_AVAILABLE", True)
        calib, _ = _gauss_spline_calibration(model=model)
        rng = np.random.default_rng(5)
        n_params = 4 if model == "spline-2d" else 5
        theta = _shared_theta(9, rng, n_params=n_params)
        expected = _crlb(theta, calib, BOX, gpu=False)

        def stub(
            coeff,
            aff,
            res,
            box,
            amp,
            xs,
            ys,
            ze,
            off,
            finite,
            mu_floor,
            mle,
            progress_callback=None,
        ):
            out = localize._spline_crlb_cpu(
                np.asarray(coeff, dtype=np.float64),
                aff,
                res,
                box,
                amp,
                xs,
                ys,
                ze,
                off,
                finite,
                mle,
            )
            failed = np.zeros(len(amp), dtype=bool)
            failed[::2] = True
            out[failed] = 12345.0  # device garbage the host must discard
            return out, failed

        monkeypatch.setattr(localize, "_spline_crlb_cuda", stub)
        np.testing.assert_allclose(
            _crlb(theta, calib, BOX, gpu=True), expected
        )

    def test_device_error_falls_back_and_warns_once(self, monkeypatch):
        """A device that is present but fails still returns the right numbers,
        but must not do it silently - a permanently broken GPU path would
        otherwise never be noticed. (Having no device at all is not an error
        and stays quiet; see test_no_cuda_uses_the_cpu_silently.)"""
        monkeypatch.setattr(localize, "SPLINE_CRLB_CUDA_AVAILABLE", True)
        monkeypatch.setattr(localize, "_crlb_gpu_fallback_warned", False)

        def boom(*args, **kwargs):
            raise RuntimeError("device on fire")

        monkeypatch.setattr(localize, "_spline_crlb_cuda", boom)
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[3000.0, 0.1, -0.1, 15.0]])
        expected = _crlb(theta, calib, BOX, gpu=False)

        with pytest.warns(RuntimeWarning, match="falling back to the CPU"):
            got = localize._spline_crlb(theta, calib, BOX)
        np.testing.assert_allclose(got, expected)

        # warned once per process, not once per call
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            np.testing.assert_allclose(
                localize._spline_crlb(theta, calib, BOX), expected
            )

    @requires_crlb_gpu
    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize(
        "model, n_channels",
        [
            ("spline-2d", 1),
            ("spline-3d", 1),
            ("spline-3d-multichannel", 2),
            ("spline-3d-multichannel", 6),
        ],
    )
    def test_matches_cpu(self, model, n_channels, mle):
        calib, _ = _gauss_spline_calibration(
            model=model, n_channels=n_channels
        )
        rng = np.random.default_rng(0)
        theta = _shared_theta(
            300, rng, n_params=4 if model == "spline-2d" else 5
        )
        np.testing.assert_allclose(
            _crlb(theta, calib, BOX, mle=mle, gpu=True),
            _crlb(theta, calib, BOX, mle=mle, gpu=False),
            rtol=self.RTOL,
        )

    @requires_crlb_gpu
    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_link_xyz_matches_cpu(self, n_channels, mle):
        rng = np.random.default_rng(1)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 200, rng)
        gpu = _crlb(theta, calib, BOX, mle=mle, gpu=True)
        cpu = _crlb(theta, calib, BOX, mle=mle, gpu=False)
        assert gpu.shape == (200, 3 + 2 * n_channels)
        np.testing.assert_allclose(gpu, cpu, rtol=self.RTOL)

    @requires_crlb_gpu
    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize("n_channels", [2, 3, 6])
    def test_channel_geometry_matches_cpu(self, n_channels, mle):
        """With a non-identity per-channel affine and non-zero ROI residuals -
        the case the plain parity tests cannot see, since at identity and zero
        the geometry terms drop out of both kernels alike."""
        rng = np.random.default_rng(11)
        calib, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        theta = _shared_theta(200, rng)
        calib, res = _with_channel_geometry(calib, len(theta), rng)
        gpu = _crlb(theta, calib, BOX, mle=mle, residuals=res, gpu=True)
        cpu = _crlb(theta, calib, BOX, mle=mle, residuals=res, gpu=False)
        np.testing.assert_allclose(gpu, cpu, rtol=self.RTOL)
        # and the geometry actually moved the answer, so this is a real test
        plain = _crlb(theta, calib, BOX, mle=mle, gpu=True)
        assert not np.allclose(gpu[:, :2], plain[:, :2], rtol=self.RTOL)

    @requires_crlb_gpu
    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize("n_channels", [2, 3, 6])
    def test_link_xyz_channel_geometry_matches_cpu(self, n_channels, mle):
        rng = np.random.default_rng(12)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 200, rng)
        calib, res = _with_channel_geometry(calib, len(theta), rng)
        gpu = _crlb(theta, calib, BOX, mle=mle, residuals=res, gpu=True)
        cpu = _crlb(theta, calib, BOX, mle=mle, residuals=res, gpu=False)
        np.testing.assert_allclose(gpu, cpu, rtol=self.RTOL)
        plain = _crlb(theta, calib, BOX, mle=mle, gpu=True)
        assert not np.allclose(gpu[:, :2], plain[:, :2], rtol=self.RTOL)

    @requires_crlb_gpu
    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_em_doubling_matches_cpu(self, model):
        """The EMCCD factor is applied by the caller, after the device hands
        back its variances - so it must land identically on both paths."""
        calib, _ = _gauss_spline_calibration(model=model)
        rng = np.random.default_rng(13)
        theta = _shared_theta(
            64, rng, n_params=4 if model == "spline-2d" else 5
        )
        gpu = _crlb(theta, calib, BOX, em=True, gpu=True)
        np.testing.assert_allclose(
            gpu, _crlb(theta, calib, BOX, em=True, gpu=False), rtol=self.RTOL
        )
        np.testing.assert_allclose(
            gpu, 2.0 * _crlb(theta, calib, BOX, gpu=True), rtol=1e-12
        )

    @requires_crlb_gpu
    def test_matches_closed_form_3d(self):
        """Against the analytic reference, not just against the CPU - a port
        that is systematically wrong could still match a co-wrong CPU path."""
        calib, splines = _gauss_spline_calibration(
            model="spline-3d", sx=1.0, sy=1.4
        )
        amp, off, z_shift = 4000.0, 20.0, -6.0
        theta = np.array([[amp, 0.2, -0.15, z_shift, off]])
        np.testing.assert_allclose(
            _crlb(theta, calib, BOX, gpu=True)[0],
            _ref_crlb(splines, BOX, amp, 0.2, -0.15, -z_shift, off),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            _crlb(theta, calib, BOX, mle=False, gpu=True)[0],
            _ref_crlb_lsq(splines, BOX, amp, 0.2, -0.15, -z_shift, off),
            rtol=1e-2,
        )

    @requires_crlb_gpu
    def test_matches_closed_form_2d(self):
        calib, splines = _gauss_spline_calibration(
            model="spline-2d", sx=1.0, sy=1.3
        )
        amp, off = 3000.0, 15.0
        theta = np.array([[amp, 0.1, -0.1, off]])
        np.testing.assert_allclose(
            _crlb(theta, calib, BOX, gpu=True)[0],
            _ref_crlb(splines, BOX, amp, 0.1, -0.1, None, off),
            rtol=1e-2,
        )

    @requires_crlb_gpu
    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_nan_theta_row_isolated(self, model):
        calib, _ = _gauss_spline_calibration(model=model)
        rng = np.random.default_rng(6)
        n_params = 4 if model == "spline-2d" else 5
        theta = _shared_theta(3, rng, n_params=n_params)
        theta[1, 0] = np.nan
        crlb = _crlb(theta, calib, BOX, gpu=True)
        assert np.all(np.isfinite(crlb[0])) and np.all(np.isfinite(crlb[2]))
        assert np.all(np.isnan(crlb[1]))

    @requires_crlb_gpu
    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_empty_theta(self, model):
        calib, _ = _gauss_spline_calibration(model=model)
        n_params = 4 if model == "spline-2d" else 5
        crlb = _crlb(np.zeros((0, n_params)), calib, BOX, gpu=True)
        assert crlb.shape == (0, n_params)

    @requires_crlb_gpu
    def test_cropped_box_matches_cpu(self):
        """The GPU must see the same centered crop the fit used."""
        calib, _ = _gauss_spline_calibration(model="spline-3d", box=9)
        calib = localize.crop_spline_calibration(calib, 7)
        rng = np.random.default_rng(7)
        theta = _shared_theta(64, rng)
        np.testing.assert_allclose(
            _crlb(theta, calib, 7, gpu=True),
            _crlb(theta, calib, 7, gpu=False),
            rtol=self.RTOL,
        )

    @requires_crlb_gpu
    def test_low_signal_stays_finite(self):
        # offset = 0 drives some model pixels to ~0; the MU_FLOOR guard keeps
        # the Fisher weight (1 / mu) finite on the device too.
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[500.0, 0.0, 0.0, 0.0]])
        crlb = _crlb(theta, calib, BOX, gpu=True)[0]
        assert np.all(np.isfinite(crlb))
        np.testing.assert_allclose(
            crlb,
            _crlb(theta, calib, BOX, gpu=False)[0],
            rtol=self.RTOL,
        )

    @requires_crlb_gpu
    def test_progress_callback_and_console(self):
        calib, _ = _gauss_spline_calibration(model="spline-3d")
        rng = np.random.default_rng(8)
        theta = _shared_theta(250, rng)
        seen = []
        _crlb(theta, calib, BOX, progress_callback=seen.append, gpu=True)
        assert seen and seen[-1] == len(theta) and seen == sorted(seen)
        _crlb(theta, calib, BOX, progress_callback="console", gpu=True)

    @pytest.mark.slow
    def test_simulator_matches_cpu(self):
        """Under Numba's CUDA simulator (no physical GPU needed) the kernels and
        the device pseudo-inverse reproduce the CPU path. Runs in a subprocess
        because the simulator must be enabled before numba is imported. Kept
        tiny - the simulator interprets every thread in Python."""
        import subprocess

        script = (
            "import os\n"
            "os.environ['NUMBA_ENABLE_CUDASIM'] = '1'\n"
            "import numpy as np\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import sys; sys.path.insert(0, '.')\n"
            # a device failure would fall back to the CPU and turn every
            # comparison below into CPU-vs-CPU, i.e. vacuously true
            "import warnings; warnings.simplefilter('error', RuntimeWarning)\n"
            "from picasso import localize\n"
            "from tests.test_localize import ("
            "_gauss_spline_calibration, _shared_theta, _crlb, BOX,"
            " _link_xyz_calib_and_theta, _with_channel_geometry)\n"
            "assert localize.SPLINE_CRLB_CUDA_AVAILABLE, 'simulator off'\n"
            "rng = np.random.default_rng(0)\n"
            "for model in ('spline-2d', 'spline-3d'):\n"
            "    calib, _ = _gauss_spline_calibration(model=model)\n"
            "    p = 4 if model == 'spline-2d' else 5\n"
            "    theta = _shared_theta(3, rng, n_params=p)\n"
            "    for mle in (True, False):\n"
            "        g = _crlb("
            "theta, calib, BOX, mle=mle, gpu=True)\n"
            "        c = _crlb("
            "theta, calib, BOX, mle=mle, gpu=False)\n"
            "        np.testing.assert_allclose(g, c, rtol=1e-5)\n"
            # the multichannel kernels, with the affine + ROI residual live
            "calib, _ = _gauss_spline_calibration("
            "model='spline-3d-multichannel', n_channels=2)\n"
            "theta = _shared_theta(2, rng)\n"
            "calib, res = _with_channel_geometry(calib, len(theta), rng)\n"
            "np.testing.assert_allclose(\n"
            "    _crlb(theta, calib, BOX, residuals=res, gpu=True),\n"
            "    _crlb(theta, calib, BOX, residuals=res, gpu=False),\n"
            "    rtol=1e-5)\n"
            "calib, theta = _link_xyz_calib_and_theta(2, 2, rng)\n"
            "calib, res = _with_channel_geometry(calib, len(theta), rng)\n"
            "np.testing.assert_allclose(\n"
            "    _crlb(theta, calib, BOX, residuals=res, gpu=True),\n"
            "    _crlb(theta, calib, BOX, residuals=res, gpu=False),\n"
            "    rtol=1e-5)\n"
            "print('SIMOK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert "SIMOK" in result.stdout, result.stdout + result.stderr


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


def _link_xyz_theta(n_channels, n_locs=3, z_center=10.0, bg=2.0):
    """``[x, y, z, N_0..N_{c-1}, bg_0..bg_{c-1}]`` with deliberately unequal
    per-channel photons, so a channel mix-up cannot pass unnoticed."""
    photons = 100.0 * (1.0 + np.arange(n_channels))
    theta = np.zeros((n_locs, 3 + 2 * n_channels))
    theta[:, 2] = -z_center
    theta[:, 3 : 3 + n_channels] = photons
    theta[:, 3 + n_channels :] = bg
    return theta, photons


class TestSplineLinkXyzColumns:
    """Output columns of the photon-decoupled (link-XYZ) multichannel spline
    fit. Pure numpy/numba - the constructor is a function of the fitted
    parameters, so no GPU is needed."""

    @staticmethod
    def _ids(n_locs):
        return pd.DataFrame(
            {
                "frame": np.arange(n_locs, dtype=np.uint32),
                "x": np.full(n_locs, 20.0),
                "y": np.full(n_locs, 30.0),
                "net_gradient": np.full(n_locs, 100.0),
            }
        )

    def _locs(self, n_channels, theta=None):
        calib, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        calib = localize._as_link_xyz_calibration(calib)
        photons = None
        if theta is None:
            theta, photons = _link_xyz_theta(
                n_channels, z_center=calib["z_center"]
            )
        locs = localize.locs_from_fits_spline(
            self._ids(len(theta)),
            theta,
            BOX,
            em=False,
            calibration=calib,
            mle=False,
        )
        return locs, photons

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_emits_one_column_per_channel(self, n_channels):
        locs, photons = self._locs(n_channels)
        for c in range(n_channels):
            assert f"photons_ch{c}" in locs.columns
            assert f"bg_ch{c}" in locs.columns
            assert f"rel_photons_ch{c}" in locs.columns
        # nothing for channels this calibration does not have
        assert f"photons_ch{n_channels}" not in locs.columns
        assert f"rel_photons_ch{n_channels}" not in locs.columns
        # the continuous readout replaced the old `color` column, and there is
        # no bare `rel_photons`
        assert "color" not in locs.columns
        assert "rel_photons" not in locs.columns

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_rel_photons_are_the_channel_shares(self, n_channels):
        locs, photons = self._locs(n_channels)
        rel = np.stack(
            [locs[f"rel_photons_ch{c}"].to_numpy() for c in range(n_channels)],
            axis=1,
        )
        # each channel's share of the total, summing to 1 per localization
        np.testing.assert_allclose(rel.sum(axis=1), 1.0, rtol=1e-5)
        np.testing.assert_allclose(
            rel, np.broadcast_to(photons / photons.sum(), rel.shape), rtol=1e-5
        )
        per_ch = np.stack(
            [locs[f"photons_ch{c}"].to_numpy() for c in range(n_channels)],
            axis=1,
        )
        np.testing.assert_allclose(
            locs["photons"].to_numpy(), per_ch.sum(axis=1), rtol=1e-5
        )
        np.testing.assert_allclose(per_ch, rel * photons.sum(), rtol=1e-5)

    def test_rel_photons_nan_without_photons(self):
        n_channels = 3
        theta, _ = _link_xyz_theta(n_channels, n_locs=2)
        theta[1, 3 : 3 + n_channels] = 0.0  # a spot that fitted no photons
        locs, _ = self._locs(n_channels, theta=theta)
        rel = np.stack(
            [locs[f"rel_photons_ch{c}"].to_numpy() for c in range(n_channels)],
            axis=1,
        )
        assert np.isfinite(rel[0]).all()
        assert np.isnan(rel[1]).all()


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
        # Least-squares fits report a chi-square instead of a likelihood.
        lsq_stats = dict(
            chi_square=np.array([120.0, 95.0], dtype=np.float32),
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
            ids, theta_e, BOX, em=False, mle=False, **lsq_stats
        )
        frames["gpufit-lse-rotated"] = localize.locs_from_fits_gpufit(
            ids, theta_r, BOX, em=False, mle=False, **lsq_stats
        )

        # CPU least-squares Gaussian, [x, y, photons, bg, sx, sy]
        theta_lq = np.array(
            [
                [3.0, 3.0, 500.0, 5.0, 1.2, 1.1],
                [3.1, 2.9, 600.0, 4.0, 1.0, 1.3],
            ]
        )
        frames["gausslq-cpu"] = gausslq.locs_from_fits(
            ids, theta_lq, BOX, em=False, chi_square=lsq_stats["chi_square"]
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
        # Photon-decoupled (link-XYZ) multichannel spline, at the largest
        # channel count there is a fit model for - it emits the widest set of
        # per-channel columns.
        n_channels = localize._LINK_XYZ_MAX_CHANNELS
        calib_mc, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        calib_link = localize._as_link_xyz_calibration(calib_mc)
        theta_link, _ = _link_xyz_theta(
            n_channels, n_locs=2, z_center=calib_link["z_center"]
        )
        frames["spline-3d-link-xyz"] = localize.locs_from_fits_spline(
            ids,
            theta_link,
            BOX,
            em=False,
            calibration=calib_link,
            mle=False,
            **lsq_stats,
        )

        # Ratiometric multichannel spline: the shared-amplitude model plus the
        # per-channel photons and the integer `color` (the assigned channel)
        # that fit_spline_multichannel_ratiometric bolts on afterwards.
        theta_mc = np.array(
            [
                [4000.0, 0.2, -0.15, -6.0, 20.0],
                [3800.0, 0.1, -0.10, -5.0, 18.0],
            ]
        )
        locs_ratio = localize.locs_from_fits_spline(
            ids,
            theta_mc,
            BOX,
            em=False,
            calibration=calib_mc,
            mle=False,
            **lsq_stats,
        )
        ratios = np.arange(1.0, n_channels + 1.0)
        ratios /= ratios.sum()
        for c in range(n_channels):
            locs_ratio[f"photons_ch{c}"] = (
                locs_ratio["photons"] * ratios[c]
            ).astype(np.float32)
        locs_ratio["color"] = np.int32(1)
        frames["spline-3d-ratiometric"] = locs_ratio
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
# Split-FOV multichannel: regions of ONE movie treated as channels
# ---------------------------------------------------------------------------


def _split_fov_bead_movie(dx=2, dy=-1, n_frames=21):
    """One movie whose left/right 48x48 halves are two channels: the right
    half is the left (reference) half shifted within its region by ``(dx, dy)``.
    Returns ``(movie, regions, bead_xy, focus)``."""
    bead_xy = [(12, 14), (30, 28), (16, 33)]
    s0 = 1.1
    focus = n_frames // 2
    yy, xx = np.mgrid[0:48, 0:48]
    base = np.zeros((n_frames, 48, 48), dtype=np.float32)
    for f in range(n_frames):
        sigma = s0 * (1.0 + 0.07 * abs(f - focus))
        img = np.full((48, 48), 100.0, dtype=np.float32)
        for bx, by in bead_xy:
            img += 3000.0 * np.exp(
                -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma**2)
            )
        base[f] = img
    base = base.astype(np.uint16)
    movie = np.zeros((n_frames, 48, 96), dtype=np.uint16)
    movie[:, :, :48] = base
    movie[:, :, 48:] = np.roll(base, shift=(dy, dx), axis=(1, 2))
    regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
    return movie, regions, bead_xy, focus


class TestSplitFovRegionAffines:
    """Region-local <-> absolute channel-transform conversion (no GPU)."""

    def test_decompose_compose_round_trip(self):
        regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
        t0 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        tc = [[1.0, 0.0, 50.0], [0.0, 1.0, -1.0]]  # 48 offset + (2, -1) fine
        affines = localize.decompose_region_affines(regions, [t0, tc])
        # region-local affine carries only the fine (2, -1) registration
        np.testing.assert_allclose(
            affines[0], [[1, 0, 0], [0, 1, 0]], atol=1e-9
        )
        np.testing.assert_allclose(
            affines[1], [[1, 0, 2], [0, 1, -1]], atol=1e-9
        )
        back = localize.compose_region_transforms(regions, affines)
        np.testing.assert_allclose(back[0], t0, atol=1e-9)
        np.testing.assert_allclose(back[1], tc, atol=1e-9)

    def test_replace_at_new_positions(self):
        regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
        tc = [[1.0, 0.0, 50.0], [0.0, 1.0, -1.0]]
        affines = localize.decompose_region_affines(
            regions, [[[1, 0, 0], [0, 1, 0]], tc]
        )
        # channels re-placed: reference at (10, 10), channel at (10, 200)
        new = [[[10, 10], [58, 58]], [[10, 200], [58, 248]]]
        t = localize.compose_region_transforms(new, affines)
        # reference is identity at its new spot; channel = new offset + fine
        np.testing.assert_allclose(t[0], [[1, 0, 0], [0, 1, 0]], atol=1e-9)
        np.testing.assert_allclose(t[1], [[1, 0, 192], [0, 1, -1]], atol=1e-9)

    def test_rotation_is_position_independent(self):
        # a small rotation in the linear part survives re-placement unchanged
        th = np.deg2rad(3.0)
        L = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
        regions = [[[0, 0], [40, 40]], [[0, 50], [40, 90]]]
        tc = [[L[0][0], L[0][1], 55.0], [L[1][0], L[1][1], 2.0]]
        affines = localize.decompose_region_affines(
            regions, [[[1, 0, 0], [0, 1, 0]], tc]
        )
        new = [[[5, 5], [45, 45]], [[5, 300], [45, 340]]]
        t = localize.compose_region_transforms(new, affines)
        # linear (rotation) part is unchanged by moving the regions
        np.testing.assert_allclose(np.asarray(t[1])[:, :2], L, atol=1e-9)


class TestFitSplineSplitFovValidation:
    """Argument validation that needs no GPU."""

    def test_requires_split_fov_calibration(self):
        calib = {"model": "spline-3d-multichannel"}  # no split_fov flag
        movie = np.zeros((1, 16, 32), np.float32)
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        with pytest.raises(ValueError, match="split-FOV"):
            localize.fit_spline_split_fov(movie, CAMERA_INFO, ids, BOX, calib)


@pytest.mark.skipif(
    not (localize.GPUFIT_INSTALLED and localize.GPUSPLINE_INSTALLED),
    reason="Gpufit (CUDA GPU) + Gpuspline not available",
)
class TestFitSplineSplitFov:
    """End-to-end split-FOV fit on a single movie built + calibrated from the
    same beads (model matches, so the fit recovers the bead positions)."""

    def _movie_and_calib(self):
        dx, dy = 2, -1
        movie, regions, bead_xy, focus = _split_fov_bead_movie(dx, dy)
        info = [{"Frames": int(movie.shape[0])}]
        calib = spline.calibrate_spline_split_fov(
            movie,
            info=info,
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            regions=regions,
        )
        return movie, calib, bead_xy, focus

    def test_fit_returns_locs_in_reference_region(self):
        movie, calib, bead_xy, focus = self._movie_and_calib()
        # detect the reference-region beads on the in-focus frames
        ids, _ = localize.identify(
            movie,
            2000.0,
            BOX,
            roi=calib["regions"][0],
            frame_bounds=(focus - 1, focus + 1),
        )
        locs = localize.fit_spline_split_fov(
            movie, CAMERA_INFO, ids, BOX, calib
        )
        assert len(locs) > 0
        # localizations are in the reference region's coordinates
        assert np.all(locs["x"] >= 0) and np.all(locs["x"] < 48)
        assert np.all(locs["y"] >= 0) and np.all(locs["y"] < 48)
        # each fitted spot sits near one of the three reference beads
        true_x = np.array([b[0] for b in bead_xy])
        true_y = np.array([b[1] for b in bead_xy])
        for x, y in zip(locs["x"], locs["y"]):
            d = np.hypot(true_x - x, true_y - y)
            assert d.min() < 2.0

    def test_confines_identifications_to_reference_region(self):
        movie, calib, bead_xy, focus = self._movie_and_calib()
        # identify over the WHOLE frame -> detections in both halves
        ids, _ = localize.identify(
            movie, 2000.0, BOX, frame_bounds=(focus, focus)
        )
        assert (ids["x"] >= 48).any()  # some detections are in region 1
        locs = localize.fit_spline_split_fov(
            movie, CAMERA_INFO, ids, BOX, calib, confine_to_reference=True
        )
        # only the reference-region molecules are fitted (region 1 is the
        # mapped partner, not an independent detection)
        assert len(locs) > 0
        assert np.all(locs["x"] < 48)

    def test_fit_at_repositioned_regions(self):
        """ROI-agnostic: the same calibration fits data whose split sits at a
        different position, when the fit-time ROIs are supplied."""
        movie, calib, bead_xy, focus = self._movie_and_calib()
        assert "channel_affines" in calib  # ROI-agnostic registration stored
        # embed the identical split (same inter-channel geometry) at a global
        # offset in a larger frame
        oy, ox = 20, 60
        n, h, w = movie.shape
        big = np.zeros((n, h + oy + 10, w + ox + 10), dtype=movie.dtype)
        big[:, oy : oy + h, ox : ox + w] = movie
        ref_region = [[oy, ox], [oy + 48, ox + 48]]
        chan_region = [[oy, ox + 48], [oy + 48, ox + 96]]
        fit_regions = [ref_region, chan_region]

        ids, _ = localize.identify(
            big,
            2000.0,
            BOX,
            roi=ref_region,
            frame_bounds=(focus - 1, focus + 1),
        )
        locs = localize.fit_spline_split_fov(
            big, CAMERA_INFO, ids, BOX, calib, regions=fit_regions
        )
        assert len(locs) > 0
        # localizations land near the shifted reference beads
        tx = np.array([ox + b[0] for b in bead_xy])
        ty = np.array([oy + b[1] for b in bead_xy])
        for x, y in zip(locs["x"], locs["y"]):
            assert np.hypot(tx - x, ty - y).min() < 2.0

        # control: without fit-time regions, the calibration's original
        # positions no longer match the data -> nothing in the reference region
        ids_all, _ = localize.identify(
            big, 2000.0, BOX, frame_bounds=(focus, focus)
        )
        locs_orig = localize.fit_spline_split_fov(
            big, CAMERA_INFO, ids_all, BOX, calib
        )
        assert len(locs_orig) < len(locs)


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

    def _run_unlinked(self, n_channels, monkeypatch, photon_ratios=None):
        """Route with photons UNLINKED, recording which fitter ran and the
        ``link_photons`` it was called with ("default" = not passed)."""
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
            lambda *a, **k: (
                calls.append(("plain", k.get("link_photons", "default"))),
                df,
            )[1],
        )
        calibration = {
            "model": "spline-3d-multichannel",
            "n_channels": n_channels,
        }
        if photon_ratios is not None:
            calibration["photon_ratios"] = photon_ratios
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        worker = localize_gui.MultichannelSplineFitWorker(
            [None] * n_channels,
            [{}] * n_channels,
            ids,
            BOX,
            calibration,
            mle=False,
            link_photons=False,
        )
        worker.run()
        return calls

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_unlinked_photons_route_to_link_xyz(self, n_channels, monkeypatch):
        assert self._run_unlinked(n_channels, monkeypatch) == [
            ("plain", False)
        ]

    def test_unlinked_photons_supersede_ratiometric(self, monkeypatch):
        # photon decoupling fits the ratio instead of scanning hypotheses, so
        # it wins over the ratiometric branch when both are possible
        calls = self._run_unlinked(
            4, monkeypatch, photon_ratios=[[0.4, 0.3, 0.2, 0.1]]
        )
        assert calls == [("plain", False)]

    def test_unlinked_photons_above_cap_fall_back_to_linked(self, monkeypatch):
        # no link-XYZ fit model is compiled beyond the cap, so the
        # shared-amplitude model (which takes any channel count) is used
        n_channels = localize._LINK_XYZ_MAX_CHANNELS + 1
        assert self._run_unlinked(n_channels, monkeypatch) == [
            ("plain", "default")
        ]

    def test_fits_only_spots_linked_across_channels(self, monkeypatch):
        """Given every channel's identifications, the worker hands the fitter
        only the reference spots detected in all channels."""
        seen = {}
        monkeypatch.setattr(
            localize,
            "fit_spline_multichannel",
            lambda movies, cams, ids, *a, **k: (
                seen.update(ids=ids),
                pd.DataFrame({"frame": [0], "x": [0.0], "y": [0.0]}),
            )[1],
        )
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ref = pd.DataFrame(
            {
                "frame": [0, 0],
                "x": [6.0, 60.0],
                "y": [6.0, 60.0],
                "net_gradient": [1.0, 1.0],
            }
        )
        # channel 1 only sees the first reference spot
        ch1 = pd.DataFrame(
            {"frame": [0], "x": [6.2], "y": [5.9], "net_gradient": [1.0]}
        )
        worker = localize_gui.MultichannelSplineFitWorker(
            [None, None],
            [{}, {}],
            ref,
            BOX,
            {
                "model": "spline-3d-multichannel",
                "n_channels": 2,
                "channel_transforms": [identity, identity],
            },
            identifications_per_channel=[ref, ch1],
        )
        linked = []
        worker.linkFinished.connect(
            lambda kept, total: linked.append((kept, total))
        )
        worker.run()
        assert linked == [(1, 2)]
        assert len(seen["ids"]) == 1
        assert float(seen["ids"]["x"].iloc[0]) == 6.0
        # progress totals follow the linked subset, not the original count
        assert worker.N == 1
