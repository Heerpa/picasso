"""Tests for picasso.spline (cubic-spline PSF calibration generation).

The GPU-independent parts (frame binning, PSF-template building, registration,
normalization) run everywhere. The final coefficient step needs Gpuspline (a
CPU library) and is gated on ``localize.GPUSPLINE_INSTALLED``.
"""

from __future__ import annotations

import numpy as np
import pytest

from picasso import localize, spline

from tests.conftest import BOX, CAMERA_INFO


# ---------------------------------------------------------------------------
# Synthetic bead z-stack
# ---------------------------------------------------------------------------


def _synthetic_bead_movie(n_frames=21, h=48, w=48, box=BOX):
    """A movie of a few static beads with a Gaussian PSF whose width is
    minimal at the central frame (focus) and grows away from it."""
    bead_xy = [(12, 14), (30, 28), (16, 33)]
    s0 = 1.1
    focus = n_frames // 2
    yy, xx = np.mgrid[0:h, 0:w]
    movie = np.zeros((n_frames, h, w), dtype=np.float32)
    for f in range(n_frames):
        sigma = s0 * (1.0 + 0.07 * abs(f - focus))
        img = np.full((h, w), 100.0, dtype=np.float32)
        for bx, by in bead_xy:
            img += 3000.0 * np.exp(
                -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma**2)
            )
        movie[f] = img
    return movie.astype(np.uint16), bead_xy, focus


class TestStepOfFrame:
    def test_one_frame_per_step(self):
        step, z_of_step, step_range = spline._step_of_frame(
            10, 20.0, 1, "fov", None
        )
        np.testing.assert_array_equal(step, np.arange(10))
        np.testing.assert_array_equal(step_range, np.arange(10))
        assert len(z_of_step) == 10

    def test_fov_order(self):
        step, _, step_range = spline._step_of_frame(10, 20.0, 2, "fov", None)
        np.testing.assert_array_equal(step, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        np.testing.assert_array_equal(step_range, np.arange(5))

    def test_z_order(self):
        step, _, _ = spline._step_of_frame(10, 20.0, 2, "z", None)
        np.testing.assert_array_equal(step, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    def test_frame_bounds_exclude(self):
        step, _, step_range = spline._step_of_frame(10, 20.0, 1, "fov", (2, 5))
        # frames outside [2, 5] are marked -1
        assert np.all(step[:2] == -1)
        assert np.all(step[6:] == -1)
        np.testing.assert_array_equal(step_range, [2, 3, 4, 5])

    def test_too_many_frames_per_step(self):
        with pytest.raises(ValueError):
            spline._step_of_frame(3, 20.0, 10, "fov", None)


class TestTemplateHelpers:
    def test_normalize_template(self):
        box, nz = BOX, 5
        vol = np.full((box, box, nz), 50.0, dtype=np.float32)
        vol[box // 2, box // 2, 2] = 50.0 + 800.0  # peak at focus slice 2
        template, bg, amp, photon_scale = spline._normalize_template(vol, 2)
        assert bg == pytest.approx(50.0, abs=1e-3)
        assert amp == pytest.approx(800.0, rel=1e-3)
        # peak of the normalized in-focus slice is ~1
        assert template[box // 2, box // 2, 2] == pytest.approx(1.0, abs=1e-3)
        assert photon_scale > 0

    def test_normalize_rejects_flat(self):
        vol = np.full((BOX, BOX, 3), 10.0, dtype=np.float32)
        with pytest.raises(ValueError):
            spline._normalize_template(vol, 1)

    def test_focus_step_picks_sharpest(self):
        box, nz = BOX, 5
        yy, xx = np.mgrid[0:box, 0:box]
        c = box // 2
        vol = np.zeros((box, box, nz), dtype=np.float32)
        sigmas = [2.4, 1.8, 1.0, 1.8, 2.4]  # sharpest at index 2
        for k, s in enumerate(sigmas):
            vol[:, :, k] = np.exp(
                -((xx - c) ** 2 + (yy - c) ** 2) / (2 * s**2)
            )
        z_center, eff_sigma = spline._focus_step(vol)
        assert z_center == 2
        assert eff_sigma == pytest.approx(1.0, abs=0.4)

    def test_register_and_average_centers_beads(self):
        box, nz = BOX, 3
        yy, xx = np.mgrid[0:box, 0:box]
        c = box // 2
        s = 1.1

        def gauss(cx, cy):
            return np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s**2)
            ).astype(np.float32)

        # two beads offset in opposite directions at the focus slice
        volumes = np.zeros((2, box, box, nz), dtype=np.float32)
        for k in range(nz):
            volumes[0, :, :, k] = gauss(c + 1.0, c)
            volumes[1, :, :, k] = gauss(c - 1.0, c)
        mean_vol = spline._register_and_average(volumes, z_center=1)
        # after centering, the averaged PSF peak should be at the box center
        focus = mean_vol[:, :, 1]
        peak_row, peak_col = np.unravel_index(np.argmax(focus), focus.shape)
        assert peak_row == c
        assert peak_col == c


class TestBuildPsfTemplate:
    """End-to-end PSF template building on a synthetic bead movie (no GPU)."""

    def test_build_template(self):
        movie, bead_xy, focus = _synthetic_bead_movie()
        built = spline.build_psf_template(
            movie,
            CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
        )
        template = built["template"]
        assert template.shape == (BOX, BOX, movie.shape[0])
        # at least some beads detected
        assert built["n_beads"] >= 2
        # focus recovered near the true central frame
        assert abs(built["z_center"] - focus) <= 2
        # normalized template: focus peak ~1, minimum ~0
        assert template[:, :, built["z_center"]].max() == pytest.approx(
            1.0, abs=0.05
        )
        assert template.min() == pytest.approx(0.0, abs=0.1)
        assert built["effective_sigma"] > 0


@pytest.mark.skipif(
    not localize.GPUSPLINE_INSTALLED, reason="Gpuspline not available"
)
class TestCalibrateSpline:
    """Full calibration including the Gpuspline coefficient step (CPU)."""

    def test_calibrate_spline_3d_roundtrip(self, tmp_path):
        from picasso import io

        movie, _, _ = _synthetic_bead_movie()
        path = str(tmp_path / "bead_spline_calib.hdf5")
        calib = spline.calibrate_spline(
            movie,
            info=[{"Frames": int(movie.shape[0])}],
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            model="spline-3d",
            path=path,
        )
        assert calib["model"] == "spline-3d"
        assert calib["coefficients"].shape[0] == 64
        assert list(calib["n_data"]) == [BOX, BOX, movie.shape[0]]
        # the saved calibration loads and can drive the fitter's packer
        loaded = io.load_spline_calibration(path)
        user_info = localize._pack_spline_user_info(loaded)
        assert user_info.dtype == np.float32

    def test_calibrate_spline_2d(self):
        movie, _, _ = _synthetic_bead_movie()
        calib = spline.calibrate_spline(
            movie,
            info=[{"Frames": int(movie.shape[0])}],
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            model="spline-2d",
        )
        assert calib["model"] == "spline-2d"
        assert calib["coefficients"].shape[0] == 16
        assert list(calib["n_data"]) == [BOX, BOX]


@pytest.mark.skipif(
    localize.GPUSPLINE_INSTALLED,
    reason="only relevant when Gpuspline is missing",
)
def test_calibrate_spline_requires_gpuspline():
    movie, _, _ = _synthetic_bead_movie()
    with pytest.raises(ImportError):
        spline.calibrate_spline(
            movie,
            info=[{"Frames": int(movie.shape[0])}],
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
        )
