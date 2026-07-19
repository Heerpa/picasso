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


# ---------------------------------------------------------------------------
# Multichannel calibration (Session D) - registration/matching without a GPU
# ---------------------------------------------------------------------------


class TestMultichannelCalibration:
    def test_match_beads(self):
        ref = np.array([[0, 0], [10, 10], [20, 20]], dtype=float)
        other = np.array([[20.3, 20.1], [0.2, -0.1], [100, 100]], dtype=float)
        ref_idx, other_idx = spline._match_beads(ref, other, 1.0)
        # ref[0]->other[1], ref[2]->other[0]; ref[1] has no match within 1 px
        assert ref_idx.tolist() == [0, 2]
        assert other_idx.tolist() == [1, 0]

    def test_match_beads_unique_targets(self):
        ref = np.array([[0, 0], [0.5, 0]], dtype=float)
        other = np.array([[0.1, 0.0]], dtype=float)
        ref_idx, other_idx = spline._match_beads(ref, other, 5.0)
        # both refs are near the single target; it must be used only once
        assert len(other_idx) == 1
        assert ref_idx.tolist() == [0]  # closest reference wins

    def test_estimate_channel_transform_recovers_shift(self):
        movie_ref, _, _ = _synthetic_bead_movie()
        dx, dy = 3, -2  # channel is the reference shifted by (dx, dy)
        movie_c = np.roll(movie_ref, shift=(dy, dx), axis=(1, 2))

        step_of_frame, _, step_range = spline._step_of_frame(
            movie_ref.shape[0], 20.0, 1, "fov", None
        )
        ref_bounds = spline._reference_frame_bounds(step_of_frame, step_range)
        mid = (ref_bounds[0] + ref_bounds[1]) // 2
        beads_ref = spline._detect_bead_positions(
            movie_ref, 2000.0, BOX, ref_bounds
        )
        transform, n_matches = spline._estimate_channel_transform(
            movie_ref,
            movie_c,
            beads_ref,
            2000.0,
            BOX,
            ref_bounds,
            mid,
            max_distance=float(BOX),
        )
        assert n_matches >= 3
        # transform maps reference (x, y) -> channel (x + dx, y + dy)
        np.testing.assert_allclose(
            transform,
            np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]]),
            atol=0.6,
        )


@pytest.mark.skipif(
    not localize.GPUSPLINE_INSTALLED, reason="Gpuspline not available"
)
class TestCalibrateSplineMultichannel:
    """Full multichannel calibration including the Gpuspline coefficient
    step (CPU). Skipped unless Gpuspline is installed."""

    def test_calibrate_multichannel(self, tmp_path):
        from picasso import io

        movie_ref, _, _ = _synthetic_bead_movie()
        movie_c = np.roll(movie_ref, shift=(2, -1), axis=(1, 2))
        info = [{"Frames": int(movie_ref.shape[0])}]
        path = str(tmp_path / "mc_spline_calib.hdf5")
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            photon_ratios=[[0.7, 0.3], [0.4, 0.6]],
            path=path,
        )
        assert calib["model"] == "spline-3d-multichannel"
        assert calib["n_channels"] == 2
        # candidate photon ratios stored for ratiometric color assignment,
        # and they survive the HDF5 round-trip (JSON metadata)
        np.testing.assert_allclose(
            calib["photon_ratios"], [[0.7, 0.3], [0.4, 0.6]]
        )
        assert io.load_spline_calibration(path)["photon_ratios"] is not None
        assert calib["coefficients"].shape[0] == 64
        assert calib["coefficients"].shape[-1] == 2
        assert len(calib["channel_transforms"]) == 2
        # round-trips and drives the multichannel user_info packer
        loaded = io.load_spline_calibration(path)
        user_info = localize._pack_spline_user_info(loaded)
        assert user_info.dtype == np.float32
        # co-focal channels: both planes at the same focus
        np.testing.assert_allclose(
            calib["plane_offsets"], [0.0, 0.0], atol=25.0
        )
        # per-channel photon_scale is stored as a list (one per channel)
        assert len(calib["photon_scale"]) == 2

    def test_calibrate_biplane_recovers_plane_offset(self, tmp_path):
        """Biplane: the second channel is the same z-stack with its focus at a
        different stage step (a frame-axis roll). The calibration must recover
        a non-zero plane offset of the right magnitude, while keeping the two
        channels laterally registered (identity transform)."""
        movie_ref, _, focus = _synthetic_bead_movie(n_frames=21)
        d_nm = 20.0
        delta_steps = 3  # channel 1 focuses 3 steps deeper
        movie_c = np.roll(movie_ref, shift=delta_steps, axis=0)
        info = [{"Frames": int(movie_ref.shape[0])}]
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=d_nm,
            path=str(tmp_path / "biplane_spline_calib.hdf5"),
        )
        offsets = calib["plane_offsets"]
        assert offsets[0] == 0.0
        # channel 1 focus offset ~ delta_steps * d (within ~1 step)
        np.testing.assert_allclose(
            offsets[1], delta_steps * d_nm, atol=1.5 * d_nm
        )


class TestSplinePhaseCalibration:
    """4Pi phase calibration: harmonic decomposition + spline build (model 12)."""

    def test_decompose_recovers_components(self):
        rng = np.random.default_rng(0)
        shape = (5, 6, 4)
        mean = rng.random(shape)
        mod = rng.random(shape)
        mod90 = rng.random(shape)
        phases = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        vols = np.stack(
            [mean + np.cos(p) * mod + np.sin(p) * mod90 for p in phases]
        )
        m, o, n = spline.decompose_phase_volumes(vols, phases)
        np.testing.assert_allclose(m, mean, atol=1e-9)
        np.testing.assert_allclose(o, mod, atol=1e-9)
        np.testing.assert_allclose(n, mod90, atol=1e-9)

    def test_decompose_requires_three_phases(self):
        with pytest.raises(ValueError):
            spline.decompose_phase_volumes(np.zeros((2, 3, 3, 3)), [0.0, 1.0])

    def test_decompose_rejects_degenerate_phases(self):
        with pytest.raises(ValueError):
            spline.decompose_phase_volumes(
                np.zeros((3, 3, 3, 3)), [1.0, 1.0, 1.0]
            )

    @pytest.mark.skipif(
        not localize.GPUSPLINE_INSTALLED, reason="Gpuspline not available"
    )
    def test_calibrate_spline_phase_builds_calibration(self, tmp_path):
        from picasso import io

        box, nz, nch, P = 13, 21, 4, 6
        zc = (nz - 1) // 2
        xg = np.arange(box)
        c0 = (box - 1) / 2
        env = np.zeros((box, box, nz), np.float32)
        for k in range(nz):
            s = 1.3 * (1 + 0.5 * abs(k - zc) / nz)
            g = np.exp(-0.5 * ((xg - c0) / s) ** 2)
            env[:, :, k] = np.outer(g, g)
        psi_c = np.arange(nch) * (2 * np.pi / nch)
        psi_p = np.linspace(0, 2 * np.pi, P, endpoint=False)
        templates = np.zeros((P, box, box, nz, nch), np.float32)
        for p in range(P):
            for c in range(nch):
                templates[p, :, :, :, c] = (
                    env * (1 + np.cos(psi_p[p] - psi_c[c])) + 10.0
                )
        path = str(tmp_path / "phase_calib.hdf5")
        calib = spline.calibrate_spline_phase(
            templates, psi_p, d=20.0, z_center_index=zc, path=path
        )
        assert calib["model"] == "spline-3d-phase-multichannel"
        assert calib["coefficients"].shape == (
            64,
            box - 1,
            box - 1,
            nz - 1,
            nch,
            3,
        )
        assert calib["n_channels"] == nch
        assert len(calib["photon_scale"]) == nch
        assert len(calib["channel_transforms"]) == nch
        assert len(calib["phases"]) == P
        # round-trips through HDF5 and drives the phase user_info packer
        loaded = io.load_spline_calibration(path)
        ui = localize._pack_spline_user_info(loaded)
        assert ui.size == 7 + 3 * nch * (box - 1) * (box - 1) * (nz - 1) * 64


# ---------------------------------------------------------------------------
# Session C: GUI + CLI wiring (no GPU required)
# ---------------------------------------------------------------------------


class TestCliWiring:
    def test_fit_method_map(self):
        from picasso import __main__ as cli

        assert cli._FIT_METHOD_MAP["spline"] == "spline-gpu"
        assert cli._FIT_METHOD_MAP["spline-mle"] == "spline-mle-gpu"

    def test_spline_calibrate_handler_exists(self):
        from picasso import __main__ as cli

        assert callable(cli._spline_calibrate)

    def test_backend_accepts_both_spline_codes(self):
        # both spline codes must be recognised model ids by the backend
        # (guards the fit2D / localize / localize_3D dispatch strings)
        import inspect

        src = inspect.getsource(localize.fit2D)
        assert "spline-gpu" in src and "spline-mle-gpu" in src


class TestGuiWiring:
    def test_fit_code_resolves_spline(self, monkeypatch):
        from picasso.gui import localize as glocalize

        models = dict(glocalize.FIT_MODELS)
        models["Experimental PSF (cubic spline)"] = {
            "optimizers": {
                "Least squares": "spline-gpu",
                "MLE": "spline-mle-gpu",
            },
            "needs_spline_calibration": True,
        }
        monkeypatch.setattr(glocalize, "FIT_MODELS", models)
        assert (
            glocalize._fit_code(
                "Experimental PSF (cubic spline)", "Least squares"
            )
            == "spline-gpu"
        )
        assert (
            glocalize._fit_code("Experimental PSF (cubic spline)", "MLE")
            == "spline-mle-gpu"
        )

    def test_fit_worker_preserves_spline_method_and_calibration(self):
        import sys
        import pandas as pd
        from PyQt6 import QtWidgets
        from picasso.gui import localize as glocalize

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        calib = {"model": "spline-3d"}
        worker = glocalize.FitWorker(
            None,
            [],
            {},
            pd.DataFrame({"x": [], "y": [], "frame": []}),
            BOX,
            "spline-mle-gpu",
            0.001,
            100,
            False,
            False,
            True,  # use_gpufit
            spline_calibration=calib,
        )
        # the "-gpu" suffix must not be appended to an already-gpu spline code
        assert worker.method == "spline-mle-gpu"
        assert worker.spline_calibration is calib
