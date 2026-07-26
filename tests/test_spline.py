"""Tests for picasso.spline (cubic-spline PSF calibration generation).

The GPU-independent parts (frame binning, PSF-template building, registration,
normalization) run everywhere. The final coefficient step needs Gpuspline (a
CPU library) and is gated on ``localize.GPUSPLINE_INSTALLED``.
"""

from __future__ import annotations

import os

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


def _synthetic_multifov_movie(
    n_fov=3, n_steps=11, h=48, w=48, order="z", box=BOX
):
    """A genuine multi-FOV bead z-stack: ``n_fov`` fields, each with beads at
    *different* positions, each scanned over ``n_steps`` z positions (focus at
    the centre). Frames are laid out in ``order`` ("z": each FOV is a full z
    stack, then the next FOV; "fov": the FOVs are interleaved at each z).

    Returns ``(movie, fov_beads, focus)`` where ``fov_beads[k]`` are the bead
    centres of FOV ``k``. The total number of physical beads is
    ``sum(len(b) for b in fov_beads)`` - more than any single field holds.
    """
    fov_beads = [
        [(12, 14), (30, 28)],
        [(18, 33), (35, 12)],
        [(22, 20), (14, 38), (40, 30)],
        [(38, 16), (11, 25)],
    ][:n_fov]
    s0 = 1.1
    focus = n_steps // 2
    yy, xx = np.mgrid[0:h, 0:w]

    def frame_img(fov, k):
        sigma = s0 * (1.0 + 0.07 * abs(k - focus))
        img = np.full((h, w), 100.0, dtype=np.float32)
        for bx, by in fov_beads[fov]:
            img += 3000.0 * np.exp(
                -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma**2)
            )
        return img

    frames = []
    if order == "z":
        for fov in range(n_fov):
            for k in range(n_steps):
                frames.append(frame_img(fov, k))
    else:  # "fov": all FOVs at z0, then all FOVs at z1, ...
        for k in range(n_steps):
            for fov in range(n_fov):
                frames.append(frame_img(fov, k))
    movie = np.stack(frames).astype(np.uint16)
    return movie, fov_beads, focus


class TestFovOfFrame:
    def test_z_order(self):
        # 2 FOVs x 5 steps, z order: each FOV is a full z stack
        fov = spline._fov_of_frame(10, 2, "z")
        np.testing.assert_array_equal(fov, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def test_fov_order(self):
        # 2 FOVs interleaved at each z position
        fov = spline._fov_of_frame(10, 2, "fov")
        np.testing.assert_array_equal(fov, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    def test_single_fov(self):
        np.testing.assert_array_equal(
            spline._fov_of_frame(5, 1, "fov"), [0, 0, 0, 0, 0]
        )

    def test_trailing_frames_marked_invalid(self):
        # 7 frames, 2 FOVs -> n_steps=3, frame 6 does not complete a step
        fov = spline._fov_of_frame(7, 2, "z")
        assert fov[-1] == -1

    def test_fov_and_step_pair_uniquely(self):
        # every (fov, step) pair maps to exactly one frame, in both orders
        n_frames, fps = 12, 3
        for order in ("fov", "z"):
            step, _, _ = spline._step_of_frame(
                n_frames, 10.0, fps, order, None
            )
            fov = spline._fov_of_frame(n_frames, fps, order)
            pairs = list(zip(fov.tolist(), step.tolist()))
            assert len(set(pairs)) == len(pairs) == n_frames


class TestMaskToSegments:
    def test_contiguous_run(self):
        mask = np.array([0, 1, 1, 1, 0, 0], dtype=bool)
        assert spline._mask_to_segments(mask) == [(1, 3)]

    def test_multiple_runs(self):
        mask = np.array([1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
        assert spline._mask_to_segments(mask) == [(0, 1), (3, 3), (5, 7)]

    def test_empty(self):
        assert spline._mask_to_segments(np.zeros(5, dtype=bool)) == []

    def test_reference_segments_split_per_fov_in_z_order(self):
        # 3 FOVs x 12 steps, z order: the in-focus middle third of each FOV is
        # a separate segment (not one giant min..max span)
        n_frames, fps = 36, 3
        step, _, step_range = spline._step_of_frame(
            n_frames, 10.0, fps, "z", None
        )
        segments = spline._reference_frame_segments(step, step_range)
        assert len(segments) == fps  # one in-focus block per FOV


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


class TestFringePreservingPhaseTemplates:
    """4Pi phase templates must SURVIVE the bead average.

    Every bead sits at its own height, so its interference fringe starts at its
    own phase; averaging the raw volumes cancels the fringe almost completely.
    The builder must recover each bead's phase and shift it onto a common one,
    so the averaged template keeps (nearly) the single-bead modulation depth.
    """

    TWO_K = 1.2  # rad per z-slice (period ~5.2 slices, as in real 4Pi data)
    DEPTH = 0.8  # single-bead modulation depth
    N_Z = 21
    N_CH = 2
    N_PH = 3
    N_BEADS = 12

    def _dataset(self, seed=0):
        import pandas as pd

        rng = np.random.default_rng(seed)
        box, nz = BOX, self.N_Z
        phases = np.arange(self.N_PH) * 2 * np.pi / self.N_PH
        chan_offset = np.array([0.0, np.pi / 2])[: self.N_CH]
        # one FOV per bead, frames ordered z-fastest
        h = w = 40
        n_frames = self.N_BEADS * nz
        step_of_frame = np.arange(n_frames) % nz
        fov_of_frame = np.arange(n_frames) // nz
        step_range = np.arange(nz)
        bead_xy = rng.integers(box, h - box, size=(self.N_BEADS, 2))
        # each bead sits at its own height -> its own fringe phase
        z_off = rng.uniform(-nz / 4, nz / 4, self.N_BEADS)
        yy, xx = np.mgrid[0:h, 0:w]
        movies = []
        for c in range(self.N_CH):
            for p in range(self.N_PH):
                mov = np.zeros((n_frames, h, w), dtype=np.float32)
                for b in range(self.N_BEADS):
                    bx, by = bead_xy[b]
                    g = np.exp(
                        -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * 1.3**2)
                    )
                    for f in np.flatnonzero(fov_of_frame == b):
                        zz = step_of_frame[f] - z_off[b] - nz // 2
                        env = 1000.0 * np.exp(-(zz**2) / (2 * 7.0**2))
                        # fringe: mod + i mod90 = env * DEPTH * exp(i*(2k z + psi))
                        angle = self.TWO_K * (step_of_frame[f] - z_off[b])
                        fringe = 1.0 + self.DEPTH * np.cos(
                            angle + chan_offset[c] - phases[p]
                        )
                        mov[f] += (env * fringe * g).astype(np.float32)
                movies.append(mov + 10.0)
        beads = pd.DataFrame(
            {
                "x": bead_xy[:, 0],
                "y": bead_xy[:, 1],
                "fov": np.arange(self.N_BEADS),
            }
        )
        cams = [dict(CAMERA_INFO) for _ in movies]
        transforms = [
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            for _ in range(self.N_CH)
        ]
        return (
            movies,
            cams,
            beads,
            transforms,
            phases,
            step_of_frame,
            step_range,
            fov_of_frame,
        )

    @staticmethod
    def _contrast(profiles, phases, lo, hi):
        mean_z, mod, mod90 = spline._central_decompose(profiles, phases)
        return float(
            np.median(np.hypot(mod, mod90)[lo:hi])
            / (np.median(mean_z[lo:hi]) + 1e-9)
        )

    def test_fringe_survives_the_bead_average(self):
        (
            movies,
            cams,
            beads,
            transforms,
            phases,
            step_of_frame,
            step_range,
            fov_of_frame,
        ) = self._dataset()
        templates, frequency, z_center = (
            spline._build_fringe_preserving_phase_templates(
                movies,
                cams,
                beads,
                transforms,
                self.N_CH,
                self.N_PH,
                phases,
                BOX,
                step_of_frame,
                step_range,
                fov_of_frame,
                0,
            )
        )
        assert templates.shape == (self.N_PH, BOX, BOX, self.N_Z, self.N_CH)
        # the fringe frequency is recovered (frequency = two_k / 2)
        assert frequency == pytest.approx(self.TWO_K / 2, rel=0.05)

        zc, lo, hi = BOX // 2, 4, self.N_Z - 4
        for c in range(self.N_CH):
            got = self._contrast(
                [templates[p, zc, zc, :, c] for p in range(self.N_PH)],
                phases,
                lo,
                hi,
            )
            # the averaged template keeps most of the single-bead depth; a
            # mis-signed or mis-demodulated alignment lands far below this
            assert got > 0.5 * self.DEPTH, f"channel {c} contrast {got:.3f}"

    def test_alignment_beats_the_naive_average(self):
        """The whole point of the alignment: it must be much better than just
        averaging the raw bead volumes."""
        (
            movies,
            cams,
            beads,
            transforms,
            phases,
            step_of_frame,
            step_range,
            fov_of_frame,
        ) = self._dataset(seed=3)
        templates, _, _ = spline._build_fringe_preserving_phase_templates(
            movies,
            cams,
            beads,
            transforms,
            self.N_CH,
            self.N_PH,
            phases,
            BOX,
            step_of_frame,
            step_range,
            fov_of_frame,
            0,
        )
        zc, lo, hi = BOX // 2, 4, self.N_Z - 4
        naive = {}
        for c in range(self.N_CH):
            for p in range(self.N_PH):
                v = spline._bead_volumes(
                    movies[c * self.N_PH + p],
                    cams[c * self.N_PH + p],
                    beads,
                    BOX,
                    step_of_frame,
                    step_range,
                    fov_of_frame=fov_of_frame,
                )
                naive[(c, p)] = v.mean(axis=0)
        for c in range(self.N_CH):
            aligned = self._contrast(
                [templates[p, zc, zc, :, c] for p in range(self.N_PH)],
                phases,
                lo,
                hi,
            )
            unaligned = self._contrast(
                [naive[(c, p)][zc, zc, :] for p in range(self.N_PH)],
                phases,
                lo,
                hi,
            )
            assert (
                aligned > 3.0 * unaligned
            ), f"channel {c}: aligned {aligned:.3f} vs naive {unaligned:.3f}"


class TestPhaseValidationPlot:
    """The opt-in 4Pi validation plot must survive failed individual fits.

    The spline fitter returns NaN for spots that diverge (typically the most
    defocused slices). Referencing each bead by a plain median then poisons the
    whole bead - and once every bead has one bad frame, the entire plot is NaN
    and matplotlib dies with "All-NaN slice encountered", losing the diagnostic
    for a fit that was mostly fine.
    """

    N_BEADS, N_Z = 6, 21

    def _inputs(self, tmp_path):
        import pandas as pd

        n_frames = self.N_BEADS * self.N_Z
        step_of_frame = np.arange(n_frames) % self.N_Z
        fov_of_frame = np.arange(n_frames) // self.N_Z
        beads = pd.DataFrame(
            {
                "x": np.full(self.N_BEADS, 20),
                "y": np.arange(self.N_BEADS) * 7 + 20,
                "fov": np.arange(self.N_BEADS),
            }
        )
        calibration = {
            "model": "spline-3d-phase-multichannel",
            "z_center": self.N_Z // 2,
            "zt_nm": 205.0,
            "frequency": 0.6,
            "z_step_nm": 50.0,
            "magnification_factor": 0.79,
        }
        movies = [np.zeros((n_frames, 60, 60), dtype=np.float32)] * 4
        cams = [dict(CAMERA_INFO)] * 4
        return (
            movies,
            cams,
            beads,
            calibration,
            step_of_frame,
            fov_of_frame,
            os.path.join(str(tmp_path), "cal.hdf5"),
        )

    def _fake_locs(self, ids, nan_every=None, all_nan=False):
        import pandas as pd

        n = len(ids)
        step = np.asarray(ids["frame"]) % self.N_Z
        z = -(step - self.N_Z // 2) * 50.0 * 0.79
        z = z + np.random.default_rng(0).normal(0, 5.0, n)
        if all_nan:
            z = np.full(n, np.nan)
        elif nan_every:
            z[::nan_every] = np.nan
        return pd.DataFrame(
            {
                "frame": np.asarray(ids["frame"]),
                "x": np.asarray(ids["x"], dtype=float),
                "y": np.asarray(ids["y"], dtype=float),
                "z": z,
                "phase": np.random.default_rng(1).uniform(0, 2 * np.pi, n),
            }
        )

    def _run(self, tmp_path, monkeypatch, **kwargs):
        (
            movies,
            cams,
            beads,
            calibration,
            step_of_frame,
            fov_of_frame,
            path,
        ) = self._inputs(tmp_path)
        monkeypatch.setattr(
            localize,
            "fit_spline_phase_multichannel",
            lambda mv, cm, ids, box, cal, **kw: self._fake_locs(ids, **kwargs),
        )
        spline._save_phase_validation(
            movies,
            cams,
            beads,
            calibration,
            1,
            4,
            BOX,
            fov_of_frame,
            step_of_frame,
            np.arange(self.N_Z),
            50.0,
            0.79,
            path,
        )
        return os.path.splitext(path)[0] + "_phase4pi_validation.png"

    def test_survives_one_failed_fit_per_bead(self, tmp_path, monkeypatch):
        # every bead has a NaN frame: a plain per-bead median would make the
        # whole plot NaN
        png = self._run(tmp_path, monkeypatch, nan_every=self.N_Z)
        assert os.path.exists(png) and os.path.getsize(png) > 10_000

    def test_all_failed_fits_raise_a_clear_error(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="non-finite z"):
            self._run(tmp_path, monkeypatch, all_nan=True)

    def test_nan_phase_uncertainty_does_not_break_the_unwrap(self):
        """A non-finite CRLB gives non-finite phase weights; the circular
        statistics must ignore them rather than return NaN for everything."""
        from picasso import fourpi

        rng = np.random.default_rng(0)
        n = 300
        z = rng.uniform(-300, 300, n)
        k = np.pi / 205.0
        phase = np.mod(2 * k * z, 2 * np.pi)
        # every weight non-finite (all-NaN CRLB)
        assert np.isfinite(
            fourpi.cyclic_average(
                np.mod(z, 205.0), 205.0, weights=np.full(n, np.nan)
            )
        )
        # a partly NaN input still gives a usable mean
        data = np.mod(z, 205.0)
        data[::7] = np.nan
        assert np.isfinite(fourpi.cyclic_average(data, 205.0))
        assert np.isfinite(fourpi.estimate_phase_reference(z, phase, k))


class TestFringePeriodUnits:
    """The fringe period is calibrated in RAW STAGE nm.

    Calibration beads sit on the coverslip and follow the stage 1:1, so no
    refractive-index focal-shift correction applies to the calibration z axis.
    Localizations, however, carry sample-space z (the fitter scales by
    ``magnification_factor``), so the conversion happens when the period is used
    against localizations.
    """

    def test_frequency_from_calibration_converts_to_localization_units(self):
        from picasso import fourpi

        mag = 0.8
        cal = {"zt_nm": 250.0, "magnification_factor": mag}
        k = fourpi.frequency_from_calibration(cal)
        # one stage nm is `mag` nm of localization z
        assert k == pytest.approx(np.pi / (250.0 * mag))
        # explicit override wins over the stored factor
        assert fourpi.frequency_from_calibration(cal, 1.0) == pytest.approx(
            np.pi / 250.0
        )

    def test_zt_nm_and_frequency_fallback_agree(self):
        from picasso import fourpi

        # zt_nm = pi / frequency * z_step_nm (stage nm), so both branches must
        # give the same rad-per-localization-nm
        d, freq_slice, mag = 50.0, 0.6, 0.79
        cal = {
            "frequency": freq_slice,
            "z_step_nm": d,
            "magnification_factor": mag,
        }
        from_slice = fourpi.frequency_from_calibration(cal)
        cal_with_zt = dict(cal, zt_nm=np.pi / freq_slice * d)
        assert from_slice == pytest.approx(
            fourpi.frequency_from_calibration(cal_with_zt), rel=1e-9
        )


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


class TestMultiFov:
    """Genuine multi-FOV z-stacks: several fields with *different* beads at
    *different* positions. Beads must be detected and extracted per FOV (each
    from its own field's frames), never averaged across fields."""

    def test_detects_beads_from_all_fovs_with_labels(self):
        movie, fov_beads, _ = _synthetic_multifov_movie(n_fov=3, order="z")
        n_fov = len(fov_beads)
        n_total = sum(len(b) for b in fov_beads)
        n_frames = movie.shape[0]
        step_of_frame, _, step_range = spline._step_of_frame(
            n_frames, 20.0, n_fov, "z", None
        )
        fov_of_frame = spline._fov_of_frame(n_frames, n_fov, "z")
        segments = spline._reference_frame_segments(step_of_frame, step_range)
        beads = spline._detect_bead_positions(
            movie, 2000.0, BOX, segments, fov_of_frame=fov_of_frame
        )
        assert "fov" in beads.columns
        # every field's beads are found (pooling would merge/lose some)
        assert len(beads) == n_total
        assert sorted(beads["fov"].unique()) == list(range(n_fov))
        for k in range(n_fov):
            assert (beads["fov"] == k).sum() == len(fov_beads[k])

    def test_bead_volume_is_isolated_to_its_own_fov(self):
        """A bead's volume must come only from its own FOV's frames: a bright
        contaminant at the same pixel in another FOV must not leak in (the bug
        that corrupted cross-FOV pixel-averaging)."""
        import pandas as pd

        n_steps, h, w = 5, 24, 24
        yy, xx = np.mgrid[0:h, 0:w]
        bx, by = 10, 10

        def blob(amp, sigma):
            return amp * np.exp(
                -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma**2)
            )

        frames = []
        # FOV0: real bead, amplitude 3000; FOV1: bright contaminant 12000 at the
        # SAME pixel (z order: FOV0 stack, then FOV1 stack)
        for k in range(n_steps):
            frames.append(blob(3000.0, 1.1 * (1 + 0.1 * abs(k - 2))))
        for k in range(n_steps):
            frames.append(blob(12000.0, 1.1))
        movie = np.stack(frames).astype(np.uint16)

        step_of_frame, _, step_range = spline._step_of_frame(
            2 * n_steps, 20.0, 2, "z", None
        )
        fov_of_frame = spline._fov_of_frame(2 * n_steps, 2, "z")
        beads = pd.DataFrame({"x": [bx], "y": [by], "fov": [0]})
        vols = spline._bead_volumes(
            movie,
            CAMERA_INFO,
            beads,
            BOX,
            step_of_frame,
            step_range,
            fov_of_frame=fov_of_frame,
        )
        peak = vols[0].max()
        # FOV0-only peak ~3000; cross-FOV averaging would give ~7500, and the
        # raw contaminant is 12000. Must be the isolated FOV0 value.
        assert peak == pytest.approx(3000.0, rel=0.15)
        assert peak < 5000.0

    def test_build_template_multifov_is_clean(self):
        """End-to-end: a multi-FOV stack yields a well-focused template built
        from all fields' beads (the per-FOV path), for both frame orders."""
        for order in ("z", "fov"):
            movie, fov_beads, focus = _synthetic_multifov_movie(
                n_fov=3, order=order
            )
            n_fov = len(fov_beads)
            n_total = sum(len(b) for b in fov_beads)
            built = spline.build_psf_template(
                movie,
                CAMERA_INFO,
                box=BOX,
                minimum_ng=2000.0,
                d=20.0,
                frames_per_step=n_fov,
                frame_order=order,
            )
            # beads pooled from all fields
            assert built["n_beads"] == n_total
            # focus recovered near the central step; clean normalized template
            assert abs(built["z_center"] - focus) <= 2
            tpl = built["template"]
            assert tpl[:, :, built["z_center"]].max() == pytest.approx(
                1.0, abs=0.05
            )
            assert tpl.min() == pytest.approx(0.0, abs=0.1)


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


class TestRansacMatch:
    """RANSAC bead matching that makes the channel registration robust to the
    coarse (ROI-origin) pre-alignment - the fix for the ROI-placement
    hypersensitivity of split-FOV calibrations."""

    @staticmethod
    def _mirrored_pair():
        # six reference beads and their images under a known y-mirror + small
        # rotation + shift (a realistic biplane registration)
        ref = np.array(
            [[10, 12], [40, 18], [25, 55], [60, 50], [15, 70], [50, 82]],
            dtype=float,
        )
        theta = np.deg2rad(3.0)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        flip = np.array([[1.0, 0.0], [0.0, -1.0]])  # mirror in y
        linear = rot @ flip
        offset = np.array([7.0, 190.0])
        c = ref @ linear.T + offset
        return ref, c, linear, offset

    @pytest.mark.parametrize("overlay_offset", [(0.0, 0.0), (14.0, -11.0)])
    def test_recovers_transform_despite_bad_overlay(self, overlay_offset):
        """A wrong coarse overlay (a misplaced ROI) must not change the result:
        the correct correspondences and transform are recovered regardless."""
        ref, c, linear, offset = self._mirrored_pair()
        # the coarse overlay maps c back near ref but is deliberately off; a plain
        # nearest-neighbour match at this offset would mis-pair some beads
        inv = np.linalg.inv(linear)
        aligned = (c - offset) @ inv.T + np.asarray(overlay_offset)
        ref_idx, c_idx = spline._ransac_match(
            ref, c, aligned, inlier_tol=3.0, radius=40.0
        )
        assert len(ref_idx) == len(ref)  # all beads matched
        # identity correspondence (c[i] is the image of ref[i]) is recovered
        order = np.argsort(ref_idx)
        assert ref_idx[order].tolist() == list(range(len(ref)))
        assert c_idx[order].tolist() == list(range(len(ref)))
        # and the transform fit on them matches the truth (mirror -> det < 0)
        M = localize.estimate_affine_transform(ref[ref_idx], c[c_idx])
        np.testing.assert_allclose(M[:, :2], linear, atol=0.02)
        assert np.linalg.det(M[:, :2]) < 0

    def test_rejects_decoy_and_is_overlay_independent(self):
        """Extra unmatched channel beads (decoys) are rejected, and two very
        different overlays give the same correspondences."""
        ref, c, _, offset = self._mirrored_pair()
        c_dec = np.vstack([c, [[200.0, 5.0], [5.0, 5.0]]])  # 2 decoys
        results = []
        for off in [(0.0, 0.0), (18.0, 16.0)]:
            aligned = np.vstack(
                [ref + np.asarray(off), [[999.0, 999.0], [-999.0, -999.0]]]
            )
            ri, ci = spline._ransac_match(
                ref, c_dec, aligned, inlier_tol=3.0, radius=45.0
            )
            results.append((ri.tolist(), ci.tolist()))
            assert len(ri) == len(ref)  # decoys excluded
            assert max(ci) < len(c)  # only real channel beads matched
        assert results[0] == results[1]  # overlay-independent

    def test_estimate_channel_transform_recovers_shift(self):
        movie_ref, _, _ = _synthetic_bead_movie()
        dx, dy = 3, -2  # channel is the reference shifted by (dx, dy)
        movie_c = np.roll(movie_ref, shift=(dy, dx), axis=(1, 2))

        step_of_frame, _, step_range = spline._step_of_frame(
            movie_ref.shape[0], 20.0, 1, "fov", None
        )
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
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

    def test_estimate_channel_transform_recovers_flip(self):
        """Separate movies where the channel is a vertically mirrored copy of
        the reference (a reflected optical path). The flip-aware coarse matching
        must still register it: all beads match and the affine is a reflection.
        """
        movie_ref, _, _ = _synthetic_bead_movie()
        movie_c = movie_ref[:, ::-1, :]  # mirror in y (up/down)

        step_of_frame, _, step_range = spline._step_of_frame(
            movie_ref.shape[0], 20.0, 1, "fov", None
        )
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
        beads_ref = spline._detect_bead_positions(
            movie_ref, 2000.0, BOX, ref_bounds
        )
        n_ref = len(beads_ref)
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
        assert n_matches == n_ref
        # a pure translation could never register a mirror: reflection -> det < 0
        assert np.linalg.det(np.asarray(transform)[:, :2]) < 0
        # ref (x, y) -> channel (x, H - 1 - y)
        h = movie_ref.shape[1]
        ref_xy = beads_ref[["x", "y"]].to_numpy(float)
        mapped = localize.apply_affine_transform(ref_xy, transform)
        np.testing.assert_allclose(mapped[:, 0], ref_xy[:, 0], atol=1.0)
        np.testing.assert_allclose(
            mapped[:, 1], h - 1 - ref_xy[:, 1], atol=1.0
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

    def test_calibrate_separate_channels_mirrored(self, tmp_path):
        """Separate-movie channels where channel 1 is a vertical mirror of the
        reference (reflected optical path). The build must register it (all
        beads matched) and store a reflection transform, so the per-channel
        template is built at the real mirrored bead positions."""
        movie_ref, _, _ = _synthetic_bead_movie()
        movie_c = movie_ref[:, ::-1, :]  # mirror in y
        info = [{"Frames": int(movie_ref.shape[0])}]
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            path=str(tmp_path / "mirrored_spline_calib.hdf5"),
        )
        assert calib["n_channels"] == 2
        # channel-1 transform is a reflection (negative determinant), not the
        # garbage a translation-only match would have produced
        t1 = np.asarray(calib["channel_transforms"][1])
        assert np.linalg.det(t1[:, :2]) < 0
        # the transform maps into the mirrored frame: y -> H - 1 - y
        h = movie_ref.shape[1]
        assert abs(t1[1, 1] + 1.0) < 0.1  # y scale ~ -1
        assert abs(t1[1, 2] - (h - 1)) < 2.0  # y offset ~ H - 1

    @pytest.mark.skipif(
        not localize.GPUFIT_INSTALLED, reason="Gpufit not available"
    )
    def test_axial_precision_multichannel_is_joint(self):
        """The multichannel axial-precision diagnostic must fit all channels
        *jointly* (the real pipeline) rather than each plane alone. This checks
        the joint contract: it stacks every channel's per-frame spots, fits them
        against the full calibration, tags the result as a joint N-channel fit
        and returns one bias/precision sample per z-step. (Degeneracy-breaking
        needs realistic aberrated PSFs; the symmetric synthetic Gaussian here is
        z-degenerate even jointly, so no tight bias bound is asserted.)"""
        import pandas as pd

        movie_ref, _, _ = _synthetic_bead_movie()
        # biplane-style: channel 1 focuses at a different stage step
        movie_c = np.roll(movie_ref, shift=2, axis=0)
        info = [{"Frames": int(movie_ref.shape[0])}]
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
        )
        # rebuild each channel's per-frame spots at its (mapped) bead positions,
        # exactly as calibrate_spline_multichannel does internally
        transforms = calib["channel_transforms"]
        movies = [movie_ref, movie_c]
        step_of_frame, _, step_range = spline._step_of_frame(
            movie_ref.shape[0], 20.0, 1, "fov", None
        )
        rb = spline._reference_frame_segments(step_of_frame, step_range)
        beads_ref = spline._detect_bead_positions(movie_ref, 2000.0, BOX, rb)
        ref_xy = beads_ref[["x", "y"]].to_numpy(float)
        per_channel = []
        for c, m in enumerate(movies):
            if c == 0:
                beads_c = beads_ref
            else:
                mp = localize.apply_affine_transform(ref_xy, transforms[c])
                beads_c = pd.DataFrame(
                    {
                        "x": np.rint(mp[:, 0]).astype(int),
                        "y": np.rint(mp[:, 1]).astype(int),
                    }
                )
            per_channel.append(
                spline.build_psf_template(
                    m,
                    CAMERA_INFO,
                    BOX,
                    2000.0,
                    20.0,
                    beads=beads_c,
                    return_spots=True,
                )
            )
        prec = spline._axial_precision_multichannel(per_channel, calib)
        assert prec is not None
        assert prec["joint"] == 2  # tagged as a joint 2-channel fit
        z = np.asarray(per_channel[0]["z_of_step"], float)
        # one bias/precision sample per z-step, and the joint fit produced
        # finite z estimates for a good fraction of spots
        assert len(prec["bias_z"]) == len(z)
        assert len(prec["precision_z"]) == len(z)
        assert np.any(np.isfinite(prec["bias_z"]))
        assert prec["n_spots"] > 0
        assert len(prec["scatter_fit"]) == len(prec["scatter_stage"]) > 0


def _synthetic_split_fov_movie(dx=2, dy=-1):
    """A single movie whose left and right 48x48 halves are two channels.

    The right half (region 1) is the left half (region 0, reference) shifted
    within its region by ``(dx, dy)`` pixels, so the ref->region-1 affine is a
    pure translation ``[[1, 0, 48 + dx], [0, 1, dy]]`` in absolute chip
    coordinates. Returns ``(movie, regions, bead_xy)`` where ``bead_xy`` are the
    reference-region bead centers.
    """
    base, bead_xy, _focus = _synthetic_bead_movie(h=48, w=48)
    n = base.shape[0]
    movie = np.zeros((n, 48, 96), dtype=np.uint16)
    movie[:, :, :48] = base
    # small shift stays well inside the region, so np.roll wrap-around is moot
    movie[:, :, 48:] = np.roll(base, shift=(dy, dx), axis=(1, 2))
    regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
    return movie, regions, bead_xy


def _synthetic_split_fov_movie_flipped(axis="y"):
    """Single movie whose right half is the left half *mirrored* (as biplane /
    4Pi / spectral splitters do with a reflected optical path).

    ``axis="y"`` flips up/down, ``"x"`` left/right. Returns ``(movie, regions)``.
    A pure-translation coarse alignment cannot register a mirrored channel, so
    this exercises the flip-aware matching in ``_estimate_channel_transform``.
    """
    base, _bead_xy, _focus = _synthetic_bead_movie(h=48, w=48)
    n = base.shape[0]
    movie = np.zeros((n, 48, 96), dtype=np.uint16)
    movie[:, :, :48] = base
    flipped = base[:, ::-1, :] if axis == "y" else base[:, :, ::-1]
    movie[:, :, 48:] = flipped
    regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
    return movie, regions


class TestSplitFovTransform:
    """Region-aware channel-transform estimation (no Gpuspline needed)."""

    def test_estimate_region_transform_recovers_shift(self):
        dx, dy = 2, -1
        movie, regions, _ = _synthetic_split_fov_movie(dx, dy)

        step_of_frame, _, step_range = spline._step_of_frame(
            movie.shape[0], 20.0, 1, "fov", None
        )
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
        ref_roi = spline._normalized_region(regions[0])
        chan_roi = spline._normalized_region(regions[1])
        beads_ref = spline._detect_bead_positions(
            movie, 2000.0, BOX, ref_bounds, roi=ref_roi
        )
        # coarse shift = (x0_ref - x0_c, y0_ref - y0_c) = (0 - 48, 0 - 0)
        transform, n_matches = spline._estimate_channel_transform(
            movie,
            movie,
            beads_ref,
            2000.0,
            BOX,
            ref_bounds,
            mid,
            max_distance=float(BOX),
            channel_roi=chan_roi,
            coarse_shift=(-48.0, 0.0),
        )
        assert n_matches >= 3
        # ref (x, y) -> region-1 (x + 48 + dx, y + dy)
        np.testing.assert_allclose(
            transform,
            np.array([[1.0, 0.0, 48 + dx], [0.0, 1.0, dy]]),
            atol=0.6,
        )

    @pytest.mark.parametrize("axis", ["y", "x"])
    def test_estimate_region_transform_recovers_flip(self, axis):
        """A mirrored channel (biplane/4Pi reflected path) must still register:
        the flip-aware coarse matching finds all correspondences and the affine
        encodes the mirror (negative determinant)."""
        movie, regions = _synthetic_split_fov_movie_flipped(axis)

        step_of_frame, _, step_range = spline._step_of_frame(
            movie.shape[0], 20.0, 1, "fov", None
        )
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
        ref_roi = spline._normalized_region(regions[0])
        chan_roi = spline._normalized_region(regions[1])
        beads_ref = spline._detect_bead_positions(
            movie, 2000.0, BOX, ref_bounds, roi=ref_roi
        )
        n_ref = len(beads_ref)
        transform, n_matches = spline._estimate_channel_transform(
            movie,
            movie,
            beads_ref,
            2000.0,
            BOX,
            ref_bounds,
            mid,
            max_distance=float(BOX),
            channel_roi=chan_roi,
            coarse_shift=(-48.0, 0.0),
        )
        # all reference beads are matched (a pure translation would find few)
        assert n_matches == n_ref
        # the linear part is a reflection -> negative determinant
        assert np.linalg.det(np.asarray(transform)[:, :2]) < 0
        # applying the transform maps ref beads into the channel region, mirrored
        ref_xy = beads_ref[["x", "y"]].to_numpy(float)
        mapped = localize.apply_affine_transform(ref_xy, transform)
        if axis == "y":
            np.testing.assert_allclose(
                mapped[:, 0], ref_xy[:, 0] + 48, atol=1.0
            )
            np.testing.assert_allclose(
                mapped[:, 1], 47 - ref_xy[:, 1], atol=1.0
            )
        else:
            np.testing.assert_allclose(
                mapped[:, 0], 47 - ref_xy[:, 0] + 48, atol=1.0
            )
            np.testing.assert_allclose(mapped[:, 1], ref_xy[:, 1], atol=1.0)


@pytest.mark.skipif(
    not localize.GPUSPLINE_INSTALLED, reason="Gpuspline not available"
)
class TestCalibrateSplitFov:
    """Full split-FOV calibration from one movie with two FOV regions."""

    def test_stores_metadata_and_region_transform(self, tmp_path):
        from picasso import io

        dx, dy = 2, -1
        movie, regions, _ = _synthetic_split_fov_movie(dx, dy)
        info = [{"Frames": int(movie.shape[0])}]
        path = str(tmp_path / "splitfov_spline_calib.hdf5")
        calib = spline.calibrate_spline_split_fov(
            movie,
            info=info,
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            regions=regions,
            path=path,
        )
        assert calib["model"] == "spline-3d-multichannel"
        assert calib["n_channels"] == 2
        assert calib["split_fov"] is True
        assert calib["reference"] == 0
        assert len(calib["regions"]) == 2
        # region-1 transform is the known translation (absolute coords)
        np.testing.assert_allclose(
            calib["channel_transforms"][1],
            [[1.0, 0.0, 48 + dx], [0.0, 1.0, dy]],
            atol=0.6,
        )
        assert calib["coefficients"].shape[-1] == 2
        # split-FOV metadata survives the HDF5 round-trip
        loaded = io.load_spline_calibration(path)
        assert loaded["split_fov"] is True
        assert len(loaded["regions"]) == 2

    def test_reference_index_is_reordered_first(self, tmp_path):
        dx, dy = 2, -1
        movie, regions, _ = _synthetic_split_fov_movie(dx, dy)
        info = [{"Frames": int(movie.shape[0])}]
        # pick region 1 as the reference; it must become channel 0 (identity)
        calib = spline.calibrate_spline_split_fov(
            movie,
            info=info,
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            regions=regions,
            reference=1,
        )
        # reference region stored first, transform now maps region-1 -> region-0
        assert calib["regions"][0] == [[0, 48], [48, 96]]
        np.testing.assert_allclose(
            calib["channel_transforms"][1],
            [[1.0, 0.0, -(48 + dx)], [0.0, 1.0, -dy]],
            atol=0.7,
        )

    def test_saves_per_channel_and_registration_diagnostics(self, tmp_path):
        movie, regions, _ = _synthetic_split_fov_movie(2, -1)
        info = [{"Frames": int(movie.shape[0])}]
        path = tmp_path / "splitfov_spline_calib.hdf5"
        spline.calibrate_spline_split_fov(
            movie,
            info=info,
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            regions=regions,
            path=str(path),
        )
        base = str(tmp_path / "splitfov_spline_calib")
        # one PSF diagnostic per channel + one registration diagnostic
        assert os.path.exists(base + "_ch0.png")
        assert os.path.exists(base + "_ch1.png")
        assert os.path.exists(base + "_registration.png")
        assert os.path.getsize(base + "_registration.png") > 5000

    def test_unequal_region_sizes_raise(self):
        movie, regions, _ = _synthetic_split_fov_movie()
        regions = [[[0, 0], [48, 48]], [[0, 48], [40, 90]]]  # different size
        info = [{"Frames": int(movie.shape[0])}]
        with pytest.raises(ValueError, match="same size"):
            spline.calibrate_spline_split_fov(
                movie,
                info=info,
                camera_info=CAMERA_INFO,
                box=BOX,
                minimum_ng=2000.0,
                d=20.0,
                regions=regions,
            )


class TestRefineSplitFovTransformsFromSignal:
    """Data-driven (no-bead) re-registration of a split-FOV calibration: pair
    blinking single-molecule signal across channels frame by frame, seeded by
    only the calibration's flip, over a bounded sample of frames."""

    @staticmethod
    def _blinking_movie(true_affine, n_frames=150, seed=0):
        """Two 48x48 regions in a 48x96 frame; each frame has a few emitters in
        the reference region that also appear in the channel region at the
        region-local ``true_affine`` mapping (shared signal, as in biplane)."""
        rng = np.random.RandomState(seed)
        H, W = 48, 96
        ref_rect = [[0, 0], [48, 48]]
        c_rect = [[0, 48], [48, 96]]
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        t_true = localize.compose_region_transforms(
            [ref_rect, c_rect], [identity, true_affine]
        )[1]

        def render(frame, x, y, amp, sigma=1.2):
            xi, yi = int(round(x)), int(round(y))
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    yy, xx = yi + dy, xi + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        frame[yy, xx] += amp * np.exp(
                            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)
                        )

        movie = np.zeros((n_frames, H, W), dtype=np.float32)
        for f in range(n_frames):
            for _ in range(rng.randint(3, 6)):
                x = rng.uniform(10, 38)
                y = rng.uniform(10, 38)
                amp = rng.uniform(2500, 4000)
                render(movie[f], x, y, amp)
                cx, cy = localize.apply_affine_transform(
                    np.array([[x, y]]), t_true
                )[0]
                render(movie[f], cx, cy, amp)
        movie = rng.poisson(np.maximum(movie, 0) + 100).astype(np.uint16)
        return movie, [ref_rect, c_rect]

    def test_recovers_true_affine_from_signal(self):
        # true fine registration: small rotation + subpixel shift
        theta = 0.02
        true_affine = np.array(
            [
                [np.cos(theta), -np.sin(theta), 1.5],
                [np.sin(theta), np.cos(theta), -1.0],
            ]
        )
        movie, regions = self._blinking_movie(true_affine)
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        # stale calibration: identity fine registration (drifted from truth)
        calib = {
            "split_fov": True,
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "regions": [[[0, 0], [48, 48]], [[0, 48], [48, 96]]],
            "channel_affines": [identity, identity],
            "channel_transforms": [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 48.0], [0.0, 1.0, 0.0]],
            ],
        }
        updated, reg_info = spline.refine_split_fov_transforms_from_signal(
            movie, calib, regions, minimum_ng=800.0, box=BOX
        )
        # only ~50 frames are sampled (not all 150), so expect fewer pairs
        assert reg_info[0]["n_matches"] >= 40
        assert reg_info[0]["rms"] < 1.0
        # the region-local affine now matches the true fine registration
        np.testing.assert_allclose(
            updated["channel_affines"][1], true_affine, atol=0.15
        )

    def test_recovers_true_affine_with_mirror(self):
        # channel is an x-mirror of the reference (as a biplane relay flips it)
        # plus a small sub-pixel shift; the calibration stores only the mirror,
        # which is the coarse seed the re-registration is allowed to trust - the
        # fine rotation/scale/shift must be recovered fresh from the signal
        w = 48
        true_affine = np.array([[-1.0, 0.0, w + 0.6], [0.0, 1.0, -0.4]])
        movie, regions = self._blinking_movie(true_affine)
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        mirror = [[-1.0, 0.0, float(w)], [0.0, 1.0, 0.0]]
        regs = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
        mirror_transform = localize.compose_region_transforms(
            regs, [np.array(identity), np.array(mirror)]
        )[1].tolist()
        calib = {
            "split_fov": True,
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "regions": regs,
            "channel_affines": [identity, mirror],
            "channel_transforms": [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                mirror_transform,
            ],
        }
        updated, reg_info = spline.refine_split_fov_transforms_from_signal(
            movie, calib, regions, minimum_ng=800.0, box=BOX
        )
        assert reg_info[0]["n_matches"] >= 40
        assert reg_info[0]["rms"] < 1.0
        # the fitted affine stays mirrored (negative determinant) and recovers
        # the true fine registration on top of the flip
        linear = np.array(updated["channel_affines"][1])[:, :2]
        assert np.linalg.det(linear) < 0
        np.testing.assert_allclose(
            updated["channel_affines"][1], true_affine, atol=0.2
        )

    def test_raises_without_shared_signal(self):
        # channel region has no correlated signal -> no pairs -> raises
        rng = np.random.RandomState(1)
        movie = rng.poisson(np.full((60, 48, 96), 100.0)).astype(np.uint16)
        # a few emitters only in the reference region
        for f in range(60):
            for _ in range(4):
                x, y = rng.uniform(10, 38), rng.uniform(10, 38)
                xi, yi = int(round(x)), int(round(y))
                movie[f, yi - 1 : yi + 2, xi - 1 : xi + 2] += 3000
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        calib = {
            "split_fov": True,
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "regions": [[[0, 0], [48, 48]], [[0, 48], [48, 96]]],
            "channel_affines": [identity, identity],
            "channel_transforms": [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 48.0], [0.0, 1.0, 0.0]],
            ],
        }
        with pytest.raises(ValueError):
            spline.refine_split_fov_transforms_from_signal(
                movie,
                calib,
                [[[0, 0], [48, 48]], [[0, 48], [48, 96]]],
                minimum_ng=800.0,
                box=BOX,
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
