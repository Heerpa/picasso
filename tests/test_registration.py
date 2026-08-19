"""
Tests for picasso.registration: building the standalone channel-registration
calibration from beads and from the experimental blinking signal.
"""

import numpy as np
import pytest

from picasso import io, registration
from picasso import transforms as tform

BOX = 7


def _matrix(entry) -> np.ndarray:
    """The 2x3 affine of a stored channel transform."""
    return tform.from_dict(entry).matrix[:2]


def _rotation(theta: float, tx: float, ty: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), tx],
            [np.sin(theta), np.cos(theta), ty],
        ]
    )


def _apply(xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([xy, np.ones(len(xy))]) @ matrix.T


def _bead_image(xy, shape=(128, 128), amp=4000.0, sigma=1.3, seed=0):
    """A single-frame movie of Gaussian beads at ``xy``, with Poisson noise."""
    rng = np.random.RandomState(seed)
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.zeros(shape, dtype=np.float32)
    for x, y in xy:
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return rng.poisson(img + 100).astype(np.uint16)[None]


def _bead_grid(start=20, stop=111, step=15):
    return np.array(
        [
            [x, y]
            for x in range(start, stop, step)
            for y in range(start, stop, step)
        ],
        dtype=float,
    )


def _blinking_movies(transforms, n_frames=80, seed=0, shape=(64, 64)):
    """Frame-synchronized movies sharing signal: every emitter in the reference
    frame also appears in each channel at that channel's transform."""
    rng = np.random.RandomState(seed)
    height, width = shape
    movies = [
        np.zeros((n_frames, height, width), dtype=np.float32)
        for _ in range(len(transforms) + 1)
    ]

    def render(frame, x, y, amp, sigma=1.2):
        xi, yi = int(round(x)), int(round(y))
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                yy, xx = yi + dy, xi + dx
                if 0 <= yy < height and 0 <= xx < width:
                    frame[yy, xx] += amp * np.exp(
                        -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)
                    )

    for f in range(n_frames):
        for _ in range(rng.randint(5, 8)):
            x = rng.uniform(16, 48)
            y = rng.uniform(16, 48)
            amp = rng.uniform(2500, 4000)
            render(movies[0][f], x, y, amp)
            for c, matrix in enumerate(transforms, start=1):
                cx, cy = _apply(np.array([[x, y]]), matrix)[0]
                render(movies[c][f], cx, cy, amp)
    return [
        rng.poisson(np.maximum(m, 0) + 100).astype(np.uint16) for m in movies
    ]


class TestBeadRegistration:
    """Registration from fiducial bead images."""

    def test_recovers_each_channel_transform(self):
        grid = _bead_grid()
        truth = [_rotation(0.015, 5.0, -4.0), _rotation(-0.02, -6.0, 3.5)]
        movies = [_bead_image(grid, seed=1)] + [
            _bead_image(_apply(grid, m), seed=10 + i)
            for i, m in enumerate(truth)
        ]

        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0
        )

        assert calibration["model"] == registration.REGISTRATION_MODEL
        assert calibration["n_channels"] == 3
        assert calibration["source"] == "beads"
        for c, matrix in enumerate(truth, start=1):
            np.testing.assert_allclose(
                _matrix(calibration["channel_transforms"][c]),
                matrix,
                atol=0.15,
            )

    def test_reference_channel_is_the_identity(self):
        grid = _bead_grid()
        truth = _rotation(0.01, 3.0, -2.0)
        movies = [
            _bead_image(grid, seed=1),
            _bead_image(_apply(grid, truth), seed=2),
        ]

        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0
        )

        np.testing.assert_allclose(
            tform.from_dict(calibration["channel_transforms"][0]).matrix,
            np.eye(3),
        )

    def test_transforms_map_reference_into_the_channel(self):
        """The stored direction is reference -> channel, the direction
        ``localize.get_spots_multichannel`` maps detections in."""
        grid = _bead_grid()
        truth = _rotation(0.0, 6.0, -5.0)  # pure, unambiguous translation
        movies = [
            _bead_image(grid, seed=1),
            _bead_image(_apply(grid, truth), seed=2),
        ]

        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0
        )

        mapped = tform.from_dict(calibration["channel_transforms"][1]).apply(
            grid
        )
        np.testing.assert_allclose(mapped, _apply(grid, truth), atol=0.15)

    def test_rejects_a_single_channel(self):
        with pytest.raises(ValueError, match="at least 2 channels"):
            registration.calibrate_channel_registration_from_beads(
                [_bead_image(_bead_grid())], box=BOX, minimum_ng=2000.0
            )

    def test_reports_too_few_bead_pairs(self):
        # two beads is below the affine minimum of three
        sparse = np.array([[30.0, 30.0], [80.0, 80.0]])
        movies = [
            _bead_image(sparse, seed=1),
            _bead_image(sparse + 4.0, seed=2),
        ]
        with pytest.raises(ValueError, match="matched bead pair"):
            registration.calibrate_channel_registration_from_beads(
                movies, box=BOX, minimum_ng=2000.0
            )


class TestSignalRegistration:
    """Registration from the experimental blinking signal."""

    def test_bootstraps_without_a_seed(self):
        """The new capability: build a registration from scratch, with no
        prior transform to pair at."""
        truth = _rotation(0.02, 4.0, -3.0)
        movies = _blinking_movies([truth])

        calibration = registration.calibrate_channel_registration_from_signal(
            movies, box=BOX, minimum_ng=800.0
        )

        assert calibration["source"] == "signal"
        np.testing.assert_allclose(
            _matrix(calibration["channel_transforms"][1]), truth, atol=0.35
        )
        assert calibration["n_pairs"][0] >= 40
        assert calibration["rms"][0] < 1.0

    def test_a_seed_refines_a_stale_registration(self):
        truth = _rotation(0.02, 4.0, -3.0)
        movies = _blinking_movies([truth])
        stale = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.0]])
        seeds = [
            tform.identity(),
            tform.AffineTransform(matrix=np.vstack([stale, [0, 0, 1]])),
        ]

        calibration = registration.calibrate_channel_registration_from_signal(
            movies, box=BOX, minimum_ng=800.0, seed_transforms=seeds
        )

        fitted = _matrix(calibration["channel_transforms"][1])
        np.testing.assert_allclose(fitted, truth, atol=0.35)
        # and it moved towards the truth rather than sitting on the seed
        assert np.abs(fitted - truth).max() < 0.5 * np.abs(stale - truth).max()

    def test_registers_several_channels(self):
        truth = [_rotation(0.02, 4.0, -3.0), _rotation(-0.015, -5.0, 2.0)]
        movies = _blinking_movies(truth)

        calibration = registration.calibrate_channel_registration_from_signal(
            movies, box=BOX, minimum_ng=800.0
        )

        assert calibration["n_channels"] == 3
        for c, matrix in enumerate(truth, start=1):
            np.testing.assert_allclose(
                _matrix(calibration["channel_transforms"][c]),
                matrix,
                atol=0.35,
            )

    def test_frame_bounds_limit_the_sample(self):
        truth = _rotation(0.0, 3.0, -2.0)
        movies = _blinking_movies([truth])

        calibration = registration.calibrate_channel_registration_from_signal(
            movies,
            box=BOX,
            minimum_ng=800.0,
            frame_bounds=[[0, 39]],
            max_frames=20,
        )

        assert calibration["n_sampled_frames"] <= 20
        np.testing.assert_allclose(
            _matrix(calibration["channel_transforms"][1]), truth, atol=0.35
        )

    def test_rejects_an_empty_frame_range(self):
        movies = _blinking_movies([_rotation(0.0, 2.0, 0.0)], n_frames=10)
        with pytest.raises(ValueError, match="No frames"):
            registration.calibrate_channel_registration_from_signal(
                movies, box=BOX, minimum_ng=800.0, frame_bounds=[[50, 60]]
            )

    def test_reports_the_channel_that_could_not_be_registered(self):
        """A channel carrying unrelated signal names itself in the error."""
        movies = _blinking_movies([_rotation(0.0, 3.0, -2.0)], n_frames=40)
        rng = np.random.RandomState(7)
        movies[1] = rng.poisson(np.full(movies[1].shape, 100.0)).astype(
            np.uint16
        )
        with pytest.raises(ValueError, match="Channel 1"):
            registration.calibrate_channel_registration_from_signal(
                movies, box=BOX, minimum_ng=800.0
            )


class TestCalibrationFile:
    def test_round_trips_through_the_generic_calibration_io(self, tmp_path):
        """The registration file has no large arrays, so it rides the existing
        YAML calibration path with no new I/O code."""
        grid = _bead_grid()
        truth = _rotation(0.01, 4.0, -3.0)
        movies = [
            _bead_image(grid, seed=1),
            _bead_image(_apply(grid, truth), seed=2),
        ]
        path = str(tmp_path / "channel_registration.yaml")

        saved = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0, path=path
        )
        loaded = io.load_any_calibration(path)

        assert loaded["model"] == registration.REGISTRATION_MODEL
        assert loaded["n_channels"] == saved["n_channels"]
        np.testing.assert_allclose(
            _matrix(loaded["channel_transforms"][1]),
            _matrix(saved["channel_transforms"][1]),
        )

    def test_transforms_are_consumable_by_the_multichannel_helpers(self):
        """The stored transforms are the same wire format the existing
        multichannel geometry helpers already take."""
        import pandas as pd

        from picasso import localize

        grid = _bead_grid()
        truth = _rotation(0.0, 6.0, -5.0)
        movies = [
            _bead_image(grid, seed=1),
            _bead_image(_apply(grid, truth), seed=2),
        ]
        calibration = registration.calibrate_channel_registration_from_beads(
            movies, box=BOX, minimum_ng=2000.0
        )

        ids = pd.DataFrame(
            {
                "frame": [0, 0],
                "x": [40, 60],
                "y": [40, 60],
                "net_gradient": [1.0, 1.0],
            }
        )
        residuals = localize.channel_roi_residuals(
            ids, calibration["channel_transforms"]
        )
        assert residuals.shape == (2, 2, 2)
        # the reference channel's box sits on the detection itself
        np.testing.assert_array_equal(residuals[:, 0, :], 0.0)
