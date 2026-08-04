"""
test_gaussfit
~~~~~~~~~~~~~

Tests for ``picasso.fitting.gaussfit``, the CPU Gaussian PSF fitter.

Unlike ``test_gaussfit_cuda``, this module *does* have a twin to compare
against, in two directions:

- **The GPU backend**, which runs the identical algorithm. Under a fixed
  iteration budget and double precision the two must agree to machine
  precision, states and iteration counts included - there is no convergence
  branch left for them to disagree on, so the comparison pins the algebra
  exactly and cannot be flaky.
- **``picasso.gausslq``**, which minimizes the *same objective* with SciPy's
  MINPACK instead. Different algorithms cannot be compared step for step, but
  they must land on the same optimum - and where they differ, the fit that
  reaches the lower chi-square is the right one. ``gausslq`` evaluates its
  model in float32, so it is the one with slack.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest

from picasso import gausslq, localize
from picasso.fitting import gaussfit, gaussfit_cuda, lmfit_cuda, splinefit

BOX = 7
CENTRE = (BOX - 1) / 2.0

MODELS = [
    pytest.param(gaussfit.SPHERICAL, id="spherical"),
    pytest.param(gaussfit.ELLIPTIC, id="elliptic"),
    pytest.param(gaussfit.ROTATED, id="rotated"),
]


def _reference(model, theta, box=BOX):
    """The closed-form model image, straight from the source papers."""
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float64)
    if model == gaussfit.SPHERICAL:
        amp, cx, cy, s, bg = theta
        return (
            amp * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / s**2) + bg
        )
    if model == gaussfit.ELLIPTIC:
        amp, cx, cy, sx, sy, bg = theta
        ex = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        return amp * ex + bg
    amp, cx, cy, sx, sy, bg, angle = theta
    ca, sa = np.cos(angle), np.sin(angle)
    a = (xx - cx) * ca - (yy - cy) * sa
    b = (xx - cx) * sa + (yy - cy) * ca
    return amp * np.exp(-0.5 * ((a / sx) ** 2 + (b / sy) ** 2)) + bg


def _truth(model):
    """A deliberately asymmetric parameter set, so an x/y swap cannot hide."""
    base = [220.0, CENTRE + 0.31, CENTRE - 0.22]
    if model == gaussfit.SPHERICAL:
        return np.array(base + [1.6, 9.0])
    if model == gaussfit.ELLIPTIC:
        return np.array(base + [1.35, 1.85, 9.0])
    return np.array(base + [1.35, 1.85, 9.0, 0.4])


def _canonical_widths(model, theta):
    """Widths as an unordered pair of magnitudes.

    The model depends on the widths only through their squares, and the
    rotated one is invariant under (swap sx/sy, rotate by pi/2), so the raw
    values are not unique. See
    ``test_gaussfit_cuda.TestWidthAndAngleDegeneracies``."""
    if model == gaussfit.SPHERICAL:
        return np.array([abs(theta[3])])
    return np.sort(np.abs(theta[3:5]))


def _seed(model, spots):
    """A starting point of the right shape for ``model``."""
    kwargs = {
        gaussfit.SPHERICAL: dict(spherical=True),
        gaussfit.ELLIPTIC: {},
        gaussfit.ROTATED: dict(rotated=True),
    }[model]
    return localize._initial_parameters_gauss(spots, BOX, **kwargs).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def lq_and_lm(noisy_batch):
    """``gausslq`` and ``gaussfit`` run to convergence on the same spots."""
    lq = gausslq.fit_spots(noisy_batch, tolerance=1e-8, max_iterations=400)
    lm = gaussfit.fit_spots(
        gaussfit.ELLIPTIC,
        noisy_batch,
        localize._initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        ),
        mle=False,
        tolerance=1e-10,
        max_iterations=400,
    )[0]
    return lq, lm


@pytest.fixture(scope="module")
def noisy_batch():
    """Poisson-noisy elliptical spots over a realistic parameter range."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:BOX, 0:BOX].astype(np.float64)
    n = 120
    spots = np.empty((n, BOX, BOX), np.float32)
    for k in range(n):
        cx = CENTRE + rng.uniform(-0.6, 0.6)
        cy = CENTRE + rng.uniform(-0.6, 0.6)
        sx, sy = rng.uniform(1.0, 1.6), rng.uniform(1.0, 1.6)
        amp, bg = rng.uniform(300, 1500), rng.uniform(5, 30)
        mu = (
            amp
            * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
            + bg
        )
        spots[k] = rng.poisson(mu)
    return spots


class TestGroundTruthRecovery:
    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_recovers_a_noiseless_spot(self, model, mle):
        theta = _truth(model)
        spots = _reference(model, theta)[None].astype(np.float32)
        fitted, _, states, _ = gaussfit.fit_spots(
            model,
            spots,
            _seed(model, spots),
            mle=mle,
            max_iterations=200,
            tolerance=1e-8,
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        np.testing.assert_allclose(fitted[0, 1:3], theta[1:3], atol=1e-2)
        np.testing.assert_allclose(fitted[0, 0], theta[0], rtol=5e-3)
        np.testing.assert_allclose(
            _canonical_widths(model, fitted[0]),
            _canonical_widths(model, theta),
            rtol=1e-2,
        )

    @pytest.mark.parametrize("model", MODELS)
    def test_fitted_model_reproduces_the_data(self, model):
        """Whatever route it took, the fit must explain the spot."""
        theta = _truth(model)
        spots = _reference(model, theta)[None].astype(np.float32)
        fitted, _, _, _ = gaussfit.fit_spots(
            model,
            spots,
            _seed(model, spots),
            max_iterations=200,
            tolerance=1e-8,
        )
        np.testing.assert_allclose(
            _reference(model, fitted[0]), spots[0], rtol=1e-3, atol=1e-2
        )


class TestGausslqParity:
    """Same model, same estimator, different optimizer.

    ``gausslq`` minimizes ``sum((data - (photons * PDF_x * PDF_y + bg))^2)``;
    the elliptical LM model is the same surface parameterized by peak height
    instead of integrated photons, so the least-squares optimum is identical.
    The parameterization differs only in how the optimizer walks there."""

    @staticmethod
    def _as_gausslq_layout(theta):
        """LM ``[peak, x, y, sx, sy, bg]`` -> gausslq's
        ``[x, y, photons, bg, sx, sy]``, x/y relative to the box centre."""
        peak, cx, cy, sx, sy, bg = theta.T
        photons = peak * 2.0 * np.pi * np.abs(sx * sy)
        return np.stack(
            [cx - CENTRE, cy - CENTRE, photons, bg, np.abs(sx), np.abs(sy)],
            axis=1,
        )

    @staticmethod
    def _chi_square(theta, spots):
        """Least-squares residual of the LM parameters, in float64."""
        yy, xx = np.mgrid[0:BOX, 0:BOX].astype(np.float64)
        out = np.empty(len(theta))
        for k, (peak, cx, cy, sx, sy, bg) in enumerate(theta):
            mu = (
                peak
                * np.exp(
                    -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
                )
                + bg
            )
            out[k] = ((spots[k].astype(np.float64) - mu) ** 2).sum()
        return out

    def test_lands_on_the_same_optimum(self, lq_and_lm):
        lq, lm = lq_and_lm
        mapped = self._as_gausslq_layout(lm)
        # Positions and widths in pixels, background in photons.
        assert np.nanmax(np.abs(lq[:, 0] - mapped[:, 0])) < 1e-3
        assert np.nanmax(np.abs(lq[:, 1] - mapped[:, 1])) < 1e-3
        assert np.nanmax(np.abs(np.abs(lq[:, 4]) - mapped[:, 4])) < 1e-3
        assert np.nanmax(np.abs(np.abs(lq[:, 5]) - mapped[:, 5])) < 1e-3
        rel_photons = np.abs(lq[:, 2] - mapped[:, 2]) / np.abs(lq[:, 2])
        assert np.nanmax(rel_photons) < 1e-2

    def test_lm_never_finds_the_worse_optimum(self, lq_and_lm, noisy_batch):
        """The load-bearing assertion. Two different algorithms on one
        objective will not stop at the same floating-point point, so compare
        what actually matters: neither may be the worse fit, and the LM
        backend must not be. ``gausslq`` evaluates its model in float32
        (``gausslq.fit_spot``), so it is the one that cannot converge tighter.
        """
        lq, lm = lq_and_lm
        chi_lm = self._chi_square(lm, noisy_batch)
        # gausslq's layout, expressed as LM parameters, so one chi-square
        # function scores both.
        peak = lq[:, 2] / (2.0 * np.pi * np.abs(lq[:, 4] * lq[:, 5]))
        as_lm = np.stack(
            [
                peak,
                lq[:, 0] + CENTRE,
                lq[:, 1] + CENTRE,
                np.abs(lq[:, 4]),
                np.abs(lq[:, 5]),
                lq[:, 3],
            ],
            axis=1,
        )
        chi_lq = self._chi_square(as_lm, noisy_batch)
        assert np.all(chi_lm <= chi_lq * (1 + 1e-6))

    def test_photon_convention_is_not_accidentally_equal(self, lq_and_lm):
        """Guard against the mapping above being a no-op: the LM amplitude is
        a peak height and must *not* equal gausslq's photon count."""
        lq, lm = lq_and_lm
        assert np.nanmedian(np.abs(lq[:, 2] - lm[:, 0])) > 1.0


@pytest.mark.skipif(not lmfit_cuda.CUDA_AVAILABLE, reason="no CUDA device")
class TestCpuGpuEquivalence:
    """The CPU and the GPU run the same algorithm, so they must agree."""

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_fixed_iterations_agree_exactly(self, model, mle, noisy_batch):
        """Fixed budget, double precision, no convergence branch: this pins
        the algebra and cannot be flaky."""
        seed = _seed(model, noisy_batch)
        cpu = gaussfit.fit_spots(
            model, noisy_batch, seed, mle=mle, tolerance=0.0, max_iterations=3
        )
        gpu = gaussfit_cuda.fit_spots(
            model,
            noisy_batch,
            seed,
            mle=mle,
            tolerance=0.0,
            max_iterations=3,
            single_precision=False,
        )
        np.testing.assert_allclose(cpu[0], gpu[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(cpu[1], gpu[1], rtol=1e-9, atol=1e-9)
        # Same algorithm under a fixed budget: even the bookkeeping matches.
        np.testing.assert_array_equal(cpu[2], gpu[2])
        np.testing.assert_array_equal(cpu[3], gpu[3])

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_converged_positions_agree(self, model, mle, noisy_batch):
        """With the default single-precision GPU evaluation, positions still
        agree far below the shot-noise floor."""
        seed = _seed(model, noisy_batch)
        cpu = gaussfit.fit_spots(model, noisy_batch, seed, mle=mle)[0]
        gpu = gaussfit_cuda.fit_spots(model, noisy_batch, seed, mle=mle)[0]
        finite = np.isfinite(cpu).all(axis=1) & np.isfinite(gpu).all(axis=1)
        assert finite.sum() > 0.9 * len(noisy_batch)
        assert np.abs(cpu[finite, 1:3] - gpu[finite, 1:3]).max() < 1e-4


class TestThreading:
    def test_threaded_matches_serial_bitwise(self, noisy_batch):
        """One thread per spot with no shared state, so the answer cannot
        depend on the scheduling."""
        seed = localize._initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        serial = gaussfit.fit_spots(
            gaussfit.ELLIPTIC, noisy_batch, seed, mle=True
        )
        fit = gaussfit.fit_spots_async(
            gaussfit.ELLIPTIC, noisy_batch, seed, mle=True
        )
        while not fit.finished():
            pass
        fit.raise_errors()
        for a, b in zip(serial, fit.results()):
            np.testing.assert_array_equal(a, b)

    def test_async_can_be_stopped(self, noisy_batch):
        seed = localize._initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        fit = gaussfit.fit_spots_async(
            gaussfit.ELLIPTIC, noisy_batch, seed, n_threads=1
        )
        fit.stop()
        while not fit.finished():
            pass
        fit.raise_errors()
        assert fit.current[0] <= len(noisy_batch)


class TestSchedule:
    def test_defaults_match_the_gpu_backend(self):
        """One definition of the schedule, imported by the GPU module."""
        assert gaussfit_cuda.TOLERANCE is gaussfit.TOLERANCE
        assert gaussfit_cuda.MAX_ITERATIONS is gaussfit.MAX_ITERATIONS

    def test_model_constants_are_shared(self):
        """The GPU module must not define its own model numbering."""
        assert gaussfit_cuda.SPHERICAL is gaussfit.SPHERICAL
        assert gaussfit_cuda.ELLIPTIC is gaussfit.ELLIPTIC
        assert gaussfit_cuda.ROTATED is gaussfit.ROTATED
        assert gaussfit_cuda.n_parameters is gaussfit.n_parameters

    def test_iteration_cap_bites(self, noisy_batch):
        seed = localize._initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        capped = gaussfit.fit_spots(
            gaussfit.ELLIPTIC, noisy_batch, seed, max_iterations=2
        )[3]
        free = gaussfit.fit_spots(
            gaussfit.ELLIPTIC, noisy_batch, seed, max_iterations=40
        )[3]
        assert capped.max() <= 2
        assert free.max() > 2

    def test_looser_tolerance_stops_earlier(self, noisy_batch):
        seed = localize._initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        loose = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            noisy_batch,
            seed,
            tolerance=1e-1,
            max_iterations=60,
        )[3]
        tight = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            noisy_batch,
            seed,
            tolerance=1e-8,
            max_iterations=60,
        )[3]
        assert loose.mean() < tight.mean()


class TestInputValidation:
    def test_rejects_non_square_spots(self):
        with pytest.raises(ValueError, match="n_spots, box, box"):
            gaussfit.fit_spots(
                gaussfit.ELLIPTIC,
                np.zeros((2, 5, 7), np.float32),
                np.zeros((2, 6)),
            )

    def test_rejects_mismatched_seed_width(self):
        with pytest.raises(ValueError, match="initial_parameters"):
            gaussfit.fit_spots(
                gaussfit.ELLIPTIC,
                np.zeros((2, BOX, BOX), np.float32),
                np.zeros((2, 5)),
            )

    def test_rejects_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown Gaussian model"):
            gaussfit.fit_spots(
                99, np.zeros((1, BOX, BOX), np.float32), np.zeros((1, 6))
            )

    def test_empty_batch(self):
        thetas, chi, states, iters = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            np.zeros((0, BOX, BOX), np.float32),
            np.zeros((0, 6)),
        )
        assert thetas.shape == (0, 6)
        assert len(chi) == len(states) == len(iters) == 0


class TestNonPositiveModelIsAbandoned:
    """A Gaussian cannot ring negative: the model only goes non-positive when
    the *background* does. Flooring such a pixel (as the spline models do)
    would zero the gradient that pushes the background back up, so the fit
    stalls and the relative convergence test accepts a badly wrong answer.
    This is why ``gaussfit`` uses the strict estimator."""

    def test_negative_background_seed_is_rejected_under_mle(self):
        theta = _truth(gaussfit.ELLIPTIC)
        spots = _reference(gaussfit.ELLIPTIC, theta)[None].astype(np.float32)
        seed = _seed(gaussfit.ELLIPTIC, spots)
        seed[0, 5] = -50.0  # background well below zero
        _, _, states, _ = gaussfit.fit_spots(
            gaussfit.ELLIPTIC, spots, seed, mle=True, max_iterations=50
        )
        assert states[0] == splinefit.FIT_STATE_NEG_CURVATURE_MLE

    def test_every_converged_mle_fit_beats_ground_truth(self, noisy_batch):
        """An MLE that converges must explain the data at least as well as
        the truth does; a floored fit would not."""
        seed = localize._initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        fitted, chi, states, _ = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            noisy_batch,
            seed,
            mle=True,
            max_iterations=200,
            tolerance=1e-8,
        )
        converged = states == splinefit.FIT_STATE_CONVERGED
        assert converged.sum() > 0.8 * len(noisy_batch)
        assert np.all(np.isfinite(chi[converged]))
        assert np.all(fitted[converged, 5] > 0)  # background stayed positive
