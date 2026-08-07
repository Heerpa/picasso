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


def _poisson_spots(box, n=60, seed=3):
    """``noisy_batch``, but at an arbitrary box size.

    The widths stay in the 1.0-1.6 range whatever the box is - a real PSF is
    set by the optics, not by how much padding the box has around it. That
    divergence between the true width and a box-proportional seed is the whole
    point of ``TestWideSigmaSeedDoesNotAbortTheFit``.
    """
    rng = np.random.default_rng(seed)
    centre = (box - 1) / 2.0
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float64)
    spots = np.empty((n, box, box), np.float32)
    for k in range(n):
        cx = centre + rng.uniform(-0.6, 0.6)
        cy = centre + rng.uniform(-0.6, 0.6)
        sx, sy = rng.uniform(1.0, 1.6), rng.uniform(1.0, 1.6)
        amp, bg = rng.uniform(300, 1500), rng.uniform(5, 30)
        mu = (
            amp
            * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
            + bg
        )
        spots[k] = rng.poisson(mu)
    return spots


class TestWideSigmaSeedDoesNotAbortTheFit:
    """A too-wide width seed must not end the fit with ``NEG_CURVATURE_MLE``.

    The regression: the driver used to treat a *trial* step whose model went
    non-positive as terminal, as Gpufit's ``LMFitCPP::calc_chi_square`` does.
    Seeded several times wider than the true PSF, the first (undamped) step
    overshoots and drives the background below zero - so the fit aborted on
    iteration 1, rolled the parameters back and wrote *the seed* out as the
    result: sigma pinned to the seed width, x and y at the exact box centre.

    Every other test in this module runs at ``BOX = 7``, where the seed lands
    within 0.4 px of the truth and the first step never overshoots, so none of
    them could see it. These parametrize the box - the axis the bug lives on -
    and drive the width seed directly.
    """

    BOXES = [7, 13, 23, 31]

    @pytest.mark.parametrize("width_seed", [2.0, 3.0, 4.0, 5.0])
    def test_a_wide_seed_is_damped_back_not_abandoned(self, width_seed):
        """The bug in isolation: one knob, the seeded sigma.

        The spots and every other seed parameter are held fixed, so a failure
        here can only be the width. The true widths are ~1.0-1.6 px, so even
        the mildest case here starts well over the truth - enough for the
        first, undamped step to overshoot the background below zero.
        """
        spots = _poisson_spots(31, n=40)
        seed = localize._initial_parameters_gauss(spots, 31).astype(np.float64)
        seed[:, 3] = seed[:, 4] = width_seed
        fitted, _, states, iterations = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            spots,
            seed,
            mle=True,
            max_iterations=200,
            tolerance=1e-8,
        )
        aborted = states == splinefit.FIT_STATE_NEG_CURVATURE_MLE
        assert not aborted.any(), (
            f"{aborted.sum()}/{len(spots)} fits abandoned at a {width_seed} px "
            "width seed"
        )
        # Recovered, not merely alive: the fit walked back to the true width.
        assert np.median(fitted[:, 3]) < 2.0
        assert (iterations > 1).all()

    def test_an_extreme_seed_costs_iterations_but_still_arrives(self):
        """Where the recovery stops being free.

        At 8 px in a 31 px box the model is nearly flat, so amplitude and
        background are strongly correlated and the damped walk back is slow -
        a median of a few hundred iterations rather than the usual ten. It
        still *arrives*, which is the point: the old driver called this
        situation fatal on iteration 1 and returned the seed. This is also why
        the width seed itself is worth getting right
        (``localize._initial_widths_gauss``) - correctness here is bought with
        an iteration budget no production run would grant.
        """
        spots = _poisson_spots(31, n=40)
        seed = localize._initial_parameters_gauss(spots, 31).astype(np.float64)
        seed[:, 3] = seed[:, 4] = 8.0
        fitted, _, states, _ = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            spots,
            seed,
            mle=True,
            max_iterations=3000,
            tolerance=1e-8,
        )
        converged = states == splinefit.FIT_STATE_CONVERGED
        assert converged.sum() >= 0.9 * len(spots)
        assert np.median(fitted[converged, 3]) < 2.0

    @pytest.mark.parametrize("box", BOXES)
    @pytest.mark.parametrize("model", MODELS)
    def test_the_production_seed_survives_every_box_size(self, box, model):
        """``size / 5`` is the seed the GUI actually uses, and on a wide box
        it is exactly the too-wide seed above."""
        spots = _poisson_spots(box)
        kwargs = {
            gaussfit.SPHERICAL: dict(spherical=True),
            gaussfit.ELLIPTIC: {},
            gaussfit.ROTATED: dict(rotated=True),
        }[model]
        seed = localize._initial_parameters_gauss(spots, box, **kwargs).astype(
            np.float64
        )
        _, _, states, _ = gaussfit.fit_spots(
            model, spots, seed, mle=True, max_iterations=200, tolerance=1e-8
        )
        aborted = (states == splinefit.FIT_STATE_NEG_CURVATURE_MLE).sum()
        assert aborted == 0, f"{aborted}/{len(spots)} fits abandoned"

    @pytest.mark.parametrize("box", BOXES)
    def test_the_seed_is_never_echoed_as_the_result(self, box):
        """The downstream signature, asserted directly.

        A fit that aborts on its first step returns its own starting point.
        That is worse than inaccurate, it is silent: the localizations look
        plausible until you notice every width is identical and every
        coordinate is a whole pixel.
        """
        spots = _poisson_spots(box)
        seed = localize._initial_parameters_gauss(spots, box).astype(
            np.float64
        )
        fitted, _, _, iterations = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            spots,
            seed,
            mle=True,
            max_iterations=200,
            tolerance=1e-8,
        )
        untouched = np.all(np.isclose(fitted, seed), axis=1)
        assert not untouched.any()
        assert (iterations > 1).all()
        assert not np.allclose(fitted[:, 1], (box - 1) / 2.0)
        assert np.abs(fitted[:, 1] - np.round(fitted[:, 1])).max() > 0.05

    def test_the_retry_still_costs_an_iteration(self):
        """The retry must stay bounded by ``max_iterations``.

        Turning the abort into a retry would spin forever on a fit that can
        never take a valid step, if a rejected attempt were free.
        """
        spots = _poisson_spots(23, n=20)
        seed = localize._initial_parameters_gauss(spots, 23).astype(np.float64)
        seed[:, 3] = seed[:, 4] = 8.0
        for budget in (1, 2, 5):
            _, _, _, iterations = gaussfit.fit_spots(
                gaussfit.ELLIPTIC,
                spots,
                seed,
                mle=True,
                max_iterations=budget,
                tolerance=1e-12,
            )
            assert iterations.max() <= budget


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
