"""
test_gaussfit_cuda
~~~~~~~~~~~~~~~~~~

Tests for ``picasso.gaussfit_cuda``, the GPU Gaussian PSF fitter.

Unlike the spline backend, these models have no CPU twin to compare against -
``gausslq`` fits a normalized Gaussian with SciPy and ``gaussmle`` an
erf-integrated one with a bespoke Newton solver, so neither is the same
estimator. The ground truth here is therefore the closed form itself: a
noiseless spot built from the model must be recovered to near machine
precision, and the analytic derivatives must match finite differences. Parity
against the ``Gpufit.dll`` these replace lives in ``test_gpufit_parity.py``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest

from picasso import gaussfit_cuda, lmfit_cuda, localize, splinefit

pytestmark = pytest.mark.skipif(
    not lmfit_cuda.CUDA_AVAILABLE, reason="no CUDA device"
)

BOX = 7
MODELS = [
    pytest.param(gaussfit_cuda.SPHERICAL, id="spherical"),
    pytest.param(gaussfit_cuda.ELLIPTIC, id="elliptic"),
    pytest.param(gaussfit_cuda.ROTATED, id="rotated"),
]


def _reference(model, theta, box=BOX):
    """The closed-form model image, straight from the Gpufit sources."""
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float64)
    if model == gaussfit_cuda.SPHERICAL:
        amp, cx, cy, s, bg = theta
        ex = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / s**2)
        return amp * ex + bg
    if model == gaussfit_cuda.ELLIPTIC:
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
    base = [220.0, BOX / 2 - 0.5 + 0.31, BOX / 2 - 0.5 - 0.22]
    if model == gaussfit_cuda.SPHERICAL:
        return np.array(base + [1.6, 9.0])
    if model == gaussfit_cuda.ELLIPTIC:
        return np.array(base + [1.35, 1.85, 9.0])
    return np.array(base + [1.35, 1.85, 9.0, 0.4])


def _canonical_widths(model, theta):
    """Widths as an unordered pair of magnitudes.

    Both symmetries of these models live here - see
    :class:`TestWidthAndAngleDegeneracies`."""
    if model == gaussfit_cuda.SPHERICAL:
        return np.array([abs(theta[3])])
    return np.sort(np.abs(theta[3:5]))


def _seed(model, spots):
    kwargs = {
        gaussfit_cuda.SPHERICAL: dict(spherical=True),
        gaussfit_cuda.ELLIPTIC: {},
        gaussfit_cuda.ROTATED: dict(rotated=True),
    }[model]
    return localize._initial_parameters_gauss(spots, BOX, **kwargs).astype(
        np.float64
    )


class TestGroundTruthRecovery:
    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_recovers_a_noiseless_spot(self, model, mle):
        theta = _truth(model)
        spots = _reference(model, theta)[None].astype(np.float32)
        fitted, chi_squares, states, _ = gaussfit_cuda.fit_spots(
            model,
            spots,
            _seed(model, spots),
            mle=mle,
            max_iterations=200,
            tolerance=1e-8,
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        # Position to a hundredth of a pixel, amplitude to 0.5 %.
        np.testing.assert_allclose(fitted[0, 1:3], theta[1:3], atol=1e-2)
        np.testing.assert_allclose(fitted[0, 0], theta[0], rtol=5e-3)
        # Widths are compared as an unordered pair of magnitudes: see
        # ``TestWidthAndAngleDegeneracies`` for the two symmetries of these
        # models that make the raw values non-unique.
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
        fitted, _, _, _ = gaussfit_cuda.fit_spots(
            model,
            spots,
            _seed(model, spots),
            max_iterations=200,
            tolerance=1e-8,
        )
        np.testing.assert_allclose(
            _reference(model, fitted[0]), spots[0], rtol=1e-3, atol=1e-2
        )


class TestNoisyBatch:
    @staticmethod
    def _batch(model, n=200, seed=0):
        rng = np.random.default_rng(seed)
        theta = _truth(model)
        spots = np.zeros((n, BOX, BOX), dtype=np.float32)
        for k in range(n):
            perturbed = theta.copy()
            perturbed[1:3] += rng.uniform(-0.8, 0.8, size=2)
            perturbed[0] *= rng.uniform(0.6, 1.8)
            spots[k] = rng.poisson(np.maximum(_reference(model, perturbed), 0))
        return spots

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_batch_converges(self, model, mle):
        spots = self._batch(model)
        fitted, chi_squares, states, _ = gaussfit_cuda.fit_spots(
            model, spots, _seed(model, spots), mle=mle
        )
        converged = states == splinefit.FIT_STATE_CONVERGED
        assert converged.mean() > 0.95
        assert np.isfinite(fitted[converged]).all()
        assert np.isfinite(chi_squares[converged]).all()
        # Magnitude, not sign: the model depends on the width only through
        # ``s**2``, so the sign is not identifiable (see
        # ``TestWidthAndAngleDegeneracies``).
        assert (np.abs(fitted[converged, 3]) > 0).all()
        assert (np.abs(fitted[converged, 3]) < BOX).all()

    @pytest.mark.parametrize("model", MODELS)
    def test_two_runs_are_bitwise_identical(self, model):
        spots = self._batch(model, n=64)
        seed = _seed(model, spots)
        first = gaussfit_cuda.fit_spots(model, spots, seed, mle=True)
        second = gaussfit_cuda.fit_spots(model, spots, seed, mle=True)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_chunking_is_transparent(self, monkeypatch):
        model = gaussfit_cuda.ELLIPTIC
        spots = self._batch(model, n=64)
        seed = _seed(model, spots)
        one = gaussfit_cuda.fit_spots(model, spots, seed)
        monkeypatch.setattr(lmfit_cuda, "chunk_rows", lambda *a, **k: 7)
        many = gaussfit_cuda.fit_spots(model, spots, seed)
        for a, b in zip(one, many):
            np.testing.assert_array_equal(a, b)


class TestWidthAndAngleDegeneracies:
    """The two ways these models parameterize the same PSF.

    Neither is a defect and neither is introduced by the port - ``Gpufit.dll``
    returns exactly the same solutions. They are pinned here because they look
    like failures the first time a fitted parameter is compared to the truth.
    """

    def test_width_sign_is_not_identifiable(self):
        """The model depends on the width only through ``s**2``.

        So ``-s`` fits precisely as well as ``+s`` and the optimizer may return
        either. Gpufit does the same, on a similar fraction of spots. Anything
        downstream that consumes a width - the photon conversion
        ``photons = amplitude * 2 pi sx sy`` in ``localize`` above all - has to
        be robust to it.
        """
        theta = _truth(gaussfit_cuda.ELLIPTIC)
        flipped = theta.copy()
        flipped[3] = -flipped[3]
        np.testing.assert_allclose(
            _reference(gaussfit_cuda.ELLIPTIC, flipped),
            _reference(gaussfit_cuda.ELLIPTIC, theta),
            rtol=1e-12,
        )

    def test_rotated_angle_is_defined_modulo_a_quarter_turn(self):
        """Rotating by pi/2 and swapping the widths is the same PSF.

        The fit converges on whichever of the two representations it reaches
        first, so the angle must never be compared to the truth directly - only
        the resulting model image is unique.
        """
        theta = _truth(gaussfit_cuda.ROTATED)
        swapped = theta.copy()
        swapped[3], swapped[4] = theta[4], theta[3]
        swapped[6] = theta[6] + np.pi / 2
        np.testing.assert_allclose(
            _reference(gaussfit_cuda.ROTATED, swapped),
            _reference(gaussfit_cuda.ROTATED, theta),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_fit_recovers_the_psf_up_to_those_symmetries(self):
        theta = _truth(gaussfit_cuda.ROTATED)
        spots = _reference(gaussfit_cuda.ROTATED, theta)[None].astype(
            np.float32
        )
        fitted, _, _, _ = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ROTATED,
            spots,
            _seed(gaussfit_cuda.ROTATED, spots),
            max_iterations=200,
            tolerance=1e-8,
        )
        np.testing.assert_allclose(
            _canonical_widths(gaussfit_cuda.ROTATED, fitted[0]),
            _canonical_widths(gaussfit_cuda.ROTATED, theta),
            rtol=1e-2,
        )
        # The angle, brought back into [0, pi/2) where it is unique.
        quarter = np.pi / 2
        assert (
            abs(
                (fitted[0, 6] - theta[6] + quarter / 2) % quarter - quarter / 2
            )
            < 0.05
        )

    def test_seed_breaks_the_width_symmetry(self):
        """With ``sx == sy`` the angle derivative is identically zero.

        The first Hessian is then singular and the fit cannot start, which is
        why ``_initial_parameters_gauss`` perturbs the two widths apart. Pinned
        here because it looks like an arbitrary fudge at the call site."""
        spots = np.zeros((1, BOX, BOX), dtype=np.float32)
        seed = _seed(gaussfit_cuda.ROTATED, spots)
        assert seed[0, 3] != seed[0, 4]


class TestNonPositiveModelIsAbandoned:
    """A Gaussian whose model goes non-positive aborts; it is never floored.

    The cubic-spline models floor such a pixel, because a cubic genuinely rings
    negative in the tails of a peaked profile - that is the basis, not the
    parameters. A Gaussian cannot ring: ``amp * exp(...) + bg`` only drops below
    zero when ``bg`` does. Flooring it there would zero exactly the gradient
    that pushes the background back up, so the chi-square would stop moving and
    the *relative* convergence test would then accept a badly wrong fit as
    converged. Reported as ``NEG_CURVATURE_MLE`` instead, matching the
    behaviour Picasso shipped before the port.
    """

    def test_negative_background_is_reported_not_floored(self):
        theta = _truth(gaussfit_cuda.ELLIPTIC)
        spots = _reference(gaussfit_cuda.ELLIPTIC, theta)[None].astype(
            np.float32
        )
        # A seed whose background is well below zero: the model is negative
        # across the whole box, so the very first evaluation is infeasible.
        seed = np.array([[theta[0], theta[1], theta[2], 1.3, 1.3, -500.0]])
        _, _, states, _ = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC, spots, seed, mle=True
        )
        assert states[0] == splinefit.FIT_STATE_NEG_CURVATURE_MLE

    def test_least_squares_is_unaffected(self):
        """Least squares has no logarithm, so a negative model is legal."""
        theta = _truth(gaussfit_cuda.ELLIPTIC)
        spots = _reference(gaussfit_cuda.ELLIPTIC, theta)[None].astype(
            np.float32
        )
        seed = np.array([[theta[0], theta[1], theta[2], 1.3, 1.3, -500.0]])
        fitted, _, states, _ = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC, spots, seed, mle=False
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        np.testing.assert_allclose(fitted[0, 5], theta[5], atol=0.5)


class TestApi:
    def test_kernel_cache(self):
        first = gaussfit_cuda._get_kernel(gaussfit_cuda.ELLIPTIC, True)
        assert first is gaussfit_cuda._get_kernel(gaussfit_cuda.ELLIPTIC, True)
        assert first is not gaussfit_cuda._get_kernel(
            gaussfit_cuda.ELLIPTIC, False
        )
        assert first is not gaussfit_cuda._get_kernel(
            gaussfit_cuda.ROTATED, True
        )

    def test_parameter_counts(self):
        assert gaussfit_cuda.n_parameters(gaussfit_cuda.SPHERICAL) == 5
        assert gaussfit_cuda.n_parameters(gaussfit_cuda.ELLIPTIC) == 6
        assert gaussfit_cuda.n_parameters(gaussfit_cuda.ROTATED) == 7

    def test_empty_input(self):
        spots = np.zeros((0, BOX, BOX), dtype=np.float32)
        initial = np.zeros((0, 6))
        thetas, chi_squares, states, iterations = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC, spots, initial
        )
        assert len(thetas) == len(chi_squares) == 0

    def test_rejects_a_non_square_box(self):
        with pytest.raises(ValueError, match="box"):
            gaussfit_cuda.fit_spots(
                gaussfit_cuda.ELLIPTIC,
                np.zeros((2, 5, 7), dtype=np.float32),
                np.zeros((2, 6)),
            )

    def test_rejects_mismatched_initial_parameters(self):
        with pytest.raises(ValueError, match="initial_parameters"):
            gaussfit_cuda.fit_spots(
                gaussfit_cuda.ELLIPTIC,
                np.zeros((2, BOX, BOX), dtype=np.float32),
                np.zeros((2, 7)),
            )

    def test_rejects_an_unknown_model(self):
        with pytest.raises((ValueError, KeyError)):
            gaussfit_cuda.fit_spots(
                99, np.zeros((1, BOX, BOX), dtype=np.float32), np.zeros((1, 6))
            )

    def test_progress_callback(self):
        model = gaussfit_cuda.ELLIPTIC
        spots = TestNoisyBatch._batch(model, n=32)
        seen = []
        gaussfit_cuda.fit_spots(
            model, spots, _seed(model, spots), progress_callback=seen.append
        )
        assert seen == sorted(seen) and seen[-1] == len(spots)

    def test_zero_width_is_reported_not_crashed(self):
        """A zero width divides by zero in every derivative."""
        spots = np.ones((1, BOX, BOX), dtype=np.float32)
        initial = np.array([[100.0, 6.0, 6.0, 0.0, 0.0, 1.0]])
        thetas, chi_squares, states, _ = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC, spots, initial
        )
        assert states[0] != splinefit.FIT_STATE_CONVERGED
        assert np.isnan(thetas[0]).all()
