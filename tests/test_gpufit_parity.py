"""
test_gpufit_parity
~~~~~~~~~~~~~~~~~~

Acceptance gate for the numba CUDA spline backend against the vendored
``Gpufit.dll`` it replaces.

This is **scaffolding**: it exists to prove the port on the machine that still
has both implementations, and it is deleted together with
``picasso/ext/pygpufit`` once the numba backend is the only one. It is skipped
wherever either backend is missing, which is most machines.

Two differences from Gpufit are *intended* and are asserted as such rather than
tolerated, because otherwise they would be chased as bugs later:

``splinefit.MU_FLOOR``
    A cubic spline rings slightly negative in the tails, so a bright, low
    background spot drives the model below zero in the corners of the box.
    Gpufit's maximum-likelihood estimator aborts such a fit with
    ``NEG_CURVATURE_MLE``; Picasso's kernels charge the pixel a bounded
    likelihood penalty and exclude it from the gradient and Hessian instead.
    The numba backend therefore converges on spots where Gpufit bails out - see
    ``splinefit.py`` for the full argument.

Precision
    ``Gpufit.dll`` is built ``REAL=float``, so its whole Levenberg-Marquardt
    loop - including the Gauss-Jordan solve - is single precision. The numba
    kernels evaluate the spline in single precision but solve in double, so
    they are strictly the more accurate of the two and exact agreement is not
    expected.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest

from picasso import gaussfit_cuda, localize, splinefit

from tests.test_splinefit import (
    BOX,
    DX,
    DY,
    NZ,
    _astigmatic_calibration,
    _flat_calibration,
    _spots_from_terms,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (localize.GPUFIT_INSTALLED and localize.GPU_FITTING_AVAILABLE),
        reason="needs both Gpufit and a CUDA device for the numba kernels",
    ),
]


def _noisy_batch(calibration, terms, n=300, seed=0, n_channels=1):
    rng = np.random.default_rng(seed)
    spots = np.zeros((n, BOX, BOX), dtype=np.float32)
    for k in range(n):
        clean = _spots_from_terms(
            terms,
            BOX,
            rng.uniform(600, 1600),
            rng.uniform(5, 25),
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.8, 0.8),
            rng.uniform(-(NZ - 2), -1.0),
        )[0, 0]
        spots[k] = rng.poisson(np.maximum(clean, 0))
    return spots


def _gpufit(spots, calibration, mle, n_z_starts=1):
    """Gpufit's spline fit, normalized to the numba backend's 4-tuple."""
    if n_z_starts == 1:
        parameters, states, chi_squares, iterations, _ = (
            localize._run_gpufit_spline(spots, calibration, mle=mle)
        )
        return parameters, chi_squares, states, iterations
    theta, chi_squares, converged, iterations = (
        localize._fit_spline_z_multistart(
            spots, calibration, mle=mle, n_z_starts=n_z_starts
        )
    )
    return theta, chi_squares, converged, iterations


def _numba(spots, calibration, mle, n_z_starts=1):
    if n_z_starts == 1:
        return localize._run_splinefit(
            spots, calibration, mle=mle, n_z_starts=1, use_gpu=True
        )
    return localize._fit_splinefit_multistart(
        spots, calibration, mle=mle, n_z_starts=n_z_starts, use_gpu=True
    )


class TestSingleStart:
    """One seed, so the two differ only in arithmetic, not in search."""

    @pytest.mark.parametrize("mle", [False, True])
    def test_noiseless_3d_agrees(self, mle):
        calibration, terms = _astigmatic_calibration()
        spots = _spots_from_terms(terms, BOX, 900.0, 12.0, DX, DY, -6.0)[:, 0]
        old = _gpufit(spots, calibration, mle)
        new = _numba(spots, calibration, mle)
        # Position to a thousandth of a pixel, photons to 0.1 %.
        np.testing.assert_allclose(new[0][:, 1:4], old[0][:, 1:4], atol=1e-3)
        np.testing.assert_allclose(new[0][:, 0], old[0][:, 0], rtol=1e-3)

    @pytest.mark.parametrize("mle", [False, True])
    def test_noiseless_2d_agrees(self, mle):
        calibration, terms = _flat_calibration()
        spots = _spots_from_terms(terms, BOX, 700.0, 8.0, DX, DY, 0.0)[:, 0]
        old = _gpufit(spots, calibration, mle)
        new = _numba(spots, calibration, mle)
        np.testing.assert_allclose(new[0][:, 1:3], old[0][:, 1:3], atol=1e-3)
        np.testing.assert_allclose(new[0][:, 0], old[0][:, 0], rtol=1e-3)

    @pytest.mark.parametrize("mle", [False, True])
    def test_noisy_batch_agrees_wherever_gpufit_converged(self, mle):
        """Per-spot agreement on a realistic Poisson batch.

        Restricted to the spots Gpufit actually fitted. Where it aborts with
        ``NEG_CURVATURE_MLE`` there is nothing to compare against - it returns
        the seed parameters and a NaN chi-square - and that case is asserted
        separately in :class:`TestIntendedDivergences`. Including those spots
        here would compare a converged fit against an untouched initial guess
        and call the difference a disagreement.
        """
        calibration, terms = _astigmatic_calibration()
        spots = _noisy_batch(calibration, terms)
        old = _gpufit(spots, calibration, mle)
        new = _numba(spots, calibration, mle)
        converged = (old[2] == splinefit.FIT_STATE_CONVERGED) & np.isfinite(
            old[0]
        ).all(axis=1)
        assert converged.mean() > 0.5
        assert np.isfinite(new[0][converged]).all()
        # Both backends do the same arithmetic on the same data, so the only
        # difference here is Gpufit's single-precision solve. A thousandth of a
        # pixel is already far below any achievable localization precision.
        dxy = np.abs(new[0][converged, 1:3] - old[0][converged, 1:3]).max()
        assert dxy < 1e-3, dxy
        dn = (
            np.abs(new[0][converged, 0] - old[0][converged, 0])
            / old[0][converged, 0]
        )
        assert dn.max() < 1e-3, dn.max()

    def test_least_squares_reaches_the_same_fit_states(self):
        """Without the model floor in play, the two agree spot for spot."""
        calibration, terms = _astigmatic_calibration()
        spots = _noisy_batch(calibration, terms)
        old = _gpufit(spots, calibration, False)
        new = _numba(spots, calibration, False)
        np.testing.assert_array_equal(new[2], old[2])


class TestMultiStart:
    @pytest.mark.parametrize("mle", [False, True])
    def test_chi_square_is_not_worse(self, mle):
        """The numba backend must not find a worse optimum than Gpufit.

        The comparison that matters, since the two run the multi-start
        differently: Gpufit re-runs whole passes and ranks on the host, the
        numba kernel runs every seed per spot. Only the outcome has to match.
        """
        calibration, terms = _astigmatic_calibration()
        spots = _noisy_batch(calibration, terms)
        old = _gpufit(spots, calibration, mle, n_z_starts=5)
        new = _numba(spots, calibration, mle, n_z_starts=5)
        good = np.isfinite(old[1]) & np.isfinite(new[1])
        assert good.mean() > 0.95
        relative = (new[1][good] - old[1][good]) / np.abs(old[1][good])
        # A per-spot allowance, plus a much tighter one on the median so a
        # systematic regression cannot hide behind the outliers.
        assert np.percentile(relative, 99) < 1e-2
        assert np.median(relative) < 1e-4

    def test_axial_positions_agree(self):
        calibration, terms = _astigmatic_calibration()
        spots = _noisy_batch(calibration, terms)
        old = _gpufit(spots, calibration, False, n_z_starts=5)
        new = _numba(spots, calibration, False, n_z_starts=5)
        good = np.isfinite(old[1]) & np.isfinite(new[1])
        dz = np.abs(new[0][good, 3] - old[0][good, 3])
        assert np.median(dz) < 0.05
        assert (dz < 0.5).mean() > 0.95


GAUSS_MODELS = [
    pytest.param(
        gaussfit_cuda.SPHERICAL,
        "GAUSS_2D",
        dict(spherical=True),
        id="spherical",
    ),
    pytest.param(
        gaussfit_cuda.ELLIPTIC, "GAUSS_2D_ELLIPTIC", {}, id="elliptic"
    ),
    pytest.param(
        gaussfit_cuda.ROTATED,
        "GAUSS_2D_ROTATED",
        dict(rotated=True),
        id="rotated",
    ),
]


def _gaussian_batch(box=13, n=300, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:box, 0:box]
    spots = np.zeros((n, box, box), dtype=np.float32)
    for k in range(n):
        cx = box / 2 - 0.5 + rng.uniform(-1, 1)
        cy = box / 2 - 0.5 + rng.uniform(-1, 1)
        sx, sy = rng.uniform(1.0, 2.0, size=2)
        angle = rng.uniform(-0.6, 0.6)
        ca, sa = np.cos(angle), np.sin(angle)
        a = (xx - cx) * ca - (yy - cy) * sa
        b = (xx - cx) * sa + (yy - cy) * ca
        clean = rng.uniform(50, 300) * np.exp(
            -0.5 * ((a / sx) ** 2 + (b / sy) ** 2)
        ) + rng.uniform(2, 15)
        spots[k] = rng.poisson(clean)
    return spots


class TestGaussianModels:
    """The three Gaussian models against the ``.cuh`` kernels they replace."""

    @pytest.mark.parametrize("model,model_id,seed_kwargs", GAUSS_MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_agrees_wherever_gpufit_converged(
        self, model, model_id, seed_kwargs, mle
    ):
        from picasso.ext.pygpufit import gpufit as gf

        box = 13
        spots = _gaussian_batch(box)
        initial = localize._initial_parameters_gpufit(
            spots, box, **seed_kwargs
        )
        data = np.maximum(spots, 0) if mle else spots
        old, states, _, old_iterations, _ = gf.fit(
            data.reshape(len(data), box * box),
            None,
            getattr(gf.ModelID, model_id),
            initial,
            tolerance=gaussfit_cuda.TOLERANCE,
            max_number_iterations=gaussfit_cuda.MAX_ITERATIONS,
            estimator_id=gf.EstimatorID.MLE if mle else gf.EstimatorID.LSE,
        )
        new, _, new_states, new_iterations = gaussfit_cuda.fit_spots(
            model, data, initial.astype(np.float64), mle=mle
        )
        converged = states == splinefit.FIT_STATE_CONVERGED
        assert converged.mean() > 0.5
        # Essentially the same Levenberg-Marquardt trajectory, step for step.
        # Not *exactly*: the convergence test is a threshold on the chi-square
        # change, so a spot sitting on that threshold can take one step more or
        # fewer when the arithmetic differs in the last bits. Under least
        # squares even that does not happen - see
        # ``test_least_squares_reaches_the_same_fit_states``.
        same = new_iterations[converged] == old_iterations[converged]
        assert same.mean() > 0.98, (
            f"{(~same).sum()} of {same.size} spots took a different number of "
            "iterations"
        )
        # Position, in pixels. Compared on a quantile plus a sanity bound
        # rather than the raw maximum: an occasional dim spot is genuinely
        # ill-conditioned, and there the extra iteration allowed for above is
        # enough to move the optimum visibly. A systematic shift would still
        # show up immediately in the quantile.
        dxy = np.abs(new[converged, 1:3] - old[converged, 1:3]).max(axis=1)
        assert np.percentile(dxy, 99) < 1e-2, np.percentile(dxy, 99)
        assert dxy.max() < 0.5, dxy.max()
        # Widths by magnitude: their sign is not identifiable, and both
        # backends pick it independently.
        ds = np.abs(np.abs(new[converged, 3]) - np.abs(old[converged, 3]))
        assert np.percentile(ds, 99) < 1e-2, np.percentile(ds, 99)

    @pytest.mark.parametrize("model,model_id,seed_kwargs", GAUSS_MODELS)
    def test_least_squares_reaches_the_same_fit_states(
        self, model, model_id, seed_kwargs
    ):
        from picasso.ext.pygpufit import gpufit as gf

        box = 13
        spots = _gaussian_batch(box)
        initial = localize._initial_parameters_gpufit(
            spots, box, **seed_kwargs
        )
        _, states, _, _, _ = gf.fit(
            spots.reshape(len(spots), box * box),
            None,
            getattr(gf.ModelID, model_id),
            initial,
            tolerance=gaussfit_cuda.TOLERANCE,
            max_number_iterations=gaussfit_cuda.MAX_ITERATIONS,
            estimator_id=gf.EstimatorID.LSE,
        )
        _, _, new_states, _ = gaussfit_cuda.fit_spots(
            model, spots, initial.astype(np.float64), mle=False
        )
        np.testing.assert_array_equal(new_states, states)


class TestIntendedDivergences:
    def test_mle_recovers_the_fits_gpufit_abandons(self):
        """``MU_FLOOR`` is why the numba MLE is usable at all on bright spots.

        A cubic spline rings negative in its tails, so a bright spot on low
        background drives the model below zero in the corners of the box.
        Gpufit's Poisson estimator treats that as fatal and aborts with
        ``NEG_CURVATURE_MLE``, returning the *seed* parameters and a NaN
        chi-square - the fit simply does not happen. On a realistic Poisson
        batch that is a substantial fraction of all spots, which is why
        ``fit_spline_multichannel_ratiometric`` warns that its MLE chi-square is
        unusable.

        This is the single most consequential behavioural difference of the
        port, so it is asserted rather than tolerated. Should Gpufit ever stop
        aborting here, the floor would no longer be doing anything and the
        reasoning in ``splinefit.py`` would need revisiting - this test failing
        is not a licence to delete it.
        """
        calibration, terms = _astigmatic_calibration()
        spots = _noisy_batch(calibration, terms)
        old = _gpufit(spots, calibration, True)
        new = _numba(spots, calibration, True)
        abandoned = old[2] == splinefit.FIT_STATE_NEG_CURVATURE_MLE
        assert abandoned.any(), (
            "Gpufit converged on every spot, so this batch no longer exercises "
            "the model floor"
        )
        # The numba kernels fit every abandoned spot, to a finite optimum, and
        # abandon none of their own.
        assert np.isfinite(new[0][abandoned]).all()
        assert np.isfinite(new[1][abandoned]).all()
        assert (new[2] != splinefit.FIT_STATE_NEG_CURVATURE_MLE).all()
        # The headline consequence: strictly more spots actually get fitted.
        old_converged = (old[2] == splinefit.FIT_STATE_CONVERGED).sum()
        new_converged = (new[2] == splinefit.FIT_STATE_CONVERGED).sum()
        assert new_converged > old_converged
