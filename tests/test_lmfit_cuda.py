"""
test_lmfit_cuda
~~~~~~~~~~~~~~~

Tests for ``picasso.lmfit_cuda``, the CUDA device machinery shared by the
numba fitting backends.

The PTX-compilation tests need no GPU: ``cuda.compile_ptx`` goes through
libNVVM, which ships with the ``numba-cuda`` wheel. They are the net that
catches ``np.isfinite`` / ``np.floor`` left in device code, which the CUDA
*simulator* cannot catch - it runs plain Python, where the NumPy forms work
fine.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import cuda, types

from picasso import lmfit_cuda, splinefit

requires_cuda = pytest.mark.skipif(
    not lmfit_cuda.CUDA_AVAILABLE, reason="no CUDA device"
)

_F64_2D = types.float64[:, ::1]
_F64_1D = types.float64[::1]
_I32_1D = types.int32[::1]


class TestPtxCompilation:
    """Every device function must compile for the real target.

    A device function that uses an unsupported NumPy scalar call fails here and
    nowhere else in a GPU-free test run.
    """

    def test_estimator_helpers_compile(self):
        cuda.compile_ptx(
            lmfit_cuda._floored_chi_square, (types.float64,), device=True
        )
        _, restype = cuda.compile_ptx(
            lmfit_cuda._estimator_terms,
            (types.boolean, types.float64, types.float64),
            device=True,
        )
        # (chi_square, weight, factor, ok)
        assert len(restype) == 4

    def test_solver_compiles(self):
        cuda.compile_ptx(
            lmfit_cuda._solve_gj_device,
            (_F64_2D, _F64_1D, types.int32, _I32_1D),
            device=True,
        )
        cuda.compile_ptx(
            lmfit_cuda._lm_solve_step_device,
            (
                _F64_2D,
                _F64_1D,
                _F64_1D,
                types.float64,
                _F64_2D,
                _F64_1D,
                types.int32,
                _I32_1D,
            ),
            device=True,
        )

    def test_numpy_scalar_calls_are_rejected_by_the_target(self):
        """Pins the reason the device code spells everything ``math.*``.

        If a future numba starts supporting these, the translation is still
        correct - but the *lint value* of the tests above would quietly
        disappear, and this test is what says so.
        """

        def uses_np_isfinite(x):
            return np.isfinite(x)

        with pytest.raises(Exception):
            cuda.compile_ptx(uses_np_isfinite, (types.float64,), device=True)


class TestChunking:
    def test_scales_with_per_row_work(self):
        """A launch of costlier fits gets fewer rows, which is what keeps a
        display-attached GPU under its watchdog timeout."""
        cheap = lmfit_cuda.chunk_rows(1000, 20)
        expensive = lmfit_cuda.chunk_rows(1000, 1500)
        assert expensive < cheap

    def test_memory_bound_for_wide_rows(self):
        assert lmfit_cuda.chunk_rows(1 << 20, 1) < lmfit_cuda.chunk_rows(
            1024, 1
        )

    def test_never_degenerate(self):
        assert lmfit_cuda.chunk_rows(1 << 30, 1 << 20) >= 1

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PICASSO_FIT_CUDA_MAX_ROWS", "5000")
        assert lmfit_cuda.max_rows_reference() == 5000

    def test_bad_env_override_warns_and_is_ignored(self, monkeypatch):
        monkeypatch.setenv("PICASSO_FIT_CUDA_MAX_ROWS", "not-a-number")
        with pytest.warns(RuntimeWarning):
            assert (
                lmfit_cuda.max_rows_reference()
                == lmfit_cuda._MAX_ROWS_REFERENCE
            )


class TestSharedConstants:
    """The two devices must agree on where a fit stops."""

    def test_states_and_damping_come_from_splinefit(self):
        assert lmfit_cuda.FIT_STATE_CONVERGED == splinefit.FIT_STATE_CONVERGED
        assert (
            lmfit_cuda.FIT_STATE_MAX_ITERATION
            == splinefit.FIT_STATE_MAX_ITERATION
        )
        assert (
            lmfit_cuda.FIT_STATE_SINGULAR_HESSIAN
            == splinefit.FIT_STATE_SINGULAR_HESSIAN
        )
        assert (
            lmfit_cuda.FIT_STATE_NEG_CURVATURE_MLE
            == splinefit.FIT_STATE_NEG_CURVATURE_MLE
        )
        assert lmfit_cuda.MU_FLOOR == splinefit.MU_FLOOR
        assert lmfit_cuda._LAMBDA_INITIAL == splinefit._LAMBDA_INITIAL
        assert lmfit_cuda._LAMBDA_DOWN == splinefit._LAMBDA_DOWN
        assert lmfit_cuda._LAMBDA_UP == splinefit._LAMBDA_UP

    def test_schedule_is_the_shared_one(self):
        for apply_seeds in (False, True):
            assert lmfit_cuda.convergence_schedule(
                apply_seeds
            ) == splinefit.convergence_schedule(apply_seeds)


@requires_cuda
class TestSolver:
    """``_solve_gj_device`` against NumPy, on the device."""

    @staticmethod
    def _solve(a, b):
        n = len(b)

        @cuda.jit
        def kernel(a, b, status):
            ipiv = cuda.local.array(16, np.int32)
            status[0] = lmfit_cuda._solve_gj_device(a, b, n, ipiv)

        d_a = cuda.to_device(np.ascontiguousarray(a, dtype=np.float64))
        d_b = cuda.to_device(np.ascontiguousarray(b, dtype=np.float64))
        d_status = cuda.to_device(np.zeros(1, dtype=np.bool_))
        kernel[1, 1](d_a, d_b, d_status)
        return d_b.copy_to_host(), bool(d_status.copy_to_host()[0])

    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        for n in (4, 5, 7, 15):
            m = rng.normal(size=(n, n))
            a = m @ m.T + n * np.eye(n)  # symmetric positive definite
            b = rng.normal(size=n)
            got, ok = self._solve(a.copy(), b.copy())
            assert ok
            np.testing.assert_allclose(got, np.linalg.solve(a, b), rtol=1e-9)

    def test_matches_the_cpu_solver(self):
        """The two ports of Gpufit's Gauss-Jordan must agree exactly."""
        rng = np.random.default_rng(1)
        n = 7
        m = rng.normal(size=(n, n))
        a = m @ m.T + n * np.eye(n)
        b = rng.normal(size=n)
        gpu, ok_gpu = self._solve(a.copy(), b.copy())
        cpu_a, cpu_b = a.copy(), b.copy()
        scratch = [np.empty(n, dtype=np.int32) for _ in range(3)]
        ok_cpu = splinefit._solve_gj(cpu_a, cpu_b, n, *scratch)
        assert ok_gpu == ok_cpu
        np.testing.assert_allclose(gpu, cpu_b, rtol=1e-12)

    def test_rejects_a_singular_matrix(self):
        a = np.zeros((4, 4))
        a[0, 0] = a[1, 1] = a[2, 2] = 1.0  # row/col 3 all zero
        _, ok = self._solve(a, np.ones(4))
        assert not ok

    def test_rejects_a_nan_pivot(self):
        """``abs(pivot) > 0`` must reject NaN, which ``pivot == 0`` would not.

        Every candidate has to be non-finite for a NaN to actually reach the
        pivot: see :meth:`test_a_single_nan_is_skipped_not_rejected`."""
        _, ok = self._solve(np.full((4, 4), np.nan), np.ones(4))
        assert not ok

    def test_a_single_nan_is_skipped_not_rejected(self):
        """A lone NaN never wins the pivot search, so the solve *succeeds*.

        ``abs(nan) >= big`` is False, so the full-pivoting search simply never
        selects that entry and eliminates around it. Surprising, but it is
        exactly what the CPU solver does, and the fit's own finiteness guards -
        not the solver - are what catch a diverged fit. Pinned here so the two
        ports cannot drift apart on it.
        """
        a = np.eye(4)
        a[2, 2] = np.nan
        _, ok_gpu = self._solve(a.copy(), np.ones(4))
        cpu_a, cpu_b = a.copy(), np.ones(4)
        scratch = [np.empty(4, dtype=np.int32) for _ in range(3)]
        ok_cpu = splinefit._solve_gj(cpu_a, cpu_b, 4, *scratch)
        assert ok_gpu == ok_cpu is True
