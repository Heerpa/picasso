"""
picasso.fitting.precision
~~~~~~~~~~~~~~~~~~~~~~~~~

Closed-form localization and width precisions for the 2D Gaussian fits.

These are *analytic* uncertainty estimates, evaluated from a fit's reported
parameters rather than from the data: given photons, widths and background,
each returns the standard error the corresponding estimator attains. They are
the cheap counterpart to the numerically inverted Fisher matrices in
``picasso.localize`` (:func:`picasso.localize._gauss_crlb`,
:func:`picasso.localize._spline_crlb`), which need the model and its
derivatives at every pixel.

Which one applies depends on the estimator that produced the fit:

============================  ====================================
function                      estimator
============================  ====================================
:func:`localization_precision`  least-squares Gaussian (position)
:func:`sigma_uncertainty_lsq`   least-squares Gaussian (width)
:func:`sigma_uncertainty_mle`   Poisson maximum likelihood (width)
============================  ====================================

They lived in ``picasso.gausslq`` and ``picasso.gaussmle`` until 0.11; those
modules are deprecated and go in Picasso 1.0.

References
----------
Mortensen, K. I., Churchman, L. S., Spudich, J. A. & Flyvbjerg, H.
"Optimized localization analysis for single-molecule tracking and
super-resolution microscopy." Nature Methods 7, 377-381 (2010).
https://doi.org/10.1038/nmeth.1447

Rieger, B. & Stallinga, S. "The lateral and axial localization uncertainty in
super-resolution light microscopy." ChemPhysChem 15, 664-670 (2014).
https://doi.org/10.1002/cphc.201300711

Kowalewski, R., Reinhardt, S. C. M. et al. Nature Communications (2026).
https://doi.org/10.1038/s41467-026-70198-5

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np

from picasso import lib


def localization_precision(
    photons: lib.FloatArray1D,
    s: lib.FloatArray1D,
    s_orth: lib.FloatArray1D,
    bg: lib.FloatArray1D,
    em: bool,
    readout_variance: lib.FloatArray1D | float = 0.0,
) -> lib.FloatArray1D:
    """Theoretical localization precision of a 2D unweighted Gaussian fit.

    Mortensen et al., Nature Methods, 2010.

    Edit v0.9.0: corrected formula for diagonal covariance Gaussian
    (i.e., sx != sy). The background term includes the orthogonal sigma.

    Parameters
    ----------
    photons : lib.FloatArray1D
        Number of photons collected for the localization.
    s : lib.FloatArray1D
        Size of the single-emitter image for each localization.
    s_orth : lib.FloatArray1D
        Size of the single-emitter image in the orthogonal direction
        for each localization.
    bg : lib.FloatArray1D
        Background signal for each localization (per pixel).
    em : bool
        Whether EMCCD was used for the localization. Its stochastic
        multiplication doubles the variance.
    readout_variance : lib.FloatArray1D or float, optional
        Mean sCMOS readout variance over the fitting box, in photoelectrons
        squared. Adds to the background term. Default 0.

    Returns
    -------
    lib.FloatArray1D
        Cramer-Rao lower bound for localization precision for each
        localization.

    Notes
    -----
    With a per-pixel sCMOS camera calibration, ``readout_variance`` adds the
    pixel readout noise to the background term, which is where a spatially
    uniform noise floor enters this closed form. That is an approximation:
    Mortensen's derivation assumes the background variance is *constant* over
    the fitting box, so replacing ``bg`` by ``bg + mean(var)`` is exact only
    when the variance map is flat there. With one hot pixel next to the
    emitter it under- or over-states the precision depending on where that
    pixel falls under the PSF. For sCMOS data with a calibration loaded,
    prefer a maximum-likelihood method, whose Cramer-Rao bound is evaluated
    pixel by pixel and is exact under the noise model
    (``picasso.localize._gauss_crlb``).
    """
    bg = bg + readout_variance
    s2 = s**2
    sa2 = s2 + 1 / 12
    sa = sa2**0.5
    sa_orth2 = s_orth**2 + 1 / 12
    sa_orth = sa_orth2**0.5
    v = sa2 * (16 / 9 + (8 * np.pi * sa * sa_orth * bg) / photons) / photons
    if em:
        v *= 2
    with np.errstate(invalid="ignore"):
        return np.sqrt(v)


def sigma_uncertainty_lsq(
    sigma: lib.SeriesOrFloatArray1D,
    sigma_orth: lib.SeriesOrFloatArray1D,
    photons: lib.SeriesOrFloatArray1D,
    bg: lib.SeriesOrFloatArray1D,
    readout_variance: lib.SeriesOrFloatArray1D | float = 0.0,
) -> lib.FloatArray1D:
    """Standard error of a **least-squares** fitted sigma.

    From the 2D Gaussian least-squares model with a diagonal covariance
    matrix, Kowalewski, Reinhardt, et al. Nature Communications, 2026.

    Parameters
    ----------
    sigma : lib.SeriesOrFloatArray1D
        Fitted sigma values in camera pixels.
    sigma_orth : lib.SeriesOrFloatArray1D
        Fitted sigma values in the orthogonal direction in camera
        pixels.
    photons : lib.SeriesOrFloatArray1D
        Number of photons.
    bg : lib.SeriesOrFloatArray1D
        Background photons per pixel.
    readout_variance : lib.SeriesOrFloatArray1D or float, optional
        Mean sCMOS readout variance over the fitting box, in photoelectrons
        squared. Adds to the background term, with the same caveat as in
        :func:`localization_precision`. Default 0.

    Returns
    -------
    se_sigma : lib.FloatArray1D
        Standard error of fitted sigma values in camera pixels.
    """
    bg = bg + readout_variance
    sa2 = sigma**2 + 1 / 12
    sa4 = sa2**2
    sa = sa2**0.5
    sa2_orth = sigma_orth**2 + 1 / 12
    sa_orth = sa2_orth**0.5
    var_sa2 = (
        sa4
        / photons
        * (512 / 81 + (64 * np.pi * sa * sa_orth * bg) / (3 * photons))
    )
    var_sigma = var_sa2 / (4 * sigma**2)
    se_sigma = np.sqrt(var_sigma)
    return se_sigma


def sigma_uncertainty_mle(
    sigma: lib.SeriesOrFloatArray1D,
    sigma_orth: lib.SeriesOrFloatArray1D,
    photons: lib.SeriesOrFloatArray1D,
    bg: lib.SeriesOrFloatArray1D,
    readout_variance: lib.SeriesOrFloatArray1D | float = 0.0,
) -> lib.FloatArray1D:
    """Standard error of a **maximum-likelihood** fitted sigma.

    From the MLE 2D Gaussian / Poisson noise model, using the
    approximation of Rieger and Stallinga, ChemPhysChem, 2014.

    Parameters
    ----------
    sigma : lib.SeriesOrFloatArray1D
        Fitted sigma values in camera pixels.
    sigma_orth : lib.SeriesOrFloatArray1D
        Unused; see above.
    photons : lib.SeriesOrFloatArray1D
        Number of photons.
    bg : lib.SeriesOrFloatArray1D
        Background photons per pixel.
    readout_variance : lib.SeriesOrFloatArray1D or float, optional
        Mean sCMOS readout variance over the fitting box, in photoelectrons
        squared. Adds to the background term, with the same caveat as in
        :func:`localization_precision`. Default 0.

    Returns
    -------
    se_sigma : lib.FloatArray1D
        Standard error of fitted sigma values in camera pixels.
    """
    bg = bg + readout_variance
    sa2 = sigma**2 + 1 / 12
    tau = (2 * np.pi * sa2 * bg) / (photons)
    delta_sigma_sq = (sigma**2 / (4 * photons)) * (
        1 + 8 * tau + np.sqrt((8 * tau) / (1 + 2 * tau))
    )
    return np.sqrt(delta_sigma_sq)
