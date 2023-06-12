"""
    picasso.gausslq
    ~~~~~~~~~~~~~~~~

    Fit spots with Gaussian least squares

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""


from scipy import optimize as _optimize
import numpy as _np
from tqdm import tqdm as _tqdm
import numba as _numba
import multiprocessing as _multiprocessing
from concurrent import futures as _futures
from . import postprocess as _postprocess

try:
    from pygpufit import gpufit as gf

    gpufit_installed = True
except ImportError:
    gpufit_installed = False


@_numba.jit(nopython=True, nogil=True)
def _gaussian(mu, sigma, grid):
    norm = 0.3989422804014327 / sigma
    return norm * _np.exp(-0.5 * ((grid - mu) / sigma) ** 2)


"""
def integrated_gaussian(mu, sigma, grid):
    norm = 0.70710678118654757 / sigma   # sq_norm = sqrt(0.5/sigma**2)
    integrated_gaussian =  0.5 *
    (erf((grid - mu + 0.5) * norm) - erf((grid - mu - 0.5) * norm))
    return integrated_gaussian
"""


@_numba.jit(nopython=True, nogil=True)
def _sum_and_center_of_mass(spot, size):
    x = 0.0
    y = 0.0
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            x += spot[i, j] * i
            y += spot[i, j] * j
            _sum_ += spot[i, j]
    x /= _sum_
    y /= _sum_
    return _sum_, y, x


@_numba.jit(nopython=True, nogil=True)
def _initial_sigmas(spot, y, x, sum, size):
    sum_deviation_y = 0.0
    sum_deviation_x = 0.0
    for i in range(size):
        for j in range(size):
            sum_deviation_y += spot[i, j] * (i - y) ** 2
            sum_deviation_x += spot[i, j] * (j - x) ** 2
    sy = _np.sqrt(sum_deviation_y / sum)
    sx = _np.sqrt(sum_deviation_x / sum)
    return sy, sx


@_numba.jit(nopython=True, nogil=True)
def _initial_parameters(spot, size, size_half):
    theta = _np.zeros(6, dtype=_np.float32)
    theta[3] = _np.min(spot)
    spot_without_bg = spot - theta[3]
    sum, theta[1], theta[0] = _sum_and_center_of_mass(spot_without_bg, size)
    theta[2] = _np.maximum(1.0, sum)
    theta[5], theta[4] = _initial_sigmas(spot - theta[3], theta[1], theta[0], sum, size)
    theta[0:2] -= size_half
    return theta


def initial_parameters_gpufit(spots, size):

    center = (size / 2.0) - 0.5
    initial_width = _np.amax([size / 5.0, 1.0])

    spot_max = _np.amax(spots, axis=(1, 2))
    spot_min = _np.amin(spots, axis=(1, 2))

    initial_parameters = _np.empty((len(spots), 6), dtype=_np.float32)

    initial_parameters[:, 0] = spot_max - spot_min
    initial_parameters[:, 1] = center
    initial_parameters[:, 2] = center
    initial_parameters[:, 3] = initial_width
    initial_parameters[:, 4] = initial_width
    initial_parameters[:, 5] = spot_min

    return initial_parameters


@_numba.jit(nopython=True, nogil=True)
def _outer(a, b, size, model, n, bg):
    for i in range(size):
        for j in range(size):
            model[i, j] = n * a[i] * b[j] + bg


@_numba.jit(nopython=True, nogil=True)
def _compute_model(theta, grid, size, model_x, model_y, model):
    model_x[:] = _gaussian(
        theta[0], theta[4], grid
    )  # sx and sy are wrong with integrated gaussian
    model_y[:] = _gaussian(
        theta[1], theta[5], grid
    )
    _outer(model_y, model_x, size, model, theta[2], theta[3])
    return model


@_numba.jit(nopython=True, nogil=True)
def _compute_residuals(theta, spot, grid, size, model_x, model_y, model, residuals):
    _compute_model(theta, grid, size, model_x, model_y, model)
    residuals[:, :] = spot - model
    return residuals.flatten()


def fit_spot(spot):
    size = spot.shape[0]
    size_half = int(size / 2)
    grid = _np.arange(-size_half, size_half + 1, dtype=_np.float32)
    model_x = _np.empty(size, dtype=_np.float32)
    model_y = _np.empty(size, dtype=_np.float32)
    model = _np.empty((size, size), dtype=_np.float32)
    residuals = _np.empty((size, size), dtype=_np.float32)
    # theta is [x, y, photons, bg, sx, sy]
    theta0 = _initial_parameters(spot, size, size_half)
    args = (spot, grid, size, model_x, model_y, model, residuals)
    result = _optimize.leastsq(
        _compute_residuals, theta0, args=args, ftol=1e-2, xtol=1e-2
    )  # leastsq is much faster than least_squares
    """
    model = compute_model(result[0], grid, size, model_x, model_y, model)
    plt.figure()
    plt.subplot(121)
    plt.imshow(spot, interpolation='none')
    plt.subplot(122)
    plt.imshow(model, interpolation='none')
    plt.colorbar()
    plt.show()
    """
    return result[0]


@_numba.jit(nopython=True, nogil=True)
def _gaussian_2D_rot(A, bkg, mu, sigma, alpha, grid_box_x, grid_box_y, model):
    """
    Args:
        A : float
            amplitude
        bkg : float
            background
        mu : tuple, len 2
            center
        sigma : tuple, len 2
            widths of axis a and b
        alpha : float, bounded [-pi/4 : pi/4]
            angle between x axis and ellipse axis
        grid_box_x : 2D int array, shape as spot box
            x coordinates
        grid_box_y : 2D int array, shape as spot box
            y coordinates
    """
    norm = 0.1591549431 / sigma[0] / sigma[1]
    a = (_np.cos(alpha)**2 / (2 * sigma[0]**2)
         + _np.sin(alpha)**2 / (2 * sigma[1]**2))
    b = (- _np.sin(2 * alpha) / (4 * sigma[0]**2)
         + _np.sin(2 * alpha) / (4 * sigma[1]**2))
    c = (_np.sin(alpha)**2 / (2 * sigma[0]**2)
         + _np.cos(alpha)**2 / (2 * sigma[1]**2))

    model[:, :] = bkg + A * norm * _np.exp(- (
        a * (grid_box_x - mu[0])**2
        + 2 * b * (grid_box_x - mu[0]) * (grid_box_y - mu[1])
        + c * (grid_box_y - mu[1])**2))
    return model


@_numba.jit(nopython=True, nogil=True)
def _compute_model_rot(theta, grid_box_x, grid_box_y, size, model):
    model = _gaussian_2D_rot(
        theta[2], theta[3], theta[:2], theta[4:6], theta[6],
        grid_box_x, grid_box_y, model)
    return model


@_numba.jit(nopython=True, nogil=True)
def _compute_residuals_rot(
        theta, spot, grid_box_x, grid_box_y, size, model, residuals):
    _compute_model_rot(theta, grid_box_x, grid_box_y, size, model)
    residuals[:, :] = spot - model
    return residuals.flatten()


def fit_spot_rot(spot, bounded=True):
    """fits a 2D gaussian to a spot, taking rotation alpha of the
    elliptic axes with respect to x/y into account.
    (sx, sy) = (sin alpha, cos alpha; cos alpha, -sin alpha) * (sa, sb)

    For compability to non-rotation fitting, alpha shall be bounded to
    the interval [-pi/4:pi/4]. Therefore, the axes sa, sb are approximately
    sx, sy, and can vary between long and short axis.

    Approach:
     * first use fit_spot to find the approximate solution (using non-bounded
        fitting)
     * then use the result as initial values to fitting with the bounded alpha
        and on a 2D model, as projections onto x/y are imprecise and grid
        transformation on the rotated axes are computationally expensive

    parameters:
        0: x-position
        1: y-position
        2: amplitude (photons)
        3: background
        4: a width (approx x width)
        5: b width (approx y width)
        6: rotation angle [rad]
    """
    size = spot.shape[0]
    size_half = int(size / 2)

    # pars_nonrot = fit_spot(spot)
    pars_nonrot = _initial_parameters(spot, size, size_half)

    theta0 = _np.empty(7, dtype=_np.float32)
    theta0[:-1] = pars_nonrot
    theta0[-1] = 0

    grid = _np.arange(-size_half, size_half + 1, dtype=_np.float32)
    grid_box_x, grid_box_y = _np.meshgrid(grid, grid)
    model = _np.empty((size, size), dtype=_np.float32)
    residuals = _np.empty((size, size), dtype=_np.float32)
    args = (spot, grid_box_x, grid_box_y, size, model, residuals)
    # bounded=False
    if bounded:
        bounds_lo = [-size_half-1, -size_half-1, 0, -1E2, 0, 0, -_np.pi / 4]
        bounds_hi = [size_half+1, size_half+1, 1E7, 1E3, size, size, _np.pi / 4]
        # print('x0', theta0)
        # print('bounds_lo', bounds_lo)
        # print('bounds_hi', bounds_hi)
        result = _optimize.least_squares(
            _compute_residuals_rot, theta0, args=args, ftol=1e-5, xtol=1e-5,
            bounds=(bounds_lo, bounds_hi)
        )  # leastsq is much faster than least_squares
        res = result.x
    else:
        result = _optimize.leastsq(
            _compute_residuals_rot, theta0, args=args, ftol=1e-5, xtol=1e-5
        )  # leastsq is much faster than least_squares
        res = result[0]
    # print('fit result', result)
    return res


def fit_spots(spots, fit_rot=False):
    npars = 7 if fit_rot else 6
    theta = _np.empty((len(spots), npars), dtype=_np.float32)
    theta.fill(_np.nan)
    for i, spot in enumerate(spots):
        if fit_rot:
            theta[i] = fit_spot_rot(spot)
        else:
            theta[i] = fit_spot(spot)
    return theta


def fit_spots_parallel(spots, asynch=False, fit_rot=False):
    n_workers = min(
        60, max(1, int(0.75 * _multiprocessing.cpu_count()))
    ) # Python crashes when using >64 cores
    n_spots = len(spots)
    n_tasks = 100 * n_workers
    spots_per_task = [
        int(n_spots / n_tasks + 1) if _ < n_spots % n_tasks else int(n_spots / n_tasks)
        for _ in range(n_tasks)
    ]
    start_indices = _np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = _futures.ProcessPoolExecutor(n_workers)
    for i, n_spots_task in zip(start_indices, spots_per_task):
        fs.append(executor.submit(fit_spots, spots[i : i + n_spots_task], fit_rot))
    if asynch:
        return fs
    with _tqdm(total=n_tasks, unit="task") as progress_bar:
        for f in _futures.as_completed(fs):
            progress_bar.update()
    return fits_from_futures(fs)


def fit_spots_gpufit(spots):
    size = spots.shape[1]
    initial_parameters = initial_parameters_gpufit(spots, size)
    spots.shape = (len(spots), (size * size))
    model_id = gf.ModelID.GAUSS_2D_ELLIPTIC

    parameters, states, chi_squares, number_iterations, exec_time = gf.fit(
        spots,
        None,
        model_id,
        initial_parameters,
        tolerance=1e-2,
        max_number_iterations=20,
    )

    parameters[:, 0] *= 2.0 * _np.pi * parameters[:, 3] * parameters[:, 4]

    return parameters


def fits_from_futures(futures):
    theta = [_.result() for _ in futures]
    return _np.vstack(theta)


def locs_from_fits(identifications, theta, box, em):
    """
        0: x-position
        1: y-position
        2: amplitude (photons)
        3: background
        4: a width (approx x width)
        5: b width (approx y width)
        6: rotation angle [rad]
    """
    # box_offset = int(box/2)
    x = theta[:, 0] + identifications.x  # - box_offset
    y = theta[:, 1] + identifications.y  # - box_offset
    lpx = _postprocess.localization_precision(
        theta[:, 2], theta[:, 4], theta[:, 3], em=em
    )
    lpy = _postprocess.localization_precision(
        theta[:, 2], theta[:, 5], theta[:, 3], em=em
    )
    a = _np.maximum(theta[:, 4], theta[:, 5])
    b = _np.minimum(theta[:, 4], theta[:, 5])
    ellipticity = (a - b) / a
    fit_rot = theta.shape[1] == 7

    if hasattr(identifications, "n_id"):
        if fit_rot:
            locs = _np.rec.array(
                (
                    identifications.frame,
                    x,
                    y,
                    theta[:, 2],
                    theta[:, 4],
                    theta[:, 5],
                    theta[:, 3],
                    lpx,
                    lpy,
                    ellipticity,
                    theta[:, 6],
                    identifications.net_gradient,
                    identifications.n_id,
                ),
                dtype=[
                    ("frame", "u4"),
                    ("x", "f4"),
                    ("y", "f4"),
                    ("photons", "f4"),
                    ("sx", "f4"),
                    ("sy", "f4"),
                    ("bg", "f4"),
                    ("lpx", "f4"),
                    ("lpy", "f4"),
                    ("ellipticity", "f4"),
                    ("alpha", "f4"),
                    ("net_gradient", "f4"),
                    ("n_id", "u4"),
                ],
            )
        else:
            locs = _np.rec.array(
                (
                    identifications.frame,
                    x,
                    y,
                    theta[:, 2],
                    theta[:, 4],
                    theta[:, 5],
                    theta[:, 3],
                    lpx,
                    lpy,
                    ellipticity,
                    identifications.net_gradient,
                    identifications.n_id,
                ),
                dtype=[
                    ("frame", "u4"),
                    ("x", "f4"),
                    ("y", "f4"),
                    ("photons", "f4"),
                    ("sx", "f4"),
                    ("sy", "f4"),
                    ("bg", "f4"),
                    ("lpx", "f4"),
                    ("lpy", "f4"),
                    ("ellipticity", "f4"),
                    ("net_gradient", "f4"),
                    ("n_id", "u4"),
                ],
            )
        locs.sort(kind="mergesort", order="n_id")
    else:
        if fit_rot:
            locs = _np.rec.array(
                (
                    identifications.frame,
                    x,
                    y,
                    theta[:, 2],
                    theta[:, 4],
                    theta[:, 5],
                    theta[:, 3],
                    lpx,
                    lpy,
                    ellipticity,
                    theta[:, 6],
                    identifications.net_gradient,
                ),
                dtype=[
                    ("frame", "u4"),
                    ("x", "f4"),
                    ("y", "f4"),
                    ("photons", "f4"),
                    ("sx", "f4"),
                    ("sy", "f4"),
                    ("bg", "f4"),
                    ("lpx", "f4"),
                    ("lpy", "f4"),
                    ("ellipticity", "f4"),
                    ("alpha", "f4"),
                    ("net_gradient", "f4"),
                ],
            )
        else:
            locs = _np.rec.array(
                (
                    identifications.frame,
                    x,
                    y,
                    theta[:, 2],
                    theta[:, 4],
                    theta[:, 5],
                    theta[:, 3],
                    lpx,
                    lpy,
                    ellipticity,
                    identifications.net_gradient,
                ),
                dtype=[
                    ("frame", "u4"),
                    ("x", "f4"),
                    ("y", "f4"),
                    ("photons", "f4"),
                    ("sx", "f4"),
                    ("sy", "f4"),
                    ("bg", "f4"),
                    ("lpx", "f4"),
                    ("lpy", "f4"),
                    ("ellipticity", "f4"),
                    ("net_gradient", "f4"),
                ],
            )
        locs.sort(kind="mergesort", order="frame")
    return locs


def locs_from_fits_gpufit(identifications, theta, box, em):
    box_offset = int(box / 2)
    x = theta[:, 1] + identifications.x - box_offset
    y = theta[:, 2] + identifications.y - box_offset
    lpx = _postprocess.localization_precision(
        theta[:, 0], theta[:, 3], theta[:, 5], em=em
    )
    lpy = _postprocess.localization_precision(
        theta[:, 0], theta[:, 4], theta[:, 5], em=em
    )
    a = _np.maximum(theta[:, 3], theta[:, 4])
    b = _np.minimum(theta[:, 3], theta[:, 4])
    ellipticity = (a - b) / a
    locs = _np.rec.array(
        (
            identifications.frame,
            x,
            y,
            theta[:, 0],
            theta[:, 3],
            theta[:, 4],
            theta[:, 5],
            lpx,
            lpy,
            ellipticity,
            identifications.net_gradient,
        ),
        dtype=[
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("bg", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("ellipticity", "f4"),
            ("net_gradient", "f4"),
        ],
    )
    locs.sort(kind="mergesort", order="frame")
    return locs
