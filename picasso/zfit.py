import numpy as _np
import numba as _numba
import multiprocessing as _multiprocessing
import concurrent.futures as _futures
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from scipy.optimize import minimize_scalar as _minimize_scalar
from scipy.optimize import minimize as _minimize
from tqdm import tqdm as _tqdm
import yaml as _yaml
from icecream import ic
import matplotlib.pyplot as _plt
from . import lib as _lib


_plt.style.use("ggplot")


def nan_index(y):
    return _np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nan(data):
    nans, x = nan_index(data)
    data[nans] = _np.interp(x(nans), x(~nans), data[~nans])
    return data


def calibrate_z(locs, info, d, magnification_factor, path=None, show=True):
    n_frames = info[0]["Frames"]
    range = (n_frames - 1) * d
    frame_range = _np.arange(n_frames)
    z_range = -(frame_range * d - range / 2)  # negative so that the first frames of
    # a bottom-to-up scan are positive z coordinates.

    mean_sx = _np.array([_np.mean(locs.sx[locs.frame == _]) for _ in frame_range])
    mean_sy = _np.array([_np.mean(locs.sy[locs.frame == _]) for _ in frame_range])
    var_sx = _np.array([_np.var(locs.sx[locs.frame == _]) for _ in frame_range])
    var_sy = _np.array([_np.var(locs.sy[locs.frame == _]) for _ in frame_range])

    keep_x = (locs.sx - mean_sx[locs.frame]) ** 2 < var_sx[locs.frame]
    keep_y = (locs.sy - mean_sy[locs.frame]) ** 2 < var_sy[locs.frame]
    keep = keep_x & keep_y
    locs = locs[keep]

    # Fits calibration curve to the mean of each frame
    mean_sx = _np.array([_np.mean(locs.sx[locs.frame == _]) for _ in frame_range])
    mean_sy = _np.array([_np.mean(locs.sy[locs.frame == _]) for _ in frame_range])

    # Fix nan
    mean_sx = interpolate_nan(mean_sx)
    mean_sy = interpolate_nan(mean_sy)

    cx = _np.polyfit(z_range, mean_sx, 6, full=False)
    cy = _np.polyfit(z_range, mean_sy, 6, full=False)

    # Fits calibration curve to each localization
    # true_z = locs.frame * d - range / 2
    # cx = _np.polyfit(true_z, locs.sx, 6, full=False)
    # cy = _np.polyfit(true_z, locs.sy, 6, full=False)

    calibration = {
        "X Coefficients": [float(_) for _ in cx],
        "Y Coefficients": [float(_) for _ in cy],
    }
    if path is not None:
        with open(path, "w") as f:
            _yaml.dump(calibration, f, default_flow_style=False)

    locs = fit_z(locs, info, calibration, magnification_factor)
    locs.z /= magnification_factor

    _plt.figure(figsize=(18, 10))

    _plt.subplot(231)
    # Plot this if calibration curve is fitted to each localization
    # _plt.plot(true_z, locs.sx, '.', label='x', alpha=0.2)
    # _plt.plot(true_z, locs.sy, '.', label='y', alpha=0.2)
    # _plt.plot(true_z, _np.polyval(cx, true_z), '0.3', lw=1.5, label='x fit')
    # _plt.plot(true_z, _np.polyval(cy, true_z), '0.3', lw=1.5, label='y fit')
    _plt.plot(z_range, mean_sx, ".-", label="x")
    _plt.plot(z_range, mean_sy, ".-", label="y")
    _plt.plot(z_range, _np.polyval(cx, z_range), "0.3", lw=1.5, label="x fit")
    _plt.plot(z_range, _np.polyval(cy, z_range), "0.3", lw=1.5, label="y fit")
    _plt.xlabel("Stage position")
    _plt.ylabel("Mean spot width/height")
    _plt.xlim(z_range.min(), z_range.max())
    _plt.legend(loc="best")

    ax = _plt.subplot(232)
    _plt.scatter(locs.sx, locs.sy, c="k", lw=0, alpha=0.1)
    _plt.plot(
        _np.polyval(cx, z_range),
        _np.polyval(cy, z_range),
        lw=1.5,
        label="calibration from fit of mean width/height",
    )
    _plt.plot()
    ax.set_aspect("equal")
    _plt.xlabel("Spot width")
    _plt.ylabel("Spot height")
    _plt.legend(loc="best")

    _plt.subplot(233)
    _plt.plot(locs.z, locs.sx, ".", label="x", alpha=0.2)
    _plt.plot(locs.z, locs.sy, ".", label="y", alpha=0.2)
    _plt.plot(z_range, _np.polyval(cx, z_range), "0.3", lw=1.5, label="calibration")
    _plt.plot(z_range, _np.polyval(cy, z_range), "0.3", lw=1.5)
    _plt.xlim(z_range.min(), z_range.max())
    _plt.xlabel("Estimated z")
    _plt.ylabel("Spot width/height")
    _plt.legend(loc="best")

    ax = _plt.subplot(234)
    _plt.plot(z_range[locs.frame], locs.z, ".k", alpha=0.1)
    _plt.plot(
        [z_range.min(), z_range.max()],
        [z_range.min(), z_range.max()],
        lw=1.5,
        label="identity",
    )
    _plt.xlim(z_range.min(), z_range.max())
    _plt.ylim(z_range.min(), z_range.max())
    ax.set_aspect("equal")
    _plt.xlabel("Stage position")
    _plt.ylabel("Estimated z")
    _plt.legend(loc="best")

    ax = _plt.subplot(235)
    deviation = locs.z - z_range[locs.frame]
    bins = _lib.calculate_optimal_bins(deviation, max_n_bins=1000)
    _plt.hist(deviation, bins)
    _plt.xlabel("Deviation to true position")
    _plt.ylabel("Occurence")

    ax = _plt.subplot(236)
    square_deviation = deviation**2
    mean_square_deviation_frame = [
        _np.mean(square_deviation[locs.frame == _]) for _ in frame_range
    ]
    rmsd_frame = _np.sqrt(mean_square_deviation_frame)
    _plt.plot(z_range, rmsd_frame, ".-", color="0.3")
    _plt.xlim(z_range.min(), z_range.max())
    _plt.gca().set_ylim(bottom=0)
    _plt.xlabel("Stage position")
    _plt.ylabel("Mean z precision")

    _plt.tight_layout(pad=2)

    if path is not None:
        dirname = path[0:-5]
        _plt.savefig(dirname + ".png", format="png", dpi=300)

    if show:
        _plt.show()

    export = False
    # Export
    if export:
        print("Exporting...")
        _np.savetxt("mean_sx.txt", mean_sx, delimiter="/t")
        _np.savetxt("mean_sy.txt", mean_sy, delimiter="/t")
        _np.savetxt("locs_sx.txt", locs.sx, delimiter="/t")
        _np.savetxt("locs_sy.txt", locs.sy, delimiter="/t")
        _np.savetxt("cx.txt", cx, delimiter="/t")
        _np.savetxt("cy.txt", cy, delimiter="/t")
        _np.savetxt("z_range.txt", z_range, delimiter="/t")
        _np.savetxt("locs_z.txt", locs.z, delimiter="/t")
        _np.savetxt("z_range_locs_frame.txt", z_range[locs.frame], delimiter="/t")
        _np.savetxt("rmsd_frame.txt", rmsd_frame, delimiter="/t")

    # np.savetxt('test.out', x, delimiter=',')   # X is an array
    # np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    # np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    return calibration


def select_index_range(arr, axis, start, end):
    # Create a tuple of slices to select the desired index range along the specified axis
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(start, end)
    
    # Use the tuple of slices to select the desired subarray
    selected = arr[tuple(slices)]
    
    return selected


def calc_shift_msqerr(arr, ref, rng, x_arr=None, x_ref=None, axis=0):
    """
    Args:
        arr : 1 or 2D array
            array to shift
        ref : 1 or 2D array
            reference data
        rng : even int
            range of shift evaluation
        axis : int
            the axis to shift over
    """
    msqerr = _np.zeros(rng)
    if x_arr is not None and x_ref is not None:
        if len(arr.shape) > 1:
            arr_new = _np.zeros_like(ref)
            for idx in range(arr.shape[1]):  # this 1 should be 'all but "axis"'
                arr_new[:, idx] = _np.interp(x_ref, x_arr, arr[:, idx], left=_np.nan, right=_np.nan)
            arr = arr_new
        else:
            arr = _np.interp(x_ref, x_arr, arr, left=_np.nan, right=_np.nan)
    for i in range(rng):
        msqerr[i] = _np.nanmean(
            (select_index_range(ref, axis, i, -rng+i)
             -select_index_range(arr, axis, int(rng/2), -int(rng/2)))**2)
    return _np.argmin(msqerr)-int(rng/2)
    

def calibrate_z_groupedshift(locs, info, d, magnification_factor, path=None, show=True, xylock=True):
    n_frames = info[0]["Frames"]
    range = (n_frames - 1) * d
    frame_range = _np.arange(n_frames)
    z_range = -(frame_range * d - range / 2)  # negative so that the first frames of
    # a bottom-to-up scan are positive z coordinates.

    # shift the frames of groups to match the sx/sy curves best
    import pandas as pd
    locsdf = pd.DataFrame(locs)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(ncols=2, nrows=2)
    # for gidx, gdf in locsdf.groupby('group'):
    #     ax[0, 0].plot(gdf['frame'].to_numpy(), gdf['sx'].to_numpy())
    #     ax[0, 1].plot(gdf['frame'].to_numpy(), gdf['sy'].to_numpy())
    if xylock:
        # get reference
        refid = locsdf.groupby('group').apply(lambda x: len(x.index)).idxmax()
        # shift x and y together
        shiftdf = locsdf.groupby('group').apply(lambda x: calc_shift_msqerr(
            x[['sx', 'sy']].to_numpy(),
            locsdf.loc[locsdf['group']==refid, ['sx', 'sy']].to_numpy(),
            int(n_frames/2),
            x_arr=x['frame'].to_numpy(),
            x_ref=locsdf.loc[locsdf['group']==refid, 'frame'].to_numpy()))
    else:
        # shift x and y separately
        raise NotImplementedError('Only shifting frames for now -> separate shifts in sx and sy don"t work')
        shiftdf = locsdf.groupby('group').apply(lambda x: calc_shift_msqerr(
            x[['sx']].to_numpy(),
            locsdf.loc[locsdf['group']==0, ['sx']].to_numpy(),
            int(n_frames/2)))
        shiftdf = locsdf.groupby('group').apply(lambda x: calc_shift_msqerr(
            x[['sy']].to_numpy(),
            locsdf.loc[locsdf['group']==0, ['sy']].to_numpy(),
            int(n_frames/2)))
    locsdf['frame'] = locsdf.apply(lambda x: x['frame']+shiftdf[x['group']], axis=1).astype(_np.int32)
    # filter
    locsdf = locsdf.loc[locsdf['frame']>=0, :]
    locsdf = locsdf.loc[locsdf['frame']<n_frames, :]
    # for gidx, gdf in locsdf.groupby('group'):
    #     ax[1, 0].plot(gdf['frame'].to_numpy(), gdf['sx'].to_numpy())
    #     ax[1, 1].plot(gdf['frame'].to_numpy(), gdf['sy'].to_numpy())
    # plt.show()
    # insert back into locs
    locs = locsdf.to_records(index=False)

    mean_sx = _np.array([_np.mean(locs.sx[locs.frame == _]) for _ in frame_range])
    mean_sy = _np.array([_np.mean(locs.sy[locs.frame == _]) for _ in frame_range])
    var_sx = _np.array([_np.var(locs.sx[locs.frame == _]) for _ in frame_range])
    var_sy = _np.array([_np.var(locs.sy[locs.frame == _]) for _ in frame_range])

    keep_x = (locs.sx - mean_sx[locs.frame]) ** 2 < var_sx[locs.frame]
    keep_y = (locs.sy - mean_sy[locs.frame]) ** 2 < var_sy[locs.frame]
    keep = keep_x & keep_y
    locs = locs[keep]

    # Fits calibration curve to the mean of each frame
    mean_sx = _np.array([_np.mean(locs.sx[locs.frame == _]) for _ in frame_range])
    mean_sy = _np.array([_np.mean(locs.sy[locs.frame == _]) for _ in frame_range])

    # Fix nan
    mean_sx = interpolate_nan(mean_sx)
    mean_sy = interpolate_nan(mean_sy)

    cx = _np.polyfit(z_range, mean_sx, 6, full=False)
    cy = _np.polyfit(z_range, mean_sy, 6, full=False)

    # Fits calibration curve to each localization
    # true_z = locs.frame * d - range / 2
    # cx = _np.polyfit(true_z, locs.sx, 6, full=False)
    # cy = _np.polyfit(true_z, locs.sy, 6, full=False)

    calibration = {
        "X Coefficients": [float(_) for _ in cx],
        "Y Coefficients": [float(_) for _ in cy],
    }
    if path is not None:
        with open(path, "w") as f:
            _yaml.dump(calibration, f, default_flow_style=False)

    locs = fit_z(locs, info, calibration, magnification_factor)
    locs.z /= magnification_factor

    _plt.figure(figsize=(18, 10))

    _plt.subplot(231)
    # Plot this if calibration curve is fitted to each localization
    # _plt.plot(true_z, locs.sx, '.', label='x', alpha=0.2)
    # _plt.plot(true_z, locs.sy, '.', label='y', alpha=0.2)
    # _plt.plot(true_z, _np.polyval(cx, true_z), '0.3', lw=1.5, label='x fit')
    # _plt.plot(true_z, _np.polyval(cy, true_z), '0.3', lw=1.5, label='y fit')
    _plt.plot(z_range, mean_sx, ".-", label="x")
    _plt.plot(z_range, mean_sy, ".-", label="y")
    _plt.plot(z_range, _np.polyval(cx, z_range), "0.3", lw=1.5, label="x fit")
    _plt.plot(z_range, _np.polyval(cy, z_range), "0.3", lw=1.5, label="y fit")
    _plt.xlabel("Stage position")
    _plt.ylabel("Mean spot width/height")
    _plt.xlim(z_range.min(), z_range.max())
    _plt.legend(loc="best")

    ax = _plt.subplot(232)
    _plt.scatter(locs.sx, locs.sy, c="k", lw=0, alpha=0.1)
    _plt.plot(
        _np.polyval(cx, z_range),
        _np.polyval(cy, z_range),
        lw=1.5,
        label="calibration from fit of mean width/height",
    )
    _plt.plot()
    ax.set_aspect("equal")
    _plt.xlabel("Spot width")
    _plt.ylabel("Spot height")
    _plt.legend(loc="best")

    _plt.subplot(233)
    _plt.plot(locs.z, locs.sx, ".", label="x", alpha=0.2)
    _plt.plot(locs.z, locs.sy, ".", label="y", alpha=0.2)
    _plt.plot(z_range, _np.polyval(cx, z_range), "0.3", lw=1.5, label="calibration")
    _plt.plot(z_range, _np.polyval(cy, z_range), "0.3", lw=1.5)
    _plt.xlim(z_range.min(), z_range.max())
    _plt.xlabel("Estimated z")
    _plt.ylabel("Spot width/height")
    _plt.legend(loc="best")

    ax = _plt.subplot(234)
    _plt.plot(z_range[locs.frame], locs.z, ".k", alpha=0.1)
    _plt.plot(
        [z_range.min(), z_range.max()],
        [z_range.min(), z_range.max()],
        lw=1.5,
        label="identity",
    )
    _plt.xlim(z_range.min(), z_range.max())
    _plt.ylim(z_range.min(), z_range.max())
    ax.set_aspect("equal")
    _plt.xlabel("Stage position")
    _plt.ylabel("Estimated z")
    _plt.legend(loc="best")

    ax = _plt.subplot(235)
    deviation = locs.z - z_range[locs.frame]
    bins = _lib.calculate_optimal_bins(deviation, max_n_bins=1000)
    _plt.hist(deviation, bins)
    _plt.xlabel("Deviation to true position")
    _plt.ylabel("Occurence")

    ax = _plt.subplot(236)
    square_deviation = deviation**2
    mean_square_deviation_frame = [
        _np.mean(square_deviation[locs.frame == _]) for _ in frame_range
    ]
    rmsd_frame = _np.sqrt(mean_square_deviation_frame)
    _plt.plot(z_range, rmsd_frame, ".-", color="0.3")
    _plt.xlim(z_range.min(), z_range.max())
    _plt.gca().set_ylim(bottom=0)
    _plt.xlabel("Stage position")
    _plt.ylabel("Mean z precision")

    _plt.tight_layout(pad=2)

    if path is not None:
        dirname = path[0:-5]
        _plt.savefig(dirname + "_groupshifted.png", format="png", dpi=300)

    if show:
        _plt.show()

    export = False
    # Export
    if export:
        print("Exporting...")
        _np.savetxt("mean_sx.txt", mean_sx, delimiter="/t")
        _np.savetxt("mean_sy.txt", mean_sy, delimiter="/t")
        _np.savetxt("locs_sx.txt", locs.sx, delimiter="/t")
        _np.savetxt("locs_sy.txt", locs.sy, delimiter="/t")
        _np.savetxt("cx.txt", cx, delimiter="/t")
        _np.savetxt("cy.txt", cy, delimiter="/t")
        _np.savetxt("z_range.txt", z_range, delimiter="/t")
        _np.savetxt("locs_z.txt", locs.z, delimiter="/t")
        _np.savetxt("z_range_locs_frame.txt", z_range[locs.frame], delimiter="/t")
        _np.savetxt("rmsd_frame.txt", rmsd_frame, delimiter="/t")

    # np.savetxt('test.out', x, delimiter=',')   # X is an array
    # np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    # np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    return calibration, locs


def calibrate_z_tilt(locs, info, d, magnification_factor, path=None, show=True):
    n_frames = info[0]["Frames"]
    range = (n_frames - 1) * d
    frame_range = _np.arange(n_frames)
    z_range = -(
        frame_range * d - range / 2
    )  # negative so that the first frames of
    # a bottom-to-up scan are positive z coordinates.

    mean_sx = _np.array(
        [_np.mean(locs.sx[locs.frame == _]) for _ in frame_range]
    )
    mean_sy = _np.array(
        [_np.mean(locs.sy[locs.frame == _]) for _ in frame_range]
    )
    var_sx = _np.array(
        [_np.var(locs.sx[locs.frame == _]) for _ in frame_range]
    )
    var_sy = _np.array(
        [_np.var(locs.sy[locs.frame == _]) for _ in frame_range]
    )

    keep_x = (locs.sx - mean_sx[locs.frame]) ** 2 < var_sx[locs.frame]
    keep_y = (locs.sy - mean_sy[locs.frame]) ** 2 < var_sy[locs.frame]
    keep = keep_x & keep_y
    locs = locs[keep]

    # fit locs.sx, locs.sy, locs.x, locs.y
    # to locs.frame * z_range
    #
    #model: 
    # z = z_frame + alpha * x + beta * y
    # sx = polynomial(z)
    # sy = polynomial(z)
    def model(p, z_frame, x, y, deg=6):
        """
        Args:
            p : list, len 2 + 2 * deg
                the parameters
            z_frame : array, len N
                the z step value
            x : array, len N
                the x position of the spot
            y : array, len N
                the y positio of the spot
            deg : int
                the degree of the polynomial
        Returns:
            sx : array, len N
                the PSF width in x
            sy : array, len N
                the PSF width in y
            z : array, len N
                the real z positions
        """
        z = z_frame + p[0] * x + p[1] * y
        sx = _np.polynomial.Polynomial(p[2:2+deg])(z)
        sy = _np.polynomial.Polynomial(p[2+deg:])(z)
        return sx, sy, z

    def model_err(p, sx, sy, z_frame, x, y, deg=6):
        sx_m, sy_m, z = model(p, z_frame, x, y, deg=6)
        err = _np.sum((sx_m-sx)**2) + _np.sum((sy_m-sy)**2)
        return err

    def model_initpars(sx, sy, z_frame, deg=6):
        """Estimate initial parameters for fitting
        """
        p = _np.zeros(2 + 2 * deg)

        # calculate mean sx and sy values for the z_frame values
        z_steps, z_indices = _np.unique(z_frame, return_inverse=True)
        sx_mean = _np.zeros_like(z_steps, dtype=_np.float64)
        sy_mean = _np.zeros_like(z_steps, dtype=_np.float64)
        for i, z in enumerate(z_steps):
            sx_mean[i] = _np.nanmean(sx[z_indices == i])
            sy_mean[i] = _np.nanmean(sy[z_indices == i])
        # now sort to make sure
        z_indices = _np.argsort(z_steps)
        sx_mean = sx_mean[z_indices]
        sy_mean = sy_mean[z_indices]
        z_steps = z_steps[z_indices]
        # and fit polynomials
        coef_x = _np.polynomial.Polynomial.fit(
            z_steps, sx_mean, deg=deg - 1,
            domain=[min(z_steps), max(z_steps)],
            window=[min(z_steps), max(z_steps)]).coef
        coef_y = _np.polynomial.Polynomial.fit(
            z_steps, sy_mean, deg=deg - 1,
            domain=[min(z_steps), max(z_steps)],
            window=[min(z_steps), max(z_steps)]).coef
        p[2:2 + deg] = coef_x
        p[2 + deg:] = coef_y

        bounds = [
            (-.005, .005),
            (-.005, .005),
            (-5, 5),
            (-5e-2, 5e-2),
            (-5e-4, 5e-4),
            (-5e-6, 5e-6),
            (-5e-8, 5e-8),
            (-5e-10, 5e-10),
            (-5, 5),
            (-5e-2, 5e-2),
            (-5e-4, 5e-4),
            (-5e-6, 5e-6),
            (-5e-8, 5e-8),
            (-5e-10, 5e-10)]
        return p, sx_mean, sy_mean, z_steps, bounds

    deg = 6
    p_init, mean_sx, mean_sy, z_range, bounds = model_initpars(
        locs.sx, locs.sy, locs.frame, deg=deg)
    ic(p_init)

    # p_opt = p_init
    optres = _minimize(
        model_err, p_init, args=(locs.sx, locs.sy, locs.frame, locs.x, locs.y),
        bounds=bounds)
    p_opt = optres.x
    print('success', optres.success, optres.message)
    print(optres)
    ic(p_opt)

    sx_m, sy_m, z = model(p_opt, locs.frame, locs.x, locs.y, deg=6)

    # cx = p_opt[2+deg-1:1:-1]
    # cy = p_opt[-1:2+deg-1:-1]
    cx = p_opt[2:2+deg]
    cy = p_opt[2+deg:]
    ic(cx)
    ic(cy)

    ic(sx_m)
    ic(sy_m)
    ic(z)

    # legacy (polyfit etc) uses reversse order of coefficients
    calibration = {
        "X Coefficients": [float(_) for _ in cx[::-1]],
        "Y Coefficients": [float(_) for _ in cy[::-1]],
    }
    if path is not None:
        with open(path, "w") as f:
            _yaml.dump(calibration, f, default_flow_style=False)

    # locs = fit_z(locs, info, calibration, magnification_factor)
    # locs.z /= magnification_factor
    # ic(locs.z)

    _plt.figure(figsize=(18, 10))

    _plt.subplot(231)
    # Plot this if calibration curve is fitted to each localization
    # _plt.plot(true_z, locs.sx, '.', label='x', alpha=0.2)
    # _plt.plot(true_z, locs.sy, '.', label='y', alpha=0.2)
    # _plt.plot(true_z, _np.polyval(cx, true_z), '0.3', lw=1.5, label='x fit')
    # _plt.plot(true_z, _np.polyval(cy, true_z), '0.3', lw=1.5, label='y fit')

    poly_init_x = _np.polynomial.Polynomial(p_init[2:2+deg])
    poly_init_y = _np.polynomial.Polynomial(p_init[2+deg:])
    poly_opt_x = _np.polynomial.Polynomial(p_opt[2:2+deg])
    poly_opt_y = _np.polynomial.Polynomial(p_opt[2+deg:])
    # _plt.scatter(locs.frame, locs.sx, c="k", lw=0, s=2, alpha=0.2)
    # _plt.scatter(locs.frame, locs.sy, c="k", lw=0, s=2, alpha=0.2)
    _plt.scatter(locs.frame, locs.sx, c=locs.y, lw=0, s=2, alpha=0.8)
    _plt.scatter(locs.frame, locs.sy, c=locs.y, lw=0, s=2, alpha=0.8)
    _plt.plot(z_range, mean_sx, ".-", label="mean x")
    _plt.plot(z_range, mean_sy, ".-", label="mean y")
    _plt.plot(z_range, poly_init_x(z_range), "0.3", lw=.5, label="x init")
    _plt.plot(z_range, poly_init_y(z_range), "0.3", lw=.5, label="y init")
    _plt.plot(z_range, poly_opt_x(z_range), "0.3", lw=1.5, label="x fit")
    _plt.plot(z_range, poly_opt_y(z_range), "0.3", lw=1.5, label="y fit")
    _plt.xlabel("Stage position")
    _plt.ylabel("spot width/height")
    _plt.xlim(z_range.min(), z_range.max())
    _plt.legend(loc="best")

    ax = _plt.subplot(232)
    # _plt.scatter(locs.sx, locs.sy, c="k", lw=0, alpha=0.1)
    _plt.scatter(locs.sx, locs.sy, c=locs.y, lw=0, s=5, alpha=0.1)
    _plt.plot(
        poly_init_x(z_range),
        poly_init_y(z_range),
        lw=0.5,
        label="calibration from fit of mean x/y",
    )
    _plt.plot(
        poly_opt_x(z_range),
        poly_opt_y(z_range),
        lw=1.5,
        label="calibration from fit of tilted calibration",
    )
    _plt.plot()
    ax.set_aspect("equal")
    _plt.xlabel("Spot width")
    _plt.ylabel("Spot height")
    _plt.legend(loc="best")

    # _plt.subplot(233)
    # _plt.plot(z, sx_m, ".", label="x", alpha=0.2)
    # _plt.plot(z, sy_m, ".", label="y", alpha=0.2)
    # _plt.plot(
    #     z_range, poly_opt_x(z_range), "0.3", lw=1.5, label="calibration"
    # )
    # _plt.plot(z_range, poly_opt_y(z_range), "0.3", lw=1.5)
    # _plt.xlim(z_range.min(), z_range.max())
    # _plt.xlabel("Estimated z")
    # _plt.ylabel("Spot width/height")
    # _plt.legend(loc="best")

    _plt.subplot(233)
    # deviation from mean position for every spot
    dev_sx = locs.sx - mean_sx[locs.frame]
    dev_sy = locs.sx - mean_sy[locs.frame]
    _plt.hexbin(locs.x, locs.y, dev_sx)
    _plt.xlabel('deviation from plane mean for sx')
    _plt.subplot(236)
    _plt.hexbin(locs.x, locs.y, dev_sy)
    _plt.xlabel('deviation from plane mean for sy')

    # ax = _plt.subplot(234)
    # _plt.plot(z_range[locs.frame], locs.z, ".k", alpha=0.1)
    # # _plt.plot(
    # #     [z_range.min(), z_range.max()],
    # #     [z_range.min(), z_range.max()],
    # #     lw=1.5,
    # #     label="identity",
    # # )
    # # _plt.xlim(z_range.min(), z_range.max())
    # # _plt.ylim(z_range.min(), z_range.max())
    # # ax.set_aspect("equal")
    # _plt.xlabel("Stage position")
    # _plt.ylabel("Estimated z")
    # _plt.legend(loc="best")

    # ax = _plt.subplot(235)
    # deviation = locs.z - z_range[locs.frame]
    # bins = _lib.calculate_optimal_bins(deviation, max_n_bins=1000)
    # _plt.hist(deviation, bins)
    # deviation = locs.z - z
    # bins = _lib.calculate_optimal_bins(deviation, max_n_bins=1000)
    # _plt.hist(deviation, bins)
    # _plt.xlabel("Deviation to true position")
    # _plt.ylabel("Occurence")

    # ax = _plt.subplot(236)
    # square_deviation = deviation ** 2
    # mean_square_deviation_frame = [
    #     _np.mean(square_deviation[locs.frame == _]) for _ in frame_range
    # ]
    # rmsd_frame = _np.sqrt(mean_square_deviation_frame)
    # _plt.plot(z_range, rmsd_frame, ".-", color="0.3")
    # _plt.xlim(z_range.min(), z_range.max())
    # _plt.gca().set_ylim(bottom=0)
    # _plt.xlabel("Stage position")
    # _plt.ylabel("Mean z precision")

    _plt.tight_layout(pad=2)

    if path is not None:
        dirname = path[0:-5]
        _plt.savefig(dirname + ".png", format='png', dpi=300)

    if show:
        _plt.show()

    export = False
    # Export
    if export:
        print("Exporting...")
        _np.savetxt("mean_sx.txt", mean_sx, delimiter="/t")
        _np.savetxt("mean_sy.txt", mean_sy, delimiter="/t")
        _np.savetxt("locs_sx.txt", locs.sx, delimiter="/t")
        _np.savetxt("locs_sy.txt", locs.sy, delimiter="/t")
        _np.savetxt("cx.txt", cx, delimiter="/t")
        _np.savetxt("cy.txt", cy, delimiter="/t")
        _np.savetxt("z_range.txt", z_range, delimiter="/t")
        _np.savetxt("locs_z.txt", locs.z, delimiter="/t")
        _np.savetxt(
            "z_range_locs_frame.txt", z_range[locs.frame], delimiter="/t"
        )
        _np.savetxt("rmsd_frame.txt", rmsd_frame, delimiter="/t")

    # np.savetxt('test.out', x, delimiter=',')   # X is an array
    # np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    # np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    return calibration


@_numba.jit(nopython=True, nogil=True)
def _fit_z_target(z, sx, sy, cx, cy):
    z2 = z * z
    z3 = z * z2
    z4 = z * z3
    z5 = z * z4
    z6 = z * z5
    wx = (
        cx[0] * z6
        + cx[1] * z5
        + cx[2] * z4
        + cx[3] * z3
        + cx[4] * z2
        + cx[5] * z
        + cx[6]
    )
    wy = (
        cy[0] * z6
        + cy[1] * z5
        + cy[2] * z4
        + cy[3] * z3
        + cy[4] * z2
        + cy[5] * z
        + cy[6]
    )
    return (sx**0.5 - wx**0.5) ** 2 + (
        sy**0.5 - wy**0.5
    ) ** 2  # Apparently this results in slightly more accurate z coordinates
    # (Huang et al. '08)
    # return (sx-wx)**2 + (sy-wy)**2


def fit_z(
    locs, 
    info, 
    calibration, 
    magnification_factor, 
    filter=2
):
    cx = _np.array(calibration["X Coefficients"])
    cy = _np.array(calibration["Y Coefficients"])
    z = _np.zeros_like(locs.x)
    square_d_zcalib = _np.zeros_like(z)
    sx = locs.sx
    sy = locs.sy
    for i in range(len(z)):
        result = _minimize_scalar(_fit_z_target, args=(sx[i], sy[i], cx, cy))
        z[i] = result.x
        square_d_zcalib[i] = result.fun
    z *= magnification_factor
    locs = _lib.append_to_rec(locs, z, "z")
    locs = _lib.append_to_rec(locs, _np.sqrt(square_d_zcalib), "d_zcalib")
    locs = _lib.ensure_sanity(locs, info)
    return filter_z_fits(locs, filter)


def fit_z_parallel(
    locs, 
    info, 
    calibration, 
    magnification_factor, 
    filter=2, 
    asynch=False,
):
    n_workers = min(
        60, max(1, int(0.75 * _multiprocessing.cpu_count()))
    ) # Python crashes when using >64 cores
    n_locs = len(locs)
    n_tasks = 100 * n_workers
    spots_per_task = [
        int(n_locs / n_tasks + 1) if _ < n_locs % n_tasks else int(n_locs / n_tasks)
        for _ in range(n_tasks)
    ]
    start_indices = _np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = _ProcessPoolExecutor(n_workers)
    for i, n_locs_task in zip(start_indices, spots_per_task):
        fs.append(
            executor.submit(
                fit_z,
                locs[i : i + n_locs_task],
                info,
                calibration,
                magnification_factor,
                filter=0,
            )
        )
    if asynch:
        return fs
    with _tqdm(total=n_tasks, unit="task") as progress_bar:
        for f in _futures.as_completed(fs):
            progress_bar.update()
    return locs_from_futures(fs, filter=filter)


def locs_from_futures(futures, filter=2):
    locs = [_.result() for _ in futures]
    locs = _np.hstack(locs).view(_np.recarray)
    return filter_z_fits(locs, filter)


def filter_z_fits(locs, range):
    if range > 0:
        rmsd = _np.sqrt(_np.nanmean(locs.d_zcalib**2))
        locs = locs[locs.d_zcalib <= range * rmsd]
    return locs
