"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in a frame sequence

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""
import numpy as _np
import dask.array as _da
import numba as _numba
import multiprocessing as _multiprocessing
import ctypes as _ctypes
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import threading as _threading
from itertools import chain as _chain
import matplotlib.pyplot as _plt
from . import gaussmle as _gaussmle
from . import io as _io
from . import postprocess as _postprocess
from . import __main__ as main
from . import multiproc as _pmultip
import os
from datetime import datetime
import time
from sqlalchemy import create_engine
import pandas as pd

_C_FLOAT_POINTER = _ctypes.POINTER(_ctypes.c_float)
LOCS_DTYPE = [
    ("frame", "u4"),
    ("x", "f4"),
    ("y", "f4"),
    ("photons", "f4"),
    ("sx", "f4"),
    ("sy", "f4"),
    ("bg", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
    ("net_gradient", "f4"),
    ("likelihood", "f4"),
    ("iterations", "i4"),
]

MEAN_COLS = ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy',
           'ellipticity', 'net_gradient', 'z', 'd_zcalib']
SET_COLS = ['Frames', 'Height', 'Width', 'Box Size', 'Min. Net Gradient', 'Pixelsize']
DRIFT_COLS = ['Drift X', 'Drift Y']

_plt.style.use("ggplot")


@_numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(frame, box):
    """ Finds pixels with maximum value within a region of interest """
    Y, X = frame.shape
    maxima_map = _np.zeros(frame.shape, _np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half: i + box_half + 1,
                j - box_half: j + box_half + 1,
            ]
            flat_max = _np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = _np.where(maxima_map)
    return y, x


@_numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(frame, y, x, i):
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@_numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(frame, y, x, box, uy, ux):
    box_half = int(box / 2)
    ng = _np.zeros(len(x), dtype=_np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(
                range(xi - box_half, xi + box_half + 1)
            ):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += (
                        gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
                    )
    return ng


@_numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(image, minimum_ng, box):
    y, x = local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = _np.zeros((box, box), dtype=_np.float32)
    uy = _np.zeros((box, box), dtype=_np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = _np.sqrt(ux ** 2 + uy ** 2)
    ux /= unorm
    uy /= unorm
    ng = net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def identify_in_frame(frame, minimum_ng, box, roi=None):
    # print('start identifying in frame')
    if roi is not None:
        frame = frame[roi[0][0]: roi[1][0], roi[0][1]: roi[1][1]]
    image = _np.float32(frame)  # otherwise numba goes crazy
    # print('start identifying in image')
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    # print('done identifying in image')
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    # print('done identifying in frame')
    return y, x, net_gradient

def identify_frame(frame, minimum_ng, box, frame_number, roi=None, resultqueue=None):
    # print('start identifying frame')
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
    # print('got result of "in frame"')
    # print('result x', x)
    # print('len {:d}'.format(len(x)))
    frame = frame_number * _np.ones(len(x))
    # print('done identifying frame')
    result = _np.rec.array(
        (frame, x, y, net_gradient),
        dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")],
    )
    if resultqueue is not None:
        resultqueue.put(result)
    return result

def identify_by_frame_number(movie, minimum_ng, box, frame_number, roi=None, lock=None):
    if lock is not None:
        with lock:
            frame = movie[frame_number]
    else:
        frame = movie[frame_number]
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
    frame = frame_number * _np.ones(len(x))
    return _np.rec.array(
        (frame, x, y, net_gradient),
        dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")],
    )


def _identify_worker(movie, current, minimum_ng, box, roi, lock):
    n_frames = len(movie)
    identifications = []
    while True:
        with lock:
            index = current[0]
            if index == n_frames:
                return identifications
            current[0] += 1
        identifications.append(
            identify_by_frame_number(movie, minimum_ng, box, index, roi, lock)
        )
    return identifications

def _identify_worker_directload(
        framequeue, resultqueue, framerequestqueue, frameloadqueue,
        framefinishedqueue,
        minimum_ng, box, roi):
    # innerlogger = _pmultip.worker_configurer(logqueue, worker_idx)
    time_getfridx = 0
    time_loadimg = 0
    time_identify = 0
    time_queues = 0
    i = 0
    while not framequeue.empty():
        tic = time.time()
        try:
            frame_idx = framequeue.get(timeout=.5)
        except:
            print('frame queue is empty')
            break
            # resultqueue.put('done')
            # return True
        time_getfridx += time.time() - tic
        tic = time.time()
        framerequestqueue.put(frame_idx)
        frame = frameloadqueue.get(timeout=5)
        time_loadimg += time.time() - tic
        tic = time.time()
        # frame = movie[frame_idx]
        result = identify_frame(frame, minimum_ng, box, frame_idx, roi)
        time_identify += time.time() - tic
        tic = time.time()
        resultqueue.put(result)
        framefinishedqueue.put(frame_idx)
        # print('finished with frame', frame_idx)
        time_queues += time.time() - tic
        tic = time.time()
        i += 1
    # print('worker is done')
    resultqueue.put('done')
    totaltime = time_getfridx + time_loadimg + time_identify + time_queues
    print('identification of {:d} imgs took {:.1f} %  getfridx, {:.1f}% loadimg, {:.1f}% identify, {:.1f}% queues'.format(
        i, 100*time_getfridx/totaltime, 100*time_loadimg/totaltime,
        100*time_identify/totaltime, 100*time_queues/totaltime
    ))
    return True

def _identify_start_processes(movie, minimum_ng, box, roi, n_workers,
        current, lock, debug_sglproc=False):
    # innerlogger = _pmultip.worker_configurer(logqueue, worker_idx)

    # create a queue from which processes can poll the next frame to work on.
    n_frames = len(movie)
    framequeue = _multiprocessing.Queue(maxsize=n_frames)
    for idx in range(n_frames):
        framequeue.put(idx)
    framefinishedqueue = _multiprocessing.Queue()

    # create processes
    tic = time.time()
    workprocesses = []
    resultqueues = []
    framerequestqueues = []
    frameloadqueues = []
    if debug_sglproc:
        resultqueue = _multiprocessing.Queue()
        _identify_worker_directload(movie, framequeue, resultqueue,
              minimum_ng, box, roi)
        print('identify worker ran through without multiprocessing')
        workprocess = _multiprocessing.Process(target=_np.min, args=([3,4 ,5]))
        workprocesses.append(workprocess)
        print('created dummy workprocess')
        resultqueues.append(resultqueue)
    else:
        for worker_idx in range(n_workers):
            resultqueue = _multiprocessing.Queue()
            framerequestqueue = _multiprocessing.Queue()
            frameloadqueue = _multiprocessing.Queue()
            workprocess = _multiprocessing.Process(
                target=_identify_worker_directload,
                args=(framequeue, resultqueue, framerequestqueue,
                      frameloadqueue, framefinishedqueue,
                      minimum_ng, box, roi))
            workprocesses.append(workprocess)
            # print('created workprocess in worker {:d}'.format(worker_idx))
            resultqueues.append(resultqueue)
            framerequestqueues.append(framerequestqueue)
            frameloadqueues.append(frameloadqueue)

    frameload_abortqueue = _multiprocessing.Queue()
    frameloader_proc = _threading.Thread(
        target=dask_movie_load_worker,
        args=(movie, framerequestqueues, frameloadqueues, frameload_abortqueue)
    )
    frameloader_proc.start()
    framemonitor_abortqueue = _multiprocessing.Queue()
    framemonitor_thread = _threading.Thread(
        target=monitor_framefinishing,
        args=(framefinishedqueue, framemonitor_abortqueue, current, lock)
    )
    framemonitor_thread.start()

    for worker_idx, workprocess in enumerate(workprocesses):
        workprocess.start()
        # print('started workprocess {:d}'.format(worker_idx))

    return (workprocesses, resultqueues, frameload_abortqueue, frameloader_proc,
            framemonitor_abortqueue, framemonitor_thread)

def _identify_stop_processes(
        workprocesses, resultqueues, frameload_abortqueue, frameloader_proc,
        framemonitor_abortqueue, framemonitor_thread):
    # identifications_dict = {}
    identifications = []
    for idx, (workprocess, resultqueue) in enumerate(
            zip(workprocesses, resultqueues)):
        # print('waiting for worker {:d} to finish.'.format(idx))
        # print('collecting results.')
        while True:  # not resultqueue.empty():
            # frame, workresult = resultqueue.get().items()
            # identifications_dict[frame] = workresult
            try:
                workresult = resultqueue.get(timeout=.1)
                if not isinstance(workresult, str):
                    identifications.append(workresult)
                else:
                    # print(workresult)  # says: 'done'
                    break
            except:
                # print('Error getting result in worker {:d}'.format(idx))
                # break
                time.sleep(1)
        # resultqueue.close()
        workprocess.join()
    # print('got all results, now frameloader')

    # stop the frame loader
    frameload_abortqueue.put('stop')
    frameloader_proc.join()
    # print('stopping framemonitor')
    # stop the frame monitor
    framemonitor_abortqueue.put('stop')
    framemonitor_thread.join()

    # identifications = []
    # for i in range(n_frames):
    #     identifications.append(identifications_dict[i])

    return identifications

def dask_movie_load_worker(movie, requestqueues, framequeues, abortqueue):
    """load frames from a dask movie as a service running in a separate
    Thread or Process, so the dask movie doesn't have to be transferred to
    each process (which takes quite long)
    Args:
        movie : A PicassoMovie with use_dask==True (e.g. ND2Movie)
            the source
        requestqueues : list of queues
            a list of queues, where other processes request frame numbers
        framequeues : list of queues, same len as requestqueues
            a list of queues, where this worker posts the frames loaded
    """
    while True:
        for requestqueue, framequeue in zip(requestqueues, framequeues):
            try:
                frame_requested = requestqueue.get(timeout=.1)
                if isinstance(frame_requested, int):
                    frame = movie[frame_requested]
                    framequeue.put(frame)
            except:
                pass
        # check whether to end this worker
        try:
            abortresult = abortqueue.get(timeout=.1)
            # print('frameloader abort queue got: ', abortresult)
            return
        except:
            pass

def monitor_framefinishing(framefinishedqueue, abortqueue, current, lock):
    """Monitor how frames get finished
    Args:
        framefinishedqueue : queue
            here, processes send frames that have been finished processing
        abortqueue : queue
            a queue where something is sent if this Thread should end
        current : array, len 1
            the current frame
        lock : a threading lock
            for writing current
    """
    while True:
        try:
            frame_finished = framefinishedqueue.get(timeout=.1)
            if isinstance(frame_finished, int):
                with lock:
                    current[0] = current[0]+1  # frame_finished
        except:
            pass
        # check whether to end this worker
        try:
            abortresult = abortqueue.get(timeout=.1)
            # print('framemonitor abort queue got: ', abortresult)
            return
        except:
            pass


def identifications_from_futures(futures):
    if isinstance(futures, list):
        ids_list_of_lists = [_.result() for _ in futures]
    elif isinstance(futures, DaskFuture):
        ids_list_of_lists = _identify_stop_processes(**futures.get_all())
    ids_list = list(_chain(*ids_list_of_lists))
    ids = _np.hstack(ids_list).view(_np.recarray)
    ids.sort(kind="mergesort", order="frame")
    return ids


class DaskFuture:
    """Data structure to hold queues etc for identifying dask arrays
    """
    def __init__(
            self,
            workprocesses, resultqueues, frameload_abortqueue, frameloader_proc,
            framemonitor_abortqueue, framemonitor_thread):
        self.workprocesses = workprocesses
        self.resultqueues = resultqueues
        self.frameload_abortqueue = frameload_abortqueue
        self.frameloader_proc = frameloader_proc
        self.framemonitor_abortqueue = framemonitor_abortqueue
        self.framemonitor_thread = framemonitor_thread

    def get_all(self):
        return {
            'workprocesses': self.workprocesses,
            'resultqueues': self.resultqueues,
            'frameload_abortqueue': self.frameload_abortqueue,
            'frameloader_proc': self.frameloader_proc,
            'framemonitor_abortqueue': self.framemonitor_abortqueue,
            'framemonitor_thread': self.framemonitor_thread,
            }


def identify_async(movie, minimum_ng, box, roi=None):
    "Use the user settings to define the number of workers that are being used"
    settings = _io.load_user_settings()
    try:
        cpu_utilization = settings["Localize"]["cpu_utilization"]
        if cpu_utilization >= 1:
            cpu_utilization = 1
    except Exception as e:
        print(e)
        print(
            "An Error occured. Setting cpu_utilization to 0.8"
        )  # TODO at some point re-write this
        cpu_utilization = 0.8
        settings["Localize"]["cpu_utilization"] = cpu_utilization
        _io.save_user_settings(settings)

    n_workers = max(1, int(cpu_utilization * _multiprocessing.cpu_count()))

    # logqueue, listener = _pmultip.start_multiprocesslogging('identify.log')
    lock = _threading.Lock()
    current = [0]
    if movie.use_dask:
        (workprocesses, resultqueues, frameload_abortqueue, frameloader_proc,
         framemonitor_abortqueue, framemonitor_thread
         ) = _identify_start_processes(
            movie, minimum_ng, box, roi, n_workers, current, lock)
        f = DaskFuture(
            workprocesses, resultqueues, frameload_abortqueue, frameloader_proc,
            framemonitor_abortqueue, framemonitor_thread)
    else:
        executor = _ThreadPoolExecutor(n_workers)
        f = [
            executor.submit(
                _identify_worker, movie, current, minimum_ng, box, roi, lock
            )
            for _ in range(n_workers)
        ]
        executor.shutdown(wait=False)
    return current, f


def identify(movie, minimum_ng, box, threaded=True):
    print('threaded', threaded)
    if threaded:
        current, futures = identify_async(movie, minimum_ng, box)
        identifications = [_.result() for _ in futures]
        identifications = [_np.hstack(_) for _ in identifications]
    else:
        identifications = [
            identify_by_frame_number(movie, minimum_ng, box, i)
            for i in range(len(movie))
        ]
    return _np.hstack(identifications).view(_np.recarray)


@_numba.jit(nopython=True, cache=False)
def _cut_spots_numba(movie, ids_frame, ids_x, ids_y, box):
    n_spots = len(ids_x)
    r = int(box / 2)
    spots = _np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[id] = movie[frame, yc - r: yc + r + 1, xc - r: xc + r + 1]
    return spots


@_numba.jit(nopython=True, cache=False)
def _cut_spots_frame(
    frame, frame_number, ids_frame, ids_x, ids_y, r, start, N, spots
):
    for j in range(start, N):
        if ids_frame[j] > frame_number:
            break
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r: yc + r + 1, xc - r: xc + r + 1]
    return j


@_numba.jit(nopython=True, cache=False)
def _cut_spots_daskmov(movie, l_mov, ids_frame, ids_x, ids_y, box, spots):
    """Cuts the spots out of a movie frame by frame.

    Args:
        movie : 3D array (t, x, y)
            the image data (can be dask or numpy array)
        l_mov : 1D array, len 1
            lenght of the movie (=t); in array to satisfy the combination of
            range() and guvectorization
        ids_frame, ids_x, ids_y : 1D array (k)
            spot positions in the image data. Length: number of spots
            identified
        box : uneven int
            the cut spot box size
        spots : 3D array (k, box, box)
            the cut spots
    Returns:
        spots : as above
            the image-data filled spots
    """
    r = int(box / 2)
    N = len(ids_frame)
    start = 0
    for frame_number in range(l_mov[0]):
        frame = movie[frame_number, :, :]
        start = _cut_spots_frame(
            frame,
            frame_number,
            ids_frame,
            ids_x,
            ids_y,
            r,
            start,
            N,
            spots,
        )
    return spots


def _cut_spots_framebyframe(movie, ids_frame, ids_x, ids_y, box, spots):
    """Cuts the spots out of a movie frame by frame.

    Args:
        movie : 3D array (t, x, y)
            the image data (can be dask or numpy array)
        ids_frame, ids_x, ids_y : 1D array (k)
            spot positions in the image data. Length: number of spots
            identified
        box : uneven int
            the cut spot box size
        spots : 3D array (k, box, box)
            the cut spots
    Returns:
        spots : as above
            the image-data filled spots
    """
    r = int(box / 2)
    N = len(ids_frame)
    start = 0
    for frame_number, frame in enumerate(movie):
        start = _cut_spots_frame(
            frame,
            frame_number,
            ids_frame,
            ids_x,
            ids_y,
            r,
            start,
            N,
            spots,
        )
    return spots


def _cut_spots(movie, ids, box):
    N = len(ids.frame)
    if isinstance(movie, _np.ndarray):
        return _cut_spots_numba(movie, ids.frame, ids.x, ids.y, box)
    # elif isinstance(movie, _io.ND2Movie):
    elif movie.use_dask:
        """ Assumes that identifications are in order of frames! """
        spots = _np.zeros((N, box, box), dtype=movie.dtype)
        spots = _da.apply_gufunc(
            _cut_spots_daskmov,
            '(p,n,m),(b),(k),(k),(k),(),(k,l,l)->(k,l,l)',
            movie.data, _np.array([len(movie)]), ids.frame, ids.x, ids.y, box, spots,
            output_dtypes=[movie.dtype], allow_rechunk=True).compute()
        return spots
    else:
        """ Assumes that identifications are in order of frames! """
        spots = _np.zeros((N, box, box), dtype=movie.dtype)
        spots = _cut_spots_framebyframe(
            movie, ids.frame, ids.x, ids.y, box, spots)
        return spots


def _to_photons(spots, camera_info):
    spots = _np.float32(spots)
    baseline = camera_info["baseline"]
    sensitivity = camera_info["sensitivity"]
    gain = camera_info["gain"]
    qe = camera_info["qe"]
    return (spots - baseline) * sensitivity / (gain * qe)


def get_spots(movie, identifications, box, camera_info):
    spots = _cut_spots(movie, identifications, box)
    return _to_photons(spots, camera_info)


def fit(
    movie,
    camera_info,
    identifications,
    box,
    eps=0.001,
    max_it=100,
    method="sigma",
):
    spots = get_spots(movie, identifications, box, camera_info)
    theta, CRLBs, likelihoods, iterations = _gaussmle.gaussmle(
        spots, eps, max_it, method=method
    )
    return locs_from_fits(
        identifications, theta, CRLBs, likelihoods, iterations, box
    )


def fit_async(
    movie,
    camera_info,
    identifications,
    box,
    eps=0.001,
    max_it=100,
    method="sigma",
):
    spots = get_spots(movie, identifications, box, camera_info)
    return _gaussmle.gaussmle_async(spots, eps, max_it, method=method)


def locs_from_fits(
    identifications, theta, CRLBs, likelihoods, iterations, box
):
    box_offset = int(box / 2)
    y = theta[:, 0] + identifications.y - box_offset
    x = theta[:, 1] + identifications.x - box_offset
    lpy = _np.sqrt(CRLBs[:, 0])
    lpx = _np.sqrt(CRLBs[:, 1])
    locs = _np.rec.array(
        (
            identifications.frame,
            x,
            y,
            theta[:, 2],
            theta[:, 5],
            theta[:, 4],
            theta[:, 3],
            lpx,
            lpy,
            identifications.net_gradient,
            likelihoods,
            iterations,
        ),
        dtype=LOCS_DTYPE,
    )
    locs.sort(kind="mergesort", order="frame")
    return locs


def localize(movie, info, parameters):
    print("localizing")
    identifications = identify(movie, parameters)
    return fit(movie, info, identifications, parameters["Box Size"])

def get_file_summary(file):

    base, ext = os.path.splitext(file)

    file_hdf = base + '_locs.hdf5'

    locs, info = _io.load_locs(file_hdf)

    summary = {}

    for col in MEAN_COLS:
        try:
            summary[col+'_mean'] = locs[col].mean()
            summary[col+'_std'] = locs[col].std()
        except ValueError:
            summary[col+'_mean'] = float('nan')
            summary[col+'_std'] = float('nan')

    for col in SET_COLS:
        col_ = col.lower()
        for inf in info:
            if col in inf:
                summary[col_] = inf[col]

    for col in SET_COLS:
        col_ = col.lower()
        if col_ not in summary:
            summary[col_] = float('nan')

    #Nena
    try:
        result, best_result = _postprocess.nena(locs, info)
        summary['nena_px'] = best_result
    except Exception as e:
        print(e)
        summary['nena_px'] = float('nan')

    summary['n_locs'] = len(locs)
    summary['locs_frame'] = len(locs)/summary['frames']

    drift_path = os.path.join(base + '_locs_undrift.hdf5')
    if os.path.isfile(drift_path):
        locs, info = _io.load_locs(drift_path)
        for col in DRIFT_COLS:
            col_ = col.lower()
            col_ = col_.replace(' ', '_')
            for inf in info:
                if col in inf:
                    summary[col_] = inf[col]

    for col in DRIFT_COLS:
        col_ = col.lower()
        col_ = col_.replace(' ', '_')
        if col_ not in summary:
            summary[col_] = float('nan')

    summary['filename'] = file
    summary['file_created'] = datetime.fromtimestamp(os.path.getmtime(file))
    summary['entry_created'] = datetime.now()

    return summary

def _db_filename():
    home = os.path.expanduser("~")
    return os.path.abspath(os.path.join(home, ".picasso", "app.db"))

def save_file_summary(summary):
    engine = create_engine("sqlite:///"+_db_filename(), echo=False)
    s  = pd.Series(summary, index=summary.keys()).to_frame().T
    s.to_sql("files", con=engine, if_exists="append", index=False)

def add_file_to_db(file):
    base, ext = os.path.splitext(file)
    out_path = base + "_locs.hdf5"

    try:
        main._undrift(out_path, 1000, display=False, fromfile=None)
    except Exception as e:
        print(e)
        print("Drift correction failed for {}".format(out_path))

    summary = get_file_summary(file)
    save_file_summary(summary)
