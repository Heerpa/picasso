"""
picasso.gui.localize
~~~~~~~~~~~~~~~~~~~~

Graphical user interface for localizing single molecules.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import glob
import os.path
import re
import sys
import threading
import time
from collections import UserDict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import imageio
from .. import (
    aim,
    CONFIG,
    gausslq,
    imageprocess,
    io,
    localize,
    lib,
    postprocess,
    spline,
    __version__,
    zfit,
)
from ..fitting import gaussfit_cuda, splinefit
from PyQt6 import QtCore, QtGui, QtWidgets
from playsound3 import playsound

GPU_FITTING_AVAILABLE = localize.GPU_FITTING_AVAILABLE
GPUSPLINE_INSTALLED = localize.GPUSPLINE_INSTALLED
CMAP_GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]
DEFAULT_PARAMETERS = {"Box Size": 7, "Min. Net Gradient": 5000}

# Distinct box colours for the cross-channel link overlay (grey is kept
# out of the palette so it reads as "unmatched"). tab20-style hues.
LINK_COLORS = [
    QtGui.QColor(*_rgb)
    for _rgb in (
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (188, 189, 34),
        (23, 190, 207),
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
        (247, 182, 210),
        (219, 219, 141),
        (158, 218, 229),
    )
]
LINK_UNMATCHED_COLOR = QtGui.QColor(150, 150, 150)


def _nearest_unique_match(
    pred_xy: np.ndarray, target_xy: np.ndarray, tol: float
) -> dict[int, int]:
    """Nearest-neighbour match within ``tol``.

    ``pred_xy`` are reference points predicted into a channel (via the
    calibration transform); ``target_xy`` are that channel's own detections.
    Returns ``{target_index: reference_index}`` with each target and each
    reference used at most once (closest pair wins) - the display counterpart
    of the per-frame pairing used in the signal re-registration.
    """
    pred_xy = np.asarray(pred_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    out: dict[int, int] = {}
    if len(pred_xy) == 0 or len(target_xy) == 0:
        return out
    dists = np.sqrt(
        ((pred_xy[:, None, :] - target_xy[None, :, :]) ** 2).sum(axis=2)
    )  # (n_reference, n_target)
    pairs = []
    for k in range(dists.shape[0]):
        j = int(np.argmin(dists[k]))
        if dists[k, j] <= tol:
            pairs.append((float(dists[k, j]), k, j))
    pairs.sort()
    used_ref: set[int] = set()
    used_target: set[int] = set()
    for _d, k, j in pairs:
        if k in used_ref or j in used_target:
            continue
        used_ref.add(k)
        used_target.add(j)
        out[j] = k
    return out


def _normalize_rect(
    rect: list,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """``[[y_a, x_a], [y_b, x_b]]`` -> ``((y_min, x_min), (y_max, x_max))``."""
    (ya, xa), (yb, xb) = rect
    return (min(ya, yb), min(xa, xb)), (max(ya, yb), max(xa, xb))


LINK_PHOTONS_TIP = (
    "2- to 6-channel fitting.\n\n"
    "CHECK to link photons + background across all channels. "
    "Appropriate when the channels share one emission.\n\n"
    "UNCHECK to decouple: each channel fits a free photon count"
    " + background, x/y/z shared. Adds photons_ch<c>, bg_ch<c> and"
    " rel_photons_ch<c> (the channel's share of the total photons)"
    " to the localizations."
)

# Fitting models offered in the GUI, decoupled from the optimizer. Each
# model maps its optimizer labels to the internal ``fit2D`` codes;
# models without an optimizer (e.g. averaging) declare a fixed ``code``
# and ``optimizers=None``. Add a new fitting algorithm by adding an
# entry here.
FIT_MODELS = {
    "2D elliptical Gaussian": {
        "optimizers": {"Least squares": "gausslq", "MLE": "gaussmle"},
    },
    "2D rotated elliptical Gaussian": {
        "optimizers": {
            "Least squares": "gausslq-rotated",
            "MLE": "gaussmle-rotated-gpu",
        },
    },
    "2D spherical Gaussian": {
        "optimizers": {
            "Least squares": "gausslq-spherical",
            "MLE": "gaussmle-spherical",
        },
    },
    "Experimental PSF (cubic spline)": {
        "optimizers": {
            "Least squares": "spline",
            "MLE": "spline-mle",
        },
        "needs_spline_calibration": True,
    },
    "Average of ROI": {
        "optimizers": None,
        "code": "avg",
    },
}
# The rotated elliptical Gaussian has a CPU least-squares implementation but
# its MLE optimizer is GPU-only. The cubic-spline PSF has both: the bare codes
# above run on the CPU (picasso.fitting.splinefit) and FitWorker appends "-gpu"
# when the GPU checkbox is ticked, exactly as for the Gaussian models.
if not GPU_FITTING_AVAILABLE:
    del FIT_MODELS["2D rotated elliptical Gaussian"]["optimizers"]["MLE"]


MODEL_TOOLTIP = (
    "Model fit to each identified spot:\n\n"
    "2D elliptical Gaussian: Gaussian with independent widths in x and y."
    " Standard choice for 2D data and required for 3D via astigmatism.\n\n"
    "2D rotated elliptical Gaussian: as above, plus a fitted rotation angle"
    " of the ellipse. Useful for tilted/anisotropic PSFs.\n\n"
    "2D spherical Gaussian: Gaussian with one common width for x and y.\n\n"
    "Experimental PSF (cubic spline): fits a cubic spline interpolation of"
    " a measured 3D PSF (requires a spline calibration file). Most accurate"
    " model and yields z directly, also for aberrated PSFs. Runs on the CPU;"
    " tick Use GPU to run it on the GPU instead (much faster).\n\n"
    "Average of ROI: reports the spot's center of mass and integrated "
    "intensity in the fit box."
)

OPTIMIZER_TOOLTIP = (
    "Optimizer used to fit the model to data:\n\n"
    "Least squares: minimizes the squared residuals between model and"
    " data. Fast and robust, but assumes Gaussian noise, so it is slightly"
    " biased for the Poisson (shot) noise of low-photon spots.\n\n"
    "MLE: maximum likelihood estimation with a Poisson noise model."
    " Statistically optimal (precision close to the Cramer-Rao lower"
    " bound) and the better choice for dim spots.\n\n"
    "Available optimizers may depend on the selected model; some are only"
    " implemented on the GPU."
)


def _fit_code(model: str, optimizer: str) -> str:
    """Resolve a (model, optimizer) selection to an internal ``fit2D`` code."""
    entry = FIT_MODELS[model]
    if entry["optimizers"] is None:
        return entry["code"]
    return entry["optimizers"][optimizer]


# Fit codes with both a CPU and a GPU implementation, selected by the "Use
# GPU" checkbox rather than by the model/optimizer comboboxes. The remaining
# codes are either GPU-only (they already end in "-gpu") or CPU-only.
_GPU_CAPABLE_CODES = frozenset(
    {
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gaussmle",
        "gaussmle-spherical",
        "spline",
        "spline-mle",
    }
)


def _effective_fit_code(code: str, use_gpu: bool) -> str:
    """The ``fit2D`` code a (code, GPU checkbox) pair actually runs."""
    if use_gpu and code in _GPU_CAPABLE_CODES:
        return code + "-gpu"
    return code


# Fit codes that iterate, and the default convergence schedule of each. Every
# method except "avg" is here: all of them run an iterative solver, so all of
# them honor the convergence criterion and the maximum-iteration count.
_GAUSSMLE_SCHEDULE = (0.001, 100)
_GAUSSLQ_SCHEDULE = (gausslq.TOLERANCE, gausslq.MAX_ITERATIONS)
_GAUSS_GPU_SCHEDULE = (gaussfit_cuda.TOLERANCE, gaussfit_cuda.MAX_ITERATIONS)
_SPLINE_SCHEDULE = (
    splinefit.TOLERANCE_MULTI_START,
    splinefit.MAX_ITERATIONS_MULTI_START,
)
_CONVERGENCE_DEFAULTS = {
    "gausslq": _GAUSSLQ_SCHEDULE,
    "gausslq-spherical": _GAUSSLQ_SCHEDULE,
    "gausslq-rotated": _GAUSSLQ_SCHEDULE,
    "gausslq-gpu": _GAUSS_GPU_SCHEDULE,
    "gausslq-spherical-gpu": _GAUSS_GPU_SCHEDULE,
    "gausslq-rotated-gpu": _GAUSS_GPU_SCHEDULE,
    "gaussmle": _GAUSSMLE_SCHEDULE,
    "gaussmle-spherical": _GAUSSMLE_SCHEDULE,
    "gaussmle-gpu": _GAUSS_GPU_SCHEDULE,
    "gaussmle-spherical-gpu": _GAUSS_GPU_SCHEDULE,
    "gaussmle-rotated-gpu": _GAUSS_GPU_SCHEDULE,
    "spline": _SPLINE_SCHEDULE,
    "spline-mle": _SPLINE_SCHEDULE,
    "spline-gpu": _SPLINE_SCHEDULE,
    "spline-mle-gpu": _SPLINE_SCHEDULE,
}
_CONVERGENCE_CODES = frozenset(_CONVERGENCE_DEFAULTS)


# Steps allotted to each file in the load progress dialog, so the bar can
# advance smoothly within a file from the loader's per-page reports.
PROGRESS_RESOLUTION = 1000


@dataclass
class Channel:
    """Per-channel state for multichannel localization.

    Localize keeps all loaded channels in ``Window.channels`` and mirrors
    the *active* channel into the flat ``Window`` attributes (``movie``,
    ``info``, ``identifications`` ...) so the existing single-movie
    identify/fit/save/draw code keeps operating on the active channel
    unchanged. Switching channels snapshots the flat state (plus the
    Parameters dialog values) into the old ``Channel`` and restores the
    new one.

    All channels' ``movie`` objects stay live simultaneously, so a future
    across-channel fitting algorithm can read every channel at once.
    """

    movie: object = None
    info: list = field(default_factory=list)
    path: str = ""
    name: str = "Channel 0"
    identifications: object = None
    locs: object = None
    locs_display: object = None
    ready_for_fit: bool = False
    last_identification_info: dict | None = None
    extra_info: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


def _sanitize_filename(name: str) -> str:
    """Turn a channel name into a filename-safe suffix."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "channel"


class _LoadCancelledError(Exception):
    """Raised inside the loader's progress callback to abort a cancelled
    load mid-file (the io calls are otherwise uninterruptible)."""


class MovieLoadWorker(QtCore.QObject):
    """Load several movie files off the GUI thread, one file per channel.

    Loading large movies on the main thread blocks Qt's event loop, so the
    window stops repainting and responding while several files are read.
    This worker runs ``io.load_movie`` for each path on a background
    thread instead.

    The ``prompt_info`` callbacks show modal dialogs and therefore *must*
    run on the GUI thread. When a loader needs one, the worker emits
    ``prompt_requested`` (delivered to the main thread via a queued
    connection) and blocks on a ``threading.Event`` until the main thread
    has filled the answer into the shared ``holder`` dict and released it.
    """

    progress = QtCore.pyqtSignal(int, str)  # index, filename
    # sub-file progress within the current file: (done, total) pages
    subprogress = QtCore.pyqtSignal(int, int)
    # callback, (args, kwargs), holder dict for the return value
    prompt_requested = QtCore.pyqtSignal(object, object, object)
    finished = QtCore.pyqtSignal(list, list, list)  # movies, infos, paths
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self, paths: list[str], prompt_for_path, load_all: bool = False
    ) -> None:
        super().__init__()
        self.paths = paths
        self._prompt_for_path = prompt_for_path
        # When True, each path is read with ``io.load_movie_all`` (every
        # channel of one multichannel file); otherwise ``io.load_movie``
        # loads one channel per file.
        self.load_all = load_all
        self._prompt_event = threading.Event()
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation; takes effect before the next file."""
        self._cancelled = True
        # Release the worker if it is currently blocked on a prompt.
        self._prompt_event.set()

    def _proxy_prompt(self, callback):
        """Wrap a GUI prompt callback so the dialog runs on the main
        thread while the worker thread blocks for the result."""

        def wrapper(*args, **kwargs):
            holder = {}
            self._prompt_event.clear()
            self.prompt_requested.emit(callback, (args, kwargs), holder)
            self._prompt_event.wait()
            return holder.get("result")

        return wrapper

    def run(self) -> None:
        movies, infos, paths = [], [], []
        try:
            for i, path in enumerate(self.paths):
                if self._cancelled:
                    break
                self.progress.emit(i, os.path.basename(path))
                prompt = self._proxy_prompt(self._prompt_for_path(path))

                # Called (queued to the GUI thread) as io scans the
                # file's IFDs, so the bar advances smoothly within a
                # file. It is also the only code of ours that runs
                # *during* the otherwise-blocking io call, so it doubles
                # as the mid-file cancellation point.
                def report(done: int, total: int) -> None:
                    if self._cancelled:
                        raise _LoadCancelledError
                    self.subprogress.emit(done, total)

                if self.load_all:
                    result = io.load_movie_all(
                        path, prompt_info=prompt, progress=report
                    )
                    if result is None:
                        continue
                    file_movies, file_infos = result
                    for movie, info in zip(file_movies, file_infos):
                        movies.append(movie)
                        infos.append(info)
                        paths.append(path)
                else:
                    result = io.load_movie(
                        path, prompt_info=prompt, progress=report
                    )
                    if result is None:
                        continue
                    movie, info = result
                    movies.append(movie)
                    infos.append(info)
                    paths.append(path)
        except Exception as e:  # noqa: BLE001 - reported to the GUI
            if not self._cancelled:
                self.failed.emit(str(e))
                return
            movies, infos, paths = [], [], []
        if self._cancelled:
            # The file being read when the user cancelled has still been
            # loaded to completion (the blocking io call cannot be
            # interrupted); discard it instead of delivering it.
            movies, infos, paths = [], [], []
        self.finished.emit(movies, infos, paths)


class RubberBand(QtWidgets.QRubberBand):
    """Red rubber band for selecting ROI."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(QtWidgets.QRubberBand.Shape.Rectangle, parent)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Change the color of the rubber band."""
        painter = QtGui.QPainter(self)
        color = QtGui.QColor(QtCore.Qt.GlobalColor.blue)
        painter.setPen(QtGui.QPen(color))
        rect = event.rect()
        rect.setHeight(int(rect.height() - 1))
        rect.setWidth(int(rect.width() - 1))
        painter.drawRect(rect)


class View(QtWidgets.QGraphicsView):
    """Central widget which shows ``Scene`` objects of individual
    frames.

    ...

    Attributes
    ----------
    hscrollbar, vscrollbar : QtWidgets.QScrollBar
        Horizontal and vertical scroll bars.
    pan : bool
        Whether the view is currently panned.
    pan_start_x, pan_start_y : int
        Starting position of the pan gesture.
    rubberband : QtWidgets.QRubberBand
        Transient rubber band shown while dragging a new ROI.
    rois : list
        Regions of interest (ROIs) selected by the user. Each ROI is
        ``[[y_min, x_min], [y_max, x_max]]``. The ROIs are kept disjoint
        (overlapping selections are clipped via
        ``localize.clip_rois``). An empty list means the whole frame.
    selected_roi : int or None
        Index of the ROI currently highlighted (selected in the
        parameters dialog table), or None.
    roi_end : QtCore.QPoint
        End point of the ROI being dragged.
    window : QtWidgets.QMainWindow
        Reference to the main window.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setAcceptDrops(True)
        self.pan = False
        self.hscrollbar = self.horizontalScrollBar()
        self.hscrollbar.valueChanged.connect(self.on_scroll)
        self.vscrollbar = self.verticalScrollBar()
        self.vscrollbar.valueChanged.connect(self.on_scroll)
        self.rubberband = RubberBand(self)
        self.rois = []
        self.selected_roi = None
        # Split-FOV region mode: ROIs are equal-size rectangular channels of one
        # movie. The first region drawn fixes the size (derived live from the
        # existing regions, so clearing them frees the size again); further
        # regions snap to it, and an existing region can be dragged (moved) to
        # fine-tune its registration. Toggled by ``window.set_split_fov_mode``.
        self.split_fov_mode = False
        self._moving_roi = None  # index of the region being dragged
        self._move_anchor = None  # (scene_dy, scene_dx) press offset in region
        # A double click fires press/release/doubleClick/release; this flag lets
        # the trailing release be ignored so deleting a region does not
        # immediately re-add one at the same spot.
        self._suppress_release = False

    def _frame_shape(self) -> tuple[int, int] | None:
        """(`height`, `width`) of the current movie frame, or None."""
        movie = getattr(self.window, "movie", None)
        if movie is None:
            return None
        return int(movie.shape[1]), int(movie.shape[2])

    def _region_size(self) -> tuple[int, int] | None:
        """Shared (`height`, `width`) of the split-FOV regions, taken from the
        first (reference) region, or None when there are no regions yet.

        Derived live rather than stored so that removing every region - by any
        means (double-click, the ROI field, the numeric dialog) - frees the
        size again and the next drawn region can define a new one."""
        if not self.rois:
            return None
        (y_min, x_min), (y_max, x_max) = self.rois[0]
        return (y_max - y_min, x_max - x_min)

    def _roi_at(self, px: float, py: float) -> int | None:
        """Index of the smallest region containing scene point (px, py)."""
        containing = [
            i
            for i, ((y_min, x_min), (y_max, x_max)) in enumerate(self.rois)
            if y_min <= py <= y_max and x_min <= px <= x_max
        ]
        if not containing:
            return None
        return min(
            containing,
            key=lambda i: (
                (self.rois[i][1][0] - self.rois[i][0][0])
                * (self.rois[i][1][1] - self.rois[i][0][1])
            ),
        )

    def _add_region(self, y0: int, x0: int) -> None:
        """Append a region of the established size with top-left (y0, x0),
        clamped inside the frame; selects it. No clipping (channels may abut).
        """
        size = self._region_size()
        if size is None:
            return
        h, w = size
        shape = self._frame_shape()
        if shape is not None:
            fh, fw = shape
            y0 = int(min(max(0, y0), max(0, fh - h)))
            x0 = int(min(max(0, x0), max(0, fw - w)))
        self.rois.append([[int(y0), int(x0)], [int(y0 + h), int(x0 + w)]])
        self.selected_roi = len(self.rois) - 1
        self.window.parameters_dialog.update_roi_display()

    def _move_region(self, idx: int, y0: int, x0: int) -> None:
        """Translate region ``idx`` so its top-left is (y0, x0), clamped."""
        (y_min, x_min), (y_max, x_max) = self.rois[idx]
        h, w = y_max - y_min, x_max - x_min
        shape = self._frame_shape()
        if shape is not None:
            fh, fw = shape
            y0 = int(min(max(0, y0), max(0, fh - h)))
            x0 = int(min(max(0, x0), max(0, fw - w)))
        self.rois[idx] = [[int(y0), int(x0)], [int(y0 + h), int(x0 + w)]]

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Start either a rubber band for selecting a ROI or panning the
        view."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.split_fov_mode:
                scene_pos = self.mapToScene(event.pos())
                idx = self._roi_at(scene_pos.x(), scene_pos.y())
                if idx is not None:
                    # begin moving an existing region
                    self._moving_roi = idx
                    self.selected_roi = idx
                    (y_min, x_min), _ = self.rois[idx]
                    self._move_anchor = (
                        scene_pos.y() - y_min,
                        scene_pos.x() - x_min,
                    )
                    self.window.parameters_dialog.update_roi_display()
                    self.window.draw_frame()
                    return
            self.roi_origin = QtCore.QPoint(event.pos())
            self.rubberband.setGeometry(
                QtCore.QRect(self.roi_origin, QtCore.QSize())
            )
            self.rubberband.show()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.pan = True
            self.pan_start_x = event.pos().x()
            self.pan_start_y = event.pos().y()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Update the rubber band or pan the view."""
        if (
            self.split_fov_mode
            and self._moving_roi is not None
            and event.buttons() == QtCore.Qt.MouseButton.LeftButton
        ):
            scene_pos = self.mapToScene(event.pos())
            dy, dx = self._move_anchor
            self._move_region(
                self._moving_roi,
                int(round(scene_pos.y() - dy)),
                int(round(scene_pos.x() - dx)),
            )
            self.window.draw_frame()
            return
        if event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.rubberband.setGeometry(
                QtCore.QRect(self.roi_origin, event.pos())
            )
        if self.pan:
            self.hscrollbar.setValue(
                self.hscrollbar.value() - event.pos().x() + self.pan_start_x
            )
            self.vscrollbar.setValue(
                self.vscrollbar.value() - event.pos().y() + self.pan_start_y
            )
            self.pan_start_x = event.pos().x()
            self.pan_start_y = event.pos().y()
            self.window.draw_frame()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Add the dragged ROI (clipping against existing ones) or stop
        panning the view."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._suppress_release:
                # trailing release of a double click that deleted a region
                self._suppress_release = False
                self.rubberband.hide()
                event.accept()
                return
            if self.split_fov_mode and self._moving_roi is not None:
                # finished dragging an existing region
                self._moving_roi = None
                self._move_anchor = None
                self.window.parameters_dialog.update_roi_display()
                self.window.draw_frame()
                return
            self.roi_end = QtCore.QPoint(event.pos())
            self.rubberband.hide()
            dx = abs(self.roi_end.x() - self.roi_origin.x())
            dy = abs(self.roi_end.y() - self.roi_origin.y())
            if self.split_fov_mode:
                self._release_split_fov_region(dx, dy)
                self.window.draw_frame()
                return
            if dx >= 10 and dy >= 10:
                roi_points = (
                    self.mapToScene(self.roi_origin),
                    self.mapToScene(self.roi_end),
                )
                new_roi = [[int(_.y()), int(_.x())] for _ in roi_points]
                box = self.window.parameters.get("Box Size", 7)
                self.rois = localize.clip_rois(
                    self.rois + [new_roi], min_size=box
                )
                self.window.parameters_dialog.update_roi_display()
            self.window.draw_frame()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.pan = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            event.ignore()

    def _release_split_fov_region(self, dx: int, dy: int) -> None:
        """Finish a left-drag/click in split-FOV mode: the first region (a real
        drag) fixes the shared size; later releases drop a same-size region
        centred on the release point."""
        self.rubberband.hide()
        origin = self.mapToScene(self.roi_origin)
        end = self.mapToScene(self.roi_end)
        size = self._region_size()
        if size is None:
            # no regions yet: this drag defines the shared size (a real drag)
            if dx < 10 or dy < 10:
                return
            y0, y1 = sorted((int(origin.y()), int(end.y())))
            x0, x1 = sorted((int(origin.x()), int(end.x())))
            self.rois.append([[y0, x0], [y1, x1]])
            self.selected_roi = len(self.rois) - 1
            self.window.parameters_dialog.update_roi_display()
        else:
            h, w = size
            cy = (origin.y() + end.y()) / 2.0
            cx = (origin.x() + end.x()) / 2.0
            self._add_region(
                int(round(cy - h / 2.0)), int(round(cx - w / 2.0))
            )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Arrow keys nudge the selected split-FOV region by 1 px for
        pixel-precise channel registration."""
        deltas = {
            QtCore.Qt.Key.Key_Left: (0, -1),
            QtCore.Qt.Key.Key_Right: (0, 1),
            QtCore.Qt.Key.Key_Up: (-1, 0),
            QtCore.Qt.Key.Key_Down: (1, 0),
        }
        if (
            self.split_fov_mode
            and self.selected_roi is not None
            and self.selected_roi < len(self.rois)
            and event.key() in deltas
        ):
            idx = self.selected_roi
            (y0, x0), _ = self.rois[idx]
            ddy, ddx = deltas[event.key()]
            self._move_region(idx, y0 + ddy, x0 + ddx)
            self.window.parameters_dialog.update_roi_display()
            self.window.draw_frame()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Remove the ROI under the cursor on a (left) double click. If
        several ROIs contain the point, the smallest one is removed."""
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self.rois:
            event.ignore()
            return
        scene_pos = self.mapToScene(event.pos())
        idx = self._roi_at(scene_pos.x(), scene_pos.y())
        if idx is None:
            event.ignore()
            return
        del self.rois[idx]
        self.selected_roi = None
        # the release that closes this double click must not re-add a region
        self._suppress_release = True
        # In split-FOV mode the shared size is derived from the remaining
        # regions (see ``_region_size``), so removing all of them frees it.
        self.window.parameters_dialog.update_roi_display()
        self.window.draw_frame()
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Zoom in/out with the mouse wheel, centered on the cursor."""
        scale = 1.008 ** (-event.angleDelta().y())
        self.window.zoom(scale, anchor=event.position().toPoint())

    def on_scroll(self) -> None:
        """Redraw the frame if scale bar is shown."""
        # draw_frame() rebuilds the scene, which can change the scrollbar
        # values and re-fire valueChanged; skip while a draw is in progress
        # to avoid unbounded re-entrant recursion.
        if self.window._drawing_frame:
            return
        if self.window.scalebar_action.isChecked():
            self.window.draw_frame()


class Scene(QtWidgets.QGraphicsScene):
    """Render individual frames, displayed in a ``View`` widget."""

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.window = window
        self.dragMoveEvent = self.dragEnterEvent

    def path_from_drop(self, event: QtGui.QDropEvent) -> tuple[str, str]:
        """Extract path of the dropped file."""
        url = event.mimeData().urls()[0]
        path = url.toLocalFile()
        base, extension = os.path.splitext(path)
        return path, extension

    def drop_has_valid_url(self, event: QtGui.QDropEvent) -> bool:
        """Check if the dropped file has a valid extension."""
        if not event.mimeData().hasUrls():
            return False
        path, extension = self.path_from_drop(event)

        if extension.lower() not in io.MOVIE_EXTENSIONS:
            return False
        return True

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept the file dragged over the widget if it has a valid
        extension."""
        if self.drop_has_valid_url(event):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Loads when dropped into the scene."""
        path, ext = self.path_from_drop(event)
        self.window.open(path)


class FitMarker(QtWidgets.QGraphicsItemGroup):
    """Marker showing fitted position."""

    def __init__(
        self,
        x: float,
        y: float,
        size: float,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        L = size / 2
        line1 = QtWidgets.QGraphicsLineItem(x - L, y - L, x + L, y + L)
        line1.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
        self.addToGroup(line1)
        line2 = QtWidgets.QGraphicsLineItem(x - L, y + L, x + L, y - L)
        line2.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
        self.addToGroup(line2)


class OddSpinBox(QtWidgets.QSpinBox):
    """Spinbox allowing only odd numbers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSingleStep(2)
        self.editingFinished.connect(self.on_editing_finished)

    def on_editing_finished(self):
        value = self.value()
        if value % 2 == 0:
            self.setValue(int(value + 1))


class CamSettingComboBox(QtWidgets.QComboBox):
    """Combo box for selecting camera settings which are relevant for
    sensitivity.

    Datasheets for different camera models specify sensitivity at
    different degrees of granularity: Some only specify one overall
    sensitivity, while for others, the sensitivity depends on the
    readout mode (faster readout leads to lower sensitivity), while
    others again have nested dependencies (e.g. depending on both
    readout rate and dynamic range). The sensitivity information is
    saved in picasso.CONFIG. The aspects the sensitivity depends on are
    termed 'Sensitivity Categories', and are listed for each camera in
    CONFIG (if applicable). Another entry for each camera,
    'Sensitivity', specifies the sensitivity as a scalar, a simple
    dict, or a nested dict, depending on the applicable sensitivity
    categories. The keys in the nested dict are the potential values of
    the respective sensitivity categories at that index of nesting.

    An example for a nested case (Andor Zyla):
        Sensitivity Categories:
          - PixelReadoutRate
          - Sensitivity/DynamicRange
        Sensitivity:
          540 MHz - fastest readout:
            12-bit (high well capacity): 7.98
            12-bit (low noise): 0.26
            16-bit (low noise & high well capacity): 0.51
          200 MHz - lowest noise:
            12-bit (high well capacity): 8.2
            12-bit (low noise): 0.24
            16-bit (low noise & high well capacity): 0.53

    This ``CamSettingComboBox`` class allows for selecting the value of
    one sensitivity category (described by its index in the list
    "Sensitivity Categories"). If the user changes the value of the
    ``CamSettingComboBox``, the entries of the lower levels of
    sensitivity categories (potentially) need to be adapted. Therefore,
    this ``CamSettingComboBox`` holds the ``CamSettingComboBoxDict``
    ``cam_combos``, which is a ``CamSettingComboBoxDict`` with
    references to the ``CamSettingComboBox``'s of all sensitivity
    category indices. This way the changed ``CamSettingComboBox`` can
    trigger the next-level ``CamSettingComboBox`` to adapt its options.

    ...

    Attributes
    ----------
    cam_combos : dict
        keys: Available cameras.

        values: list of CamSettingComboBoxes
            one for each sensitivity category, described in the CONFIG
            entry for the respective camera.
    camera : str
        Camera name this CamSettingComboBox belongs to.
    categories : list of str
        Sensitivity categories of the camera.
    index : int
        Index of sensitivity category this CamSettingComboBox belongs
        to.
    """

    def __init__(
        self,
        cam_combos: dict,
        camera: str,
        index: int,
        sensitivity_categories: list[str] = [],
    ) -> None:
        super().__init__()
        self.cam_combos = cam_combos
        self.camera = camera
        self.index = index
        self.categories = sensitivity_categories

    def change_target_choices(self, index: int) -> None:
        """Update the target choices based on the selected camera
        settings."""
        cam_combos = self.cam_combos[self.camera]
        sensitivity = CONFIG["Cameras"][self.camera]["Sensitivity"]
        for i in range(self.index + 1):
            sensitivity = sensitivity[cam_combos[i].currentText()]
        if len(cam_combos) > self.index + 1:
            target = cam_combos[self.index + 1]
            target.blockSignals(True)
            target.clear()
            target.blockSignals(False)
            target.addItems(sorted(list(sensitivity.keys())))


class CamSettingComboBoxDict(UserDict):
    """Dictionary holding ``CamSettingComboBox``'s for different cameras
    and sensitivity categories.

    keys: str
        Camera names.
    values: list of CamSettingComboBoxes
        one for each sensitivity category of this camera.

    Attributes
    ----------
    sensitivity_categories : dict
        keys: str
            Camera names.
        values: list of str
            Sensitivity categories.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sensitivity_categories = {}

    def add_categories(self, cam: str, categories: list[str]) -> None:
        """Call this when setting combo boxes for a new camera, to
        accompany it with the corresponding sensitivity categories."""
        self.sensitivity_categories[cam] = categories

    def set_camcombo_value(self, cam: str, category: str, value: str) -> None:
        """Set the selected value of one combo box.

        Parameters
        ----------
        cam : str
            Camera name to set.
        category : str
            Category combo box to set.
        value : str
            Value to set.
        """
        cat_idx = self.sensitivity_categories[cam].index(category)
        cam_combo = self.data[cam][cat_idx]
        for index in range(cam_combo.count()):
            if cam_combo.itemText(index) == value:
                cam_combo.setCurrentIndex(index)
                break

    def set_camcombo_values(self, cam: str, values: dict) -> None:
        """Set the values of all combo boxes of a camera.

        Parameters
        ----------
        cam : str
            Camera name to set.
        values : dict
            keys: Sensitivity categories.

            values: The values to set.
        """
        for i, cat in enumerate(self.sensitivity_categories[cam]):
            if cat in values:
                cam_combo = self.data[cam][i]
                for index in range(cam_combo.count()):
                    if cam_combo.itemText(index) == values[cat]:
                        cam_combo.setCurrentIndex(index)
                        break


class EmissionComboBoxDict(UserDict):
    """Dictionary holding ``QComboBox``'s for different cameras,
    each having the potential emission wavelengths as options.
    The ComboBox is only shown if the quantum efficiency is
    given in the CONFIG, otherwise it is irrelevant for localizing.

    keys: str
        Camera names.
    values: QtWidgets.QComboBox
        Wavelengths.
    """

    def __init__(self):
        super().__init__()

    def set_emcombo_value(self, cam: str, wavelength: str):
        """Sets the selected value of one combo box

        Parameters
        ----------
        cam : str
            Camera name to set.
        wavelength : str
            Wavelength to set.
        """
        em_combo = self.data[cam]
        for index in range(em_combo.count()):
            if em_combo.itemText(index) == wavelength:
                em_combo.setCurrentIndex(index)
                break


class PromptInfoDialog(lib.Dialog):
    """Enter movie metadata.

    ...

    Attributes
    ----------
    byte_order : QtWidgets.QComboBox
        Combo box for selecting byte order (little or big endian).
    buttons : QtWidgets.QDialogButtonBox
        Button box for the dialog (OK/Cancel).
    dtype : QtWidgets.QComboBox
        Combo box for selecting data type (float/int, number of bytes).
    frames : QtWidgets.QSpinBox
        Spin box for selecting the number of frames.
    movie_height, movie_width : QtWidgets.QSpinBox
        Spin boxes for selecting the height and width of the movie.
    save : QtWidgets.QCheckBox
        Check box for selecting whether to save the info to a YAML file.
    window : QtWidgets.QWidget
        The parent window for the dialog.
    """

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter movie info")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Byte Order:"), 0, 0)
        self.byte_order = QtWidgets.QComboBox()
        self.byte_order.addItems(
            ["Little Endian (loads faster)", "Big Endian"]
        )
        grid.addWidget(self.byte_order, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Data Type:"), 1, 0)
        self.dtype = QtWidgets.QComboBox()
        self.dtype.addItems(
            [
                "float16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "uint8",
                "uint16",
                "uint32",
            ]
        )
        grid.addWidget(self.dtype, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Frames:"), 2, 0)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(1, int(1e9))
        grid.addWidget(self.frames, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Height:"), 3, 0)
        self.movie_height = QtWidgets.QSpinBox()
        self.movie_height.setRange(1, int(1e9))
        grid.addWidget(self.movie_height, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Width"), 4, 0)
        self.movie_width = QtWidgets.QSpinBox()
        self.movie_width.setRange(1, int(1e9))
        grid.addWidget(self.movie_width, 4, 1)
        self.save = QtWidgets.QCheckBox("Save info to yaml file")
        self.save.setChecked(True)
        grid.addWidget(self.save, 5, 0, 1, 2)
        vbox.addLayout(grid)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getMovieSpecs(
        parent: QtWidgets.QWidget | None = None,
    ) -> tuple[dict, bool, bool]:
        dialog = PromptInfoDialog(parent)
        result = dialog.exec()
        info = {}
        info["Byte Order"] = (
            ">" if dialog.byte_order.currentText() == "Big Endian" else "<"
        )
        info["Data Type"] = dialog.dtype.currentText()
        info["Frames"] = dialog.frames.value()
        info["Height"] = dialog.movie_height.value()
        info["Width"] = dialog.movie_width.value()
        save = dialog.save.isChecked()
        return info, save, result == QtWidgets.QDialog.DialogCode.Accepted


class PromptMovieInfoDialog(lib.Dialog):
    """Enter movie metadata manually when it cannot be read from the
    file.

    Used as a fallback for ``.tif``/``.stk``/``.nd2`` movies whose pixel
    data could be opened but whose embedded metadata could not be
    parsed. The dialog is pre-filled with whatever dimensions could
    still be read from the file structure; the user supplies the rest
    (notably the pixel size). See ``docs/files.rst`` for the required
    metadata keys (Width, Height, Frames, Pixelsize).

    Attributes
    ----------
    frames : QtWidgets.QSpinBox
        Spin box for the number of frames.
    movie_height, movie_width : QtWidgets.QSpinBox
        Spin boxes for the height and width of the movie.
    pixelsize : QtWidgets.QSpinBox
        Spin box for the effective camera pixel size in nm.
    save : QtWidgets.QCheckBox
        Whether to save the entered metadata to a YAML file next to the
        movie, so it is reused the next time the movie is opened.
    """

    def __init__(
        self,
        window: QtWidgets.QWidget,
        partial_info: dict | None = None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter movie info")
        partial_info = partial_info or {}
        vbox = QtWidgets.QVBoxLayout(self)
        message = QtWidgets.QLabel(
            "The movie metadata could not be read from the file.\n"
            "Please enter the required information manually."
        )
        message.setWordWrap(True)
        vbox.addWidget(message)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Frames:"), 0, 0)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(1, int(1e9))
        self.frames.setValue(int(partial_info.get("Frames", 1)))
        grid.addWidget(self.frames, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Height:"), 1, 0)
        self.movie_height = QtWidgets.QSpinBox()
        self.movie_height.setRange(1, int(1e9))
        self.movie_height.setValue(int(partial_info.get("Height", 1)))
        grid.addWidget(self.movie_height, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Width:"), 2, 0)
        self.movie_width = QtWidgets.QSpinBox()
        self.movie_width.setRange(1, int(1e9))
        self.movie_width.setValue(int(partial_info.get("Width", 1)))
        grid.addWidget(self.movie_width, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Pixel size (nm):"), 3, 0)
        self.pixelsize = QtWidgets.QSpinBox()
        self.pixelsize.setRange(1, 10000)
        self.pixelsize.setValue(int(partial_info.get("Pixelsize", 130)))
        grid.addWidget(self.pixelsize, 3, 1)
        self.save = QtWidgets.QCheckBox("Save info to yaml file")
        self.save.setChecked(True)
        grid.addWidget(self.save, 4, 0, 1, 2)
        vbox.addLayout(grid)
        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getMovieSpecs(
        parent: QtWidgets.QWidget | None = None,
        partial_info: dict | None = None,
    ) -> tuple[dict, bool, bool]:
        dialog = PromptMovieInfoDialog(parent, partial_info)
        result = dialog.exec()
        # Preserve any extra readable keys (e.g. "File") already present.
        info = dict(partial_info or {})
        info["Frames"] = dialog.frames.value()
        info["Height"] = dialog.movie_height.value()
        info["Width"] = dialog.movie_width.value()
        info["Pixelsize"] = dialog.pixelsize.value()
        info["Generated by"] = "Picasso Localize (manual metadata)"
        save = dialog.save.isChecked()
        return info, save, result == QtWidgets.QDialog.DialogCode.Accepted


class PromptChannelDialog(lib.Dialog):
    """Dialog for selecting a channel. Used for .IMS files."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Select channel")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Channel:"), 0, 0)
        self.byte_order = QtWidgets.QComboBox()

        grid.addWidget(self.byte_order, 0, 1)

        vbox.addLayout(grid)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getMovieSpecs(
        parent: QtWidgets.QWidget | None = None,
        channels: list[str] | None = None,
    ) -> tuple[dict, bool, bool]:
        dialog = PromptChannelDialog(parent)
        dialog.byte_order.addItems(channels)
        result = dialog.exec()
        channel = dialog.byte_order.currentText()
        return channel, result == QtWidgets.QDialog.DialogCode.Accepted


class Calibrate3DDialog(lib.Dialog):
    """Dialog for entering the parameters of a 3D (astigmatism)
    calibration: the z step size, the number of frames acquired per z
    (stage) position and, if more than one, the order in which those
    frames were acquired."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("3D Calibration")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)

        # Step size
        grid.addWidget(QtWidgets.QLabel("Calibration step size (nm):"), 0, 0)
        self.step = QtWidgets.QDoubleSpinBox()
        self.step.setRange(0.01, 1e6)
        self.step.setDecimals(2)
        self.step.setValue(5)
        grid.addWidget(self.step, 0, 1)

        # Number of frames per z (stage) position
        frames_label = QtWidgets.QLabel("Number of frames per step size:")
        frames_label.setToolTip(
            "Number of frames acquired at each z (stage) position.\n"
            "Acquiring several frames per position increases the number\n"
            "of localizations per position and thus the confidence of\n"
            "the calibration fit."
        )
        grid.addWidget(frames_label, 1, 0)
        self.frames_per_step = QtWidgets.QSpinBox()
        self.frames_per_step.setRange(1, 1000)
        self.frames_per_step.setValue(1)
        grid.addWidget(self.frames_per_step, 1, 1)

        # Frame order (only relevant when frames_per_step > 1)
        self.order_label = QtWidgets.QLabel("Frame order:")
        grid.addWidget(self.order_label, 2, 0)
        self.frame_order = QtWidgets.QComboBox()
        self.frame_order.addItem("Different FOVs first", userData="fov")
        self.frame_order.addItem("Different z positions first", userData="z")
        self.frame_order.setToolTip(
            "Order in which the frames were acquired when more than one\n"
            "frame per step size is used.\n"
            "'Same z position, different FOVs': the z position is held\n"
            "constant while several fields of view are imaged, i.e.,\n"
            "consecutive frames share the same z position.\n"
            "'Same FOV, z positions sequentially': the full z stack is\n"
            "scanned and then repeated, i.e., frames cycle through all z\n"
            "positions."
        )
        grid.addWidget(self.frame_order, 2, 1)

        # The order only matters when more than one frame per step
        self.frames_per_step.valueChanged.connect(self._update_order_enabled)
        self._update_order_enabled(self.frames_per_step.value())

        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def _update_order_enabled(self, n_frames: int) -> None:
        """Enable the frame order choice only if more than one frame is
        acquired per z position."""
        enabled = n_frames > 1
        self.order_label.setEnabled(enabled)
        self.frame_order.setEnabled(enabled)

    @staticmethod
    def getCalibrationSpecs(
        parent: QtWidgets.QWidget | None = None,
    ) -> tuple[float, int, str, bool]:
        """Show the dialog and return the chosen step size, number of
        frames per step, frame order and whether the dialog was
        accepted."""
        dialog = Calibrate3DDialog(parent)
        result = dialog.exec()
        step = dialog.step.value()
        frames_per_step = dialog.frames_per_step.value()
        frame_order = dialog.frame_order.currentData()
        accepted = result == QtWidgets.QDialog.DialogCode.Accepted
        return step, frames_per_step, frame_order, accepted


class CalibrateSplineDialog(lib.Dialog):
    """Dialog for entering the parameters of a cubic-spline PSF calibration
    built from a bead z-stack: the z step size, the number of frames acquired
    per z (stage) position, the acquisition order of those frames, and whether
    to build a 3D (z-recovering) or 2D (single-plane) spline PSF. The box size
    and minimum net gradient are taken from the main parameters."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Spline PSF Calibration")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)

        # Step size
        grid.addWidget(QtWidgets.QLabel("Calibration step size (nm):"), 0, 0)
        self.step = QtWidgets.QDoubleSpinBox()
        self.step.setRange(0.01, 1e6)
        self.step.setDecimals(2)
        self.step.setValue(5)
        grid.addWidget(self.step, 0, 1)

        # Number of frames per z (stage) position
        frames_label = QtWidgets.QLabel("Number of frames per step size:")
        frames_label.setToolTip(
            "Number of frames acquired at each z (stage) position.\n"
            "Acquiring several frames per position (e.g., different fields\n"
            "of view) increases the number of beads averaged into the PSF."
        )
        grid.addWidget(frames_label, 1, 0)
        self.frames_per_step = QtWidgets.QSpinBox()
        self.frames_per_step.setRange(1, 1000)
        self.frames_per_step.setValue(1)
        grid.addWidget(self.frames_per_step, 1, 1)

        # Frame order (only relevant when frames_per_step > 1)
        self.order_label = QtWidgets.QLabel("Frame order:")
        grid.addWidget(self.order_label, 2, 0)
        self.frame_order = QtWidgets.QComboBox()
        self.frame_order.addItem("Different FOVs first", userData="fov")
        self.frame_order.addItem("Different z positions first", userData="z")
        self.frame_order.setToolTip(
            "Order in which the frames were acquired when more than one\n"
            "frame per step size is used (see the 3D astigmatism dialog)."
        )
        grid.addWidget(self.frame_order, 2, 1)

        # Spline PSF dimensionality / model
        grid.addWidget(QtWidgets.QLabel("Spline PSF model:"), 3, 0)
        self.model = QtWidgets.QComboBox()
        self.model.addItem("3D (recovers z)", userData="spline-3d")
        self.model.addItem("2D (single plane)", userData="spline-2d")
        self.model.setToolTip(
            "3D / 2D: single- or multichannel cubic-spline PSF."
        )
        grid.addWidget(self.model, 3, 1)

        # Magnification factor (applied to the fitted z, as in astigmatism)
        magnification_label = QtWidgets.QLabel("Magnification factor:")
        magnification_label.setToolTip(
            "Factor used to correct for z-position abberation due to\n"
            "refractive index mismatch, see Huang B, et al. Science. 2008."
        )
        grid.addWidget(magnification_label, 4, 0)
        self.magnification_factor = QtWidgets.QDoubleSpinBox()
        self.magnification_factor.setRange(0, 1e6)
        self.magnification_factor.setDecimals(4)
        self.magnification_factor.setValue(0.79)
        grid.addWidget(self.magnification_factor, 4, 1)

        # Optional z-bias correction (astigmatism)
        self.correct_z_bias = QtWidgets.QCheckBox(
            "Set z = 0 at max. intensity"
        )
        self.correct_z_bias.setToolTip(
            "Define z = 0 at the axial intensity peak of the averaged PSF,\n"
            "correcting a potential z bias in the raw stage scan.\n"
            "Only meaningful for a PSF with a single,\n"
            "well-defined intensity focus (e.g. astigmatism)."
        )
        self.correct_z_bias.setChecked(False)
        grid.addWidget(self.correct_z_bias, 5, 0, 1, 2)

        # Multichannel (2- to 6-channel) default fit mode. Stored in the
        # calibration
        self.link_photons = QtWidgets.QCheckBox(
            "Link photon counts across channels"
        )
        self.link_photons.setToolTip(LINK_PHOTONS_TIP)
        self.link_photons.setChecked(True)
        grid.addWidget(self.link_photons, 6, 0, 1, 2)

        self.frames_per_step.valueChanged.connect(self._update_order_enabled)
        self._update_order_enabled(self.frames_per_step.value())

        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def _update_order_enabled(self, n_frames: int) -> None:
        """Enable the frame order choice only if more than one frame is
        acquired per z position."""
        enabled = n_frames > 1
        self.order_label.setEnabled(enabled)
        self.frame_order.setEnabled(enabled)

    @staticmethod
    def getCalibrationSpecs(
        parent: QtWidgets.QWidget | None = None,
    ) -> tuple[float, int, str, str, float, bool, bool, bool]:
        """Show the dialog and return the chosen step size, number of frames
        per step, frame order, spline model, magnification factor, whether to
        correct the z bias, whether to link photons across channels, and whether
        it was accepted."""
        dialog = CalibrateSplineDialog(parent)
        result = dialog.exec()
        step = dialog.step.value()
        frames_per_step = dialog.frames_per_step.value()
        frame_order = dialog.frame_order.currentData()
        model = dialog.model.currentData()
        magnification_factor = dialog.magnification_factor.value()
        correct_z_bias = dialog.correct_z_bias.isChecked()
        link_photons = dialog.link_photons.isChecked()
        accepted = result == QtWidgets.QDialog.DialogCode.Accepted
        return (
            step,
            frames_per_step,
            frame_order,
            model,
            magnification_factor,
            correct_z_bias,
            link_photons,
            accepted,
        )


class RefineRegistrationDialog(lib.Dialog):
    """Dialog for choosing which frames the signal-based channel re-alignment
    considers.

    The refinement pairs shared single-molecule signal frame by frame, so the
    frames it samples matter: early frames of a movie are often too dense (or
    still bleaching) to pair unambiguously, while late frames may be too
    sparse. The user picks the frame window (1-indexed, inclusive) and how many
    frames are evenly sampled from it."""

    def __init__(
        self,
        window: QtWidgets.QWidget,
        n_frames: int,
        frame_range: list | None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Re-align channels (signal)")
        vbox = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "Frames used to pair the shared single-molecule signal.\n"
            "The sampled frames are spread evenly over the range."
        )
        vbox.addWidget(info)
        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)

        # frame window (1-indexed, inclusive), seeded from the identification
        # frame range when a single segment is set
        lo, hi = 1, max(1, int(n_frames))
        if frame_range:
            first = frame_range[0]
            if not np.isscalar(first):
                lo = max(1, int(first[0]) + 1)
                hi = min(hi, int(frame_range[-1][1]) + 1)
        grid.addWidget(QtWidgets.QLabel("First frame:"), 0, 0)
        self.first_frame = QtWidgets.QSpinBox()
        self.first_frame.setRange(1, max(1, int(n_frames)))
        self.first_frame.setValue(lo)
        grid.addWidget(self.first_frame, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Last frame:"), 1, 0)
        self.last_frame = QtWidgets.QSpinBox()
        self.last_frame.setRange(1, max(1, int(n_frames)))
        self.last_frame.setValue(max(lo, hi))
        grid.addWidget(self.last_frame, 1, 1)

        sampled_label = QtWidgets.QLabel("Frames to sample:")
        sampled_label.setToolTip(
            "Number of frames evenly sampled from the range above.\n"
            "Only these frames are detected on, so this bounds the runtime.\n"
            "Increase it for sparse data (fewer blinks per frame)."
        )
        grid.addWidget(sampled_label, 2, 0)
        self.max_frames = QtWidgets.QSpinBox()
        self.max_frames.setRange(2, 10000)
        self.max_frames.setValue(50)
        grid.addWidget(self.max_frames, 2, 1)

        # keep the window ordered
        self.first_frame.valueChanged.connect(
            lambda v: self.last_frame.setValue(max(v, self.last_frame.value()))
        )
        self.last_frame.valueChanged.connect(
            lambda v: self.first_frame.setValue(
                min(v, self.first_frame.value())
            )
        )

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getFrameSpecs(
        parent: QtWidgets.QWidget,
        n_frames: int,
        frame_range: list | None = None,
    ) -> tuple[list, int, bool]:
        """Show the dialog and return the chosen ``frame_bounds`` (a single
        0-indexed inclusive segment), the number of frames to sample and
        whether the dialog was accepted."""
        dialog = RefineRegistrationDialog(parent, n_frames, frame_range)
        result = dialog.exec()
        bounds = [
            [dialog.first_frame.value() - 1, dialog.last_frame.value() - 1]
        ]
        return (
            bounds,
            dialog.max_frames.value(),
            result == QtWidgets.QDialog.DialogCode.Accepted,
        )


class ROIDialog(lib.Dialog):
    """Sub-dialog for managing several regions of interest (ROIs)
    numerically.

    The dialog edits ``window.view.rois`` directly (clipping overlaps via
    ``localize.clip_rois``) and keeps the compact ROI field in the
    parameters dialog in sync. It is modeless, so the user can keep
    drawing ROIs in the preview while it is open.

    Attributes
    ----------
    table : QtWidgets.QTableWidget
        Table listing the ROIs, one rectangle per row.
    window : QtWidgets.QMainWindow
        Reference to the main window.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Regions of interest")
        self.setModal(False)
        self._updating = False

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        info = QtWidgets.QLabel(
            "Each row is a rectangular ROI (y_min, x_min, y_max, x_max, "
            "in camera pixels). Drag a rectangle in the preview or use "
            "Add, then edit the cells. Overlapping ROIs are clipped "
            "automatically so they never cover a pixel twice. Clear the "
            "list to analyze the whole frame."
        )
        info.setWordWrap(True)
        header.addWidget(info, 1)
        help_button = lib.HelpButton(ParametersDialog.ROI_URL)
        header.addWidget(help_button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["y_min", "x_min", "y_max", "x_max"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.itemChanged.connect(self.on_table_changed)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add")
        self.add_button.clicked.connect(self.on_add)
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.clicked.connect(self.on_remove)
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.on_clear)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.clear_button)
        layout.addLayout(buttons)

        self.resize(360, 280)
        self.update_table()

    def _box(self) -> int:
        """Current box size, used as the minimum ROI side length."""
        return self.window.parameters.get("Box Size", 7)

    def _commit(self, rois: list) -> None:
        """Clip ``rois`` and store them on the view, refreshing the
        compact field in the parameters dialog."""
        self.window.view.rois = localize.clip_rois(rois, min_size=self._box())
        self.window.parameters_dialog.update_roi_display(skip_dialog=True)
        self.window.draw_frame()

    def update_table(self) -> None:
        """Repopulate the table from the view's ROIs."""
        view = self.window.view
        self._updating = True
        self.table.setRowCount(len(view.rois))
        for row, ((y_min, x_min), (y_max, x_max)) in enumerate(view.rois):
            for col, val in enumerate((y_min, x_min, y_max, x_max)):
                item = QtWidgets.QTableWidgetItem(str(int(val)))
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
        if view.selected_roi is not None and view.selected_roi < len(
            view.rois
        ):
            self.table.selectRow(view.selected_roi)
        self._updating = False

    def on_table_changed(self, item: object = None) -> None:
        """Rebuild the view's ROIs from the table, clipping overlaps."""
        if self._updating:
            return
        rois = []
        for row in range(self.table.rowCount()):
            try:
                vals = [int(self.table.item(row, c).text()) for c in range(4)]
            except (AttributeError, ValueError):
                return  # incomplete row, wait for the user to finish
            y_min, x_min, y_max, x_max = vals
            rois.append([[y_min, x_min], [y_max, x_max]])
        self._commit(rois)
        self.update_table()

    def on_add(self) -> None:
        """Add a default ROI row that the user can edit."""
        view = self.window.view
        if self.window.movie is not None:
            height, width = self.window.movie.shape[1:]
            side = int(min(height, width) / 4)
        else:
            side = 50
        offset = 10 * len(view.rois)  # avoid fully overlapping the last one
        new_roi = [[offset, offset], [offset + side, offset + side]]
        self._commit(view.rois + [new_roi])
        self.update_table()

    def on_remove(self) -> None:
        """Remove the ROI(s) selected in the table."""
        view = self.window.view
        rows = sorted(
            {idx.row() for idx in self.table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            if 0 <= row < len(view.rois):
                del view.rois[row]
        view.selected_roi = None
        self._commit(view.rois)
        self.update_table()

    def on_clear(self) -> None:
        """Remove all ROIs (analyze the whole frame)."""
        self.window.view.selected_roi = None
        self._commit([])
        self.update_table()

    def on_selection_changed(self) -> None:
        """Highlight the ROI selected in the table."""
        if self._updating:
            return
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        self.window.view.selected_roi = min(rows) if rows else None
        self.window.draw_frame()


class ParametersDialog(lib.Dialog):
    """Choose analysis parameters.

    ...

    Attributes
    ----------
    baseline : QtWidgets.QDoubleSpinBox
        Spin box for selecting camera baseline (background amplitude).
    box_spinbox : OddSpinBox
        Spin box for selecting the box size.
    camera : QtWidgets.QComboBox
        Combo box for selecting the camera.
    cam_combos : CamSettingComboBoxDict
        Combo boxes for selecting channels.
    convergence_criterion : QtWidgets.QDoubleSpinBox
        Spin box for setting the convergence criterion. Only used for
        MLE fitting.
    emission_combos : EmissionSettingComboBoxDict
        Combo boxes for selecting emission wavelengths.
    fit_model : QtWidgets.QComboBox
        Combo box for selecting the fitting model (e.g. 2D elliptical
        Gaussian or average of ROI).
    fit_optimizer : QtWidgets.QComboBox
        Combo box for selecting the optimizer (Least squares or MLE).
        Hidden for models that do not use an optimizer.
    fit_z_checkbox : QtWidgets.QCheckBox
        Checkbox for enabling/disabling fitting in the z-dimension using
        astigmatism.
    gain : QtWidgets.QSpinBox
        Spin box for selecting camera EM gain.
    gpu_checkbox : QtWidgets.QCheckBox
        Checkbox for enabling/disabling GPU fitting. Only shown if a
        CUDA-capable GPU is available. Also selects the device for the
        astigmatism z fit, which has its own CUDA kernel
        (``picasso.zfit``).
    magnification_factor : QtWidgets.QDoubleSpinBox
        Spin box for setting the magnification factor for 3D fitting.
    max_it : QtWidgets.QSpinBox
        Spin box for selecting the max. number of iterations. Only used
        for MLE fitting.
    mng_min_spinbox : QtWidgets.QSpinBox
        Spin box for selecting the minimum net gradient (lower bound).
    mng_max_spinbox : QtWidgets.QSpinBox
        Spin box for selecting the minimum net gradient (upper bound).
    mng_slider : QtWidgets.QSlider
        Slider for selecting the minimum net gradient.
    mng_spinbox : QtWidgets.QSpinBox
        Spin box for selecting the minimum net gradient.
    pixelsize : QtWidgets.QDoubleSpinBox
        Spin box for setting camera pixel size (nm).
    preview_checkbox : QtWidgets.QCheckBox
        Checkbox for enabling/disabling preview of identified spots.
    roi_field : QtWidgets.QLineEdit
        Compact field summarizing the regions of interest (ROIs): empty
        for the whole frame, the four coordinates of the single ROI when
        there is exactly one (editable), or a count when there are
        several.
    roi_dialog : ROIDialog or None
        Sub-dialog (lazily created) for managing several ROIs in a table.
    sensitivity : QtWidgets.QDoubleSpinBox
        Spin box for setting camera sensitivity.
    qe : QtWidgets.QDoubleSpinBox
        Spin box for setting camera quantum efficiency (QE). **Note**:
        QE value is not used in the analysis, only present for
        backward compatibility.
    quality_check : QtWidgets.QCheckBox
        Checkbox for enabling/disabling quality check (mean bright time,
        drift and NeNA estimation).
    quality_grid_labels : list[QtWidgets.QLabel]
        Labels for displaying quality checks.
    quality_grid_values : list[QtWidgets.QLabel]
        Values for displaying quality checks.
    window : QtWidgets.QWidget
        The main window of the application.
    """

    CALIB_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#d-calibration"  # noqa: E501
    IDENT_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#identification-and-fitting-of-single-molecule-spots"  # noqa: E501
    ROI_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#regions-of-interest-rois"  # noqa: E501
    SPLINE_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#experimental-psf-cubic-spline-fitting"  # noqa: E501

    def __init__(  # noqa: C901
        self, parent: QtWidgets.QMainWindow | None = None
    ) -> None:
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Parameters")
        self.setModal(False)

        self.z_calibration = {}
        self.z_calibration_path = None
        self.spline_calibration = {}
        self.spline_calibration_path = None
        # calibration group boxes, toggled by the selected fit model; set up
        # further below
        self.z_groupbox = None
        self.spline_groupbox = None
        # last resolved fit2D code, so the convergence defaults are only
        # reapplied when the method actually changes
        self._last_fit_code = None

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        self.scroll_area.setWidget(container)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll_area)

        identification_groupbox = QtWidgets.QGroupBox("Identification")
        vbox.addWidget(identification_groupbox)
        identification_grid = QtWidgets.QGridLayout(identification_groupbox)

        # Box Size
        first_row = QtWidgets.QHBoxLayout()
        identification_grid.addLayout(first_row, 0, 0, 1, 2)
        first_row.addWidget(lib.HelpButton(self.IDENT_URL))
        boxsize_label = QtWidgets.QLabel("Box side length:")
        boxsize_label.setToolTip(
            "Box size in camera pixels for identification."
        )
        first_row.addWidget(boxsize_label)
        self.box_spinbox = OddSpinBox()
        self.box_spinbox.setKeyboardTracking(False)
        self.box_spinbox.setValue(DEFAULT_PARAMETERS["Box Size"])
        self.box_spinbox.valueChanged.connect(self.on_box_changed)
        first_row.addWidget(self.box_spinbox)

        # Min. Net Gradient
        mng_label = QtWidgets.QLabel("Min. net gradient:")
        mng_label.setToolTip(
            "Threshold (related to brightness) for spot identification."
        )
        identification_grid.addWidget(mng_label, 1, 0)
        self.mng_spinbox = QtWidgets.QSpinBox()
        self.mng_spinbox.setRange(0, int(1e9))
        self.mng_spinbox.setValue(DEFAULT_PARAMETERS["Min. Net Gradient"])
        self.mng_spinbox.setKeyboardTracking(False)
        self.mng_spinbox.valueChanged.connect(self.on_mng_spinbox_changed)
        identification_grid.addWidget(self.mng_spinbox, 1, 1)

        # Slider
        self.mng_slider = QtWidgets.QSlider()
        self.mng_slider.setToolTip(
            "Adjust the minimum net gradient for spot identification."
        )
        self.mng_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.mng_slider.setRange(0, 10000)
        self.mng_slider.setValue(DEFAULT_PARAMETERS["Min. Net Gradient"])
        self.mng_slider.setSingleStep(1)
        self.mng_slider.setPageStep(20)
        self.mng_slider.valueChanged.connect(self.on_mng_slider_changed)
        identification_grid.addWidget(self.mng_slider, 2, 0, 1, 2)

        hbox = QtWidgets.QHBoxLayout()
        identification_grid.addLayout(hbox, 3, 0, 1, 2)

        # Min SpinBox
        self.mng_min_spinbox = QtWidgets.QSpinBox()
        self.mng_min_spinbox.setToolTip(
            "Minimum value for the minimum net gradient slider."
        )
        self.mng_min_spinbox.setRange(0, 999999)
        self.mng_min_spinbox.setKeyboardTracking(False)
        self.mng_min_spinbox.setValue(0)
        self.mng_min_spinbox.valueChanged.connect(self.on_mng_min_changed)
        hbox.addWidget(self.mng_min_spinbox)

        hbox.addStretch(1)

        # Max SpinBox
        self.mng_max_spinbox = QtWidgets.QSpinBox()
        self.mng_max_spinbox.setToolTip(
            "Maximum value for the minimum net gradient slider."
        )
        self.mng_max_spinbox.setKeyboardTracking(False)
        self.mng_max_spinbox.setRange(0, 999999)
        self.mng_max_spinbox.setValue(10000)
        self.mng_max_spinbox.valueChanged.connect(self.on_mng_max_changed)
        hbox.addWidget(self.mng_max_spinbox)

        # preview identifications + cross-channel link colour overlay
        preview_row = QtWidgets.QHBoxLayout()
        self.preview_checkbox = QtWidgets.QCheckBox("Preview")
        self.preview_checkbox.setToolTip(
            "Show identified spots in the current frame?"
        )
        self.preview_checkbox.setTristate(False)
        self.preview_checkbox.stateChanged.connect(self.on_preview_changed)
        preview_row.addWidget(self.preview_checkbox)
        self.link_colors_checkbox = QtWidgets.QCheckBox("Link colors")
        self.link_colors_checkbox.setToolTip(
            "Color-code the identification boxes by their cross-channel link.\n\n"
            "Spots paired across channels share a color; unmatched spots are\n"
            "gray. Pairing uses the loaded multichannel / split-FOV spline\n"
            "calibration's inter-channel transform. With no calibration\n"
            "loaded, the transform is estimated from the identifications "
            "themselves.\n"
            "Identify every channel first."
        )
        self.link_colors_checkbox.setTristate(False)
        self.link_colors_checkbox.stateChanged.connect(
            self.on_link_colors_changed
        )
        preview_row.addWidget(self.link_colors_checkbox)
        preview_row.addStretch(1)
        identification_grid.addLayout(preview_row, 4, 0)

        # ROIs
        label = QtWidgets.QLabel("ROIs:")
        label.setToolTip(
            "Restrict the analysis to one or more rectangular regions.\n"
            "Drag a rectangle in the preview to add a ROI, double-click a\n"
            "ROI to remove it, or click 'Edit ROIs...' to enter them\n"
            "numerically. With a single ROI its coordinates\n"
            "(y_min, x_min, y_max, x_max, in camera pixels) can be edited\n"
            "directly here. Leave empty to analyze the whole frame."
        )
        roi_label_layout = QtWidgets.QHBoxLayout()
        roi_label_layout.addWidget(lib.HelpButton(self.ROI_URL))
        roi_label_layout.addWidget(label)
        roi_label_layout.addStretch(1)
        # Split-FOV: treat the drawn ROIs as separate channels of one movie.
        self.split_fov_checkbox = QtWidgets.QCheckBox("Regions = channels")
        self.split_fov_checkbox.setToolTip(
            "Split-FOV mode: treat the drawn ROIs as separate channels imaged\n"
            "side-by-side on one camera (spectral / biplane split\n"
            "optics). The first region is the reference channel and all\n"
            "regions are kept the same size: drag once to set the size, click\n"
            "to drop more regions, drag a region (or use the arrow keys) to\n"
            "fine-tune its registration. 'Calibrate spline PSF' and the\n"
            "spline fit then use these regions as channels of this movie."
        )
        self.split_fov_checkbox.setTristate(False)
        self.split_fov_checkbox.stateChanged.connect(self.on_split_fov_changed)
        roi_label_layout.addWidget(self.split_fov_checkbox)
        identification_grid.addLayout(roi_label_layout, 5, 0)

        self._updating_roi_field = False
        self.roi_dialog = None
        roi_layout = QtWidgets.QHBoxLayout()
        self.roi_field = QtWidgets.QLineEdit()
        self.roi_field.setPlaceholderText("Whole frame")
        regex = r"\d+,\d+,\d+,\d+"  # 4 integers separated by commas
        validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(regex)
        )
        self.roi_field.setValidator(validator)
        self.roi_field.editingFinished.connect(self.on_roi_field_finished)
        self.roi_field.textChanged.connect(self.on_roi_field_changed)
        roi_layout.addWidget(self.roi_field)
        self.roi_edit_button = QtWidgets.QPushButton("Edit ROIs...")
        self.roi_edit_button.clicked.connect(self.on_edit_rois)
        roi_layout.addWidget(self.roi_edit_button)
        identification_grid.addLayout(roi_layout, 5, 1)

        # min/max frames
        label = QtWidgets.QLabel("Frames (min,max):")
        label.setToolTip(
            "Specify the first and last frame (inclusive) to be analyzed;\n"
            "by default, all frames are analyzed.\n"
            "Several disjoint segments can be given as min,max pairs\n"
            "separated by semicolons, e.g. '1,100; 200,300'."
        )
        identification_grid.addWidget(label, 6, 0)
        self.frames_edit = QtWidgets.QLineEdit()
        # one or more "min,max" pairs separated by semicolons, with
        # optional surrounding whitespace
        regex = r"\s*\d+\s*,\s*\d+\s*(;\s*\d+\s*,\s*\d+\s*)*"
        validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(regex)
        )
        self.frames_edit.setValidator(validator)
        self.frames_edit.editingFinished.connect(self.on_frames_edit_finished)
        self.frames_edit.textChanged.connect(self.on_frames_edit_changed)
        identification_grid.addWidget(self.frames_edit, 6, 1)

        # Multichannel: optionally use the same key settings for every channel.
        # Shown only when more than one channel is loaded (see the Window's
        # _populate_channel_combo). Checking a box makes that group shared, so
        # switching channels no longer overwrites it.
        self.link_groupbox = QtWidgets.QGroupBox(
            "Same settings across channels"
        )
        vbox.addWidget(self.link_groupbox)
        link_layout = QtWidgets.QHBoxLayout(self.link_groupbox)
        self.link_box_checkbox = QtWidgets.QCheckBox("Box size")
        self.link_box_checkbox.setToolTip(
            "Use the same box size for every channel."
        )
        self.link_mng_checkbox = QtWidgets.QCheckBox("Min. net gradient")
        self.link_mng_checkbox.setToolTip(
            "Use the same minimum net gradient for every channel."
        )
        self.link_camera_checkbox = QtWidgets.QCheckBox("Camera settings")
        self.link_camera_checkbox.setToolTip(
            "Use the same camera and photon-conversion settings (camera,\n"
            "baseline, EM gain, sensitivity, QE, pixel size) for every "
            "channel."
        )
        for cb in (
            self.link_box_checkbox,
            self.link_mng_checkbox,
            self.link_camera_checkbox,
        ):
            cb.setTristate(False)
            cb.stateChanged.connect(self.on_link_params_changed)
            link_layout.addWidget(cb)
        link_layout.addStretch(1)
        self.link_groupbox.hide()  # shown by the window for >1 channel

        # Camera:
        if "Cameras" in CONFIG:
            # Experiment settings
            exp_groupbox = QtWidgets.QGroupBox("Experiment Settings")
            vbox.addWidget(exp_groupbox)
            exp_grid = QtWidgets.QGridLayout(exp_groupbox)
            exp_grid.addWidget(QtWidgets.QLabel("Camera:"), 0, 0)
            self.camera = QtWidgets.QComboBox()
            exp_grid.addWidget(self.camera, 0, 1)
            cameras = sorted(list(CONFIG["Cameras"].keys()))
            if "CameraPriority" in CONFIG:
                cam_prio_list = CONFIG["CameraPriority"]
                # remove the prio cameras from the sorted list
                for cam in cam_prio_list:
                    if cam in cameras:
                        cameras.remove(cam)
                cameras = cam_prio_list + cameras
            self.camera.addItems(cameras)
            self.camera.currentIndexChanged.connect(self.on_camera_changed)

            self.cam_settings = QtWidgets.QStackedWidget()
            self.cam_settings.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            exp_grid.addWidget(self.cam_settings, 1, 0, 1, 2)
            self.cam_combos = CamSettingComboBoxDict()
            self.emission_combos = EmissionComboBoxDict()
            for cam in cameras:
                cam_widget = QtWidgets.QWidget()
                cam_grid = QtWidgets.QGridLayout(cam_widget)
                self.cam_settings.addWidget(cam_widget)
                cam_config = CONFIG["Cameras"][cam]
                if "Sensitivity" in cam_config:
                    sensitivity = cam_config["Sensitivity"]
                    if "Sensitivity Categories" in cam_config:
                        self.cam_combos[cam] = []
                        categories = cam_config["Sensitivity Categories"]
                        self.cam_combos.add_categories(cam, categories)
                        for i, category in enumerate(categories):
                            row_count = cam_grid.rowCount()
                            cam_grid.addWidget(
                                QtWidgets.QLabel(category + ":"), row_count, 0
                            )
                            cat_combo = CamSettingComboBox(
                                self.cam_combos,
                                cam,
                                i,
                            )
                            cam_grid.addWidget(cat_combo, row_count, 1)
                            self.cam_combos[cam].append(cat_combo)
                        self.cam_combos[cam][0].addItems(
                            sorted(list(sensitivity.keys()))
                        )
                        for cam_combo in self.cam_combos[cam][:-1]:
                            cam_combo.currentIndexChanged.connect(
                                cam_combo.change_target_choices
                            )
                        self.cam_combos[cam][0].change_target_choices(0)
                        self.cam_combos[cam][-1].currentIndexChanged.connect(
                            self.update_sensitivity
                        )
                if "Quantum Efficiency" in cam_config:
                    try:
                        qes = cam_config["Quantum Efficiency"].keys()
                    except AttributeError:
                        pass
                    else:
                        row_count = cam_grid.rowCount()
                        cam_grid.addWidget(
                            QtWidgets.QLabel("Emission Wavelength:"),
                            row_count,
                            0,
                        )
                        emission_combo = QtWidgets.QComboBox()
                        cam_grid.addWidget(emission_combo, row_count, 1)
                        wavelengths = sorted([str(_) for _ in qes])
                        emission_combo.addItems(wavelengths)
                        emission_combo.currentIndexChanged.connect(
                            self.on_emission_changed
                        )
                        self.emission_combos[cam] = emission_combo
                spacer = QtWidgets.QWidget()
                spacer.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Preferred,
                    QtWidgets.QSizePolicy.Policy.Expanding,
                )
                cam_grid.addWidget(spacer, cam_grid.rowCount(), 0)

        # Photon conversion
        photon_groupbox = QtWidgets.QGroupBox("Photon Conversion")
        vbox.addWidget(photon_groupbox)
        photon_grid = QtWidgets.QGridLayout(photon_groupbox)

        # EM Gain
        em_label = QtWidgets.QLabel("EM gain:")
        em_label.setToolTip(
            "Electron multiplying gain of a EMCCD camera (=1 for sCMOS)."
        )
        photon_grid.addWidget(em_label, 0, 0)
        self.gain = QtWidgets.QSpinBox()
        self.gain.setRange(1, int(1e6))
        self.gain.setValue(1)
        photon_grid.addWidget(self.gain, 0, 1)

        # Baseline
        baseline_label = QtWidgets.QLabel("Baseline:")
        baseline_label.setToolTip("Mean pixel value in the absence of light.")
        photon_grid.addWidget(baseline_label, 1, 0)
        self.baseline = QtWidgets.QDoubleSpinBox()
        self.baseline.setRange(0, 1e6)
        self.baseline.setValue(100.0)
        self.baseline.setDecimals(1)
        self.baseline.setSingleStep(0.1)
        photon_grid.addWidget(self.baseline, 1, 1)

        # Sensitivity
        sensitivity_label = QtWidgets.QLabel("Sensitivity:")
        sensitivity_label.setToolTip(
            "Camera sensitivity in counts per photon (conversion factor)."
        )
        photon_grid.addWidget(sensitivity_label, 2, 0)
        self.sensitivity = QtWidgets.QDoubleSpinBox()
        self.sensitivity.setRange(0, 1e6)
        self.sensitivity.setValue(1.0)
        self.sensitivity.setDecimals(4)
        self.sensitivity.setSingleStep(0.01)
        photon_grid.addWidget(self.sensitivity, 2, 1)

        # QE
        qe_label = QtWidgets.QLabel("Quantum efficiency:")
        qe_label.setToolTip(
            "To be deprecated in v1.0; not used in the analysis."
        )
        photon_grid.addWidget(qe_label, 3, 0)
        self.qe = QtWidgets.QDoubleSpinBox()
        self.qe.setRange(0, 1)
        self.qe.setValue(1)
        self.qe.setDecimals(2)
        self.qe.setSingleStep(0.1)
        photon_grid.addWidget(self.qe, 3, 1)

        # Camera pixel size
        px_label = QtWidgets.QLabel("Pixel size (nm):")
        px_label.setToolTip(
            "Effective camera pixel size in nm (after magnification)."
        )
        photon_grid.addWidget(px_label, 4, 0)
        self.pixelsize = QtWidgets.QSpinBox()
        self.pixelsize.setRange(0, 10000)
        self.pixelsize.setValue(130)
        self.pixelsize.setSingleStep(1)
        self.pixelsize.valueChanged.connect(self.on_pixelsize_changed)
        photon_grid.addWidget(self.pixelsize, 4, 1)

        # Fit Settings
        fit_groupbox = QtWidgets.QGroupBox("Fit Settings")
        vbox.addWidget(fit_groupbox)
        fit_grid = QtWidgets.QGridLayout(fit_groupbox)

        model_label = QtWidgets.QLabel("Model:")
        model_label.setToolTip(MODEL_TOOLTIP)
        fit_grid.addWidget(model_label, 1, 0)
        self.fit_model = QtWidgets.QComboBox()
        self.fit_model.addItems(list(FIT_MODELS.keys()))
        self.fit_model.setCurrentIndex(0)
        self.fit_model.setToolTip(MODEL_TOOLTIP)
        fit_grid.addWidget(self.fit_model, 1, 1)

        self.optimizer_label = QtWidgets.QLabel("Optimizer:")
        self.optimizer_label.setToolTip(OPTIMIZER_TOOLTIP)
        fit_grid.addWidget(self.optimizer_label, 2, 0)
        self.fit_optimizer = QtWidgets.QComboBox()
        self.fit_optimizer.setToolTip(OPTIMIZER_TOOLTIP)
        fit_grid.addWidget(self.fit_optimizer, 2, 1)

        self.fit_stack = QtWidgets.QStackedWidget()
        fit_grid.addWidget(self.fit_stack, 3, 0, 1, 2)
        fit_stack = self.fit_stack

        # MLE
        mle_widget = QtWidgets.QWidget()

        mle_grid = QtWidgets.QGridLayout(mle_widget)
        cc_label = QtWidgets.QLabel("Convergence criterion:")
        cc_label.setToolTip(
            "Tolerance for testing if fitting has converged: the fit stops\n"
            " once the chi-square changes by less than this, relative to\n"
            " its own magnitude."
        )
        mle_grid.addWidget(cc_label, 0, 0)
        self.convergence_criterion = QtWidgets.QDoubleSpinBox()
        # Never 0: fit2D requires a positive tolerance, and below ~1e-6 the
        # test is answering noise in the chi-square rather than convergence.
        self.convergence_criterion.setRange(1e-6, 1e6)
        self.convergence_criterion.setDecimals(6)
        self.convergence_criterion.setValue(0.001)
        self.convergence_criterion.setSingleStep(0.0001)
        mle_grid.addWidget(self.convergence_criterion, 0, 1)
        mi_label = QtWidgets.QLabel("Max. iterations:")
        mi_label.setToolTip("Maximum number of iterations per spot.")
        mle_grid.addWidget(mi_label, 1, 0)
        self.max_it = QtWidgets.QSpinBox()
        self.max_it.setRange(1, int(1e6))
        self.max_it.setValue(100)
        mle_grid.addWidget(self.max_it, 1, 1)

        # Shown for the one method that does not iterate ("Average of ROI");
        # an empty page keeps the stack indices stable.
        lq_widget = QtWidgets.QWidget()

        # 0 -> no optimizer parameters, 1 -> convergence criterion/max_it.
        fit_stack.addWidget(lq_widget)
        fit_stack.addWidget(mle_widget)

        # Picasso implements both least-squares and MLE estimators on the
        # GPU, so the checkbox applies to both optimizers and sits below
        # the optimizer parameter stack.
        self.gpu_checkbox = QtWidgets.QCheckBox("Use GPU")
        self.gpu_checkbox.setToolTip(
            "Perform fitting on a CUDA-capable GPU.\n\n"
            "The algorithm is a port of Gpufit; the kernels are compiled "
            "at run time by Numba.\n\n"
            "Przybylski, A., Thiel, B., Keller-Findeisen, J. et al. "
            "Gpufit: An open-source toolkit for GPU-accelerated curve "
            "fitting. Sci Rep 7, 15722 (2017). "
        )
        self.gpu_checkbox.setTristate(False)
        self.gpu_checkbox.setDisabled(True)
        self.gpu_checkbox.stateChanged.connect(self.on_gpu_fitting_changed)

        if not GPU_FITTING_AVAILABLE:
            self.gpu_checkbox.hide()
        else:
            self.gpu_checkbox.setDisabled(False)
        fit_grid.addWidget(self.gpu_checkbox, 4, 0, 1, 2)

        self.fit_model.currentIndexChanged.connect(self.on_fit_model_changed)
        self.fit_optimizer.currentIndexChanged.connect(
            self.on_fit_optimizer_changed
        )
        # Populate the optimizer combobox and set visibility for the default
        # model.
        self.on_fit_model_changed()

        # 3D via astigmatism (Gaussian models); shown for non-spline fits
        self.z_groupbox = z_groupbox = QtWidgets.QGroupBox(
            "3D via Astigmatism"
        )
        vbox.addWidget(z_groupbox)

        z_grid = QtWidgets.QGridLayout(z_groupbox)
        load_z_calib = QtWidgets.QPushButton("Load calibration")
        load_z_calib.setToolTip(
            "Load a 3D calibration file (.yaml).\n"
            "Please visit the documentation:\n"
            "https://picassosr.readthedocs.io/en/latest/\n"
            "for instructions on how to obtain it."
        )
        load_z_calib.setAutoDefault(False)
        load_z_calib.clicked.connect(self.load_z_calib)
        z_grid.addWidget(load_z_calib, 0, 1)
        self.fit_z_checkbox = QtWidgets.QCheckBox("Fit Z")
        self.fit_z_checkbox.setToolTip("Fit z coordinates?")
        self.fit_z_checkbox.setEnabled(False)
        self.z_calib_label = QtWidgets.QLabel("-- no calibration loaded --")
        self.z_calib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.z_calib_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        z_grid.addWidget(self.z_calib_label, 0, 0)
        magnification_label = QtWidgets.QLabel("Magnification factor:")
        magnification_label.setToolTip(
            "Factor used to correct for z-position abberation due to\n"
            "refractive index mismatch, see Huang B, et al. Science. 2008."
        )
        z_grid.addWidget(magnification_label, 1, 0)
        self.magnification_factor = QtWidgets.QDoubleSpinBox()
        self.magnification_factor.setRange(0, 1e6)
        self.magnification_factor.setDecimals(4)
        self.magnification_factor.setValue(0.79)
        z_grid.addWidget(self.magnification_factor, 1, 1)

        # The astigmatism z fit runs on whichever device the fit box's
        # 'Use GPU' checkbox selects
        fit_z_row = QtWidgets.QHBoxLayout()
        fit_z_row.addWidget(lib.HelpButton(self.CALIB_URL))
        fit_z_row.addWidget(self.fit_z_checkbox)
        z_grid.addLayout(fit_z_row, 2, 0, 1, 2)

        # Experimental PSF (cubic spline). The calibration is loaded here and
        # passed to the fit when the "Experimental PSF (cubic spline)" model
        # is chosen. Always available: the fit runs on the CPU
        # (picasso.fitting.splinefit) and, with a CUDA GPU, on the GPU
        # (picasso.fitting.splinefit_cuda).
        self.spline_groupbox = spline_groupbox = QtWidgets.QGroupBox(
            "Experimental PSF (spline)"
        )
        spline_groupbox.setToolTip(
            "Fit an experimentally measured PSF, modelled as a cubic "
            "spline. Runs on the CPU, or on a CUDA-capable GPU.\n\n"
            "Li, Y., Mund, M., Hoess, P. et al. Real-time 3D "
            "single-molecule localization using experimental point "
            "spread functions. Nat Methods 15, 367-369 (2018). "
            "https://doi.org/10.1038/nmeth.4661\n\n"
            "Babcock, H.P., Zhuang, X. Analyzing Single Molecule "
            "Localization Microscopy Data Using Cubic Splines. Sci Rep "
            "7, 552 (2017). https://doi.org/10.1038/s41598-017-00622-w"
            "\n\n"
            "Przybylski, A., Thiel, B., Keller-Findeisen, J. et al. "
            "Gpufit: An open-source toolkit for GPU-accelerated curve "
            "fitting. Sci Rep 7, 15722 (2017). "
            "https://doi.org/10.1038/s41598-017-15313-9"
            "\n\n"
            "Multichannel (global) fitting follows globLoc:\n"
            "Li, Y., Shi, W., Liu, S. et al. Global fitting for "
            "high-accuracy multi-channel single-molecule localization. "
            "Nat Commun 13, 3133 (2022). "
            "https://doi.org/10.1038/s41467-022-30719-4"
        )
        vbox.addWidget(spline_groupbox)
        spline_grid = QtWidgets.QGridLayout(spline_groupbox)
        load_spline_calib = QtWidgets.QPushButton("Load calibration")
        load_spline_calib.setToolTip(
            "Load a cubic-spline PSF calibration (.hdf5), built via\n"
            "3D > Calibrate spline PSF. Used by the 'Experimental PSF\n"
            "(cubic spline)' fit model."
        )
        load_spline_calib.setAutoDefault(False)
        load_spline_calib.clicked.connect(self.load_spline_calib)
        spline_grid.addWidget(load_spline_calib, 0, 1)
        spline_grid.addWidget(lib.HelpButton(self.SPLINE_URL), 0, 2)
        self.spline_calib_label = QtWidgets.QLabel(
            "-- no calibration loaded --"
        )
        self.spline_calib_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.spline_calib_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        spline_grid.addWidget(self.spline_calib_label, 0, 0)

        self.link_photons_checkbox = QtWidgets.QCheckBox(
            "Link photon counts across channels"
        )
        self.link_photons_checkbox.setToolTip(LINK_PHOTONS_TIP)
        self.link_photons_checkbox.setTristate(False)
        self.link_photons_checkbox.setChecked(True)
        self.link_photons_checkbox.hide()  # shown for 2-6 channel calibs
        spline_grid.addWidget(self.link_photons_checkbox, 1, 0, 1, 3)

        # show the calibration box that matches the initial fit model
        self._update_calib_group_visibility()

        if "Cameras" in CONFIG:
            camera = self.camera.currentText()
            if camera in CONFIG["Cameras"]:
                self.on_camera_changed(0)
                camera_config = CONFIG["Cameras"][camera]
                if (
                    "Sensitivity" in camera_config
                    and "Sensitivity Categories" in camera_config
                ):
                    self.update_sensitivity()

        # Sample quality
        quality_groupbox = QtWidgets.QGroupBox(
            "Postprocessing and Sample Quality"
        )
        quality_groupbox.setToolTip(
            "Drift correction + drift, bright time and NeNA estimates."
        )
        vbox.addWidget(quality_groupbox)
        quality_grid = QtWidgets.QGridLayout(quality_groupbox)

        # drift correction
        self.aim_undrift_checkbox = QtWidgets.QCheckBox("Apply AIM")
        self.aim_undrift_checkbox.setToolTip(
            "Apply the AIM for drift correction"
        )
        self.aim_undrift_checkbox.setChecked(False)
        self.aim_undrift_checkbox.stateChanged.connect(
            self.on_aim_undrift_changed
        )
        quality_grid.addWidget(self.aim_undrift_checkbox, 0, 0)

        aim_segmentation_label = QtWidgets.QLabel("Segmentation")
        aim_segmentation_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        aim_segmentation_label.setToolTip(
            "Select the number of frames in a segment for AIM."
        )
        quality_grid.addWidget(aim_segmentation_label, 0, 1)
        self.aim_segmentation = QtWidgets.QSpinBox()
        self.aim_segmentation.setRange(1, 100_000)
        self.aim_segmentation.setValue(1000)
        self.aim_segmentation.setSingleStep(100)
        self.aim_segmentation.setEnabled(False)
        quality_grid.addWidget(self.aim_segmentation, 0, 2)

        self.fiducial_check = QtWidgets.QCheckBox("Use fiducials")
        self.fiducial_check.setToolTip(
            "Use fiducial markers for drift correction?\n"
        )
        self.fiducial_check.setChecked(False)
        quality_grid.addWidget(self.fiducial_check, 0, 3)

        # sample quality
        self.quality_check = QtWidgets.QPushButton(
            "Estimate and add to database"
        )
        self.quality_check.setEnabled(False)
        quality_grid.addWidget(self.quality_check, 1, 0, 1, 4)
        self.quality_check.clicked.connect(self.check_quality)

        self.quality_grid_labels = [
            QtWidgets.QLabel("Locs/frame"),
            QtWidgets.QLabel("NeNA"),
            QtWidgets.QLabel("Mean drift"),
            QtWidgets.QLabel("Bright time (frames)"),
        ]
        for idx, _ in enumerate(self.quality_grid_labels):
            quality_grid.addWidget(_, idx + 1, 1)

        self.quality_grid_values = [
            QtWidgets.QLabel(""),
            QtWidgets.QLabel(""),
            QtWidgets.QLabel(""),
            QtWidgets.QLabel(""),
        ]

        for idx, _ in enumerate(self.quality_grid_values):
            quality_grid.addWidget(_, idx + 1, 2)

        self.reset_quality_check()

        # adjust the size of the dialog to fit its contents
        hint = container.sizeHint()
        lib.adjust_widget_size(self, hint)

    def reset_quality_check(self) -> None:
        """Reset the quality check UI elements."""
        self.quality_check.setEnabled(False)
        self.quality_check.setVisible(True)

        for _ in self.quality_grid_labels:
            _.setVisible(False)

        for _ in self.quality_grid_values:
            _.setVisible(False)
            _.setText("")

    def on_pixelsize_changed(self) -> None:
        """If the movie is loaded and scale bar is shown, update it."""
        if (
            hasattr(self.window, "movie")
            and self.window.movie is not None
            and hasattr(self.window, "scalebar_action")
            and self.window.scalebar_action.isChecked()
        ):
            self.window.draw_frame()

    def update_roi_display(self, skip_dialog: bool = False) -> None:
        """Refresh the compact ROI field (and the ROI sub-dialog) from
        the view's ROIs.

        The field shows nothing (placeholder "Whole frame") when there
        are no ROIs, the four editable coordinates when there is exactly
        one, and a read-only count when there are several.

        Parameters
        ----------
        skip_dialog : bool, optional
            If True, do not refresh the ROI sub-dialog's table (used when
            the call originates from that dialog to avoid recursion).
        """
        view = self.window.view
        n = len(view.rois)
        self._updating_roi_field = True
        if n == 1:
            self.roi_field.setReadOnly(False)
            (y_min, x_min), (y_max, x_max) = view.rois[0]
            self.roi_field.setText(
                f"{int(y_min)},{int(x_min)},{int(y_max)},{int(x_max)}"
            )
        elif n == 0:
            self.roi_field.setReadOnly(False)
            self.roi_field.setText("")
        else:
            self.roi_field.setReadOnly(True)
            self.roi_field.setText(f"{n} ROIs")
        self._updating_roi_field = False
        if not skip_dialog and self.roi_dialog is not None:
            self.roi_dialog.update_table()

    def on_split_fov_changed(self) -> None:
        """Toggle split-FOV region mode (ROIs become channels of one movie)."""
        self.window.set_split_fov_mode(self.split_fov_checkbox.isChecked())

    def on_roi_field_changed(self) -> None:
        """Clear the ROIs when the user empties the field."""
        if self._updating_roi_field or self.roi_field.isReadOnly():
            return
        if self.roi_field.text() == "":
            self.window.view.rois = []
            self.window.view.selected_roi = None
            if self.roi_dialog is not None:
                self.roi_dialog.update_table()
            self.window.draw_frame()

    def on_roi_field_finished(self) -> None:
        """Parse the single ROI typed into the compact field."""
        if self._updating_roi_field or self.roi_field.isReadOnly():
            return
        text = self.roi_field.text()
        if text == "":
            return
        try:
            y_min, x_min, y_max, x_max = (int(v) for v in text.split(","))
        except ValueError:
            return  # incomplete input, wait for the user to finish
        box = self.window.parameters.get("Box Size", 7)
        self.window.view.rois = localize.clip_rois(
            [[[y_min, x_min], [y_max, x_max]]], min_size=box
        )
        self.window.view.selected_roi = None
        self.update_roi_display()
        self.window.draw_frame()

    def on_edit_rois(self) -> None:
        """Open (or raise) the ROI management sub-dialog."""
        if self.roi_dialog is None:
            self.roi_dialog = ROIDialog(self.window)
        self.roi_dialog.update_table()
        self.roi_dialog.show()
        self.roi_dialog.raise_()
        self.roi_dialog.activateWindow()

    def on_fit_model_changed(self) -> None:
        """Repopulate the optimizer combobox for the selected model and
        show/hide the optimizer controls. Models without an optimizer
        (e.g. averaging) hide the optimizer row and its parameters."""
        model = self.fit_model.currentText()
        optimizers = FIT_MODELS[model]["optimizers"]
        if optimizers is None:
            self.optimizer_label.hide()
            self.fit_optimizer.hide()
            self.fit_stack.hide()
            self._apply_convergence_defaults()
        else:
            self.optimizer_label.show()
            self.fit_optimizer.show()
            self.fit_stack.show()
            self.fit_optimizer.blockSignals(True)
            self.fit_optimizer.clear()
            self.fit_optimizer.addItems(list(optimizers.keys()))
            self.fit_optimizer.setCurrentIndex(0)
            self.fit_optimizer.blockSignals(False)
            self.on_fit_optimizer_changed()
        self._update_calib_group_visibility()

    def _update_calib_group_visibility(self) -> None:
        """Show only the calibration box relevant to the selected fit model:
        the astigmatism z-calibration for Gaussian models, or the spline PSF
        calibration for the experimental-PSF (spline) model. Guarded so the
        initial ``on_fit_model_changed`` call (before the boxes exist) is a
        no-op."""
        needs_spline = FIT_MODELS[self.fit_model.currentText()].get(
            "needs_spline_calibration", False
        )
        if self.z_groupbox is not None:
            self.z_groupbox.setVisible(not needs_spline)
        if self.spline_groupbox is not None:
            self.spline_groupbox.setVisible(needs_spline)

    def current_fit_code(self) -> str:
        """The ``fit2D`` code the current selection would run."""
        model = self.fit_model.currentText()
        optimizer = self.fit_optimizer.currentText()
        if not optimizer and FIT_MODELS[model]["optimizers"] is not None:
            return ""
        return _effective_fit_code(
            _fit_code(model, optimizer), self.gpu_checkbox.isChecked()
        )

    def _apply_convergence_defaults(self) -> None:
        """Show the selected method's own convergence schedule."""
        code = self.current_fit_code()
        if not code:
            return
        self.fit_stack.setCurrentIndex(1 if code in _CONVERGENCE_CODES else 0)
        if code == self._last_fit_code:
            return
        self._last_fit_code = code
        defaults = _CONVERGENCE_DEFAULTS.get(code)
        if defaults is not None:
            self.convergence_criterion.setValue(defaults[0])
            self.max_it.setValue(defaults[1])

    def on_fit_optimizer_changed(self) -> None:
        """Enable/disable the GPU fitting checkbox based on the selected
        model and optimizer, then show that method's convergence schedule."""
        model = self.fit_model.currentText()
        optimizer = self.fit_optimizer.currentText()
        if optimizer and _fit_code(model, optimizer).endswith("-gpu"):
            # GPU-only method (e.g. the rotated elliptical Gaussian):
            # fitting always runs on the GPU, so pin the checkbox.
            self.gpu_checkbox.setChecked(True)
            self.gpu_checkbox.setDisabled(True)
        elif optimizer in ("Least squares", "MLE"):
            # Both estimators are implemented on the GPU
            self.gpu_checkbox.setDisabled(not GPU_FITTING_AVAILABLE)
        else:
            self.gpu_checkbox.setChecked(False)
            self.gpu_checkbox.setDisabled(True)
        self.on_gpu_fitting_changed()

    def on_frames_edit_changed(self) -> None:
        """Handle changes when text is deleted."""
        if self.frames_edit.text() == "":
            self.window.frame_range = None

    def on_frames_edit_finished(self) -> None:
        """Handle the completion of frames editing. Parses one or more
        semicolon-separated ``min,max`` segments (1-indexed, inclusive),
        clamps each to the movie length, and stores them as a list of
        0-indexed ``[lo, hi]`` segments in ``window.frame_range``."""
        if not self.window.movie or self.frames_edit.text() == "":
            return
        n_frames = len(self.window.movie)
        segments = []
        for part in self.frames_edit.text().split(";"):
            min_frame, max_frame = [int(_) for _ in part.split(",")]
            min_frame = max(1, min_frame)
            max_frame = min(n_frames, max_frame)
            segments.append([min_frame, max_frame])
        # store as 0-indexed inclusive segments
        self.window.frame_range = [[lo - 1, hi - 1] for lo, hi in segments]
        self.frames_edit.setText(
            "; ".join(f"{lo},{hi}" for lo, hi in segments)
        )

    def load_z_calib(self) -> None:
        """Load the 3D calibration from a user-selected YAML file."""
        if self.z_calibration_path:
            dialog_directory, _ = os.path.split(self.z_calibration_path)
        else:
            dialog_directory = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load 3d calibration",
            directory=dialog_directory,
            filter="*.yaml",
        )
        if path:
            self.update_z_calib(path)

    def load_spline_calib(self) -> None:
        """Load a cubic-spline PSF calibration from a user-selected HDF5."""
        if self.spline_calibration_path:
            dialog_directory, _ = os.path.split(self.spline_calibration_path)
        else:
            dialog_directory = None
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load spline PSF calibration",
            directory=dialog_directory,
            filter="*.hdf5",
        )
        if path:
            self.update_spline_calib(path)

    def update_spline_calib_with_config_path(self) -> None:
        """Retrieve the spline PSF calibration path that corresponds to the
        selected camera and emission wavelength, from the config."""
        if self.spline_groupbox is None:  # fit UI not built yet
            return
        if "spline-calibrations" not in CONFIG:
            return
        camera = self.camera.currentText()
        fp_calib_lam = CONFIG["spline-calibrations"].get(camera)
        if fp_calib_lam is not None:
            em_combo = self.emission_combos[camera]
            wavelength = int(em_combo.currentText())
            fp_calib = fp_calib_lam.get(wavelength)
            if fp_calib is not None:
                self.update_spline_calib(fp_calib)

    def update_spline_calib(self, path: str | None) -> None:
        """Load (or clear) a cubic-spline PSF calibration from an HDF5 file."""
        if path:
            if os.path.exists(path):
                try:
                    self.spline_calibration = io.load_spline_calibration(path)
                except Exception as e:
                    self.update_spline_calib(None)
                    self.spline_calib_label.setText(
                        "-- invalid calibration --"
                    )
                    self.spline_calib_label.setToolTip(str(e))
                    return
                self.spline_calibration_path = path
            else:
                self.update_spline_calib(None)
                self.spline_calib_label.setText(
                    "-- calibration path not found --"
                )
                self.spline_calib_label.setToolTip("")
                return
            self.spline_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
            )
            self.spline_calib_label.setText(os.path.basename(path))
            self.spline_calib_label.setToolTip(path)
            # split-FOV: drop the calibration's channel regions into the view
            # and enter split-FOV mode, so the registration can be inspected and
            # fine-tuned on this data (arrow-key nudge / drag) and re-drawn if
            # the split moved. The fit then uses whatever regions are shown.
            if self.spline_calibration.get("split_fov"):
                regions = self.spline_calibration.get("regions") or []
                self.window.view.rois = [
                    [
                        [int(r[0][0]), int(r[0][1])],
                        [int(r[1][0]), int(r[1][1])],
                    ]
                    for r in regions
                ]
                self.window.view.selected_roi = None
                self.split_fov_checkbox.blockSignals(True)
                self.split_fov_checkbox.setChecked(True)
                self.split_fov_checkbox.blockSignals(False)
                self.window.set_split_fov_mode(True)
                self.update_roi_display()
                self.window.draw_frame()
        else:
            self.spline_calibration = {}
            self.spline_calibration_path = None
            self.spline_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.spline_calib_label.setText("-- no calibration loaded --")
        self._update_link_photons_visibility()

    def _update_link_photons_visibility(self) -> None:
        """Show the 'Link photons across channels' checkbox only for a
        multichannel spline calibration with 2 to
        ``localize._LINK_XYZ_MAX_CHANNELS`` channels - the range for which
        the photon-decoupled (link-XYZ) fit is supported."""
        # A config auto-load may call update_spline_calib during startup before
        # the fit UI (and this checkbox) is built; ignore until it exists.
        if not hasattr(self, "link_photons_checkbox"):
            return
        cal = self.spline_calibration or {}
        # (see _link_photons_enabled for how the state reaches the fit workers)
        n_channels = int(
            cal.get("n_channels", len(cal.get("channel_transforms", []) or []))
        )
        is_multichannel = cal.get("model") == "spline-3d-multichannel"
        show = is_multichannel and (
            2 <= n_channels <= localize._LINK_XYZ_MAX_CHANNELS
        )
        if show:
            # default the toggle to the mode chosen when the calibration was
            # built (stored as "link_photons"); the user can still flip it.
            self.link_photons_checkbox.setChecked(
                bool(cal.get("link_photons", True))
            )
        self.link_photons_checkbox.setVisible(show)

    def _link_photons_enabled(self) -> bool:
        """Whether the multichannel spline fit should link photons across
        channels (shared amplitude, model 11). True when the checkbox is absent
        (no GPU / non-multichannel) so behaviour is unchanged by default."""
        cb = getattr(self, "link_photons_checkbox", None)
        return cb.isChecked() if cb is not None else True

    def update_z_calib_with_config_path(self):
        """Retrieve the z calibration path that corresponds to the
        selected camera and emission wavelength, from the config"""
        if "z-calibrations" not in CONFIG:
            return
        camera = self.camera.currentText()
        fp_calib_lam = CONFIG["z-calibrations"].get(camera)
        if fp_calib_lam is not None:
            em_combo = self.emission_combos[camera]
            wavelength = int(em_combo.currentText())
            fp_calib = fp_calib_lam.get(wavelength)
            if fp_calib is not None:
                self.update_z_calib(fp_calib)
                # To avoid the situation where the user runs a 3D localization
                # just because the calib file was loaded, uncheck the "Fit Z"
                # checkbox;
                self.fit_z_checkbox.setChecked(False)
            else:
                self.update_z_calib(None)
        else:
            self.update_z_calib(None)

    def update_z_calib(self, path: str) -> None:
        """Load the 3D calibration from a YAML file."""
        if path:
            if os.path.exists(path):
                self.z_calibration = io.load_calibration(path)
                self.z_calibration_path = path
            else:
                self.update_z_calib(None)
                self.z_calib_label.setText("-- calibration path not found --")
                self.z_calib_label.setToolTip("")
                return
            self.z_calib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self.z_calib_label.setText(os.path.basename(path))
            self.z_calib_label.setToolTip(path)
            self.fit_z_checkbox.setEnabled(True)
            self.fit_z_checkbox.setChecked(True)
        else:
            self.z_calibration = {}
            self.z_calibration_path = None
            self.z_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.z_calib_label.setText("-- no calibration loaded --")
            self.fit_z_checkbox.setChecked(False)
            self.fit_z_checkbox.setEnabled(False)

    def quality_progress(self, msg: str, index: int, result: str) -> None:
        """Update the quality progress UI elements."""
        if msg != "":
            self.window.status_bar.showMessage(msg)
        else:
            self.quality_grid_values[index].setText(result)

    def quality_progress_finished(self, msg: str) -> None:
        """Handle the completion of the quality progress."""
        self.window.status_bar.showMessage(msg)

    def check_quality(self) -> None:
        """Start the quality check worker thread."""
        self.quality_check.setVisible(False)
        for _ in self.quality_grid_labels:
            _.setVisible(True)
        for _ in self.quality_grid_values:
            _.setVisible(True)

        self.q_worker = QualityWorker(
            self.window.locs,
            self.window.info,
            self.window.movie_path,
            self.pixelsize,
        )
        self.q_worker.progressMade.connect(self.quality_progress)
        self.q_worker.finished.connect(self.quality_progress_finished)
        self.q_worker.start()

    def on_aim_undrift_changed(self, state: int) -> None:
        """Enable/disable AIM segmentation spinbox."""
        if state == 0:  # unchecked
            self.aim_segmentation.setEnabled(False)
        else:  # checked
            self.aim_segmentation.setEnabled(True)

    def on_link_params_changed(self, _state: int = 0) -> None:
        """A cross-channel link toggle changed: converge every channel to the
        current (shared) settings so it takes effect immediately, not only on
        the next channel switch."""
        self.window.propagate_linked_params()

    def on_box_changed(self) -> None:
        """Handle changes to the parameter boxes."""
        self.window.on_parameters_changed()

    def on_camera_changed(self, index: int) -> None:
        """Handle changes to the camera selection."""
        self.gain.setValue(1)
        self.cam_settings.setCurrentIndex(index)
        camera = self.camera.currentText()
        cam_config = CONFIG["Cameras"][camera]
        if "Baseline" in cam_config:
            self.baseline.setValue(cam_config["Baseline"])
        if "DefaultGain" in cam_config:
            self.gain.setValue(cam_config["DefaultGain"])
        if "Pixelsize" in cam_config:
            self.pixelsize.setValue(cam_config["Pixelsize"])
        self.update_sensitivity()
        self.update_qe()

        # load 3D calibration
        self.update_z_calib_with_config_path()
        # load spline PSF calibration
        self.update_spline_calib_with_config_path()

    def update_qe(self) -> None:
        """Update QE. Note that QE is not used in the analysis, the
        method is kept for backward compatibility."""
        camera = self.camera.currentText()
        cam_config = CONFIG["Cameras"][camera]
        if "Quantum Efficiency" in cam_config:
            qe = cam_config["Quantum Efficiency"]
            try:
                self.qe.setValue(qe)
            except TypeError:
                # qe is not a number
                em_combo = self.emission_combos[camera]
                wavelength = float(em_combo.currentText())
                qe = cam_config["Quantum Efficiency"][wavelength]
                self.qe.setValue(qe)

    def on_emission_changed(self) -> None:
        """Update QE due to change in emission wavelength."""
        self.update_qe()
        self.update_z_calib_with_config_path()
        self.update_spline_calib_with_config_path()

    def on_mng_spinbox_changed(self, value: int) -> None:
        """Handle change to the min. net gradient spinbox."""
        if value < self.mng_slider.minimum():
            self.mng_min_spinbox.setValue(value)
        if value > self.mng_slider.maximum():
            self.mng_max_spinbox.setValue(value)
        self.mng_slider.setValue(value)

    def on_mng_slider_changed(self, value: int) -> None:
        """Handle change to the min. net gradient slider."""
        self.mng_spinbox.setValue(value)
        if self.preview_checkbox.isChecked():
            self.window.on_parameters_changed()

    def on_mng_min_changed(self, value: int) -> None:
        self.mng_slider.setMinimum(value)

    def on_mng_max_changed(self, value: int) -> None:
        self.mng_slider.setMaximum(value)

    def on_preview_changed(self) -> None:
        """Update the frame with/without indentification preview."""
        self.window.draw_frame()

    def on_link_colors_changed(self) -> None:
        """Redraw with/without cross-channel link colour-coding of the
        identification boxes."""
        self.window.draw_frame()

    def on_gpu_fitting_changed(self) -> None:
        """Handle changes to the GPU fitting option."""
        self._apply_convergence_defaults()
        # this dialog is created before the window's movie attribute
        if getattr(self.window, "movie", None) is not None:
            self.window.draw_frame()

    def get_camera(self, info: dict) -> tuple[str, list[str]]:
        """Get the camera name from the provided camera info."""
        if "Camera" in info and "Cameras" in CONFIG:
            cameras = [
                self.camera.itemText(_) for _ in range(self.camera.count())
            ]
            camera = info["Camera"]
            if camera in cameras:
                return camera, cameras
        return None, None

    def set_gain(self, camera: str, mm_info: dict, cam_config: dict) -> None:
        """Set EM gain if the relevant information is available in the
        config and metadata."""
        if "Gain Property Name" in cam_config:
            gain_property_name = cam_config["Gain Property Name"]
            gain = mm_info[camera + "-" + gain_property_name]
            if "EM Switch Property" in cam_config:
                switch_property_name = cam_config["EM Switch Property"]["Name"]
                switch_property_value = mm_info[
                    camera + "-" + switch_property_name
                ]
                if (
                    switch_property_value
                    == cam_config["EM Switch Property"][True]
                ):
                    self.gain.setValue(int(gain))
                else:
                    self.gain.setValue(1)

    def set_sensitivity(
        self, camera: str, mm_info: dict, cam_config: dict
    ) -> None:
        if "Sensitivity Categories" in cam_config:
            cam_combos = self.cam_combos[camera]
            categories = cam_config["Sensitivity Categories"]
            for i, category in enumerate(categories):
                property_name = camera + "-" + category
                if property_name in mm_info:
                    e_setting = mm_info[camera + "-" + category]
                    cam_combo = cam_combos[i]
                    for index in range(cam_combo.count()):
                        if cam_combo.itemText(index) == e_setting:
                            cam_combo.setCurrentIndex(index)
                            break

    def set_wavelength(
        self, camera: str, mm_info: dict, cam_config: dict
    ) -> None:
        if "Channel Device" in cam_config:
            channel_device_name = cam_config["Channel Device"]["Name"]
            channel = mm_info[channel_device_name]
            channels = cam_config["Channel Device"]["Emission Wavelengths"]
            if channel in channels:
                wavelength = str(channels[channel])
                em_combo = self.emission_combos[camera]
                for index in range(em_combo.count()):
                    if em_combo.itemText(index) == wavelength:
                        em_combo.setCurrentIndex(index)
                        break

    def set_camera_parameters(self, info: dict) -> None:
        """Set the camera parameters based on the provided camera
        info."""
        camera, cameras = self.get_camera(info)
        if camera is None:
            return

        index = cameras.index(camera)
        self.camera.setCurrentIndex(index)
        if "Micro-Manager Metadata" in info:
            mm_info = info["Micro-Manager Metadata"]
            cam_config = CONFIG["Cameras"][camera]
            self.set_gain(camera, mm_info, cam_config)
            self.set_sensitivity(camera, mm_info, cam_config)
            self.set_wavelength(camera, mm_info, cam_config)

    def update_sensitivity(self) -> None:
        """Update the sensitivity settings for the current camera."""
        camera = self.camera.currentText()
        cam_config = CONFIG["Cameras"][camera]
        sensitivity = cam_config["Sensitivity"]
        if "Sensitivity" in cam_config:
            try:
                self.sensitivity.setValue(sensitivity)
            except TypeError:
                # sensitivity is not a number
                categories = cam_config["Sensitivity Categories"]
                for i, category in enumerate(categories):
                    cat_combo = self.cam_combos[camera][i]
                    sensitivity = sensitivity[cat_combo.currentText()]
                self.sensitivity.setValue(sensitivity)


class ContrastDialog(lib.Dialog):
    """Choose display contrast."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Contrast")
        self.resize(200, 0)
        self.setModal(False)
        grid = QtWidgets.QGridLayout(self)
        black_label = QtWidgets.QLabel("Black:")
        black_label.setToolTip("Min. intensity rendered")
        grid.addWidget(black_label, 0, 0)
        self.black_spinbox = lib.LogDoubleSpinBox()
        self.black_spinbox.setDecimals(0)
        self.black_spinbox.setKeyboardTracking(False)
        self.black_spinbox.setRange(1, 999999)
        self.black_spinbox.valueChanged.connect(self.on_contrast_changed)
        grid.addWidget(self.black_spinbox, 0, 1)
        white_label = QtWidgets.QLabel("White:")
        white_label.setToolTip("Max. intensity rendered")
        grid.addWidget(white_label, 1, 0)
        self.white_spinbox = lib.LogDoubleSpinBox()
        self.white_spinbox.setDecimals(0)
        self.white_spinbox.setKeyboardTracking(False)
        self.white_spinbox.setRange(1, 999999)
        self.white_spinbox.valueChanged.connect(self.on_contrast_changed)
        grid.addWidget(self.white_spinbox, 1, 1)
        self.auto_checkbox = QtWidgets.QCheckBox("Auto")
        self.auto_checkbox.setToolTip(
            "Set the range automatically for each frame?"
        )
        self.auto_checkbox.setTristate(False)
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.stateChanged.connect(self.on_auto_changed)
        grid.addWidget(self.auto_checkbox, 2, 0, 1, 2)
        self.silent_contrast_change = False

    def change_contrast_silently(self, black: int, white: int) -> None:
        """Change the contrast values without emitting signals."""
        self.silent_contrast_change = True
        self.black_spinbox.setValue(black)
        self.white_spinbox.setValue(white)
        self.silent_contrast_change = False

    def on_contrast_changed(self, value: int) -> None:
        if not self.silent_contrast_change:
            self.auto_checkbox.setChecked(False)
            self.window.draw_frame()

    def on_auto_changed(self, state: int) -> None:
        if state:
            movie = self.window.movie
            frame_number = self.window.curr_frame_number
            frame = movie[frame_number]
            self.change_contrast_silently(frame.min(), frame.max())
            self.window.draw_frame()


class LocColumnSelectionDialog(lib.Dialog):
    """Dialog for selecting which columns to save in the localization
    file."""

    # Target height of one on-screen column of checkboxes
    _MAX_ROWS_PER_COLUMN = 20

    def __init__(self, parent: QtWidgets.QMainWindow) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select columns to save")
        self.setModal(True)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # one titled group box per LOCALIZATION_COLUMNS key, packed into as
        # many side-by-side columns as it takes to stay under the row target
        self.column_checkboxes = {}
        screen_column = None
        rows_used = 0
        for key, columns in localize.LOCALIZATION_COLUMNS.items():
            if (
                screen_column is None
                or rows_used + len(columns) > self._MAX_ROWS_PER_COLUMN
            ):
                screen_column = QtWidgets.QVBoxLayout()
                screen_column.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
                layout.addLayout(screen_column)
                rows_used = 0
            group = QtWidgets.QGroupBox(key)
            group_layout = QtWidgets.QVBoxLayout(group)
            for column in columns:
                checkbox = QtWidgets.QCheckBox(column)
                checkbox.setChecked(True)
                if column in lib.REQUIRED_COLUMNS:
                    checkbox.setDisabled(True)
                self.column_checkboxes[column] = checkbox
                group_layout.addWidget(checkbox)
            screen_column.addWidget(group)
            rows_used += len(columns)

        self.load_user_settings()

    def load_user_settings(self) -> None:
        """Reads the columns to save from user settings."""
        settings = io.load_user_settings()
        columns = settings["Localize"].get("Columns to save", None)
        if columns is None:
            columns = {}
            for subcols in localize.LOCALIZATION_COLUMNS.values():
                for col in subcols:
                    columns[col] = True
        for column, checkbox in self.column_checkboxes.items():
            is_checked = columns.get(column, True)
            checkbox.setChecked(is_checked)


class Window(QtWidgets.QMainWindow):
    """The main window.

    ...

    Attributes
    ----------
    camera_info : dict
        Camera information, such as gain, sensitivity, etc.
    contrast_dialog : ContrastDialog
        The dialog for adjusting display contrast.
    channels : list of Channel
        Per-channel state for multichannel data. The active channel
        (``current_channel``) is mirrored into the flat attributes
        (``movie``, ``info``, ``movie_path`` ...). A single movie is held
        as a one-element list.
    current_channel : int
        Index of the active channel within ``channels``.
    extra_info : list of dict
        Movie metadata in a format of a dictionary, with Localize info.
    identifications : pd.DataFrame
        Identified spots - frame, position, net gradient.
    last_identification_info : dict
        A dictionary of analysis parameters used for the last operation.
        Used to save user settings when closing the application.
    locs : pd.DataFrame
        Resulting localizations.
    info : list of dict
        Movie metadata in a format of a dictionary, without Localize
        info, see self.extra_info.
    movie : np.memmap or None
        Loaded movie (frame, y, x) of the active channel.
    movie_path : str or list
        Path to the active channel's movie file (empty list when nothing
        is loaded).
    parameters_dialog : ParametersDialog
        The dialog for adjusting parameters.
    ready_for_fit : bool
        If True, spots were identified and are ready to be fitted.
    scene : Scene
        The scene for displaying the image.
    status_bar : QtWidgets.QStatusBar
        Status bar displayed in the bottom of the window.
    view : View
        The main view for displaying the image.
    """

    DOCS_URL = "https://picassosr.readthedocs.io/en/latest/localize.html"

    def __init__(self) -> None:
        super().__init__()
        # Init GUI
        self.setWindowTitle(f"Picasso v{__version__}: Localize")

        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "localize.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(768, 768)
        self.parameters_dialog = ParametersDialog(self)
        self.contrast_dialog = ContrastDialog(self)
        self.columns_dialog = LocColumnSelectionDialog(self)
        self.metadata_dialog = lib.MetadataDialog(self)
        self.user_settings_dialog = lib.UserSettingsDialog(self)
        self.init_menu_bar()
        self.view = View(self)
        self.scene = Scene(self)
        self.view.setScene(self.scene)
        # Slider below the movie for quickly navigating between frames.
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setMaximumHeight(15)
        self.frame_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 4px;
                background: #b0b0b0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 10px;
                height: 12px;
                margin: -5px 0;
                border-radius: 3px;
                background: #5a5a5a;
            }
            """
        )
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        # Channel selector (hidden unless several channels are loaded).
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.setVisible(False)
        self.channel_combo.currentIndexChanged.connect(
            self.on_channel_combo_changed
        )
        central_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.channel_combo)
        central_layout.addWidget(self.view)
        central_layout.addWidget(self.frame_slider)
        self.setCentralWidget(central_widget)
        self.status_bar = self.statusBar()
        self.status_bar_frame_indicator = QtWidgets.QLabel()
        self.status_bar.addPermanentWidget(self.status_bar_frame_indicator)

        # re-entrancy guard for draw_frame (see on_scroll)
        self._drawing_frame = False
        # Holds the curr movie as a numpy memmap in the format
        # (frame, y, x)
        self.movie = None
        # Dictionary of analysis parameters used for the last operation
        self.last_identification_info = None
        # Dataframe of identifcations with fields frame, x and y
        self.identifications = None
        self.ready_for_fit = False
        self.locs = None
        # Snapshot of ``self.locs`` used to draw the FitMarker overlay.
        # Kept separate so drift correction can mutate ``self.locs``
        # without moving the markers shown on screen.
        self.locs_display = None
        self.movie_path = []
        # None analyzes all frames; otherwise a list of 0-indexed
        # inclusive [lo, hi] segments to restrict identification to
        self.frame_range = None
        self.info = []
        self.extra_info = []
        self._active_worker = None
        # Bookkeeping for a multichannel "Identify" (Ctrl+I) batch that runs
        # identification on every channel in turn; None when not running.
        self._multi_identify = None
        # Multichannel state. ``self.channels[self.current_channel]`` is
        # the active channel, mirrored into the flat attributes above.
        # Single movies are stored as a one-element list.
        self.channels = []
        self.current_channel = 0
        # Guards the parameter/contrast handlers while restoring a
        # channel's dialog values, so they don't wipe the channel's
        # locs/markers.
        self._switching_channel = False

        # Background movie-loading worker state (see open_channels_from_files).
        self._load_thread = None
        self._load_worker = None
        self._load_progress = None
        # Load request made while a cancelled load is still winding down;
        # started as soon as the old worker thread is torn down.
        self._pending_load = None
        self._load_t0 = 0.0
        self._load_multi_file = False
        # Make sure a running loader thread is stopped before the
        # QApplication is destroyed; destroying a live QThread aborts the
        # process. ``aboutToQuit`` fires on every quit path (including
        # sys.exit through the event loop), unlike ``closeEvent``.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_movie_load)

        self.load_user_settings()

    def load_user_settings(self) -> None:
        """Load user settings based on the last-used parameters."""
        settings = io.load_user_settings()
        pwd = []
        box_size = []
        gradient = []
        try:
            pwd = settings["Localize"]["PWD"]
            box_size = settings["Localize"]["box_size"]
            gradient = settings["Localize"]["gradient"]
        except Exception as e:
            print(e)
            pass
        if len(pwd) == 0:
            pwd = []
        if type(box_size) is int:
            self.parameters_dialog.box_spinbox.setValue(box_size)
        if type(gradient) is int:
            self.parameters_dialog.mng_slider.setValue(gradient)

        # Restore the last-used fitting model and optimizer. The model must
        # be set first, since it repopulates the optimizer combobox.
        fit_model = settings["Localize"].get("fit_model", None)
        if fit_model is not None:
            index = self.parameters_dialog.fit_model.findText(fit_model)
            if index >= 0:
                self.parameters_dialog.fit_model.setCurrentIndex(index)
        fit_optimizer = settings["Localize"].get("fit_optimizer", None)
        if fit_optimizer is not None:
            index = self.parameters_dialog.fit_optimizer.findText(
                fit_optimizer
            )
            if index >= 0:
                self.parameters_dialog.fit_optimizer.setCurrentIndex(index)

        self.pwd = pwd

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Close the application, save user settings."""
        self._stop_movie_load()
        settings = io.load_user_settings()
        if self.movie_path != []:
            settings["Localize"]["PWD"] = os.path.dirname(self.movie_path)
            settings["Localize"][
                "box_size"
            ] = self.parameters_dialog.box_spinbox.value()
            settings["Localize"][
                "gradient"
            ] = self.parameters_dialog.mng_slider.value()
        settings["Localize"][
            "fit_model"
        ] = self.parameters_dialog.fit_model.currentText()
        settings["Localize"][
            "fit_optimizer"
        ] = self.parameters_dialog.fit_optimizer.currentText()
        settings["Localize"]["Columns to save"] = {
            column: checkbox.isChecked()
            for column, checkbox in (
                self.columns_dialog.column_checkboxes.items()
            )
        }
        io.save_user_settings(settings)
        QtWidgets.QApplication.instance().closeAllWindows()

    def init_menu_bar(self) -> None:
        """Initialize the menu bar."""
        menu_bar = self.menuBar()

        """ File """
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open movie")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        open_multichannel_action = file_menu.addAction(
            "Open one multichannel movie"
        )
        open_multichannel_action.setShortcut("Ctrl+Shift+O")
        open_multichannel_action.triggered.connect(
            self.open_multichannel_file_dialog
        )
        file_menu.addAction(open_multichannel_action)
        open_channels_action = file_menu.addAction(
            "Open channels from several movies"
        )
        open_channels_action.triggered.connect(
            self.open_channels_from_files_dialog
        )
        file_menu.addAction(open_channels_action)
        open_mm_folder_action = file_menu.addAction(
            "Open MicroManager image folder"
        )
        open_mm_folder_action.triggered.connect(self.open_mm_folder_dialog)
        file_menu.addAction(open_mm_folder_action)
        save_identifications_action = file_menu.addAction(
            "Save identifications"
        )
        save_identifications_action.triggered.connect(
            self.save_identifications_dialog
        )
        load_identifications_action = file_menu.addAction(
            "Load identifications"
        )
        load_identifications_action.triggered.connect(
            self.open_identifications
        )
        load_picks_action = file_menu.addAction(
            "Load picks as identifications"
        )
        load_picks_action.triggered.connect(self.open_picks)
        load_locs_action = file_menu.addAction("Load locs as identifications")
        load_locs_action.triggered.connect(self.open_locs)
        save_action = file_menu.addAction("Save localizations")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_locs_dialog)
        file_menu.addAction(save_action)
        save_spots_action = file_menu.addAction("Save spots")
        save_spots_action.setShortcut("Ctrl+Shift+S")
        save_spots_action.triggered.connect(self.save_spots_dialog)
        file_menu.addAction(save_spots_action)
        select_columns_action = file_menu.addAction("Select columns to save")
        select_columns_action.triggered.connect(self.columns_dialog.show)
        file_menu.addAction(select_columns_action)
        file_menu.addSeparator()
        export_current_action = file_menu.addAction("Export current view")
        export_current_action.setShortcut("Ctrl+E")
        export_current_action.triggered.connect(self.export_current)
        metadata_action = file_menu.addAction("Metadata")
        metadata_action.setShortcut("Ctrl+M")
        metadata_action.triggered.connect(self.show_metadata)
        file_menu.addAction(metadata_action)

        file_menu.addSeparator()
        sounds_menu = file_menu.addMenu("Sound notifications")
        sounds_actiongroup = QtGui.QActionGroup(file_menu)
        default_sound_path = lib.get_sound_notification_path()  # last used
        default_sound_name = os.path.basename(str(default_sound_path))
        for sound in lib.get_available_sound_notifications():
            sound_name = os.path.splitext(str(sound))[0].replace("_", " ")
            action = sounds_actiongroup.addAction(
                QtGui.QAction(sound_name, sounds_menu, checkable=True)
            )
            action.setObjectName(sound)  # store full name
            if default_sound_name == sound:
                action.setChecked(True)
            sounds_menu.addAction(action)
        sounds_actiongroup.triggered.connect(lib.set_sound_notification)
        sounds_menu.addSeparator()
        open_sounds_action = sounds_menu.addAction(
            "Open notification sounds folder..."
        )
        open_sounds_action.triggered.connect(
            lib.open_sound_notifications_folder
        )
        picasso_settings_action = file_menu.addAction("Picasso settings")
        picasso_settings_action.triggered.connect(
            self.user_settings_dialog.show
        )
        open_config_action = file_menu.addAction(
            "Open camera config file location"
        )
        open_config_action.triggered.connect(self.open_config_location)
        help_action = file_menu.addAction("Help")
        help_action.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.DOCS_URL))
        )

        """ View """
        view_menu = menu_bar.addMenu("View")
        previous_frame_action = view_menu.addAction("Previous frame")
        previous_frame_action.setShortcut("Left")
        previous_frame_action.triggered.connect(lambda: self.previous_frame(1))
        view_menu.addAction(previous_frame_action)
        next_frame_action = view_menu.addAction("Next frame")
        next_frame_action.setShortcut("Right")
        next_frame_action.triggered.connect(lambda: self.next_frame(1))
        view_menu.addAction(next_frame_action)
        # Jump multiple frames at once using modifier keys with the arrows.
        # Shift -> 10, Ctrl/Cmd -> 100
        for step, modifier in (
            (10, "Shift"),
            (100, "Ctrl"),
        ):
            jump_back_action = view_menu.addAction(
                "Previous {} frames".format(step)
            )
            jump_back_action.setShortcut("{}+Left".format(modifier))
            jump_back_action.triggered.connect(
                lambda *_, s=step: self.previous_frame(s)
            )
            view_menu.addAction(jump_back_action)
            jump_forward_action = view_menu.addAction(
                "Next {} frames".format(step)
            )
            jump_forward_action.setShortcut("{}+Right".format(modifier))
            jump_forward_action.triggered.connect(
                lambda *_, s=step: self.next_frame(s)
            )
            view_menu.addAction(jump_forward_action)
        view_menu.addSeparator()
        first_frame_action = view_menu.addAction("First frame")
        first_frame_action.setShortcut("Home")
        first_frame_action.triggered.connect(self.first_frame)
        view_menu.addAction(first_frame_action)
        last_frame_action = view_menu.addAction("Last frame")
        last_frame_action.setShortcut("End")
        last_frame_action.triggered.connect(self.last_frame)
        view_menu.addAction(last_frame_action)
        go_to_frame_action = view_menu.addAction("Go to frame")
        go_to_frame_action.setShortcut("Ctrl+G")
        go_to_frame_action.triggered.connect(self.to_frame)
        view_menu.addAction(go_to_frame_action)
        view_menu.addSeparator()
        # Channel navigation (multichannel only). Disabled - and therefore
        # not stealing the Up/Down keys from the split-FOV region nudging in
        # View.keyPressEvent - until several channels are loaded (see
        # _populate_channel_combo).
        self.previous_channel_action = view_menu.addAction("Previous channel")
        self.previous_channel_action.setShortcut("Up")
        self.previous_channel_action.triggered.connect(self.previous_channel)
        self.previous_channel_action.setEnabled(False)
        view_menu.addAction(self.previous_channel_action)
        self.next_channel_action = view_menu.addAction("Next channel")
        self.next_channel_action.setShortcut("Down")
        self.next_channel_action.triggered.connect(self.next_channel)
        self.next_channel_action.setEnabled(False)
        view_menu.addAction(self.next_channel_action)
        view_menu.addSeparator()
        zoom_in_action = view_menu.addAction("Zoom in")
        zoom_in_action.setShortcuts(["Ctrl++", "Ctrl+="])
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = view_menu.addAction("Zoom out")
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        fit_in_view_action = view_menu.addAction("Fit image to window")
        fit_in_view_action.setShortcut("Ctrl+W")
        fit_in_view_action.triggered.connect(self.fit_in_view)
        view_menu.addAction(fit_in_view_action)
        view_menu.addSeparator()
        constract_action = view_menu.addAction("Contrast")
        constract_action.setShortcut("Ctrl+C")
        constract_action.triggered.connect(self.contrast_dialog.show)
        view_menu.addAction(constract_action)
        self.scalebar_action = view_menu.addAction("Show scale bar")
        self.scalebar_action.setCheckable(True)
        self.scalebar_action.setChecked(False)
        self.scalebar_action.triggered.connect(self.draw_frame)
        view_menu.addAction(self.scalebar_action)

        """ Analyze """
        analyze_menu = menu_bar.addMenu("Analyze")
        parameters_action = analyze_menu.addAction("Parameters")
        parameters_action.setShortcut("Ctrl+P")
        parameters_action.triggered.connect(self.parameters_dialog.show)
        analyze_menu.addAction(parameters_action)
        analyze_menu.addSeparator()
        identify_action = analyze_menu.addAction("Identify")
        identify_action.setShortcut("Ctrl+I")
        identify_action.triggered.connect(self.identify)
        analyze_menu.addAction(identify_action)
        fit_action = analyze_menu.addAction("Fit")
        fit_action.setShortcut("Ctrl+F")
        fit_action.triggered.connect(self.fit)
        analyze_menu.addAction(fit_action)
        localize_action = analyze_menu.addAction("Localize (Identify && Fit)")
        localize_action.setShortcut("Ctrl+L")
        localize_action.triggered.connect(self.localize)
        analyze_menu.addAction(localize_action)
        analyze_menu.addSeparator()
        self.abort_action = analyze_menu.addAction("Abort")
        self.abort_action.setShortcut("Ctrl+.")
        self.abort_action.triggered.connect(self.abort)
        self.abort_action.setEnabled(False)
        analyze_menu.addAction(self.abort_action)

        """ Calibration """
        threed_menu = menu_bar.addMenu("Calibration")

        calibrate_z_action = threed_menu.addAction(
            "Calibrate astigmatism (Gaussian)"
        )
        calibrate_z_action.triggered.connect(self.calibrate_z)

        # The spline actions need the compiled Gpuspline library; offer them
        # only when it loaded, rather than showing actions that can only fail.
        if GPUSPLINE_INSTALLED:
            calibrate_spline_action = threed_menu.addAction(
                "Calibrate spline PSF"
            )
            calibrate_spline_action.triggered.connect(self.calibrate_spline)

            reregister_signal_action = threed_menu.addAction(
                "Re-align channels (current signal)"
            )
            reregister_signal_action.triggered.connect(
                self.reregister_channels_from_signal
            )

        self.plugin_menu = menu_bar.addMenu("Plugins")  # do not delete

    def open_config_location(self) -> None:
        """Open the folder holding the camera config file in the system
        file browser."""
        from .. import config_filename, _resolve_config_path

        path = _resolve_config_path()
        if path is None:
            # No config yet: point the user at the intended location.
            path = config_filename()
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    @property
    def camera_info(self) -> dict[str, float]:
        """Camera information, baseline, EM gain, sensitivity and QE."""
        camera_info = {}
        camera_info["Baseline"] = self.parameters_dialog.baseline.value()
        camera_info["Gain"] = self.parameters_dialog.gain.value()
        camera_info["Sensitivity"] = self.parameters_dialog.sensitivity.value()
        camera_info["Qe"] = self.parameters_dialog.qe.value()
        camera_info["Pixelsize"] = self.parameters_dialog.pixelsize.value()
        return camera_info

    def calibrate_z(self) -> None:
        """Use the loaded movie to obtain z-calibration data for 3D
        fitting using astigmatism."""
        self.localize(calibrate_z=True)

    def calibrate_spline(self) -> None:
        """Build a cubic-spline PSF calibration from the loaded bead z-stack
        movie (Gpuspline). Detects beads, averages them into a PSF volume and
        computes the spline coefficients, saved as an HDF5 calibration."""
        if self.movie is None:
            QtWidgets.QMessageBox.information(
                self, "Spline PSF Calibration", "No file loaded."
            )
            return
        if not GPUSPLINE_INSTALLED:
            QtWidgets.QMessageBox.warning(
                self,
                "Spline PSF Calibration",
                "Gpuspline could not be loaded. See "
                "picasso/ext/pygpuspline/README.txt for how to add the "
                "compiled library. Only Windows and Linux are supported.",
            )
            return

        # identify spots first so the beads are displayed (like the
        # astigmatic 3D calibration); the calibration is built afterwards.
        self.identify(calibrate_spline=True)

    def set_split_fov_mode(self, enabled: bool) -> None:
        """Enable/disable split-FOV region mode, where the drawn ROIs are the
        channels of one movie. The shared region size is derived live from the
        existing regions (see ``View._region_size``); enabling the mode only
        gives the view keyboard focus for arrow-key nudging.

        Split-FOV (regions = channels of one movie) is mutually exclusive with
        separate-movie multichannel data: when several movies are loaded the
        request is rejected and the checkbox reverted."""
        if enabled and len(self.channels) > 1:
            QtWidgets.QMessageBox.information(
                self,
                "Split-FOV mode",
                "'Regions = channels' treats the ROIs of a single movie as "
                "channels and cannot be used when several movies are loaded "
                "as separate channels.",
            )
            checkbox = self.parameters_dialog.split_fov_checkbox
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            self.view.split_fov_mode = False
            return
        self.view.split_fov_mode = bool(enabled)
        if enabled:
            self.view.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.draw_frame()

    def build_spline_calibration(self) -> None:
        """Prompt for the spline PSF calibration parameters and build the
        calibration from the loaded bead z-stack in a background thread."""
        split_fov = self.view.split_fov_mode
        regions = list(self.view.rois) if split_fov else None
        if split_fov and len(regions) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Spline PSF Calibration",
                "Split-FOV mode is on: draw at least 2 equal-size regions "
                "(channels) before calibrating. The first region is the "
                "reference channel.",
            )
            return

        # Separate-channel multichannel: several channels are loaded (separate
        # files or a multichannel file) and split-FOV mode is off. The first
        # loaded channel is the reference.
        multichannel = not split_fov and len(self.channels) > 1
        if multichannel:
            n_frames = {int(c.movie.shape[0]) for c in self.channels}
            if len(n_frames) != 1:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Spline PSF Calibration",
                    "The loaded channels have different frame counts "
                    f"({sorted(n_frames)}). A multichannel calibration needs "
                    "every channel to share the same z-scan layout.",
                )
                return

        specs = CalibrateSplineDialog.getCalibrationSpecs(self)
        (
            step,
            frames_per_step,
            frame_order,
            model,
            magnification_factor,
            correct_z_bias,
            link_photons,
            accepted,
        ) = specs
        if not accepted:
            return

        base = os.path.splitext(self.movie_path)[0] if self.movie_path else ""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save spline PSF calibration",
            base + "_spline_calib.hdf5",
            filter="*.hdf5",
        )
        if not path:
            return

        parameters = self.parameters
        # Separate-channel multichannel build: pass every channel's movie/info
        # (reference first) so the worker calls calibrate_spline_multichannel.
        # Camera info is shared (Localize keeps a single camera-parameter set).
        movies = infos = camera_infos = None
        # The frame range is shared across channels (window-level), so it
        # applies to the reference channel and every other channel alike.
        frame_bounds = self.frame_range
        if multichannel:
            movies = [c.movie for c in self.channels]
            infos = [c.info for c in self.channels]
            camera_infos = [self.camera_info for _ in self.channels]
        self.spline_calibration_worker = SplineCalibrationWorker(
            movie=self.movie,
            info=self.info,
            camera_info=self.camera_info,
            box=parameters["Box Size"],
            minimum_ng=parameters["Min. Net Gradient"],
            step=step,
            frames_per_step=frames_per_step,
            frame_order=frame_order,
            frame_bounds=frame_bounds,
            model=model,
            magnification_factor=magnification_factor,
            correct_z_bias=correct_z_bias,
            link_photons=link_photons,
            roi=self.view.rois,
            regions=regions,
            movies=movies,
            infos=infos,
            camera_infos=camera_infos,
            path=path,
        )
        self.spline_calibration_worker.statusChanged.connect(
            self.status_bar.showMessage
        )
        self.spline_calibration_worker.finished.connect(
            self.on_spline_calibration_finished
        )
        self.spline_calibration_worker.failed.connect(
            self.on_spline_calibration_failed
        )
        if multichannel:
            msg = (
                f"Building multichannel spline PSF calibration from "
                f"{len(self.channels)} channels (reference: "
                f"{self.channels[0].name}) ..."
            )
        elif regions:
            msg = (
                f"Building split-FOV spline PSF calibration from "
                f"{len(regions)} regions ..."
            )
        else:
            msg = "Building spline PSF calibration ..."
        self.status_bar.showMessage(msg)
        self.spline_calibration_worker.start()

    def on_spline_calibration_finished(self, path: str, n_beads: int) -> None:
        """Report a successful spline PSF calibration, listing the diagnostic
        images that were written next to it (so a missing one is obvious)."""
        self.status_bar.showMessage("")
        base = os.path.splitext(path)[0]
        written = [
            os.path.basename(p) for p in sorted(glob.glob(base + "_*.png"))
        ]
        lines = [
            f"Spline PSF calibration built from {n_beads} beads and saved to:",
            path,
        ]
        if written:
            lines += ["", "Diagnostics written:"] + [f"  {w}" for w in written]
        QtWidgets.QMessageBox.information(
            self, "Spline PSF Calibration", "\n".join(lines)
        )

    def on_spline_calibration_failed(self, message: str) -> None:
        """Report a failed spline PSF calibration."""
        self.status_bar.showMessage("")
        QtWidgets.QMessageBox.critical(self, "Spline PSF Calibration", message)

    def show_metadata(self) -> None:
        """Open the metadata dialog."""
        if self.movie is None:
            QtWidgets.QMessageBox.information(
                self, "Metadata", "No file loaded."
            )
            return
        infos = self.extra_info if self.extra_info else self.info
        label = os.path.basename(self.movie_path) if self.movie_path else None
        self.metadata_dialog.set_infos(infos, labels=label)
        self.metadata_dialog.show()
        self.metadata_dialog.raise_()

    def open_file_dialog(self) -> None:
        """Open a file dialog to select a movie file to load."""
        if self.pwd == []:
            dir = None
        else:
            dir = self.pwd

        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open image sequence",
            directory=dir,
            filter=(
                "All supported formats ("
                + " ".join("*" + e for e in io.MOVIE_EXTENSIONS)
                + ")"
                ";;Raw files (*.raw)"
                ";;Tif images (*.tif *.tiff)"
                ";;BigTiff (*.btf *.tf8 *.tf2)"
                ";;Zeiss LSM (*.lsm)"
                ";;Zeiss CZI (*.czi)"
                ";;Leica LIF (*.lif)"
                ";;ImaRIS IMS (*.ims)"
                ";;Nd2 files (*.nd2)"
                ";;STK files (*.stk)"
            ),
        )
        if path:
            self.pwd = path
            self.open(path)

    def open_mm_folder_dialog(self) -> None:
        """Open a MicroManager "separate image files" acquisition folder.

        MicroManager can save each frame of a movie as its own TIFF
        (``img_*.tif``) inside one folder. This picks such a folder and
        loads the whole sequence as a single movie.
        """
        dir = None if self.pwd == [] else self.pwd
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open MicroManager image folder", directory=dir
        )
        if not directory:
            return
        path = io.find_mm_separate_first(directory)
        if path is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No image sequence found",
                "No MicroManager 'separate image files' sequence "
                "(img_*.tif) was found in the selected folder.",
            )
            return
        self.pwd = path
        self.open(path)

    def _prompt_for_path(self, path: str):
        """Return the metadata prompt callback appropriate for ``path``."""
        if path.lower().endswith((".ims", ".czi", ".lif")):
            # Multi-channel .ims/.czi/.lif files prompt for a channel.
            return self.prompt_channel
        elif path.lower().endswith(io.TIFF_EXTENSIONS + (".stk", ".nd2")):
            # For these formats the metadata may fail to parse; prompt the
            # user to enter it manually as a fallback.
            return self.prompt_movie_info
        return self.prompt_info

    def open(self, path: str) -> None:
        """Open a single movie file as one channel."""
        self._start_movie_load([path], self._prompt_for_path)

    def open_multichannel_file_dialog(self) -> None:
        """Open a single file and load *all* of its channels."""
        dir = None if self.pwd == [] else self.pwd
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open multichannel file",
            directory=dir,
            filter="Multichannel files (*.ims *.czi *.lif *.nd2)",
        )
        if path:
            self.pwd = path
            self.open_multichannel_file(path)

    def open_multichannel_file(self, path: str) -> None:
        """Load every channel of a single multichannel file."""
        self._start_movie_load(
            [path], lambda _p: self.prompt_movie_info, load_all=True
        )

    def open_channels_from_files_dialog(self) -> None:
        """Open several movie files, each loaded as one channel."""
        dir = None if self.pwd == [] else self.pwd
        paths, exe = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open channels from files",
            directory=dir,
            filter=(
                "All supported formats ("
                + " ".join("*" + e for e in io.MOVIE_EXTENSIONS)
                + ")"
            ),
        )
        if paths:
            self.pwd = paths[0]
            self.open_channels_from_files(paths)

    def open_channels_from_files(self, paths: list[str]) -> None:
        """Load several movie files as channels (one file per channel)."""
        self._start_movie_load(paths, self._prompt_for_path, multi_file=True)

    def _start_movie_load(
        self,
        paths: list[str],
        prompt_for_path,
        load_all: bool = False,
        multi_file: bool = False,
    ) -> None:
        """Load movies on a background thread (see ``MovieLoadWorker``) so
        the GUI keeps repainting and responding while files are read.

        ``load_all`` reads every channel of each file (single multichannel
        file); otherwise one channel is loaded per file. ``multi_file``
        controls channel naming when several separate files are loaded.
        """
        if self._load_thread is not None:
            if self._load_worker is not None and self._load_worker._cancelled:
                # The previous load was cancelled but its worker is still
                # finishing the blocking io call. Remember this request
                # and start it as soon as the old thread is torn down.
                self._pending_load = (
                    paths,
                    prompt_for_path,
                    load_all,
                    multi_file,
                )
                self.status_bar.showMessage(
                    "Finishing cancelled load, the new file will open "
                    "right after..."
                )
            # Otherwise a load is already in progress; ignore re-entrant
            # requests.
            return

        self._load_t0 = time.time()
        self._load_multi_file = multi_file
        # Each file spans PROGRESS_RESOLUTION steps so the bar can advance
        # smoothly *within* a file from the worker's per-page reports,
        # instead of only ticking once per file.
        self._load_index = 0
        progress = QtWidgets.QProgressDialog(
            "Loading movie...",
            "Cancel",
            0,
            len(paths) * PROGRESS_RESOLUTION,
            self,
        )
        progress.setWindowTitle("Opening movie")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._load_progress = progress

        thread = QtCore.QThread(self)
        worker = MovieLoadWorker(paths, prompt_for_path, load_all=load_all)
        worker.moveToThread(thread)
        self._load_thread = thread
        self._load_worker = worker

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_load_progress)
        worker.subprogress.connect(self._on_load_subprogress)
        worker.prompt_requested.connect(self._on_load_prompt_requested)
        worker.finished.connect(self._on_load_finished)
        worker.failed.connect(self._on_load_failed)
        # Direct connection so the flag is set from the GUI thread
        # immediately; the worker thread is busy in run() and would not
        # service a queued slot until the current file finished.
        progress.canceled.connect(
            worker.cancel, QtCore.Qt.ConnectionType.DirectConnection
        )
        progress.canceled.connect(self._on_load_canceled)
        thread.start()

    def _on_load_progress(self, index: int, filename: str) -> None:
        """Update the progress dialog as each file starts loading.

        The bar cannot move yet: before per-page reporting begins, the
        loader indexes the whole file (opening it and counting frames),
        which happens inside a single tifffile call with no sub-steps.
        Say so, so the user does not think the load has stalled."""
        self._load_index = index
        if self._load_progress is not None:
            self._load_progress.setLabelText(
                f"Opening {filename}...\n"
                "Indexing frames — the bar starts once they are counted."
            )
            self._load_progress.setValue(index * PROGRESS_RESOLUTION)

    def _on_load_subprogress(self, done: int, total: int) -> None:
        """Advance the bar within the current file from the worker's
        per-page reports (see ``MovieLoadWorker.subprogress``)."""
        if self._load_progress is not None and total > 0:
            fraction = min(done / total, 1.0)
            value = int((self._load_index + fraction) * PROGRESS_RESOLUTION)
            self._load_progress.setValue(value)

    def _on_load_prompt_requested(
        self, callback, args_kwargs, holder: dict
    ) -> None:
        """Run a worker-requested metadata prompt on the GUI thread and
        hand the result back, then unblock the worker."""
        args, kwargs = args_kwargs
        try:
            holder["result"] = callback(*args, **kwargs)
        finally:
            self._load_worker._prompt_event.set()

    def _on_load_canceled(self) -> None:
        """Discard the progress dialog as soon as the user cancels. The
        worker keeps emitting progress while it finishes the current
        (uninterruptible) io call, and ``setValue()`` on a cancelled
        ``QProgressDialog`` re-shows it."""
        if self._load_progress is not None:
            self._load_progress.close()
            self._load_progress = None
        self.status_bar.showMessage("Cancelling load...")

    def _stop_movie_load(self) -> None:
        """Cancel any in-progress load and block until the worker thread
        has stopped. Called before the window/app is destroyed, because
        destroying a still-running QThread aborts the process."""
        if self._load_worker is not None:
            self._load_worker.cancel()
        if self._load_thread is not None:
            self._load_thread.quit()
            self._load_thread.wait()

    def _finish_load(self) -> None:
        """Tear down the worker thread and progress dialog."""
        if self._load_progress is not None:
            self._load_progress.close()
            self._load_progress = None
        if self._load_thread is not None:
            self._load_thread.quit()
            self._load_thread.wait()
            self._load_thread.deleteLater()
            self._load_thread = None
        if self._load_worker is not None:
            self._load_worker.deleteLater()
            self._load_worker = None
        if self._pending_load is not None:
            # A load requested while the cancelled one was winding down;
            # start it now that the old thread is gone.
            pending, self._pending_load = self._pending_load, None
            self._start_movie_load(*pending)

    def _on_load_finished(
        self, movies: list, infos: list, paths: list
    ) -> None:
        """Activate the loaded channels once the worker is done."""
        self._finish_load()
        if not movies:
            return
        names = [
            self._channel_name(
                infos[i], paths[i], i, multi_file=self._load_multi_file
            )
            for i in range(len(movies))
        ]
        self._set_channels(movies, infos, paths, names)
        dt = time.time() - self._load_t0
        if len(movies) == 1:
            self.status_bar.showMessage(f"Opened movie in {dt:.2f} seconds.")
        else:
            self.status_bar.showMessage(
                f"Opened {len(movies)} channel(s) in {dt:.2f} seconds."
            )

    def _on_load_failed(self, message: str) -> None:
        """Report a load error and tear down the worker."""
        self._finish_load()
        QtWidgets.QMessageBox.warning(self, "Could not load file", message)

    def _channel_name(
        self,
        info: list,
        path: str,
        idx: int,
        multi_file: bool = False,
    ) -> str:
        """Pick a display name for a channel: the metadata ``"Channel"``
        key if present, else the filename stem (separate files), else
        ``"Channel N"``."""
        try:
            channel = info[0].get("Channel")
        except (AttributeError, IndexError, TypeError):
            channel = None
        if channel:
            return str(channel)
        if multi_file:
            return os.path.splitext(os.path.basename(path))[0]
        return f"Channel {idx}"

    def _warn_if_channel_lengths_differ(self, infos: list) -> None:
        """Warn (non-blocking) when the channels have different frame
        counts. Multichannel identify/fit pair spots frame-by-frame across
        channels and the shared frame slider / frame range are clamped to
        each channel's length, so unequal lengths are almost always a
        mistake. The load still proceeds so the movies can be inspected."""
        if len(infos) < 2:
            return
        try:
            frame_counts = [
                int(lib.get_from_metadata(info, "Frames")) for info in infos
            ]
        except Exception:
            return
        if len(set(frame_counts)) > 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Channels differ in length",
                "The loaded channels have different numbers of frames "
                f"({', '.join(str(n) for n in frame_counts)}).\n\n"
                "Channels should have the same length: multichannel "
                "identification and fitting pair spots frame-by-frame "
                "across channels, and the shared frame slider and frame "
                "range are clamped to each channel's length. Please load "
                "channels with equal frame counts.",
            )

    def _set_channels(
        self,
        movies: list,
        infos: list,
        paths: list,
        names: list,
    ) -> None:
        """Build the channel list from loaded movies and activate the
        first one. The single funnel used by every load path."""
        if not movies:
            return
        self._warn_if_channel_lengths_differ(infos)
        # Build channels and seed each channel's parameter/contrast
        # snapshot from the current dialog state, applying any camera /
        # pixel-size hints from that channel's metadata. Guarded so the
        # value changes don't wipe localizations.
        self._switching_channel = True
        try:
            self.channels = []
            for movie, info, path, name in zip(movies, infos, paths, names):
                channel = Channel(movie=movie, info=info, path=path, name=name)
                self.parameters_dialog.set_camera_parameters(info[0])
                if "Pixelsize" in info[0]:
                    self.parameters_dialog.pixelsize.setValue(
                        int(info[0]["Pixelsize"])
                    )
                channel.params = self._capture_params()
                self.channels.append(channel)
            self.current_channel = 0
        finally:
            self._switching_channel = False
        self._populate_channel_combo()
        self.frame_slider.setEnabled(True)
        # A fresh load starts at frame 0; contrast is shared and keeps its
        # current dialog state.
        self.curr_frame_number = 0
        self._restore_current_channel()
        self.parameters_dialog.reset_quality_check()

    def _populate_channel_combo(self) -> None:
        """Refresh the channel selector; hidden unless several channels."""
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems([c.name for c in self.channels])
        self.channel_combo.setCurrentIndex(self.current_channel)
        self.channel_combo.blockSignals(False)
        multichannel = len(self.channels) > 1
        self.channel_combo.setVisible(multichannel)
        self.previous_channel_action.setEnabled(multichannel)
        self.next_channel_action.setEnabled(multichannel)
        self.parameters_dialog.link_groupbox.setVisible(multichannel)
        # Split-FOV (regions = channels of one movie) is incompatible with
        # separate-movie multichannel data
        checkbox = self.parameters_dialog.split_fov_checkbox
        if multichannel and checkbox.isChecked():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            self.view.split_fov_mode = False
            self.draw_frame()
        checkbox.setEnabled(not multichannel)

    def on_channel_combo_changed(self, index: int) -> None:
        """Switch the active channel from the selector."""
        self.set_current_channel(index)

    def previous_channel(self) -> None:
        """Activate the channel above the current one (Up arrow)."""
        self.set_current_channel(self.current_channel - 1)

    def next_channel(self) -> None:
        """Activate the channel below the current one (Down arrow)."""
        self.set_current_channel(self.current_channel + 1)

    def set_current_channel(self, index: int) -> None:
        """Make channel ``index`` the active one, swapping the flat state
        and the Parameters/Contrast dialog values."""
        if (
            index == self.current_channel
            or index < 0
            or index >= len(self.channels)
        ):
            return
        self._snapshot_current_channel()
        self.current_channel = index
        # Keep the selector in sync for switches that did not come from it
        # (keyboard navigation, multichannel identify batches).
        self.channel_combo.blockSignals(True)
        self.channel_combo.setCurrentIndex(index)
        self.channel_combo.blockSignals(False)
        self._restore_current_channel()
        self.parameters_dialog.reset_quality_check()

    def _snapshot_current_channel(self) -> None:
        """Store the active flat state + dialog values into its Channel."""
        if not self.channels:
            return
        channel = self.channels[self.current_channel]
        channel.movie = self.movie
        channel.info = self.info
        channel.path = self.movie_path
        channel.identifications = self.identifications
        channel.locs = self.locs
        channel.locs_display = self.locs_display
        channel.ready_for_fit = self.ready_for_fit
        channel.last_identification_info = self.last_identification_info
        channel.extra_info = self.extra_info
        channel.params = self._capture_params()
        # Contrast, the current frame and the frame range are shared across
        # channels (see _restore_current_channel), so they are not
        # snapshotted per channel.

    def _restore_current_channel(self) -> None:
        """Load the active Channel's state into the flat attrs + dialogs.

        Contrast and the current frame are shared across channels, so they
        are intentionally *not* restored per channel: the contrast dialog is
        left as-is and the shared frame number is reused (only clamped to
        this channel's length)."""
        channel = self.channels[self.current_channel]
        self._switching_channel = True
        try:
            self.movie = channel.movie
            self.info = channel.info
            self.movie_path = channel.path
            self.identifications = channel.identifications
            self.locs = channel.locs
            self.locs_display = channel.locs_display
            self.ready_for_fit = channel.ready_for_fit
            self.last_identification_info = channel.last_identification_info
            self.extra_info = channel.extra_info
            self._apply_params(channel.params)
            # Contrast and the frame range are shared: keep the dialogs
            # as-is (do not restore per-channel values).
        finally:
            self._switching_channel = False
        # The current frame is shared across channels too.
        self._apply_channel_to_ui(getattr(self, "curr_frame_number", 0))

    def _apply_channel_to_ui(self, frame_number: int = 0) -> None:
        """Sync the frame slider, displayed frame, zoom and title to the
        active channel."""
        last_frame = lib.get_from_metadata(self.info, "Frames") - 1
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, last_frame))
        self.frame_slider.blockSignals(False)
        frame_number = min(max(0, frame_number), max(0, last_frame))
        self.set_frame(frame_number)
        self.fit_in_view()
        base = os.path.basename(self.movie_path) if self.movie_path else ""
        title = f"Picasso v{__version__}: Localize. File: {base}"
        if len(self.channels) > 1:
            title += f" [{self.channels[self.current_channel].name}]"
        self.setWindowTitle(title)

    def _capture_params(self) -> dict:
        """Snapshot the analysis-relevant Parameters dialog values."""
        pd = self.parameters_dialog
        params = {
            "box": pd.box_spinbox.value(),
            "mng": pd.mng_slider.value(),
            "mng_min": pd.mng_min_spinbox.value(),
            "mng_max": pd.mng_max_spinbox.value(),
            "fit_model": pd.fit_model.currentIndex(),
            "fit_optimizer": pd.fit_optimizer.currentIndex(),
            "baseline": pd.baseline.value(),
            "gain": pd.gain.value(),
            "sensitivity": pd.sensitivity.value(),
            "qe": pd.qe.value(),
            "pixelsize": pd.pixelsize.value(),
            "convergence": pd.convergence_criterion.value(),
            "max_it": pd.max_it.value(),
            "magnification": pd.magnification_factor.value(),
            "fit_z": pd.fit_z_checkbox.isChecked(),
            "fit_z_enabled": pd.fit_z_checkbox.isEnabled(),
            "z_calibration": pd.z_calibration,
            "z_calibration_path": pd.z_calibration_path,
            "z_calib_label": pd.z_calib_label.text(),
            "use_gpu": pd.gpu_checkbox.isChecked(),
        }
        if hasattr(pd, "camera"):
            params["camera"] = pd.camera.currentIndex()
        return params

    def _apply_params(self, params: dict) -> None:
        """Restore a channel's analysis parameters into the dialog. Caller
        sets ``self._switching_channel`` to suppress side effects."""
        if not params:
            return
        pd = self.parameters_dialog
        # Settings whose "Same across channels" box is ticked are shared, so
        # they are not restored per channel (the current dialog value stays).
        link_box = pd.link_box_checkbox.isChecked()
        link_mng = pd.link_mng_checkbox.isChecked()
        link_cam = pd.link_camera_checkbox.isChecked()
        # Camera selection first: its cascade fills baseline/gain/pixelsize
        # from the config, which we then override with the channel's values.
        if not link_cam and hasattr(pd, "camera") and "camera" in params:
            pd.camera.setCurrentIndex(params["camera"])
        if not link_box:
            pd.box_spinbox.setValue(params["box"])
        if not link_mng:
            pd.mng_min_spinbox.setValue(params["mng_min"])
            pd.mng_max_spinbox.setValue(params["mng_max"])
            pd.mng_slider.setValue(params["mng"])
            pd.mng_spinbox.setValue(params["mng"])
        # Set the model first so its handler repopulates the optimizer list,
        # then restore the optimizer selection.
        pd.fit_model.setCurrentIndex(params.get("fit_model", 0))
        pd.fit_optimizer.setCurrentIndex(params.get("fit_optimizer", 0))
        if not link_cam:
            pd.baseline.setValue(params["baseline"])
            pd.gain.setValue(params["gain"])
            pd.sensitivity.setValue(params["sensitivity"])
            pd.qe.setValue(params["qe"])
            pd.pixelsize.setValue(params["pixelsize"])
        pd.convergence_criterion.setValue(params["convergence"])
        pd.max_it.setValue(params["max_it"])
        pd.magnification_factor.setValue(params["magnification"])
        pd.z_calibration = params["z_calibration"]
        pd.z_calibration_path = params["z_calibration_path"]
        pd.z_calib_label.setText(params["z_calib_label"])
        pd.fit_z_checkbox.setEnabled(params["fit_z_enabled"])
        pd.fit_z_checkbox.setChecked(params["fit_z"])
        # Saved parameter files written before the Numba CUDA port use the
        # old "gpufit" key.
        pd.gpu_checkbox.setChecked(
            params.get("use_gpu", params.get("gpufit", False))
        )

    def propagate_linked_params(self) -> None:
        """Copy each linked (shared) parameter group from the active channel's
        current dialog state into every channel, so enabling a link takes
        effect immediately rather than only on the next channel switch."""
        if len(self.channels) < 2:
            return
        pd = self.parameters_dialog
        cur = self._capture_params()
        keys: list[str] = []
        if pd.link_box_checkbox.isChecked():
            keys += ["box"]
        if pd.link_mng_checkbox.isChecked():
            keys += ["mng", "mng_min", "mng_max"]
        if pd.link_camera_checkbox.isChecked():
            keys += [
                "camera",
                "baseline",
                "gain",
                "sensitivity",
                "qe",
                "pixelsize",
            ]
        for channel in self.channels:
            if not channel.params:
                continue
            for key in keys:
                if key in cur:
                    channel.params[key] = cur[key]

    def _channels_share_file(self) -> bool:
        """True when several channels were loaded from one file (so their
        output names need a channel suffix to avoid overwriting)."""
        return len(self.channels) > 1 and all(
            c.path == self.channels[0].path for c in self.channels
        )

    def channel_output_base(self) -> str:
        """Base path (no extension) for the active channel's output files,
        suffixed with the channel name when channels share one file."""
        base, _ = os.path.splitext(self.movie_path)
        if self._channels_share_file():
            name = self.channels[self.current_channel].name
            base = base + "_" + _sanitize_filename(name)
        return base

    def open_picks(self) -> None:
        """Open a file dialog to select a picks (from Picasso: Render)
        file to load."""
        if self.movie_path != []:
            dir = os.path.dirname(self.movie_path)
        else:
            dir = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open picks", directory=dir, filter="*.yaml"
        )
        if path:
            self.load_picks(path)

    def load_picks(self, path: str) -> None:
        """Load picks from a YAML file from Picasso: Render."""
        # ask for drift correction
        driftpath, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open drift file",
            directory=os.path.dirname(path),
            filter="*.txt",
        )
        drift = None
        if driftpath:
            try:
                drift = io.load_drift(driftpath)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could not load drift file",
                    f"Drift file could not be loaded, error: {e}\n."
                    "No drift correction will be applied.",
                )
                drift = None
        picks, shape, _ = io.load_picks(path)
        if shape != "Circle":
            QtWidgets.QMessageBox.warning(
                self,
                "Unsupported shape",
                f"Only circle picks are supported, but got {shape}.",
            )
            return
        # convert
        self.identifications = localize.picks_to_identifications(
            picks,
            n_frames=lib.get_from_metadata(self.info, "Frames"),
            drift=drift,
        )
        self._clean_up_external_ids()

    def open_locs(self) -> None:
        """Open localizations for refitting data. Provide spot
        identifications."""
        if self.movie_path != []:
            dir = os.path.dirname(self.movie_path)
        else:
            dir = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open locs", directory=dir, filter="*.hdf5"
        )
        if path:
            self.load_locs(path)

    def load_locs(self, path: str) -> None:
        """Load localizations from a HDF5 file. Provide spot
        identifications."""
        try:
            locs, _ = io.load_locs(path)
            n_frames, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Input Dialog",
                "Enter number of frames around localization event:",
                10,
                min=0,
            )
            if not ok:
                return
        except io.NoMetadataFileError:
            return

        self.identifications = localize.locs_to_identifications(
            locs=locs,
            movie_info=self.info,
            n_frames=n_frames,
        )
        self._clean_up_external_ids()

    def open_identifications(self) -> None:
        """Open identifications previously saved as a HDF5 file."""
        if self.movie_path != []:
            dir = os.path.dirname(self.movie_path)
        else:
            dir = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open identifications",
            directory=dir,
            filter="*.hdf5",
        )
        if path:
            self.load_identifications(path)

    def load_identifications(self, path: str) -> None:
        """Load identifications from a HDF5 file."""
        try:
            identifications, info = io.load_identifications(
                path, qt_parent=self
            )
        except io.NoMetadataFileError:
            return
        except KeyError:
            return
        self.identifications = identifications
        box = lib.get_from_metadata(info, "Box Size")
        min_ng = lib.get_from_metadata(info, "Min. Net Gradient")
        if box or min_ng:
            self.last_identification_info = {}
            if box is not None:
                self.last_identification_info["Box Size"] = box
                self.parameters_dialog.box_spinbox.setValue(box)
            if min_ng is not None:
                self.last_identification_info["Min. Net Gradient"] = min_ng
                self.parameters_dialog.mng_slider.setValue(min_ng)
        self._clean_up_external_ids()

    def _clean_up_external_ids(self) -> None:
        if self.identifications is None:
            return
        # remove all identifications that are oob
        box = self.parameters["Box Size"]
        m_size = self.movie.shape
        r = int(box / 2)
        self.identifications = self.identifications[
            (self.identifications.y - r > 0)
            & (self.identifications.x - r > 0)
            & (self.identifications.x + r < m_size[0])
            & (self.identifications.y + r < m_size[1])
        ]
        # assign gui attributes
        self.locs = None
        self.locs_display = None
        self.loaded_picks = True
        self.last_identification_info = {
            "Box Size": self.parameters_dialog.box_spinbox.value(),
            "Min. Net Gradient": self.parameters_dialog.mng_slider.value(),
            "ROI": self.view.rois,
            "Frame bounds": self.frame_range,
        }
        self.ready_for_fit = True
        self.draw_frame()
        self.status_bar.showMessage(
            f"Created a total of {len(self.identifications):,} "
            "identifications."
        )

    def prompt_info(self) -> tuple[dict, bool] | None:
        """Prompt for movie information."""
        info, save, ok = PromptInfoDialog.getMovieSpecs(self)
        if ok:
            return info, save

    def prompt_movie_info(
        self, partial_info: dict | None = None
    ) -> tuple[dict, bool] | None:
        """Prompt for movie metadata when it cannot be read from the
        file (fallback for .tif/.stk/.nd2 movies). Pre-filled with the
        dimensions that could still be read. Returns ``(info, save)`` or
        None if cancelled."""
        info, save, ok = PromptMovieInfoDialog.getMovieSpecs(
            self, partial_info
        )
        if ok:
            return info, save

    def prompt_channel(self, channels: list[str]) -> str | None:
        """Prompt for channel selection for multi-channel movies
        (IMARIS .ims, Zeiss .czi, Leica .lif)."""
        channel, ok = PromptChannelDialog.getMovieSpecs(self, channels)
        if ok:
            return channel

    def previous_frame(self, step: int = 1) -> None:
        """Navigate backwards by ``step`` frames and display the result."""
        if self.movie is not None:
            if self.curr_frame_number > 0:
                self.set_frame(max(0, self.curr_frame_number - step))

    def next_frame(self, step: int = 1) -> None:
        """Navigate forwards by ``step`` frames and display the result."""
        if self.movie is not None:
            last_frame = self.info[0]["Frames"] - 1
            if self.curr_frame_number < last_frame:
                self.set_frame(min(last_frame, self.curr_frame_number + step))

    def first_frame(self) -> None:
        """Navigate to the first frame and display it."""
        if self.movie is not None:
            self.set_frame(0)

    def last_frame(self) -> None:
        """Navigate to the last frame and display it."""
        if self.movie is not None:
            self.set_frame(self.info[0]["Frames"] - 1)

    def to_frame(self) -> None:
        """Navigate to a specific frame and display it."""
        if self.movie is not None:
            frames = self.info[0]["Frames"]
            number, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Go to frame",
                "Frame number:",
                self.curr_frame_number + 1,
                1,
                frames,
            )
            if ok:
                self.set_frame(number - 1)

    def set_frame(self, number: int) -> None:
        """Set the current frame to the specified number."""
        self.curr_frame_number = number
        if self.contrast_dialog.auto_checkbox.isChecked():
            black = self.movie[number].min()
            white = self.movie[number].max()
            self.contrast_dialog.change_contrast_silently(black, white)
        self.draw_frame()
        self.status_bar_frame_indicator.setText(
            "{:,}/{:,}".format(
                number + 1, lib.get_from_metadata(self.info, "Frames")
            )
        )
        # Keep the slider in sync without re-triggering set_frame.
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(number)
        self.frame_slider.blockSignals(False)

    def on_frame_slider_changed(self, value: int) -> None:
        """Navigate to the frame selected with the slider."""
        if self.movie is not None and value != self.curr_frame_number:
            self.set_frame(value)

    def draw_frame(self) -> None:
        """Draw the current frame - show the movie frame, apply
        contrast, add identifications and fit markers, if applicable."""
        if self.movie is None:
            return
        # rebuilding the scene can change scrollbar values and re-fire
        # valueChanged -> on_scroll -> draw_frame; guard against unbounded
        # re-entrant recursion.
        if self._drawing_frame:
            return
        self._drawing_frame = True
        try:
            self._draw_frame()
        finally:
            self._drawing_frame = False

    def _draw_frame(self) -> None:
        """Actual frame-drawing implementation, wrapped by ``draw_frame``
        with a re-entrancy guard."""
        if self.movie is not None:
            frame = self.movie[self.curr_frame_number]
            frame = frame.astype("float32")
            if self.contrast_dialog.auto_checkbox.isChecked():
                frame -= frame.min()
                frame /= frame.max()
            else:
                frame -= self.contrast_dialog.black_spinbox.value()
                frame /= self.contrast_dialog.white_spinbox.value()
            frame *= 255.0
            frame = np.maximum(frame, 0)
            frame = np.minimum(frame, 255)
            frame = frame.astype("uint8")
            height, width = frame.shape
            image = QtGui.QImage(
                frame.data,
                width,
                height,
                width,
                QtGui.QImage.Format.Format_Indexed8,
            )
            image.setColorTable(CMAP_GRAYSCALE)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.scene = Scene(self)
            self.scene.addPixmap(pixmap)
            # pin the scene rect to the image bounds so overlay items that
            # extend past the edge (e.g. the fixed-size scale bar text) do
            # not enlarge the scene and shift/re-center the view
            self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
            self.view.setScene(self.scene)
            # draw the ROI rectangles (in scene/pixel coordinates)
            split_fov = self.view.split_fov_mode
            for i, ((y_min, x_min), (y_max, x_max)) in enumerate(
                self.view.rois
            ):
                if i == self.view.selected_roi:
                    color = QtGui.QColor("cyan")
                elif split_fov:
                    # split-FOV: highlight the reference channel (index 0)
                    color = (
                        QtGui.QColor("lime")
                        if i == 0
                        else QtGui.QColor("orange")
                    )
                else:
                    color = QtGui.QColor("blue")
                pen = QtGui.QPen(color)
                pen.setCosmetic(True)  # constant width regardless of zoom
                self.scene.addRect(
                    QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min),
                    pen,
                )
                if split_fov:
                    # label each region by its channel index (0 = reference)
                    text = self.scene.addSimpleText(
                        "ref" if i == 0 else f"ch{i}"
                    )
                    text.setBrush(QtGui.QBrush(color))
                    text.setPos(float(x_min), float(y_min))
                    text.setFlag(
                        QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
                    )
            if self.ready_for_fit:
                box = self.last_identification_info["Box Size"]
                if not self._draw_linked_identifications(
                    self.curr_frame_number, box
                ):
                    identifications_frame = self.identifications[
                        self.identifications.frame == self.curr_frame_number
                    ]
                    self.draw_identifications(
                        identifications_frame, box, QtGui.QColor("yellow")
                    )
            else:
                if self.parameters_dialog.preview_checkbox.isChecked():
                    identifications_frame = localize.identify_by_frame_number(
                        self.movie,
                        self.parameters["Min. Net Gradient"],
                        self.parameters["Box Size"],
                        self.curr_frame_number,
                        roi=self.view.rois,
                        frame_bounds=self.frame_range,
                    )
                    box = self.parameters["Box Size"]
                    self.status_bar.showMessage(
                        f"Found {len(identifications_frame):,} spots in "
                        "current frame."
                    )
                    self.draw_identifications(
                        identifications_frame, box, QtGui.QColor("red")
                    )
                else:
                    self.status_bar.showMessage("")
            if self.locs_display is not None:
                locs_frame = self.locs_display[
                    self.locs_display.frame == self.curr_frame_number
                ]
                for _, loc in locs_frame.iterrows():
                    self.scene.addItem(
                        FitMarker(loc["x"] + 0.5, loc["y"] + 0.5, 1)
                    )
            self.draw_scalebar()

    def draw_identifications(
        self,
        identifications: pd.DataFrame,
        box: int,
        color: QtGui.QColor,
    ) -> None:
        """Draw identification boxes in the scene."""
        box_half = int(box / 2)
        for _, identification in identifications.iterrows():
            x = identification["x"]
            y = identification["y"]
            self.scene.addRect(x - box_half, y - box_half, box, box, color)

    def _draw_linked_identifications(
        self, frame_number: int, box: int
    ) -> bool:
        """Draw this frame's identification boxes colour-coded by cross-channel
        link, when 'Link colors' is on and a multichannel / split-FOV spline
        calibration is loaded.

        Spots paired across channels (matched to the reference channel via the
        calibration's inter-channel transform, as the signal re-registration
        does) share a colour; unmatched spots are grey. Returns True if it
        handled the drawing, False to fall back to plain single-colour boxes.
        """
        pdialog = self.parameters_dialog
        if not getattr(pdialog, "link_colors_checkbox", None):
            return False
        if not pdialog.link_colors_checkbox.isChecked():
            return False
        cal = pdialog.spline_calibration or {}
        n_channels = int(cal.get("n_channels", 0))
        if n_channels >= 2:
            # a loaded calibration always provides the registration - the
            # colours then show exactly what the fit will pair, so a bad
            # registration is visible and can be re-registered deliberately
            # (Postprocess > re-register from signal)
            cal = self._link_calibration_for_mode(cal, n_channels)
        else:
            # nothing loaded: register the channels from the identifications
            cal = self._estimated_link_calibration(box)
        if cal is None:
            return False
        n_channels = int(cal["n_channels"])
        tol = 1.5 * float(box)
        try:
            if cal.get("split_fov"):
                boxes = self._linked_boxes_split_fov(
                    cal, n_channels, frame_number, tol
                )
            else:
                boxes = self._linked_boxes_multichannel(
                    cal, n_channels, frame_number, tol
                )
        except Exception:
            # a malformed calibration must never break the viewer; fall back
            return False
        if boxes is None:
            return False
        box_half = int(box / 2)
        for x, y, color in boxes:
            self.scene.addRect(x - box_half, y - box_half, box, box, color)
        return True

    def _link_calibration_for_mode(
        self, cal: dict, n_channels: int
    ) -> dict | None:
        """The loaded calibration's registration, adapted to how the data are
        currently laid out (split-FOV regions vs. separate channels).

        The calibration is *always* the source of the inter-channel transform
        when one is loaded - the link colours then show exactly the pairing the
        fit will use, so a stale registration shows up as grey boxes and can be
        re-registered on purpose rather than being silently papered over. Only
        the placement is adapted when the layout differs from the calibration's:

        * split-FOV mode with a separate-movie calibration: both its channels
          start at the frame origin, so its transforms already *are* the
          region-local registration; they are placed at the drawn ROIs.
        * separate channels with a split-FOV calibration: the region-local
          affines *are* the inter-channel registration, so they apply directly
          to the whole frame of each movie.

        Returns None if the layout cannot be mapped onto the calibration's
        channels at all (e.g. a different number of ROIs).
        """
        split_fov_cal = bool(cal.get("split_fov"))
        if self.view.split_fov_mode == split_fov_cal:
            return cal
        if self.view.split_fov_mode:
            transforms = cal.get("channel_transforms")
            if not transforms or len(self.view.rois) != n_channels:
                return None
            return {
                "n_channels": n_channels,
                "split_fov": True,
                "regions": [list(map(list, r)) for r in self.view.rois],
                "channel_affines": [
                    np.asarray(t, dtype=float).tolist()
                    for t in transforms[:n_channels]
                ],
            }
        if len(self.channels) < 2:
            return cal  # single movie: keep the calibration's own regions
        affines = cal.get("channel_affines")
        if affines is None:
            regions = cal.get("regions")
            transforms = cal.get("channel_transforms")
            if not regions or not transforms:
                return None
            affines = [
                a.tolist()
                for a in localize.decompose_region_affines(
                    [_normalize_rect(r) for r in regions], transforms
                )
            ]
        return {
            "n_channels": n_channels,
            "channel_transforms": [
                np.asarray(a, dtype=float).tolist()
                for a in affines[:n_channels]
            ],
        }

    def _estimated_link_calibration(self, box: int) -> dict | None:
        """A calibration-shaped dict whose inter-channel transforms are
        estimated from the identifications themselves, for link colouring
        without a loaded spline calibration.

        The transforms come from
        :func:`spline.estimate_transforms_from_identifications`, which searches
        the mirror orientations - so a flipped channel (image splitter, mirrored
        quadrant) links just as it does with a calibration. Channels that cannot
        be registered fall back to the identity, i.e. the plain overlay.
        Estimating is not free, so the result is cached until the detections,
        the box size or the ROIs change. Returns None if there is nothing to
        link.
        """
        split_fov = self.view.split_fov_mode
        if split_fov:
            regions = [list(map(list, r)) for r in self.view.rois]
            if len(regions) < 2 or self.identifications is None:
                return None
            n_channels = len(regions)
        else:
            if len(self.channels) < 2:
                return None
            n_channels = len(self.channels)
            regions = None
        # the active channel's live detections are not mirrored back into
        # ``self.channels`` until the channel is switched
        ids_per_channel = (
            None
            if split_fov
            else [
                (
                    self.identifications
                    if c == self.current_channel
                    else self.channels[c].identifications
                )
                for c in range(n_channels)
            ]
        )
        key = (
            split_fov,
            int(box),
            n_channels,
            (
                tuple(np.asarray(regions).ravel().tolist())
                if regions is not None
                else None
            ),
            tuple(
                0 if ids is None else len(ids)
                for ids in (ids_per_channel or [self.identifications])
            ),
        )
        cached = getattr(self, "_link_cal_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        if split_fov:
            region_rects = [_normalize_rect(r) for r in regions]
            ids_per_channel = [
                self._identifications_in_region(rect) for rect in region_rects
            ]
        try:
            transforms = spline.estimate_transforms_from_identifications(
                ids_per_channel,
                box,
                regions=regions,
                frame_shape=(
                    None
                    if split_fov or self.movie is None
                    else (int(self.movie.shape[1]), int(self.movie.shape[2]))
                ),
            )
        except Exception:
            transforms = None
        if split_fov:
            # region-local affines: the ROI placement is stripped out, so the
            # boxes follow the ROIs if the user nudges them
            if transforms is None:
                affines = [identity for _ in range(n_channels)]
            else:
                affines = [
                    (
                        identity
                        if t is None
                        else localize.decompose_region_affines(
                            [region_rects[0], region_rects[c]],
                            [np.asarray(transforms[0]), np.asarray(t)],
                        )[1].tolist()
                    )
                    for c, t in enumerate(transforms)
                ]
            cal = {
                "n_channels": n_channels,
                "split_fov": True,
                "regions": regions,
                "channel_affines": affines,
            }
        else:
            cal = {
                "n_channels": n_channels,
                "channel_transforms": [
                    (
                        identity
                        if transforms is None or transforms[c] is None
                        else np.asarray(transforms[c]).tolist()
                    )
                    for c in range(n_channels)
                ],
            }
        self._link_cal_cache = (key, cal)
        return cal

    def _identifications_in_region(self, rect: tuple) -> pd.DataFrame | None:
        """The identifications inside a ``((y_min, x_min), (y_max, x_max))``
        region (split-FOV: one region per channel)."""
        ids = self.identifications
        if ids is None or len(ids) == 0:
            return None
        (y0, x0), (y1, x1) = rect
        x = np.asarray(ids["x"], dtype=float)
        y = np.asarray(ids["y"], dtype=float)
        return ids[(x >= x0) & (x < x1) & (y >= y0) & (y < y1)]

    def _linked_boxes_split_fov(
        self, cal: dict, n_channels: int, frame_number: int, tol: float
    ) -> list | None:
        """Colour-coded boxes for a split-FOV calibration: every region lives in
        one frame, so paired boxes across regions get the same colour. Returns a
        list of ``(x, y, QColor)`` for all this-frame spots, or None to fall
        back."""
        ids = self.identifications
        if ids is None or len(ids) == 0:
            return None
        # place the channels at the drawn ROIs when they match the channel
        # count (reference first), else the calibration's stored regions
        if self.view.split_fov_mode and len(self.view.rois) == n_channels:
            regions = [list(map(list, r)) for r in self.view.rois]
        else:
            regions = cal.get("regions")
        if not regions or len(regions) != n_channels:
            return None
        region_rects = [_normalize_rect(r) for r in regions]
        affines = cal.get("channel_affines")
        if affines is None:
            affines = [
                a.tolist()
                for a in localize.decompose_region_affines(
                    cal["regions"], cal["channel_transforms"]
                )
            ]
        transforms = localize.compose_region_transforms(
            region_rects, [np.asarray(a, dtype=float) for a in affines]
        )
        m = np.asarray(ids["frame"]) == frame_number
        xy = np.column_stack(
            [np.asarray(ids["x"])[m], np.asarray(ids["y"])[m]]
        ).astype(float)
        if len(xy) == 0:
            return []
        # assign each spot to the region that contains it (first match wins)
        region_of = np.full(len(xy), -1, dtype=int)
        for ci, ((y0, x0), (y1, x1)) in enumerate(region_rects):
            inside = (
                (xy[:, 0] >= x0)
                & (xy[:, 0] < x1)
                & (xy[:, 1] >= y0)
                & (xy[:, 1] < y1)
            )
            region_of[(region_of < 0) & inside] = ci
        ref_local = np.where(region_of == 0)[0]
        ref_xy = xy[ref_local]
        colors = [LINK_UNMATCHED_COLOR] * len(xy)
        # A reference spot links only if matched in EVERY other region/channel
        # (the bead is found in all channels). Count per reference spot, then
        # colour; spots missing from any channel stay grey.
        n_ref = len(ref_local)
        match_count = np.zeros(n_ref, dtype=int)
        per_channel: list = []
        n_checked = 0
        for c in range(1, n_channels):
            chan_local = np.where(region_of == c)[0]
            n_checked += 1
            if len(chan_local) == 0 or n_ref == 0:
                per_channel.append((chan_local, {}))
                continue
            pred = localize.apply_affine_transform(ref_xy, transforms[c])
            matches = _nearest_unique_match(pred, xy[chan_local], tol)
            per_channel.append((chan_local, matches))
            for rk in set(matches.values()):
                match_count[rk] += 1
        complete = (
            match_count == n_checked
            if n_checked
            else np.zeros(n_ref, dtype=bool)
        )
        # colour non-reference spots that pair with a fully-linked ref spot
        for chan_local, matches in per_channel:
            for tj, rk in matches.items():
                if complete[rk]:
                    colors[chan_local[tj]] = LINK_COLORS[rk % len(LINK_COLORS)]
        # a reference spot is coloured only if it links across all channels
        for rk, gi in enumerate(ref_local):
            if complete[rk]:
                colors[gi] = LINK_COLORS[rk % len(LINK_COLORS)]
        return [(xy[i, 0], xy[i, 1], colors[i]) for i in range(len(xy))]

    def _linked_boxes_multichannel(
        self, cal: dict, n_channels: int, frame_number: int, tol: float
    ) -> list | None:
        """Colour-coded boxes for a multichannel calibration (separate movies /
        one multichannel file): only the current channel is on screen, so a spot
        keeps its group colour as the user switches channels. Returns a list of
        ``(x, y, QColor)`` for the current channel's this-frame spots, or None to
        fall back."""
        transforms = cal.get("channel_transforms")
        if not transforms or len(transforms) < n_channels:
            return None
        if len(self.channels) < 2:
            return None
        reference = self.channels[0]
        if reference.identifications is None:
            return None

        def frame_xy(ids) -> np.ndarray:
            if ids is None or len(ids) == 0:
                return np.empty((0, 2), dtype=float)
            m = np.asarray(ids["frame"]) == frame_number
            return np.column_stack(
                [np.asarray(ids["x"])[m], np.asarray(ids["y"])[m]]
            ).astype(float)

        ref_xy = frame_xy(reference.identifications)

        # A reference spot counts as linked only if it is matched in EVERY
        # other channel (i.e. the bead is found in all channels). Count, per
        # reference spot, the channels it matches; "complete" requires a match
        # in each channel checked. Spots missing from any channel stay grey.
        n_ref = len(ref_xy)
        match_count = np.zeros(n_ref, dtype=int)
        n_checked = 0
        for c2 in range(1, n_channels):
            if c2 >= len(self.channels):
                continue
            n_checked += 1
            chan_xy = frame_xy(self.channels[c2].identifications)
            if n_ref == 0 or len(chan_xy) == 0:
                continue
            pred = localize.apply_affine_transform(
                ref_xy, np.asarray(transforms[c2], dtype=float)
            )
            for rk in set(_nearest_unique_match(pred, chan_xy, tol).values()):
                match_count[rk] += 1
        complete = (
            match_count == n_checked
            if n_checked
            else np.zeros(n_ref, dtype=bool)
        )

        c = self.current_channel
        if c == 0:
            # colour reference spots only if they pair in ALL other channels
            return [
                (
                    ref_xy[rk, 0],
                    ref_xy[rk, 1],
                    (
                        LINK_COLORS[rk % len(LINK_COLORS)]
                        if complete[rk]
                        else LINK_UNMATCHED_COLOR
                    ),
                )
                for rk in range(n_ref)
            ]
        # a non-reference channel: colour its detections by the reference spot
        # they pair with, but only when that reference spot links across ALL
        # channels; otherwise grey (matches that spot's box in channel 0)
        cur_xy = frame_xy(self.identifications)
        if len(cur_xy) == 0:
            return []
        matches = {}
        if n_ref:
            pred = localize.apply_affine_transform(
                ref_xy, np.asarray(transforms[c], dtype=float)
            )
            matches = _nearest_unique_match(pred, cur_xy, tol)
        return [
            (
                cur_xy[j, 0],
                cur_xy[j, 1],
                (
                    LINK_COLORS[matches[j] % len(LINK_COLORS)]
                    if (j in matches and complete[matches[j]])
                    else LINK_UNMATCHED_COLOR
                ),
            )
            for j in range(len(cur_xy))
        ]

    def draw_scalebar(self) -> None:
        """Draw a scale bar if the option is checked."""
        if not self.scalebar_action.isChecked():
            return

        scene_pixelsize = self.parameters_dialog.pixelsize.value()

        # length (nm) - set optimal size (~1/8 of image width)
        rect = self.view.viewport().rect()
        visible_scene_rect = self.view.mapToScene(rect).boundingRect()
        width = visible_scene_rect.width()
        # the view may not be laid out yet (e.g. slow/network loads), in
        # which case ``width`` is 0 and there is nothing to draw
        if width <= 0:
            return
        width_nm = width * scene_pixelsize
        optimal_scalebar = width_nm / 8

        # approximate to the nearest thousands, hundreds, tens or ones
        if optimal_scalebar > 10_000:
            scalebar = 10_000
        elif optimal_scalebar > 1_000:
            scalebar = int(1_000 * round(optimal_scalebar / 1_000))
        elif optimal_scalebar > 100:
            scalebar = int(100 * round(optimal_scalebar / 100))
        elif optimal_scalebar > 10:
            scalebar = int(10 * round(optimal_scalebar / 10))
        else:
            scalebar = int(round(optimal_scalebar))

        # position against the viewport (the drawable area) rather than
        # the whole view, so the bar is not covered by the scrollbar
        # gutters (opaque on Windows, overlaid on macOS)
        viewport_width = self.view.viewport().width()
        viewport_height = self.view.viewport().height()
        length_displaypxl = int(
            round(viewport_width * (scalebar / scene_pixelsize) / width)
        )
        # when zoomed in far enough the scale bar rounds down to 0 nm /
        # 0 display pixels; skip drawing to avoid a division by zero below
        if scalebar <= 0 or length_displaypxl <= 0:
            return
        height_displaypxl = 10

        # draw a rectangle
        x = viewport_width - length_displaypxl - 40
        y = viewport_height - height_displaypxl - 20
        pen = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
        brush = QtGui.QBrush(QtGui.QColor("white"))
        polygon = self.view.mapToScene(
            x,
            y,
            length_displaypxl,
            height_displaypxl,
        )
        x_scene = polygon.boundingRect().x()
        y_scene = polygon.boundingRect().y()
        length_scene = polygon.boundingRect().width()
        height_scene = polygon.boundingRect().height()
        self.scene.addRect(
            x_scene,
            y_scene,
            length_scene,
            height_scene,
            pen,
            brush,
        )

        # add scale bar text
        font = QtGui.QFont()
        font.setPointSize(20)
        text_item = self.scene.addText(f"{scalebar} nm", font)
        text_item.setDefaultTextColor(QtGui.QColor("white"))
        # scene units per device pixel (uniform zoom, but keep x/y
        # separate to be safe)
        scene_per_px_x = length_scene / length_displaypxl
        scene_per_px_y = height_scene / height_displaypxl
        # position the text centered above the scale bar, with a fixed
        # device-pixel gap so the spacing looks the same at every zoom
        text_rect = text_item.boundingRect()
        text_width = text_rect.width() * scene_per_px_x
        text_height = text_rect.height() * scene_per_px_y
        gap = 8 * scene_per_px_y
        text_x = x_scene + (length_scene - text_width) / 2
        text_y = y_scene - gap - text_height
        text_item.setPos(text_x, text_y)
        text_item.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,  # noqa: E501
            True,
        )

    @property
    def parameters(self) -> dict:
        """Dictionary with box size and min. net gradient."""
        return {
            "Box Size": self.parameters_dialog.box_spinbox.value(),
            "Min. Net Gradient": self.parameters_dialog.mng_slider.value(),
        }

    def on_parameters_changed(self) -> None:
        """Reset ``self.locs`` and draw frame."""
        # Ignore the value changes emitted while restoring a channel's
        # stored parameters - otherwise switching channels would wipe the
        # channel's localizations and fit markers.
        if self._switching_channel:
            return
        self.locs = None
        self.locs_display = None
        self.ready_for_fit = False
        self.draw_frame()

    def abort(self) -> None:
        """Abort the currently running async process."""
        if self._active_worker is not None:
            self._active_worker.requestInterruption()

    def on_worker_aborted(self) -> None:
        """Handle the abortion of any worker thread."""
        self._active_worker = None
        self.abort_action.setEnabled(False)
        # restore the GPU checkbox state for the selected model
        self.parameters_dialog.on_fit_optimizer_changed()
        self.status_bar.showMessage("Aborted.")

    def identify(
        self,
        fit_afterwards: bool = False,
        calibrate_z: bool = False,
        calibrate_spline: bool = False,
    ) -> None:
        """Identify spots in the loaded movie.

        Parameters
        ----------
        fit_afterwards : bool, optional
            Whether to automatically fit the identified spots
            afterwards. Default is False.
        calibrate_z : bool, optional
            Whether to run z-calibration for 3D fitting after
            identification. Default is False.
        calibrate_spline : bool, optional
            Whether to build a cubic-spline PSF calibration after
            identification (see ``build_spline_calibration``). Default is
            False.
        """
        if len(self.channels) > 1:
            self._identify_all_channels(
                fit_afterwards=fit_afterwards,
                calibrate_z=calibrate_z,
                calibrate_spline=calibrate_spline,
            )
            return
        if self.movie is not None:
            self.status_bar.showMessage("Preparing identification...")
            self.identification_worker = IdentificationWorker(
                self, fit_afterwards, calibrate_z, calibrate_spline
            )
            self.identification_worker.progressMade.connect(
                self.on_identify_progress
            )
            self.identification_worker.finished.connect(
                self.on_identify_finished
            )
            self.identification_worker.aborted.connect(self.on_worker_aborted)
            self._active_worker = self.identification_worker
            self.abort_action.setEnabled(True)
            self.identification_worker.start()

    def on_identify_progress(
        self,
        frame_number: int,
        parameters: dict,
    ) -> None:
        """Update the status bar with the current identification
        progress."""
        n_frames = self.info[0]["Frames"]
        box = parameters["Box Size"]
        mng = parameters["Min. Net Gradient"]
        message = (
            f"Identifying in frame {frame_number:,} / {n_frames:,}"
            f" (Box Size: {box}; Min. Net Gradient: {mng:,}) ..."
        )
        self.status_bar.showMessage(message)

    def _linked_count_phrase(self, n_detections: int) -> str | None:
        """How many identified spots a joint spline fit would actually fit.

        A multichannel (or split-FOV) spline fit fits one spot per molecule:
        the reference detections that are found in every channel / region are
        fitted jointly, everything else is dropped (see
        ``localize.filter_linked_identifications``).
        """
        cal = self.parameters_dialog.spline_calibration or {}
        box = self.parameters["Box Size"]
        split_fov = bool(cal.get("split_fov"))
        self.status_bar.showMessage(
            "Linking spots across "
            + ("regions" if split_fov else "channels")
            + " ..."
        )
        self.status_bar.repaint()  # not processEvents: no re-entrant slots
        try:
            if split_fov:
                ids = self.identifications
                if ids is None or len(ids) == 0:
                    return None
                n_channels = int(
                    cal.get("n_channels") or len(cal.get("regions") or [])
                )
                regions = None
                if (
                    self.view.split_fov_mode
                    and len(self.view.rois) == n_channels
                ):
                    regions = [list(map(list, r)) for r in self.view.rois]
                _, n_kept, _ = (
                    localize.filter_linked_identifications_split_fov(
                        ids, cal, box, regions=regions
                    )
                )
                where = "regions"
            else:
                transforms = cal.get("channel_transforms")
                n_channels = min(
                    int(cal.get("n_channels", len(self.channels))),
                    len(self.channels),
                )
                if (
                    n_channels < 2
                    or not transforms
                    or len(transforms) < n_channels
                ):
                    return None
                # the flat state holds the active channel's detections
                self._snapshot_current_channel()
                ids_per_channel = [
                    c.identifications for c in self.channels[:n_channels]
                ]
                reference = ids_per_channel[0]
                if reference is None or len(reference) == 0:
                    return None
                if all(i is None or len(i) == 0 for i in ids_per_channel[1:]):
                    return None
                _, n_kept, _ = localize.filter_linked_identifications(
                    ids_per_channel, transforms, box
                )
                where = "channels"
        except (ValueError, KeyError, IndexError):
            return None
        return (
            f"{n_kept:,} spots linked across {n_channels} {where} "
            f"({n_detections:,} in total)"
        )

    def on_identify_finished(
        self,
        parameters: dict,
        roi: list,
        elapsed_time: float,
        identifications: pd.DataFrame,
        fit_afterwards: bool,
        calibrate_z: bool,
        calibrate_spline: bool,
    ) -> None:
        """Handle the completion of the identification process. Save
        the parameters used, and localize/calibrate if requested."""
        self._active_worker = None
        self.abort_action.setEnabled(False)
        if len(identifications):
            self.locs = None
            self.locs_display = None
            self.last_identification_info = parameters.copy()
            self.last_identification_info["ROI"] = roi
            self.last_identification_info["Frame bounds"] = self.frame_range
            n_identifications = len(identifications)
            box = parameters["Box Size"]
            mng = parameters["Min. Net Gradient"]
            self.identifications = identifications
            self.ready_for_fit = True
            # for split-FOV data the detections of every region sit in this one
            # table, but the joint fit only fits molecules linked across all of
            # them - report that count (see _linked_count_phrase)
            counted = (
                self._linked_count_phrase(n_identifications)
                or f"{n_identifications:,} spots"
            )
            message = (
                f"Identified {counted} in {elapsed_time:.2f}"
                f" seconds. (Box Size: {box}; Min. Net Gradient: {mng}). "
                "Ready for fit."
            )
            self.status_bar.showMessage(message)
            self.draw_frame()
            # sound notification
            if elapsed_time > lib.SOUND_NOTIFICATION_DURATION:
                sound_path = lib.get_sound_notification_path()
                if sound_path is not None:
                    playsound(sound_path, block=False)
            if calibrate_spline:
                self.build_spline_calibration()
            elif fit_afterwards:
                self.fit(calibrate_z=calibrate_z)
        elif calibrate_spline:
            self.status_bar.showMessage("")
            QtWidgets.QMessageBox.information(
                self,
                "Spline PSF Calibration",
                "No beads were identified. Lower the minimum net gradient "
                "or check the selected frame range and try again.",
            )

    def _identify_all_channels(
        self,
        fit_afterwards: bool = False,
        calibrate_z: bool = False,
        calibrate_spline: bool = False,
    ) -> None:
        """Identify spots in every channel in turn (multichannel Identify).

        Each channel is activated and identified with its own box / min. net
        gradient (shared when the matching 'Same across channels' link is on)
        and the shared ROI / frame range; the results are stored per channel.
        The originally active channel is restored when the batch finishes, and
        the requested follow-up (fit or calibration, as passed to
        :meth:`identify`) then runs once - so 'Localize (Identify && Fit)'
        behaves like Identify-then-Fit over all channels."""
        self._multi_identify = {
            "return_channel": self.current_channel,
            "queue": list(range(len(self.channels))),
            "total": len(self.channels),
            "done": 0,
            "sum": 0,
            "fit_afterwards": bool(fit_afterwards),
            "calibrate_z": bool(calibrate_z),
            "calibrate_spline": bool(calibrate_spline),
        }
        self.status_bar.showMessage("Identifying all channels...")
        self._identify_run_next()

    def _identify_run_next(self) -> None:
        """Start identification on the next queued channel, or finish the
        multichannel Identify batch and restore the active channel."""
        state = self._multi_identify
        if state is None:
            return
        if not state["queue"]:
            self._multi_identify = None
            self.set_current_channel(state["return_channel"])
            # persist the last channel's results (set_current_channel only
            # snapshots on an actual switch)
            self._snapshot_current_channel()
            self._active_worker = None
            self.abort_action.setEnabled(False)
            self.draw_frame()
            # the joint multichannel fit only fits molecules found in every
            # channel, so report those rather than the raw detection total
            counted = self._linked_count_phrase(state["sum"]) or (
                f"{state['sum']:,} spots across {state['total']} channels"
            )
            self.status_bar.showMessage(
                f"Identified {counted}. Ready for fit."
            )
            # the follow-up requested by the entry point (Localize / a
            # calibration run), now that every channel is identified
            if state["sum"]:
                if state["calibrate_spline"]:
                    self.build_spline_calibration()
                elif state["fit_afterwards"]:
                    self.fit(calibrate_z=state["calibrate_z"])
            elif state["calibrate_spline"]:
                QtWidgets.QMessageBox.information(
                    self,
                    "Spline PSF Calibration",
                    "No beads were identified in any channel. Lower the "
                    "minimum net gradient or check the selected frame range "
                    "and try again.",
                )
            return
        idx = state["queue"].pop(0)
        self.set_current_channel(idx)
        if self.movie is None:
            self._identify_run_next()
            return
        worker = IdentificationWorker(self, False, False, False)
        worker.progressMade.connect(self.on_identify_progress)
        worker.finished.connect(self._on_multi_identify_finished)
        worker.aborted.connect(self._on_multi_identify_aborted)
        self.identification_worker = worker
        self._active_worker = worker
        self.abort_action.setEnabled(True)
        worker.start()

    def _on_multi_identify_finished(
        self,
        parameters: dict,
        roi: list,
        elapsed_time: float,
        identifications: pd.DataFrame,
        *_: object,
    ) -> None:
        """Store one channel's identifications, then start the next channel."""
        self._active_worker = None
        state = self._multi_identify
        if len(identifications):
            self.locs = None
            self.locs_display = None
            self.last_identification_info = parameters.copy()
            self.last_identification_info["ROI"] = roi
            self.last_identification_info["Frame bounds"] = self.frame_range
            self.identifications = identifications
            self.ready_for_fit = True
            if state is not None:
                state["sum"] += len(identifications)
        else:
            self.identifications = None
            self.ready_for_fit = False
        if state is not None:
            state["done"] += 1
            name = self.channels[self.current_channel].name
            self.status_bar.showMessage(
                f"Identified channel {state['done']}/{state['total']} "
                f"({name}): {len(identifications):,} spots ..."
            )
        self._identify_run_next()

    def _on_multi_identify_aborted(self) -> None:
        """Abort the multichannel Identify batch and restore the view."""
        state = self._multi_identify
        self._multi_identify = None
        self._active_worker = None
        self.abort_action.setEnabled(False)
        if state is not None:
            self.set_current_channel(state["return_channel"])
        self.draw_frame()
        self.status_bar.showMessage("Aborted.")

    def _check_spline_box_size(self, spline_calibration: dict) -> bool:
        """Check that the identification box size is not larger than in
        the spline calibration."""
        n_data = spline_calibration.get("n_data")
        if not n_data:
            return True  # nothing to compare against; let the fit proceed
        calib_box = int(n_data[0])
        box = self.parameters["Box Size"]
        # equal or smaller: supported by fitting against a centered crop
        if box <= calib_box:
            return True
        detail = (
            f"The selected box size ({box} px) is larger than this spline "
            f"calibration, which was built with a box size of "
            f"{calib_box} px. Use the box of the same size or smaller."
        )
        reply = QtWidgets.QMessageBox.question(
            self,
            "Spline PSF fit — box size",
            f"{detail}\n\nSet the box size to {calib_box} px now?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.parameters_dialog.box_spinbox.setValue(calib_box)
        return False

    def fit(self, calibrate_z: bool = False) -> None:
        """Fit identified spots (single molecules).

        Parameters
        ----------
        calibrate_z : bool, optional
            Whether to perform z-calibration during fitting. Default is
            False.
        """
        if self.movie is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Fit",
                "Load a movie before fitting.",
            )
            return
        if not self.ready_for_fit:
            QtWidgets.QMessageBox.warning(
                self,
                "Fit",
                "No identifications available. Run Identify (or load "
                "identifications) before fitting.",
            )
            return
        self.status_bar.showMessage("Preparing fit...")
        model = self.parameters_dialog.fit_model.currentText()
        optimizer = self.parameters_dialog.fit_optimizer.currentText()
        method = _fit_code(model, optimizer)
        # get the convergence criterion and max iterations
        iterates = (
            _effective_fit_code(
                method, self.parameters_dialog.gpu_checkbox.isChecked()
            )
            in _CONVERGENCE_CODES
        )
        eps = (
            self.parameters_dialog.convergence_criterion.value()
            if iterates
            else None
        )
        max_it = self.parameters_dialog.max_it.value() if iterates else None
        fit_z = self.parameters_dialog.fit_z_checkbox.isChecked()
        use_gpu = self.parameters_dialog.gpu_checkbox.isChecked()
        spline_calibration = None
        if method.startswith("spline"):
            spline_calibration = self.parameters_dialog.spline_calibration
            if not spline_calibration:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Spline PSF fit",
                    "Load a spline PSF calibration first (Experimental "
                    "PSF (spline) > Load calibration), or build one via "
                    "3D > Calibrate spline PSF.",
                )
                self.status_bar.showMessage("")
                return
            # The spline fit requires the box size to match the one the
            # calibration was built with.
            if not self._check_spline_box_size(spline_calibration):
                self.status_bar.showMessage("")
                return
            # A 3D spline fit recovers z directly, so the separate
            # astigmatism z-fitting step must not run.
            fit_z = False
            if spline_calibration.get("model") == "spline-3d-multichannel":
                self._start_multichannel_spline_fit(
                    spline_calibration, method, eps, max_it
                )
                return
        self.fit_worker = FitWorker(
            self.movie,
            self.info,
            self.camera_info,
            self.identifications,
            self.parameters["Box Size"],
            method,
            eps,
            max_it,
            fit_z,
            calibrate_z,
            use_gpu,
            spline_calibration=spline_calibration,
        )
        self.fit_worker.progressMade.connect(self.on_fit_progress)
        self.fit_worker.cutProgressMade.connect(self.on_cut_progress)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = self.fit_worker
        self.abort_action.setEnabled(True)
        self.fit_worker.start()

    def _start_multichannel_spline_fit(
        self,
        calibration: dict,
        method: str,
        eps: float | None = None,
        max_it: int | None = None,
    ) -> None:
        """Fit a multichannel spline PSF across all loaded channels
        simultaneously. The first loaded channel is the reference; its
        identifications are mapped into every channel via the calibration's
        stored transforms."""
        if calibration.get("split_fov"):
            self._start_split_fov_spline_fit(calibration, method, eps, max_it)
            return
        n_channels = int(calibration.get("n_channels", 0))
        # persist the displayed channel's identifications, so the per-channel
        # tables read below are complete even if no channel switch happened
        # since the last Identify run
        self._snapshot_current_channel()
        if len(self.channels) < n_channels:
            QtWidgets.QMessageBox.warning(
                self,
                "Multichannel spline fit",
                f"This calibration expects {n_channels} channels, but "
                f"{len(self.channels)} are loaded. Load them with "
                "'File > Open channels from several movies' in the same "
                "order as the calibration.",
            )
            self.status_bar.showMessage("")
            return
        reference = self.channels[0]
        if reference.identifications is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Multichannel spline fit",
                "Identify spots in the reference channel (the first loaded "
                "channel) before running a multichannel spline fit.",
            )
            self.status_bar.showMessage("")
            return
        movies = [self.channels[c].movie for c in range(n_channels)]
        # Use each channel's own camera info when available (needed for correct
        # per-channel photon conversion); fall back to the shared one.
        camera_infos = [
            getattr(self.channels[c], "camera_info", None) or self.camera_info
            for c in range(n_channels)
        ]
        # Use only linked identifications
        ids_per_channel = [
            getattr(self.channels[c], "identifications", None)
            for c in range(n_channels)
        ]
        n_missing = sum(
            1 for ids in ids_per_channel[1:] if ids is None or len(ids) == 0
        )
        if n_missing:
            QtWidgets.QMessageBox.information(
                self,
                "Multichannel spline fit",
                f"{n_missing} of the {n_channels - 1} non-reference channels "
                "have no identifications, so localizations cannot be linked "
                "across all channels. Run 'Analyze > Identify' (Ctrl+I) first - "
                "with several channels loaded it identifies every channel - for "
                "a fully linked fit; continuing with the channels that are "
                "identified.",
            )
        self.fit_worker = MultichannelSplineFitWorker(
            movies,
            camera_infos,
            reference.identifications,
            self.parameters["Box Size"],
            calibration,
            mle=method in ("spline-mle", "spline-mle-gpu"),
            use_gpu=method.endswith("-gpu"),
            link_photons=self.parameters_dialog._link_photons_enabled(),
            identifications_per_channel=ids_per_channel,
            eps=eps,
            max_it=max_it,
        )
        self.fit_worker.linkMade.connect(self.on_link_progress)
        self.fit_worker.linkFinished.connect(self.on_link_finished)
        self.fit_worker.progressMade.connect(self.on_fit_progress)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = self.fit_worker
        self.abort_action.setEnabled(True)
        self.fit_worker.start()

    def _start_split_fov_spline_fit(
        self,
        calibration: dict,
        method: str,
        eps: float | None = None,
        max_it: int | None = None,
    ) -> None:
        """Fit a split-FOV multichannel spline PSF from the single loaded movie.
        The channels are placed at the drawn ROIs when they match the
        calibration's channel count (so a moved split can be re-registered by
        re-drawing), otherwise at the calibration's stored regions. The
        reference region's identifications are mapped into every region via the
        stored inter-channel affine (see ``localize.fit_spline_split_fov``)."""
        if self.identifications is None or len(self.identifications) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Split-FOV spline fit",
                "Identify spots first (they are confined to the reference "
                "region automatically).",
            )
            self.status_bar.showMessage("")
            return
        n_channels = int(calibration.get("n_channels", 0))
        # use the drawn ROIs to place the channels when they match the channel
        # count (reference first); else fall back to the calibration positions
        regions = None
        if self.view.split_fov_mode and len(self.view.rois) == n_channels:
            regions = [list(map(list, r)) for r in self.view.rois]
        # guard: warn if no identification falls in the reference region (a
        # moved/rescaled split or wrong ROIs would otherwise fit nothing)
        ref_rect = (regions or calibration.get("regions"))[
            0 if regions else int(calibration.get("reference", 0))
        ]
        (ry0, rx0), (ry1, rx1) = ref_rect
        rx = np.asarray(self.identifications["x"], dtype=float)
        ry = np.asarray(self.identifications["y"], dtype=float)
        n_in = int(
            np.count_nonzero(
                (rx >= min(rx0, rx1))
                & (rx < max(rx0, rx1))
                & (ry >= min(ry0, ry1))
                & (ry < max(ry0, ry1))
            )
        )
        if n_in == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Split-FOV spline fit",
                f"{len(self.identifications)} spots were identified but none "
                "fall inside the reference region, so there is nothing to fit. "
                "The reference region is probably in the wrong place for this "
                "data: enable 'Regions = channels' and drag the ROIs onto the "
                "channels (reference first), or run Calibration > Refine "
                "split-FOV registration.",
            )
            self.status_bar.showMessage("")
            return
        self.fit_worker = MultichannelSplineFitWorker(
            [self.movie],
            [self.camera_info],
            self.identifications,
            self.parameters["Box Size"],
            calibration,
            mle=method in ("spline-mle", "spline-mle-gpu"),
            use_gpu=method.endswith("-gpu"),
            split_fov=True,
            regions=regions,
            link_photons=self.parameters_dialog._link_photons_enabled(),
            eps=eps,
            max_it=max_it,
        )
        self.fit_worker.linkMade.connect(self.on_link_progress)
        self.fit_worker.linkFinished.connect(self.on_link_finished)
        self.fit_worker.progressMade.connect(self.on_fit_progress)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = self.fit_worker
        self.abort_action.setEnabled(True)
        self.fit_worker.start()

    def reregister_channels_from_signal(self) -> None:
        """Re-estimate the inter-channel registration of the loaded spline
        calibration directly from the current blinking data.

        Both channel layouts pair shared single-molecule signal frame by frame
        and re-fit the inter-channel affines, updating the loaded calibration;
        this dispatches on the calibration:

        * **Split-FOV** (regions of one movie): the signal is read inside the
          drawn ROIs (or the calibration's regions) and each channel is seeded
          by only the calibration's flip (see
          :func:`spline.refine_split_fov_transforms_from_signal`).
        * **Multichannel** (a separate movie per channel): the loaded channel
          movies are paired frame by frame, seeded from the calibration's
          existing transforms (see
          :func:`spline.refine_multichannel_transforms_from_signal`).

        The frame window and the number of frames sampled from it are asked for
        in :class:`RefineRegistrationDialog` (seeded from the identification
        frame range), since which part of the movie pairs best is
        data-dependent.
        """
        title = "Re-align channels (signal)"
        calibration = self.parameters_dialog.spline_calibration
        if not (calibration and calibration.get("channel_transforms")):
            QtWidgets.QMessageBox.warning(
                self,
                title,
                "Load a multichannel or split-FOV spline calibration first "
                "(Experimental PSF (spline) > Load calibration).",
            )
            return
        split_fov = bool(calibration.get("split_fov"))
        parameters = self.parameters

        # gather the layout-specific inputs and pick the refiner
        if split_fov:
            if self.movie is None:
                QtWidgets.QMessageBox.information(
                    self, title, "No movie loaded."
                )
                return
            n_channels = int(calibration.get("n_channels", 0))
            if self.view.split_fov_mode and len(self.view.rois) == n_channels:
                regions = [list(map(list, r)) for r in self.view.rois]
            else:
                regions = calibration.get("regions")
            if not regions or len(regions) != n_channels:
                QtWidgets.QMessageBox.warning(
                    self,
                    title,
                    f"Draw one ROI per channel (reference first): {n_channels} "
                    "regions are needed for this calibration.",
                )
                return

            n_movie_frames = len(self.movie)

            def _refine(frame_bounds, max_frames):
                return spline.refine_split_fov_transforms_from_signal(
                    self.movie,
                    calibration,
                    regions,
                    minimum_ng=parameters["Min. Net Gradient"],
                    box=parameters["Box Size"],
                    frame_bounds=frame_bounds,
                    max_frames=max_frames,
                )

        else:
            n_channels = int(calibration.get("n_channels", len(self.channels)))
            if len(self.channels) < n_channels:
                QtWidgets.QMessageBox.warning(
                    self,
                    title,
                    f"This calibration has {n_channels} channels, but "
                    f"{len(self.channels)} movies are loaded. Load them with "
                    "'File > Open channels from several movies' in the same "
                    "order as the calibration (reference first).",
                )
                return
            # persist the active channel so every channel's movie is current
            self._snapshot_current_channel()
            movies = [self.channels[c].movie for c in range(n_channels)]
            # the channels are frame-synchronized, so only frames present in
            # every movie can be paired
            n_movie_frames = min(len(m) for m in movies)

            def _refine(frame_bounds, max_frames):
                return spline.refine_multichannel_transforms_from_signal(
                    movies,
                    calibration,
                    minimum_ng=parameters["Min. Net Gradient"],
                    box=parameters["Box Size"],
                    frame_bounds=frame_bounds,
                    max_frames=max_frames,
                )

        # let the user pick the frames considered: the first frames of a movie
        # are often too dense (or still bleaching) for unambiguous pairing
        frame_bounds, max_frames, ok = RefineRegistrationDialog.getFrameSpecs(
            self, n_movie_frames, self.frame_range
        )
        if not ok:
            return

        self.status_bar.showMessage("Re-aligning channels from signal ...")
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor
        )
        try:
            _, reg_info = _refine(frame_bounds, max_frames)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.status_bar.showMessage("")
            QtWidgets.QMessageBox.critical(
                self, title, f"Re-alignment failed: {e}"
            )
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.status_bar.showMessage("")

        # split-FOV: reflect the (possibly re-ordered) regions in the view
        if split_fov:
            self.view.rois = [
                [[int(r[0][0]), int(r[0][1])], [int(r[1][0]), int(r[1][1])]]
                for r in (calibration.get("regions") or [])
            ]
            self.parameters_dialog.update_roi_display()
            self.draw_frame()

        rows = [
            f"ch{r['channel']}: {r['n_matches']} paired signals, "
            f"RMS {r['rms']:.2f} px"
            for r in reg_info
        ]
        QtWidgets.QMessageBox.information(
            self,
            title,
            "Channels re-aligned from the current signal:\n\n"
            + "\n".join(rows),
        )

    def fit_z(self) -> None:
        """Fit z coordinates of the fitted localizations based on the
        calibration data."""
        self.status_bar.showMessage("Fitting z position...")
        model = self.parameters_dialog.fit_model.currentText()
        optimizer = self.parameters_dialog.fit_optimizer.currentText()
        fitting_method = _fit_code(model, optimizer)
        # zfit only knows gausslq/gaussmle; map the GPU/rotated/avg
        # codes to the corresponding CPU noise model
        fitting_method = (
            "gaussmle" if fitting_method.startswith("gaussmle") else "gausslq"
        )
        self.fit_z_worker = FitZWorker(
            self.locs,
            self.info + [self.camera_info],  # ensure pixel size in info
            self.parameters_dialog.z_calibration,
            self.parameters_dialog.magnification_factor.value(),
            self.parameters_dialog.pixelsize.value(),
            fitting_method,
            self.parameters_dialog.gpu_checkbox.isChecked(),
        )
        self.fit_z_worker.progressMade.connect(self.on_fit_z_progress)
        self.fit_z_worker.finished.connect(self.on_fit_z_finished)
        self.fit_z_worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = self.fit_z_worker
        self.abort_action.setEnabled(True)
        self.fit_z_worker.start()

    def on_cut_progress(self, curr: int, total: int) -> None:
        """Update the status bar with the spot cutting progress."""
        message = f"Extracting spot {curr:,} / {total:,} ..."
        self.status_bar.showMessage(message)

    def _link_target_name(self) -> str:
        """'regions' for a split-FOV fit (the channels are regions of the one
        loaded movie), 'channels' otherwise."""
        worker = getattr(self, "fit_worker", None)
        return "regions" if getattr(worker, "split_fov", False) else "channels"

    def on_link_progress(self, curr: int, total: int) -> None:
        """Update the status bar with the cross-channel linking progress."""
        self.status_bar.showMessage(
            f"Linking spots across {self._link_target_name()}: "
            f"{curr:,} / {total:,} ..."
        )

    def on_link_finished(self, n_kept: int, n_total: int) -> None:
        """Report how many detections survived the cross-channel linking."""
        pct = 100 * n_kept / n_total if n_total else 0.0
        self.status_bar.showMessage(
            f"{n_kept:,} of {n_total:,} spots ({pct:.1f}%) linked across all "
            f"{self._link_target_name()}; fitting those ..."
        )

    def on_fit_progress(self, curr: int, total: int) -> None:
        """Update the status bar with the fitting progress."""
        worker = getattr(self, "fit_worker", None)
        if isinstance(worker, MultichannelSplineFitWorker):
            # extraction, GPU fit and per-spot CRLB share this callback
            message = f"Fitting multichannel spline: {curr:,} / {total:,} ..."
            self.status_bar.showMessage(message)
        elif getattr(worker, "method", "").endswith("-gpu") and getattr(
            worker, "method", ""
        ).startswith("spline"):
            # The GPU spline fit is one launch per chunk, so this callback
            # only ever reports the per-spot CRLB pass that follows it. The
            # CPU spline reports the fit itself and falls through to the
            # generic per-spot message below.
            self.status_bar.showMessage("Calculating localization precision")
        elif self.parameters_dialog.gpu_checkbox.isChecked():
            self.status_bar.showMessage("Fitting spots on the GPU...")
        else:
            message = f"Fitting spot {curr:,} / {total:,} ..."
            self.status_bar.showMessage(message)

    def on_fit_finished(
        self,
        locs: pd.DataFrame,
        elapsed_time: float,
        fit_z: bool,
        calibrate_z: bool,
    ) -> None:
        """Handle the completion of the fitting process. Draw fit
        markers, fit/calibration z coordinates, if requested, save
        localizations."""
        self._active_worker = None
        self.abort_action.setEnabled(False)
        self.status_bar.showMessage(
            f"Fitted {len(locs):,} spots in {elapsed_time:.2f} seconds."
        )
        self.locs = locs
        self.locs_display = locs
        self._distribute_multichannel_fit(locs)
        self.draw_frame()
        # sound notification
        if elapsed_time > lib.SOUND_NOTIFICATION_DURATION:
            sound_path = lib.get_sound_notification_path()
            if sound_path is not None:
                playsound(sound_path, block=False)
        base = self.channel_output_base()
        if calibrate_z:
            # restore the GPU checkbox state for the selected model
            self.parameters_dialog.on_fit_optimizer_changed()
            step, frames_per_step, frame_order, ok = (
                Calibrate3DDialog.getCalibrationSpecs(self)
            )
            if ok:
                base = self.channel_output_base()
                out_path = base + "_3d_calib.yaml"
                path, exe = lib.get_save_filename_ext_dialog(
                    self, "Save 3D calibration", out_path, filter="*.yaml"
                )
                if path:
                    t0 = time.time()
                    zfit.calibrate_z(
                        locs,
                        self.info,
                        step,
                        self.parameters_dialog.magnification_factor.value(),
                        path=path,
                        frame_bounds=self.frame_range,
                        frames_per_step=frames_per_step,
                        frame_order=frame_order,
                    )
                    dt = time.time() - t0
                    if dt > lib.SOUND_NOTIFICATION_DURATION:
                        sound_path = lib.get_sound_notification_path()
                        if sound_path is not None:
                            playsound(sound_path, block=False)
                    self.status_bar.showMessage(
                        f"3D calibrated in {dt:.2f} seconds."
                    )
        else:
            if fit_z:
                self.fit_z()
            else:
                self.save_locs_after_fit()

    def _locs_in_channel(
        self, locs: pd.DataFrame, transform: np.ndarray
    ) -> pd.DataFrame:
        """Map reference-channel localizations into another channel's pixel
        coordinates via the calibration's affine, for cross-channel display."""
        xy = localize.apply_affine_transform(
            np.column_stack(
                [
                    np.asarray(locs["x"], dtype=np.float64),
                    np.asarray(locs["y"], dtype=np.float64),
                ]
            ),
            np.asarray(transform, dtype=np.float64),
        )
        mapped = locs.copy()
        mapped["x"] = xy[:, 0].astype(np.float32)
        mapped["y"] = xy[:, 1].astype(np.float32)
        return mapped

    def _distribute_multichannel_fit(self, locs: pd.DataFrame) -> bool:
        """Show a multichannel spline fit on every loaded channel.

        The fit is in the reference channel's (channel 0) coordinates. Each
        other channel's ``locs_display`` gets the same localizations mapped
        into its own pixel frame via the calibration's ``channel_transforms``,
        so switching channels overlays the fit on that channel's movie. The
        fit itself (saveable ``locs``) stays with the reference channel, which
        is made active so it displays and saves in reference coordinates.

        Returns False (a no-op) for single-movie data, including split-FOV.
        """
        if len(self.channels) <= 1:
            return False
        calibration = self.parameters_dialog.spline_calibration
        transforms = (calibration or {}).get("channel_transforms")
        for c, channel in enumerate(self.channels):
            if c == 0 or not transforms or c >= len(transforms):
                channel.locs_display = locs
            else:
                channel.locs_display = self._locs_in_channel(
                    locs, transforms[c]
                )
        self.channels[0].locs = locs
        if self.current_channel != 0:
            # the fit belongs to the reference channel: activate it so the
            # overlay lands on the reference movie and saving uses reference
            # coordinates. Restore directly (no snapshot) - the just-set
            # per-channel locs_display would otherwise be clobbered by the
            # stale flat state.
            self.current_channel = 0
            self._populate_channel_combo()
            self._restore_current_channel()
        return True

    def on_fit_z_progress(self, curr: int, total: int) -> None:
        """Update the status bar with the fitting progress."""
        message = f"Fitting z coordinate {curr:,} / {total:,} ..."
        self.status_bar.showMessage(message)

    def on_fit_z_finished(
        self,
        locs: pd.DataFrame,
        elapsed_time: float,
    ) -> None:
        """Handle the completion of the z fitting process."""
        self._active_worker = None
        self.abort_action.setEnabled(False)
        self.status_bar.showMessage(
            f"Fitted {len(locs):,} z coordinates in {elapsed_time:.2f} "
            "seconds."
        )
        self.locs = locs
        self.locs_display = locs
        self.save_locs_after_fit()
        # sound notification
        if elapsed_time > lib.SOUND_NOTIFICATION_DURATION:
            sound_path = lib.get_sound_notification_path()
            if sound_path is not None:
                playsound(sound_path, block=False)

    def save_locs_after_fit(self) -> None:
        """Save localizations after fitting to an .hdf5 file."""
        base = self.channel_output_base()
        self.save_locs(base + "_locs.hdf5")

        if not self.parameters_dialog.quality_check.isEnabled():
            self.parameters_dialog.quality_check.setEnabled(True)

        # restore the GPU checkbox state for the selected model
        self.parameters_dialog.on_fit_optimizer_changed()
        self.status_bar.showMessage(f"Saved {len(self.locs):,} localizations.")

        # apply drift if requested
        aim_check = self.parameters_dialog.aim_undrift_checkbox.isChecked()
        aim_segmentation = self.parameters_dialog.aim_segmentation.value()
        fiducial_check = self.parameters_dialog.fiducial_check.isChecked()
        if aim_check or fiducial_check:
            self.drift_correction(aim_check, aim_segmentation, fiducial_check)

    def drift_correction(
        self, aim_check: bool, aim_segmentation: int, fiducial_check: bool
    ) -> None:
        """Apply drift correction to the fitted localizations and save
        the drift-corrected localizations and the .txt drift files.

        ``self.locs_display`` is left untouched so the on-screen
        FitMarker overlays stay at their pre-drift positions."""
        drift = None

        if aim_check:
            self.status_bar.showMessage("Applying AIM drift correction...")
            undrift_locs, new_info, drift = aim.aim(
                self.locs,
                self.extra_info,
                aim_segmentation,
                progress=None,
            )
            self.locs = undrift_locs
            self.extra_info = new_info
            self.status_bar.showMessage("AIM finished.")

        if fiducial_check:
            self.status_bar.showMessage(
                "Finding fiducials for drift correction..."
            )
            fiducial_picks, box = imageprocess.find_fiducials(
                self.locs, self.extra_info
            )
            if len(fiducial_picks) == 0:
                if drift is not None:
                    # save the AIM drift-corrected localizations
                    base = self.channel_output_base()
                    self.save_locs(base + "_locs_undrifted.hdf5")
                    # save txt drift file
                    np.savetxt(base + "_locs_drift.txt", drift, newline="\r\n")
                    self.status_bar.showMessage(
                        "Saved AIM drift-corrected localizations. No "
                        "fiducials found, only AIM drift correction "
                        "applied."
                    )
                else:
                    self.status_bar.showMessage(
                        "No fiducials found. Skipping fiducial-based "
                        "drift correction."
                    )
                return
            self.status_bar.showMessage(
                f"Found {len(fiducial_picks)} fiducials. Applying drift "
                "correction..."
            )
            self.locs, self.extra_info, fiducials_drift = (
                postprocess.undrift_from_fiducials(
                    locs=self.locs,
                    info=self.extra_info,
                    picks=fiducial_picks,
                    pick_size=box,
                )
            )
            if drift is not None:
                drift["x"] += fiducials_drift["x"]
                drift["y"] += fiducials_drift["y"]
                if "z" in fiducials_drift.columns:
                    drift["z"] += fiducials_drift["z"]
            else:
                drift = fiducials_drift
            self.status_bar.showMessage(
                "Fiducial-based drift correction finished."
            )
        # save the drift-corrected localizations
        base = self.channel_output_base()
        self.save_locs(base + "_locs_undrifted.hdf5")
        # save txt drift file
        np.savetxt(base + "_locs_drift.txt", drift, newline="\r\n")
        self.status_bar.showMessage(
            "Saved drift-corrected localizations and drift file."
        )

    def fit_in_view(self) -> None:
        """Reset the zoom in the scene."""
        rectangle = QtCore.QRectF(
            0,
            0,
            self.movie.shape[2],
            self.movie.shape[1],
        )
        self.view.fitInView(
            rectangle, QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        self.draw_frame()

    def zoom_in(self) -> None:
        """Zoom in the view."""
        self.zoom(10 / 7)

    def zoom_out(self) -> None:
        """Zoom out the view."""
        self.zoom(7 / 10)

    def zoom(self, factor: float, anchor: QtCore.QPoint | None = None) -> None:
        """Zoom in or out the view by a specific factor. Anchor can
        specify the cursor position, otherwise zooms to/from the
        viewport's center."""
        if not hasattr(self, "movie") or self.movie is None:
            return
        # do not allow zooming out too much
        if factor < 1:
            rect = self.view.viewport().rect()
            visible_scene_rect = self.view.mapToScene(rect).boundingRect()
            if visible_scene_rect.width() / factor > self.movie.shape[2]:
                self.fit_in_view()
                return
        # fall back to the viewport center if no anchor is given
        # (e.g. zoom in/out from the menu or keyboard shortcuts)
        if anchor is None:
            anchor = self.view.viewport().rect().center()
        # adjust the transform directly so the scene point under the
        # anchor stays fixed (NoAnchor avoids scrollbar rounding drift)
        self.view.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor
        )
        old_scene_pos = self.view.mapToScene(anchor)
        self.view.scale(factor, factor)
        new_scene_pos = self.view.mapToScene(anchor)
        delta = new_scene_pos - old_scene_pos
        self.view.translate(delta.x(), delta.y())

        self.draw_frame()

    def save_spots(self, path: str) -> None:
        """Save identified spots as an .hdf5 file."""
        if self.identifications is None:
            message = (
                "No identifications to save. Please run identification first."
            )
            QtWidgets.QMessageBox.warning(self, "No identifications", message)
            return
        box = self.parameters["Box Size"]
        spots = localize.get_spots(
            self.movie, self.identifications, box, self.camera_info
        )
        info = self.info + [self.last_identification_info | self.camera_info]
        info_path = os.path.splitext(path)[0] + ".yaml"
        if path.endswith(".npy"):
            np.save(path, spots)
            io.save_info(info_path, info)
        elif path.endswith(".tif"):
            imageio.mimwrite(path, spots.astype("float32"))
            io.save_info(info_path, info)

    def save_spots_dialog(self) -> None:
        """Get the path for saving identified spots."""
        if self.movie_path != []:
            base = self.channel_output_base()
            path = base + "_spots.tif"
            path, exe = lib.get_save_filename_ext_dialog(
                self,
                "Save spots",
                path,
                filter="*.tif;;*.npy",
                check_ext=".yaml",
            )
            if path:
                self.save_spots(path)

    def export_current(self) -> None:
        """Export current view as .png or .tif."""
        try:
            base = self.channel_output_base()
        except AttributeError:
            return
        out_path = base + "_view.png"
        path, ext = lib.get_save_filename_ext_dialog(
            self, "Save image", out_path, filter="*.png;;*.tif"
        )
        if path:
            visible_scene_rect = self.view.mapToScene(
                self.view.viewport().rect()
            ).boundingRect()
            scene_rect = visible_scene_rect.intersected(
                self.scene.itemsBoundingRect()
            )
            scale = self.view.transform().m11()
            size = QtCore.QSize(
                max(1, int(round(scene_rect.width() * scale))),
                max(1, int(round(scene_rect.height() * scale))),
            )
            qimage = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
            qimage.fill(QtGui.QColor("transparent"))
            painter = QtGui.QPainter(qimage)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self.scene.render(
                painter,
                QtCore.QRectF(qimage.rect()),
                scene_rect,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            )
            painter.end()
            qimage.save(path)
        self.view.setMinimumSize(1, 1)

    def save_locs(self, path: str) -> None:
        """Save localizations and their metadata."""
        localize_info = self.last_identification_info.copy()
        localize_info["Generated by"] = f"Picasso v{__version__} Localize"
        model = self.parameters_dialog.fit_model.currentText()
        if FIT_MODELS[model]["optimizers"] is None:
            localize_info["Fit method"] = model
        else:
            optimizer = self.parameters_dialog.fit_optimizer.currentText()
            localize_info["Fit method"] = f"{model}, {optimizer}"
        if self.parameters_dialog.fit_z_checkbox.isChecked():
            localize_info["Z Calibration Path"] = (
                self.parameters_dialog.z_calibration_path
            )
            localize_info["Z Calibration"] = (
                self.parameters_dialog.z_calibration
            )
        if FIT_MODELS.get(model, {}).get("needs_spline_calibration"):
            # Record the path and model only; the coefficient table is far
            # too large to embed in the metadata.
            localize_info["Spline Calibration Path"] = (
                self.parameters_dialog.spline_calibration_path
            )
            localize_info["Spline Calibration Model"] = (
                self.parameters_dialog.spline_calibration.get("model")
            )
        self.extra_info = self.info + [localize_info | self.camera_info]
        self.select_locs_columns()  # save only selected columns
        io.save_locs(path, self.locs, self.extra_info)

    def save_locs_dialog(self) -> None:
        """Get the path to save localizations."""
        if self.movie_path != []:
            base = self.channel_output_base()
            locs_path = base + "_locs.hdf5"
            path, exe = lib.get_save_filename_ext_dialog(
                self,
                "Save localizations",
                locs_path,
                filter="*.hdf5",
                check_ext=".yaml",
            )
            if path:
                self.save_locs(path)

    def save_identifications(self, path: str) -> None:
        """Save identifications and their metadata to an HDF5 file."""
        ids_info = {
            "Generated by": f"Picasso v{__version__} Localize",
            "Box Size": self.parameters_dialog.box_spinbox.value(),
            "Min. Net Gradient": (self.parameters_dialog.mng_slider.value()),
        }
        info = self.info + [ids_info]
        io.save_identifications(path, self.identifications, info)

    def save_identifications_dialog(self) -> None:
        """Get the path to save identifications."""
        if self.identifications is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No identifications",
                "No identifications to save. Run identification first.",
            )
            return
        if self.movie_path != []:
            base = self.channel_output_base()
            ids_path = base + "_identifications.hdf5"
            path, exe = lib.get_save_filename_ext_dialog(
                self,
                "Save identifications",
                ids_path,
                filter="*.hdf5",
                check_ext=".yaml",
            )
            if path:
                self.save_identifications(path)

    def localize(self, calibrate_z: bool = False) -> None:
        """Identify and fit, see ``identify`` and ``fit``.

        Parameters
        ----------
        calibrate_z : bool, optional
            Whether to run z-calibration for 3D fitting afterwards
            Default is False.
        """
        self.parameters_dialog.gpu_checkbox.setDisabled(True)
        if (
            calibrate_z
            and self.identifications is not None
            and self.ready_for_fit
            and not self.identifications_outdated()
        ):
            # reuse existing identifications (e.g., loaded picks/locs)
            self.fit(calibrate_z=calibrate_z)
        else:
            self.identify(fit_afterwards=True, calibrate_z=calibrate_z)

    def identifications_outdated(self) -> bool:
        """Check whether the identification settings (box size, min. net
        gradient, ROI, frame range) have changed since
        ``self.identifications`` was created."""
        last = self.last_identification_info
        if last is None:
            return True
        if any(
            last.get(key) != value for key, value in self.parameters.items()
        ):
            return True
        if last.get("ROI") != self.view.rois:
            return True
        if last.get("Frame bounds") != self.frame_range:
            return True
        return False

    def select_locs_columns(self) -> None:
        """Select only the columns that are checked in the corresponding
        dialog."""
        to_keep = []
        for column, checkbox in self.columns_dialog.column_checkboxes.items():
            if checkbox.isChecked() and column in self.locs.columns:
                to_keep.append(column)
        self.locs = self.locs[to_keep]


class IdentificationWorker(QtCore.QThread):
    """Identify spots in the movie using multiprocessing.

    Loads the user parameters and updates the status bar about the
    progress."""

    progressMade = QtCore.pyqtSignal(int, dict)
    finished = QtCore.pyqtSignal(
        dict, object, float, pd.DataFrame, bool, bool, bool
    )
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        fit_afterwards: bool,
        calibrate_z: bool,
        calibrate_spline: bool = False,
    ) -> None:
        super().__init__()
        self.window = window
        self.movie = window.movie
        self.rois = window.view.rois
        self.frame_range = window.frame_range
        self.parameters = window.parameters
        self.fit_afterwards = fit_afterwards
        self.calibrate_z = calibrate_z
        self.calibrate_spline = calibrate_spline

    def on_progress(self, frame_number: int) -> None:
        self.progressMade.emit(frame_number, self.parameters)

    def run(self) -> None:
        t0 = time.time()
        # we ignore info since we will merge the metadata from identification
        # as well when saving localizations
        result = localize.identify(
            movie=self.movie,
            minimum_ng=self.parameters["Min. Net Gradient"],
            box=self.parameters["Box Size"],
            roi=self.rois,
            frame_bounds=self.frame_range,
            threaded=True,
            progress_callback=self.on_progress,
            abort_callback=self.isInterruptionRequested,
        )
        if result is None:  # handle aborted process
            self.aborted.emit()
            return
        identifications, info = result
        elapsed_time = time.time() - t0
        self.finished.emit(
            self.parameters,
            self.rois,
            elapsed_time,
            identifications,
            self.fit_afterwards,
            self.calibrate_z,
            self.calibrate_spline,
        )


class FitWorker(QtCore.QThread):
    """Fit single molecules to the identified spots using
    multiprocessing and update the status bar accordingly."""

    progressMade = QtCore.pyqtSignal(int, int)
    cutProgressMade = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(pd.DataFrame, float, bool, bool)
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        movie: np.memmap,
        movie_info: list[dict],
        camera_info: dict,
        identifications: pd.DataFrame,
        box: int,
        method: Literal[
            "gausslq",
            "gausslq-spherical",
            "gausslq-rotated",
            "gausslq-rotated-gpu",
            "gausslq-spherical-gpu",
            "gaussmle",
            "gaussmle-spherical",
            "gaussmle-rotated-gpu",
            "gaussmle-spherical-gpu",
            "spline",
            "spline-mle",
            "spline-gpu",
            "spline-mle-gpu",
            "avg",
        ],
        eps: float | None,
        max_it: int | None,
        fit_z: bool,
        calibrate_z: bool,
        use_gpu: bool,
        spline_calibration: dict | None = None,
    ) -> None:
        super().__init__()
        self.movie = movie
        self.movie_info = movie_info
        self.camera_info = camera_info
        self.identifications = identifications
        self.box = box
        self.eps = eps
        self.max_it = max_it
        self.fit_z = fit_z
        self.calibrate_z = calibrate_z
        self.spline_calibration = spline_calibration
        self.N = len(identifications)
        self._last_cut_emit = 0
        self.method = _effective_fit_code(method, use_gpu)

    def on_progress(self, n_done: int) -> None:
        self.progressMade.emit(n_done, self.N)

    def on_cut_progress(self, n_done: int) -> None:
        # The underlying cut loop may call back very frequently (e.g. once
        # per frame), so throttle GUI updates to ~1% increments to avoid
        # flooding the main thread's event queue. Always emit the last one.
        step = max(1, self.N // 1000)
        if n_done - self._last_cut_emit >= step or n_done >= self.N:
            self._last_cut_emit = n_done
            self.cutProgressMade.emit(n_done, self.N)

    def run(self) -> None:
        t0 = time.time()
        # we ignore info since we will merge the metadata from identification
        # as well when saving localizations
        locs, info = localize.fit2D(
            movie=self.movie,
            movie_info=self.movie_info,
            camera_info=self.camera_info,
            identifications=self.identifications,
            box=self.box,
            fitting_method=self.method,
            eps=self.eps,
            max_it=self.max_it,
            mle_method="sigmaxy",
            spline_calibration=self.spline_calibration,
            multiprocess=True,
            progress_callback=self.on_progress,
            abort_callback=self.isInterruptionRequested,
            cut_progress_callback=self.on_cut_progress,
        )
        if locs is None:  # handle aborted process
            self.aborted.emit()
            return
        self.progressMade.emit(self.N + 1, self.N)
        dt = time.time() - t0
        self.finished.emit(locs, dt, self.fit_z, self.calibrate_z)


class MultichannelSplineFitWorker(QtCore.QThread):
    """Fit a multichannel cubic-spline PSF across several registered channels
    simultaneously. The reference channel's identifications are mapped into
    every channel via the calibration's stored affine transforms."""

    progressMade = QtCore.pyqtSignal(int, int)
    linkMade = QtCore.pyqtSignal(int, int)
    linkFinished = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(pd.DataFrame, float, bool, bool)
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        movies: list,
        camera_infos: list,
        identifications: pd.DataFrame,
        box: int,
        calibration: dict,
        mle: bool = False,
        split_fov: bool = False,
        regions: list | None = None,
        link_photons: bool = True,
        identifications_per_channel: list | None = None,
        use_gpu: bool | None = None,
        eps: float | None = None,
        max_it: int | None = None,
    ) -> None:
        super().__init__()
        self.movies = movies
        self.camera_infos = camera_infos
        self.identifications = identifications
        self.identifications_per_channel = identifications_per_channel
        self.box = box
        self.calibration = calibration
        self.mle = mle
        self.use_gpu = use_gpu
        self.eps = eps
        self.max_it = max_it
        # Link photons across channels (shared amplitude, model 11). When False
        # and the calibration has 2 to 6 channels, fit the photon-decoupled
        # link-XYZ model: per-channel
        # free photons/background
        self.link_photons = link_photons
        # Split-FOV: ``movies``/``camera_infos`` hold a single entry (one loaded
        # movie); the channels are regions of that movie, handled by
        # ``fit_spline_split_fov`` (which confines to the reference region).
        # ``regions`` (optional) places the channels at the current ROIs.
        self.split_fov = split_fov
        self.regions = regions
        self.N = len(identifications)

    def on_progress(self, n_done: int) -> None:
        self.progressMade.emit(n_done, self.N)

    def on_link_progress(self, n_done: int) -> None:
        self.linkMade.emit(n_done, self.N)

    def _link_across_channels(self, n_channels: int) -> bool:
        """Restrict the reference detections to those found in every channel.
        Returns False if nothing survives (the caller then aborts).

        ``self.N`` - the denominator of the fit progress - is set to the number
        of spots that are actually fitted, i.e. the linked molecules (one fit
        per molecule across all channels), not the raw detection count."""
        if self.split_fov:
            return self._link_across_regions()
        ids_per_channel = self.identifications_per_channel
        if not ids_per_channel or len(ids_per_channel) < 2:
            return True
        ids_per_channel = list(ids_per_channel[:n_channels])
        transforms = self.calibration.get("channel_transforms")
        if not transforms or len(transforms) < len(ids_per_channel):
            return True
        linked, n_kept, n_total = localize.filter_linked_identifications(
            ids_per_channel,
            transforms,
            self.box,
            progress_callback=self.on_link_progress,
        )
        self.linkFinished.emit(n_kept, n_total)
        if n_kept == 0:
            return False
        self.identifications = linked
        self.N = len(linked)
        return True

    def _link_across_regions(self) -> bool:
        """Split-FOV counterpart of :meth:`_link_across_channels`: the single
        identification table holds every region's detections, so it is split by
        region and linked across them. Keeps the reference-region detections
        found in all other regions - one fitted spot per molecule."""
        all_regions = self.identifications
        try:
            fit_regions, reference, _ = localize.split_fov_fit_geometry(
                self.calibration, self.regions
            )
        except (ValueError, KeyError):
            # geometry unavailable (e.g. no channel_affines to re-place at the
            # drawn ROIs): fit as before, on the whole identification table
            return True
        # only the reference region is fitted (the other regions are cut from
        # it via the transforms), so it - not the movie-wide detection count -
        # is the denominator of the linking and fit progress
        self.identifications = localize.confine_to_region(
            all_regions, fit_regions[reference]
        )
        self.N = len(self.identifications)
        linked, n_kept, n_total = (
            localize.filter_linked_identifications_split_fov(
                all_regions,
                self.calibration,
                self.box,
                regions=self.regions,
                progress_callback=self.on_link_progress,
            )
        )
        self.linkFinished.emit(n_kept, n_total)
        if n_kept == 0:
            return False
        self.identifications = linked
        self.N = len(linked)
        return True

    def run(self) -> None:
        t0 = time.time()
        try:
            n_channels = int(
                self.calibration.get("n_channels", len(self.movies))
            )
            if not self._link_across_channels(n_channels):
                where = "regions" if self.split_fov else "channels"
                print(
                    f"Multichannel spline fit: no detection is linked across "
                    f"all {where} - check the channel registration."
                )
                self.aborted.emit()
                return
            if self.split_fov:
                locs = localize.fit_spline_split_fov(
                    self.movies[0],
                    self.camera_infos[0],
                    self.identifications,
                    self.box,
                    self.calibration,
                    regions=self.regions,
                    mle=self.mle,
                    use_gpu=self.use_gpu,
                    link_photons=self.link_photons,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                )
            elif not self.link_photons and (
                2 <= n_channels <= localize._LINK_XYZ_MAX_CHANNELS
            ):
                # Photon decoupling (globLoc link-XYZ): free per-channel photons
                # and background, shared x/y/z. Supersedes the ratiometric scan.
                locs = localize.fit_spline_multichannel(
                    self.movies,
                    self.camera_infos,
                    self.identifications,
                    self.box,
                    self.calibration,
                    mle=self.mle,
                    use_gpu=self.use_gpu,
                    link_photons=False,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                )
            elif self.calibration.get("photon_ratios") is not None:
                # Ratiometric color assignment: the calibration carries
                # candidate per-channel photon ratios (one per dye/color). Each
                # localization is assigned the max-likelihood ratio as `color`.
                locs = localize.fit_spline_multichannel_ratiometric(
                    self.movies,
                    self.camera_infos,
                    self.identifications,
                    self.box,
                    self.calibration,
                    mle=self.mle,
                    use_gpu=self.use_gpu,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                )
            else:
                locs = localize.fit_spline_multichannel(
                    self.movies,
                    self.camera_infos,
                    self.identifications,
                    self.box,
                    self.calibration,
                    mle=self.mle,
                    use_gpu=self.use_gpu,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                )
        except Exception as e:
            print(f"Multichannel spline fit failed: {e}")
            self.aborted.emit()
            return
        self.progressMade.emit(self.N + 1, self.N)
        dt = time.time() - t0
        # fit_z / calibrate_z are always False for the multichannel spline
        # path (z comes from the fit; there is no astigmatism step)
        self.finished.emit(locs, dt, False, False)


class FitZWorker(QtCore.QThread):
    """Fit the z coordinates to fitted localizations based on the
    calibration file using multiprocessing."""

    progressMade = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(pd.DataFrame, float)
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        locs: pd.DataFrame,
        info: list[dict],
        calibration: dict,
        magnification_factor: float,
        pixelsize: float,
        fitting_method: Literal["gausslq", "gaussmle"],
        gpu: bool = False,
    ) -> None:
        super().__init__()
        self.locs = locs
        self.info = info
        self.calibration = calibration
        self.magnification_factor = magnification_factor
        self.pixelsize = pixelsize
        self.fitting_method = fitting_method
        self.gpu = gpu

    def on_progress(self, n_done: int) -> None:
        self.progressMade.emit(n_done, len(self.locs))

    def run(self) -> None:
        t0 = time.time()
        # we ignore info since we will merge the metadata from 2D
        # localization as well when saving localizations
        locs, info = zfit.zfit(
            locs=self.locs,
            info=self.info,
            calibration=self.calibration,
            magnification_factor=self.magnification_factor,
            pixelsize=self.pixelsize,
            fitting_method=self.fitting_method,
            multiprocess=not self.gpu,
            gpu=self.gpu,
            progress_callback=self.on_progress,
            abort_callback=self.isInterruptionRequested,
        )
        dt = time.time() - t0
        self.finished.emit(locs, dt)


class SplineCalibrationWorker(QtCore.QThread):
    """Build a cubic-spline PSF calibration from a bead z-stack movie in a
    background thread (bead detection + averaging on the CPU, coefficients via
    Gpuspline)."""

    finished = QtCore.pyqtSignal(str, int)  # (path, n_beads)
    failed = QtCore.pyqtSignal(str)
    statusChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        movie,
        info: list[dict],
        camera_info: dict,
        box: int,
        minimum_ng: float,
        step: float,
        frames_per_step: int,
        frame_order: str,
        model: str,
        path: str,
        frame_bounds=None,
        magnification_factor: float = 0.79,
        correct_z_bias: bool = False,
        link_photons: bool = True,
        roi=None,
        regions=None,
        movies=None,
        infos=None,
        camera_infos=None,
    ) -> None:
        super().__init__()
        self.movie = movie
        self.info = info
        self.camera_info = camera_info
        self.box = box
        self.minimum_ng = minimum_ng
        self.step = step
        self.frames_per_step = frames_per_step
        self.frame_order = frame_order
        self.frame_bounds = frame_bounds
        self.model = model
        self.magnification_factor = magnification_factor
        self.correct_z_bias = correct_z_bias
        self.link_photons = link_photons
        self.roi = roi
        # When set (>= 2 rectangles), the ROIs are treated as channels of one
        # movie (split-FOV) and a multichannel calibration is built instead.
        self.regions = regions
        # When set (>= 2 movies), the channels are separate movies (separate
        # files or a multichannel file) registered from bead correspondences;
        # movies[0]/infos[0] is the reference channel.
        self.movies = movies
        self.infos = infos
        self.camera_infos = camera_infos
        self.path = path

    def run(self) -> None:
        try:
            if self.movies:
                calibration = spline.calibrate_spline_multichannel(
                    self.movies,
                    infos=self.infos,
                    camera_infos=self.camera_infos,
                    box=self.box,
                    minimum_ng=self.minimum_ng,
                    d=self.step,
                    frames_per_step=self.frames_per_step,
                    frame_bounds=self.frame_bounds,
                    frame_order=self.frame_order,
                    magnification_factor=self.magnification_factor,
                    correct_z_bias=self.correct_z_bias,
                    link_photons=self.link_photons,
                    reference=0,
                    path=self.path,
                )
            elif self.regions:
                calibration = spline.calibrate_spline_split_fov(
                    self.movie,
                    info=self.info,
                    camera_info=self.camera_info,
                    box=self.box,
                    minimum_ng=self.minimum_ng,
                    d=self.step,
                    regions=self.regions,
                    frames_per_step=self.frames_per_step,
                    frame_bounds=self.frame_bounds,
                    frame_order=self.frame_order,
                    magnification_factor=self.magnification_factor,
                    correct_z_bias=self.correct_z_bias,
                    link_photons=self.link_photons,
                    path=self.path,
                )
            else:
                calibration = spline.calibrate_spline(
                    self.movie,
                    info=self.info,
                    camera_info=self.camera_info,
                    box=self.box,
                    minimum_ng=self.minimum_ng,
                    d=self.step,
                    frames_per_step=self.frames_per_step,
                    frame_bounds=self.frame_bounds,
                    frame_order=self.frame_order,
                    model=self.model,
                    magnification_factor=self.magnification_factor,
                    correct_z_bias=self.correct_z_bias,
                    roi=self.roi,
                    path=self.path,
                )
        except Exception as e:  # surface any failure to the GUI
            self.failed.emit(str(e))
            return
        self.finished.emit(self.path, int(calibration.get("n_beads", 0)))


class QualityWorker(QtCore.QThread):
    """Run quality checks on the localized data, i.e., calculate the
    number of localizations, experimental localization precision (NeNA),
    drift and mean bright time."""

    progressMade = QtCore.pyqtSignal(str, int, str)
    finished = QtCore.pyqtSignal(str)

    def __init__(
        self,
        locs: pd.DataFrame,
        info: list[dict],
        path: str,
        pixelsize: QtWidgets.QDoubleSpinBox,
    ) -> None:
        super().__init__()
        self.locs = locs
        self.info = info
        self.path = path
        self.pixelsize = pixelsize

    def run(self) -> None:
        # Sanity of locs
        sane_locs = lib.ensure_sanity(self.locs, self.info)

        # Locs
        self.progressMade.emit("Checking Quality (1/4) Locs ..", 0, "")
        locs_per_frame = len(sane_locs) / self.info[0]["Frames"]
        self.progressMade.emit("", 0, f"{locs_per_frame:.1f}")

        # NeNA
        self.progressMade.emit("Checking Quality (2/4) NeNA ..", 0, "")

        def nena_callback(x):
            self.progressMade.emit(
                f"Checking Quality (2/4) NeNA: {x} %",
                0,
                "",
            )

        nena_px = localize.check_nena(sane_locs, self.info, nena_callback)
        nena_nm = float(self.pixelsize.value() * nena_px)
        self.progressMade.emit("", 1, f"{nena_px:.2f} px / {nena_nm:.2f} nm")

        # Drift
        self.progressMade.emit("Checking Quality (3/4) Drift ..", 0, "")

        def drift_callback(x):
            self.progressMade.emit(
                f"Checking Quality (3/4) Drift {x} %",
                0,
                "",
            )

        drift_x, drift_y = localize.check_drift(
            sane_locs, self.info, callback=drift_callback
        )
        self.progressMade.emit(
            "",
            2,
            f"X: {drift_x:.3f} px / Y: {drift_y:.3f} px",
        )

        # Kinetics
        self.progressMade.emit("Checking Quality (4/4) Kinetics ..", 0, "")
        len_mean = localize.check_kinetics(sane_locs, self.info)
        self.progressMade.emit("", 3, f"{len_mean:.3f}")

        localize.add_file_to_db(
            self.path,
            None,
            drift=(drift_x, drift_y),
            len_mean=len_mean,
            nena=nena_px,
        )
        self.finished.emit("Quality parameters complete.")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Window()

    # load plugins from ~/.picasso/plugins
    from .plugins_loader import load_plugins, add_plugins_menu_actions

    load_plugins(window, "localize")
    add_plugins_menu_actions(window, "localize")

    window.show()

    from ..updater import setup_gui_update_check

    setup_gui_update_check(window)

    lib.install_excepthook(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
