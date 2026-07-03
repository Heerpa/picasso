"""
picasso.gui.localize
~~~~~~~~~~~~~~~~~~~~

Graphical user interface for localizing single molecules.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

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
    imageprocess,
    io,
    localize,
    lib,
    postprocess,
    __version__,
    zfit,
)
from PyQt6 import QtCore, QtGui, QtWidgets
from playsound3 import playsound

try:
    from picasso.ext.pygpufit import gpufit

    GPUFIT_INSTALLED = bool(gpufit.cuda_available())
except Exception:
    GPUFIT_INSTALLED = False
CMAP_GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]
DEFAULT_PARAMETERS = {"Box Size": 7, "Min. Net Gradient": 5000}

# Fitting models offered in the GUI, decoupled from the optimizer. Each
# model maps its optimizer labels to the internal ``fit2D`` codes;
# models without an optimizer (e.g. averaging) declare a fixed ``code``
# and ``optimizers=None``. Add a new fitting algorithm by adding an
# entry here.
FIT_MODELS = {
    "2D elliptical Gaussian": {
        "optimizers": {"Least squares": "gausslq", "MLE": "gaussmle"},
    },
    "Average of ROI": {
        "optimizers": None,
        "code": "avg",
    },
}


def _fit_code(model: str, optimizer: str) -> str:
    """Resolve a (model, optimizer) selection to an internal ``fit2D`` code."""
    entry = FIT_MODELS[model]
    if entry["optimizers"] is None:
        return entry["code"]
    return entry["optimizers"][optimizer]


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
    Parameters/Contrast dialog values) into the old ``Channel`` and
    restores the new one.

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
    frame_range: object = None
    curr_frame_number: int = 0
    params: dict = field(default_factory=dict)
    contrast: dict | None = None


def _sanitize_filename(name: str) -> str:
    """Turn a channel name into a filename-safe suffix."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "channel"


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
                # Emitted (queued to the GUI thread) as io scans the
                # file's IFDs, so the bar advances smoothly within a file.
                report = self.subprogress.emit
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
            self.failed.emit(str(e))
            return
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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Start either a rubber band for selecting a ROI or panning the
        view."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
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
            self.roi_end = QtCore.QPoint(event.pos())
            self.rubberband.hide()
            dx = abs(self.roi_end.x() - self.roi_origin.x())
            dy = abs(self.roi_end.y() - self.roi_origin.y())
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

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Remove the ROI under the cursor on a (left) double click. If
        several ROIs contain the point, the smallest one is removed."""
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self.rois:
            event.ignore()
            return
        scene_pos = self.mapToScene(event.pos())
        px, py = scene_pos.x(), scene_pos.y()
        containing = [
            i
            for i, ((y_min, x_min), (y_max, x_max)) in enumerate(self.rois)
            if y_min <= py <= y_max and x_min <= px <= x_max
        ]
        if not containing:
            event.ignore()
            return
        idx = min(
            containing,
            key=lambda i: (
                (self.rois[i][1][0] - self.rois[i][0][0])
                * (self.rois[i][1][1] - self.rois[i][0][1])
            ),
        )
        del self.rois[idx]
        self.selected_roi = None
        self.window.parameters_dialog.update_roi_display()
        self.window.draw_frame()
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Zoom in/out with the mouse wheel, centered on the cursor."""
        scale = 1.008 ** (-event.angleDelta().y())
        self.window.zoom(scale, anchor=event.position().toPoint())

    def on_scroll(self) -> None:
        """Redraw the frame if scale bar is shown."""
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
        self.frames_per_step.setRange(1, 100)
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
    fit_z_gpu_checkbox : QtWidgets.QCheckBox
        Checkbox for fitting z coordinates on a CUDA-capable GPU
        (numba.cuda). Only shown if a compatible GPU is available.
    gain : QtWidgets.QSpinBox
        Spin box for selecting camera EM gain.
    gpufit_checkbox : QtWidgets.QCheckBox
        Checkbox for enabling/disabling GPU fitting. Only shown if a GPU
        is available and ``pygpufit`` is installed.
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

    def __init__(  # noqa: C901
        self, parent: QtWidgets.QMainWindow | None = None
    ) -> None:
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Parameters")
        self.setModal(False)

        self.z_calibration = {}
        self.z_calibration_path = None

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

        # preview identifications
        self.preview_checkbox = QtWidgets.QCheckBox("Preview")
        self.preview_checkbox.setToolTip(
            "Show identified spots in the current frame?"
        )
        self.preview_checkbox.setTristate(False)
        self.preview_checkbox.stateChanged.connect(self.on_preview_changed)
        identification_grid.addWidget(self.preview_checkbox, 4, 0)

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
        model_label.setToolTip("Model fit to each identified spot.")
        fit_grid.addWidget(model_label, 1, 0)
        self.fit_model = QtWidgets.QComboBox()
        self.fit_model.addItems(list(FIT_MODELS.keys()))
        self.fit_model.setCurrentIndex(0)
        fit_grid.addWidget(self.fit_model, 1, 1)

        self.optimizer_label = QtWidgets.QLabel("Optimizer:")
        self.optimizer_label.setToolTip("Optimizer used to fit the model.")
        fit_grid.addWidget(self.optimizer_label, 2, 0)
        self.fit_optimizer = QtWidgets.QComboBox()
        fit_grid.addWidget(self.fit_optimizer, 2, 1)

        self.fit_stack = QtWidgets.QStackedWidget()
        fit_grid.addWidget(self.fit_stack, 3, 0, 1, 2)
        fit_stack = self.fit_stack

        # MLE
        mle_widget = QtWidgets.QWidget()

        mle_grid = QtWidgets.QGridLayout(mle_widget)
        cc_label = QtWidgets.QLabel("Convergence criterion:")
        cc_label.setToolTip("Tolerance for testing if fitting has converged.")
        mle_grid.addWidget(cc_label, 0, 0)
        self.convergence_criterion = QtWidgets.QDoubleSpinBox()
        self.convergence_criterion.setRange(0, 1e6)
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

        # LQ
        lq_widget = QtWidgets.QWidget()
        lq_grid = QtWidgets.QGridLayout(lq_widget)

        self.gpufit_checkbox = QtWidgets.QCheckBox("Use GPUfit")
        self.gpufit_checkbox.setTristate(False)
        self.gpufit_checkbox.setDisabled(True)
        self.gpufit_checkbox.stateChanged.connect(self.on_gpufit_changed)

        if not GPUFIT_INSTALLED:
            self.gpufit_checkbox.hide()
        else:
            self.gpufit_checkbox.setDisabled(False)
        lq_grid.addWidget(self.gpufit_checkbox)

        # Stack pages are ordered to match the optimizer combobox indices:
        # 0 -> "Least squares" (GPU option), 1 -> "MLE" (convergence/max_it).
        fit_stack.addWidget(lq_widget)
        fit_stack.addWidget(mle_widget)

        self.fit_model.currentIndexChanged.connect(self.on_fit_model_changed)
        self.fit_optimizer.currentIndexChanged.connect(
            self.on_fit_optimizer_changed
        )
        # Populate the optimizer combobox and set visibility for the default
        # model.
        self.on_fit_model_changed()

        # 3D
        z_groupbox = QtWidgets.QGroupBox("3D via Astigmatism")
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
        z_grid.addWidget(lib.HelpButton(self.CALIB_URL), 2, 0)
        self.fit_z_checkbox = QtWidgets.QCheckBox("Fit Z")
        self.fit_z_checkbox.setToolTip("Fit z coordinates?")
        self.fit_z_checkbox.setEnabled(False)
        z_grid.addWidget(self.fit_z_checkbox, 2, 1)
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

        self.fit_z_gpu_checkbox = QtWidgets.QCheckBox("Use GPU")
        self.fit_z_gpu_checkbox.setTristate(False)
        self.fit_z_gpu_checkbox.setToolTip("Fit z coordinates on a GPU?")
        self.fit_z_gpu_checkbox.setEnabled(self.fit_z_checkbox.isChecked())
        self.fit_z_checkbox.toggled.connect(self.fit_z_gpu_checkbox.setEnabled)
        if not zfit.CUDA_AVAILABLE:
            self.fit_z_gpu_checkbox.hide()
        z_grid.addWidget(self.fit_z_gpu_checkbox, 2, 2)

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

    def on_fit_optimizer_changed(self) -> None:
        """Switch the optimizer parameter page and enable/disable the GPU
        fitting checkbox based on the selected optimizer."""
        index = self.fit_optimizer.currentIndex()
        if index >= 0:
            self.fit_stack.setCurrentIndex(index)
        if self.fit_optimizer.currentText() == "Least squares":
            self.gpufit_checkbox.setDisabled(not GPUFIT_INSTALLED)
        else:
            self.gpufit_checkbox.setChecked(False)
            self.gpufit_checkbox.setDisabled(True)

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

    def on_gpufit_changed(self) -> None:
        """Handle changes to the GPU fitting option."""
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

    def __init__(self, parent: QtWidgets.QMainWindow) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select columns to save")
        self.setMinimumWidth(250)
        self.setModal(True)
        vbox = QtWidgets.QVBoxLayout(self)

        # add checkboxes
        self.column_checkboxes = {}
        for column in localize.LOCALIZATION_COLUMNS["Base"]:
            checkbox = QtWidgets.QCheckBox(column)
            checkbox.setChecked(True)
            self.column_checkboxes[column] = checkbox
            vbox.addWidget(checkbox)
            if column in lib.REQUIRED_COLUMNS:
                checkbox.setDisabled(True)
        for key, column in localize.LOCALIZATION_COLUMNS.items():
            if key == "Base":
                continue
            for col in column:
                checkbox = QtWidgets.QCheckBox(f"{col} ({key})")
                checkbox.setChecked(True)
                self.column_checkboxes[col] = checkbox
                vbox.addWidget(checkbox)
                if col in lib.REQUIRED_COLUMNS:
                    checkbox.setDisabled(True)

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
        self.parameters_dialog.fit_z_gpu_checkbox.setChecked(
            bool(settings["Localize"]["fit_z_gpu"])
        )

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
            "fit_z_gpu"
        ] = self.parameters_dialog.fit_z_gpu_checkbox.isChecked()
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

        """ 3D """
        threed_menu = menu_bar.addMenu("3D")

        calibrate_z_action = threed_menu.addAction("Calibrate 3D")
        calibrate_z_action.triggered.connect(self.calibrate_z)

        self.plugin_menu = menu_bar.addMenu("Plugins")  # do not delete

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
            # A load is already in progress; ignore re-entrant requests.
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
                channel.contrast = self._capture_contrast()
                self.channels.append(channel)
            self.current_channel = 0
        finally:
            self._switching_channel = False
        self._populate_channel_combo()
        self.frame_slider.setEnabled(True)
        self._restore_current_channel()
        self.parameters_dialog.reset_quality_check()

    def _populate_channel_combo(self) -> None:
        """Refresh the channel selector; hidden unless several channels."""
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems([c.name for c in self.channels])
        self.channel_combo.setCurrentIndex(self.current_channel)
        self.channel_combo.blockSignals(False)
        self.channel_combo.setVisible(len(self.channels) > 1)

    def on_channel_combo_changed(self, index: int) -> None:
        """Switch the active channel from the selector."""
        self.set_current_channel(index)

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
        channel.frame_range = self.frame_range
        channel.curr_frame_number = getattr(self, "curr_frame_number", 0)
        channel.params = self._capture_params()
        channel.contrast = self._capture_contrast()

    def _restore_current_channel(self) -> None:
        """Load the active Channel's state into the flat attrs + dialogs."""
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
            self.frame_range = channel.frame_range
            self._apply_params(channel.params)
            self._apply_contrast(channel.contrast)
        finally:
            self._switching_channel = False
        self._apply_channel_to_ui(channel.curr_frame_number)

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
            "fit_z_gpu": pd.fit_z_gpu_checkbox.isChecked(),
            "z_calibration": pd.z_calibration,
            "z_calibration_path": pd.z_calibration_path,
            "z_calib_label": pd.z_calib_label.text(),
            "gpufit": pd.gpufit_checkbox.isChecked(),
            "frames_edit": pd.frames_edit.text(),
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
        # Camera selection first: its cascade fills baseline/gain/pixelsize
        # from the config, which we then override with the channel's values.
        if hasattr(pd, "camera") and "camera" in params:
            pd.camera.setCurrentIndex(params["camera"])
        pd.box_spinbox.setValue(params["box"])
        pd.mng_min_spinbox.setValue(params["mng_min"])
        pd.mng_max_spinbox.setValue(params["mng_max"])
        pd.mng_slider.setValue(params["mng"])
        pd.mng_spinbox.setValue(params["mng"])
        # Set the model first so its handler repopulates the optimizer list,
        # then restore the optimizer selection.
        pd.fit_model.setCurrentIndex(params.get("fit_model", 0))
        pd.fit_optimizer.setCurrentIndex(params.get("fit_optimizer", 0))
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
        pd.fit_z_gpu_checkbox.setChecked(params.get("fit_z_gpu", False))
        pd.gpufit_checkbox.setChecked(params["gpufit"])
        pd.frames_edit.setText(params["frames_edit"])

    def _capture_contrast(self) -> dict:
        """Snapshot the contrast dialog state."""
        cd = self.contrast_dialog
        return {
            "black": cd.black_spinbox.value(),
            "white": cd.white_spinbox.value(),
            "auto": cd.auto_checkbox.isChecked(),
        }

    def _apply_contrast(self, contrast: dict | None) -> None:
        """Restore a channel's contrast settings into the dialog."""
        if not contrast:
            return
        cd = self.contrast_dialog
        cd.auto_checkbox.blockSignals(True)
        cd.auto_checkbox.setChecked(contrast["auto"])
        cd.auto_checkbox.blockSignals(False)
        cd.change_contrast_silently(contrast["black"], contrast["white"])

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
            self.view.setScene(self.scene)
            # draw the ROI rectangles (in scene/pixel coordinates)
            for i, ((y_min, x_min), (y_max, x_max)) in enumerate(
                self.view.rois
            ):
                color = (
                    QtGui.QColor("cyan")
                    if i == self.view.selected_roi
                    else QtGui.QColor("blue")
                )
                pen = QtGui.QPen(color)
                pen.setCosmetic(True)  # constant width regardless of zoom
                self.scene.addRect(
                    QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min),
                    pen,
                )
            if self.ready_for_fit:
                identifications_frame = self.identifications[
                    self.identifications.frame == self.curr_frame_number
                ]
                box = self.last_identification_info["Box Size"]
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

    def draw_scalebar(self) -> None:
        """Draw a scale bar if the option is checked."""
        if not self.scalebar_action.isChecked():
            return

        scene_pixelsize = self.parameters_dialog.pixelsize.value()

        # length (nm) - set optimal size (~1/8 of image width)
        rect = self.view.viewport().rect()
        visible_scene_rect = self.view.mapToScene(rect).boundingRect()
        width = visible_scene_rect.width()
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

        length_displaypxl = int(
            round(self.view.width() * (scalebar / scene_pixelsize) / width)
        )
        height_displaypxl = 10

        # draw a rectangle
        x = self.view.width() - length_displaypxl - 40
        y = self.view.height() - height_displaypxl - 20
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
        # position the text centered below the scale bar
        text_rect = text_item.boundingRect()
        text_width = text_rect.width() / (length_displaypxl / length_scene)
        text_x = x_scene + (length_scene - text_width) / 2
        text_y = (
            y_scene + height_scene - 45 / (height_displaypxl / height_scene)
        )
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
        self.parameters_dialog.gpufit_checkbox.setDisabled(False)
        self.status_bar.showMessage("Aborted.")

    def identify(
        self,
        fit_afterwards: bool = False,
        calibrate_z: bool = False,
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
        """
        if self.movie is not None:
            self.status_bar.showMessage("Preparing identification...")
            self.identification_worker = IdentificationWorker(
                self, fit_afterwards, calibrate_z
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

    def on_identify_finished(
        self,
        parameters: dict,
        roi: list,
        elapsed_time: float,
        identifications: pd.DataFrame,
        fit_afterwards: bool,
        calibrate_z: bool,
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
            message = (
                f"Identified {n_identifications:,} spots in {elapsed_time:.2f}"
                f" seconds. (Box Size: {box}; Min. Net Gradient: {mng}). "
                "Ready for fit."
            )
            self.status_bar.showMessage(message)
            self.identifications = identifications
            self.ready_for_fit = True
            self.draw_frame()
            # sound notification
            if elapsed_time > lib.SOUND_NOTIFICATION_DURATION:
                sound_path = lib.get_sound_notification_path()
                if sound_path is not None:
                    playsound(sound_path, block=False)
            if fit_afterwards:
                self.fit(calibrate_z=calibrate_z)

    def fit(self, calibrate_z: bool = False) -> None:
        """Fit identified spots (single molecules).

        Parameters
        ----------
        calibrate_z : bool, optional
            Whether to perform z-calibration during fitting. Default is
            False.
        """
        if self.movie is not None and self.ready_for_fit:
            self.status_bar.showMessage("Preparing fit...")
            model = self.parameters_dialog.fit_model.currentText()
            optimizer = self.parameters_dialog.fit_optimizer.currentText()
            method = _fit_code(model, optimizer)
            eps = self.parameters_dialog.convergence_criterion.value()
            max_it = self.parameters_dialog.max_it.value()
            fit_z = self.parameters_dialog.fit_z_checkbox.isChecked()
            use_gpufit = self.parameters_dialog.gpufit_checkbox.isChecked()
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
                use_gpufit,
            )
            self.fit_worker.progressMade.connect(self.on_fit_progress)
            self.fit_worker.cutProgressMade.connect(self.on_cut_progress)
            self.fit_worker.finished.connect(self.on_fit_finished)
            self.fit_worker.aborted.connect(self.on_worker_aborted)
            self._active_worker = self.fit_worker
            self.abort_action.setEnabled(True)
            self.fit_worker.start()

    def fit_z(self) -> None:
        """Fit z coordinates of the fitted localizations based on the
        calibration data."""
        self.status_bar.showMessage("Fitting z position...")
        model = self.parameters_dialog.fit_model.currentText()
        optimizer = self.parameters_dialog.fit_optimizer.currentText()
        fitting_method = _fit_code(model, optimizer)
        # avgroi won't really work for z; fall back to gausslq for compatibility
        if fitting_method == "avg":
            fitting_method = "gausslq"
        self.fit_z_worker = FitZWorker(
            self.locs,
            self.info + [self.camera_info],  # ensure pixel size in info
            self.parameters_dialog.z_calibration,
            self.parameters_dialog.magnification_factor.value(),
            self.parameters_dialog.pixelsize.value(),
            fitting_method,
            self.parameters_dialog.fit_z_gpu_checkbox.isChecked(),
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

    def on_fit_progress(self, curr: int, total: int) -> None:
        """Update the status bar with the fitting progress."""
        if self.parameters_dialog.gpufit_checkbox.isChecked():
            self.status_bar.showMessage("Fitting spots by GPUfit...")
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
        self.draw_frame()
        # sound notification
        if elapsed_time > lib.SOUND_NOTIFICATION_DURATION:
            sound_path = lib.get_sound_notification_path()
            if sound_path is not None:
                playsound(sound_path, block=False)
        base = self.channel_output_base()
        if calibrate_z:
            self.parameters_dialog.gpufit_checkbox.setDisabled(False)
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

        self.parameters_dialog.gpufit_checkbox.setDisabled(False)
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
        self.parameters_dialog.gpufit_checkbox.setDisabled(True)
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
    finished = QtCore.pyqtSignal(dict, object, float, pd.DataFrame, bool, bool)
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        fit_afterwards: bool,
        calibrate_z: bool,
    ) -> None:
        super().__init__()
        self.window = window
        self.movie = window.movie
        self.rois = window.view.rois
        self.frame_range = window.frame_range
        self.parameters = window.parameters
        self.fit_afterwards = fit_afterwards
        self.calibrate_z = calibrate_z

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
        method: Literal["gausslq", "gaussmle", "avg"],
        eps: float,
        max_it: int,
        fit_z: bool,
        calibrate_z: bool,
        use_gpufit: bool,
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
        self.N = len(identifications)
        self._last_cut_emit = 0
        if use_gpufit and method == "gausslq":
            method = "gausslq-gpu"
        self.method = method

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
