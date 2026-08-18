"""
picasso.lib_qt
~~~~~~~~~~~~~~

Qt-dependent (PyQt6) handy classes and functions, split out of
``picasso.lib`` so that importing ``picasso.lib`` does not require
PyQt6. All names defined here remain accessible as ``lib.<name>`` -
``picasso.lib`` forwards them lazily via a module ``__getattr__``, so
PyQt6 is only imported on first use.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import TypeAlias

import yaml
import matplotlib.pyplot as plt
from PyQt6 import QtCore, QtWidgets, QtGui
from playsound3 import playsound

from picasso import io
from picasso.lib import (
    _dialogs,
    SOUND_NOTIFICATION_DURATION,
    REQUIRED_COLUMNS,
    MockProgress,
    TqdmProgress,
    get_sound_notification_path,
    is_path_available,
)


class Dialog(QtWidgets.QDialog):
    """Base class for dialogs without 'What's this?' help."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._focus_buttons = ["OK"]
        self.setWindowFlag(
            QtCore.Qt.WindowType.WindowContextHelpButtonHint, False
        )

    def showEvent(self, event):
        """Remove focus from any QPushButton when the dialog is shown,
        so that pressing Enter does not trigger any button by default
        (unless it's called "OK").

        Parameters
        ----------
        event : QtGui.QShowEvent
            The Qt show event.
        """
        super().showEvent(event)
        for button in self.findChildren(QtWidgets.QPushButton):
            if button.text() in self._focus_buttons:
                continue
            button.setDefault(False)
            button.setAutoDefault(False)


class UserSettingsDialog(Dialog):
    """Dialog for inspecting and editing the user settings YAML file."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("User Settings")
        self.setModal(False)
        self.resize(600, 500)

        layout = QtWidgets.QVBoxLayout(self)

        path_label = QtWidgets.QLabel(
            f"Settings file: {io._user_settings_filename()}\n"
            "Warning: editing this file can affect the behavior of Picasso.\n"
            "Clearing the file will reset all settings to their default "
            "values."
        )
        path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(path_label)

        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setFont(QtGui.QFont("Helvetica", 12))
        layout.addWidget(self.editor)

        button_layout = QtWidgets.QHBoxLayout()
        reload_button = QtWidgets.QPushButton("Reload")
        reload_button.clicked.connect(self.load_settings)
        button_layout.addWidget(reload_button)
        button_layout.addStretch()
        save_button = QtWidgets.QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Re-read the settings from disk each time the dialog opens.

        Parameters
        ----------
        event : QtGui.QShowEvent
            The Qt show event.
        """
        super().showEvent(event)
        self.load_settings()

    def load_settings(self) -> None:
        """Read the settings file and display its contents."""
        filename = io._user_settings_filename()
        try:
            with open(filename, "r") as f:
                self.editor.setPlainText(f.read())
        except FileNotFoundError:
            self.editor.setPlainText(
                "# No settings file found. Edit and save to create one."
            )

    def save_settings(self) -> None:
        """Validate YAML and write back to the settings file."""
        text = self.editor.toPlainText()
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid YAML",
                f"Cannot save — the YAML is invalid:\n\n{e}",
            )
            return
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid settings",
                "Settings must be a YAML mapping (key: value pairs).",
            )
            return
        io.save_user_settings(parsed)
        QtWidgets.QMessageBox.information(
            self, "Saved", "User settings saved successfully."
        )


class MetadataDialog(Dialog):
    """Dialog for inspecting YAML metadata (list of lists of dicts).

    Can be used standalone with any ``infos`` data, making it reusable
    across Picasso modules.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        self.setModal(False)
        self.resize(700, 500)

        layout = QtWidgets.QVBoxLayout(self)

        # channel selector
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.channel_box = QtWidgets.QComboBox()
        self.channel_box.currentIndexChanged.connect(self._on_channel_changed)
        selector_layout.addWidget(self.channel_box)
        selector_layout.addStretch(1)

        # copy button
        copy_button = QtWidgets.QPushButton("Copy to clipboard")
        copy_button.clicked.connect(self._copy_to_clipboard)
        selector_layout.addWidget(copy_button)

        layout.addLayout(selector_layout)

        # tree widget for structured metadata display
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Key", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setStretchLastSection(True)
        self.tree.setColumnWidth(0, 250)
        layout.addWidget(self.tree)

        self._infos: list[list[dict]] = []
        self._labels: list[str] = []

    def set_infos(
        self,
        infos: list[list[dict]] | list[dict],
        labels: list[str] | str | None = None,
    ) -> None:
        """Set metadata and refresh the display. The user can provide
        the metadata and the label for a single channel as a list of
        dicts and a single string, respectively, or for multiple
        channels as a list of lists of dicts and a list of strings,
        respectively.

        Parameters
        ----------
        infos : list of list of dict or list of dict
            Metadata for each channel. Each element is a list of dicts
            as loaded from a YAML file.
        labels : list of str, optional
            Display labels for each channel (e.g., file paths).
        """
        if isinstance(infos, list) and all(isinstance(i, dict) for i in infos):
            infos = [infos]  # wrap single list of dicts into a list
        if isinstance(labels, str):
            labels = [labels]  # wrap single label into a list
        self._infos = infos
        self._labels = labels or [f"Channel {i}" for i in range(len(infos))]
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItems(self._labels)
        self.channel_box.blockSignals(False)
        if infos:
            self._on_channel_changed(0)

    def _on_channel_changed(self, index: int) -> None:
        """Populate tree with metadata from the selected channel."""
        self.tree.clear()
        if index < 0 or index >= len(self._infos):
            return
        info_list = self._infos[index]
        for i, info_dict in enumerate(info_list):
            section_label = info_dict.get("Generated by", f"Section {i}")
            section_item = QtWidgets.QTreeWidgetItem(
                [f"[{i}] {section_label}", ""]
            )
            section_item.setExpanded(True)
            font = section_item.font(0)
            font.setBold(True)
            section_item.setFont(0, font)
            self._add_dict_to_tree(section_item, info_dict)
            self.tree.addTopLevelItem(section_item)
        self.tree.expandAll()

    def _add_dict_to_tree(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        data: dict | list | object,
    ) -> None:
        """Recursively add dict/list contents to a tree item."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    child = QtWidgets.QTreeWidgetItem([str(key), ""])
                    self._add_dict_to_tree(child, value)
                    parent.addChild(child)
                else:
                    child = QtWidgets.QTreeWidgetItem([str(key), str(value)])
                    parent.addChild(child)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                if isinstance(value, (dict, list)):
                    child = QtWidgets.QTreeWidgetItem([f"[{i}]", ""])
                    self._add_dict_to_tree(child, value)
                    parent.addChild(child)
                else:
                    child = QtWidgets.QTreeWidgetItem([f"[{i}]", str(value)])
                    parent.addChild(child)

    def _copy_to_clipboard(self) -> None:
        """Copy the current channel's metadata to clipboard as YAML."""

        index = self.channel_box.currentIndex()
        if index < 0 or index >= len(self._infos):
            return
        text = yaml.dump_all(
            self._infos[index], default_flow_style=False, sort_keys=False
        )
        QtWidgets.QApplication.clipboard().setText(text)


class ProgressDialog(QtWidgets.QProgressDialog):
    """ProgressDialog displays a progress dialog with a progress bar."""

    def __init__(self, description, minimum, maximum, parent):
        # append time estimate to description
        super().__init__(
            description,
            None,
            minimum,
            maximum,
            parent,
            QtCore.Qt.WindowType.CustomizeWindowHint,
        )
        self.description_base = description  # without time estimate
        self.initalized = None

    def init(self):
        """Arm the dialog: register it, make it modal and start the clock.

        Called lazily on the first :meth:`set_value` so that a fast operation
        never flashes a dialog.
        """
        _dialogs.append(self)
        self.setMinimumDuration(500)
        self.setModal(True)
        self.t0 = time.time()
        self.app = QtCore.QCoreApplication.instance()
        self.initalized = True
        self.count_started = False
        self.finished = False
        # sound notification
        self.sound_notification_path = get_sound_notification_path()

    def set_value(self, value):
        """Advance the bar, arming the dialog and the time estimate as needed.

        Parameters
        ----------
        value : int
            Cumulative progress so far.
        """
        if not self.initalized:
            self.init()
        self.setValue(value)
        if self.count_started:
            # estimate time left
            elapsed = time.time() - self.t0_est
            remaining = int(
                (self.maximum() - value) * elapsed / (value + 1e-6)
            )
            # convert to hh-mm-ss
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            # format time estimate
            if hours > 0:
                hours = min(10, hours)  # limit hours to 10 for display
                time_estimate = f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"
            else:
                time_estimate = f"{minutes:02d}m:{seconds:02d}s"
            # set label text with time estimate
            description = (
                f"{self.description_base}"
                f"\nEstimated time remaining: {time_estimate}"
            )
            self.setLabelText(description)
        # sound notification
        if value >= self.maximum() and self.finished is False:
            self.finished = True
            self.play_sound_notification()
        # if value is above zero, count has started, enabling time estimate
        if not self.count_started:
            if value > 0:
                self.count_started = True
                self.t0_est = time.time()
        self.app.processEvents()

    def closeEvent(self, event):
        """Unregister the dialog and play the finish sound if it was not
        already played.

        Parameters
        ----------
        event : QtGui.QCloseEvent
            The Qt close event.
        """
        _dialogs.remove(self)
        if self.finished is False:
            self.finished = True
            self.play_sound_notification()

    def zero_progress(self, description=None):
        """Set progress dialog to zero and change the title if given.

        Parameters
        ----------
        description : str, optional
            Label of the new phase. None keeps the current one.
        """
        if description:
            self.setLabelText(description)
            self.description_base = description
        if self.initalized:
            # restart the time-estimate baseline so the next non-zero
            # set_value re-arms the timer for the new phase
            self.count_started = False
        self.set_value(0)

    def play_sound_notification(self):
        """Play a sound notification if a sound file is specified and
        at least a minute has passed since the dialog was opened."""
        if self.sound_notification_path is not None:
            if time.time() - self.t0 > SOUND_NOTIFICATION_DURATION:
                playsound(self.sound_notification_path, block=False)

    def get_iterator(self, start=None, end=None):
        """Get an iterator that spans the dialog's remaining progress.

        Parameters
        ----------
        start, end : int, optional
            First and one-past-last value. None uses the dialog's current
            value and maximum.

        Returns
        -------
        iterator : range
        """
        start = self.value() if start is None else start
        end = self.maximum() if end is None else end
        return range(start, end)


class StatusDialog(Dialog):
    """StatusDialog displays the description string in a dialog."""

    def __init__(self, description, parent):
        super(StatusDialog, self).__init__(
            parent,
            QtCore.Qt.WindowType.CustomizeWindowHint,
        )
        _dialogs.append(self)
        vbox = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(description)
        vbox.addWidget(label)
        self.sound_notification_path = get_sound_notification_path()
        self.t0 = time.time()
        self.show()
        QtCore.QCoreApplication.instance().processEvents()

    def closeEvent(self, event):
        """Unregister the dialog and play the notification sound if the task
        ran long enough to warrant one.

        Parameters
        ----------
        event : QtGui.QCloseEvent
            The Qt close event.
        """
        _dialogs.remove(self)
        if self.sound_notification_path is not None:
            if time.time() - self.t0 > SOUND_NOTIFICATION_DURATION:
                playsound(self.sound_notification_path, block=False)


# type alias for the progress dialogs
ProgressType: TypeAlias = ProgressDialog | MockProgress | TqdmProgress


class ScrollableGroupBox(QtWidgets.QGroupBox):
    """QGroupBox with QScrollArea as the top widget that enables
    scrolling."""

    def __init__(self, title, parent=None, layout="grid"):
        super().__init__(title, parent=parent)

        # Create a layout for the content of the group box
        if layout == "grid":
            self.content_layout = QtWidgets.QGridLayout(self)
        elif layout == "form":
            self.content_layout = QtWidgets.QFormLayout(self)
        self.content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(10)
        self.content_layout.setContentsMargins(10, 10, 10, 10)

        # Create a scroll area and set its content to the content layout
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(QtWidgets.QWidget(self))
        self.scroll_area.widget().setLayout(self.content_layout)

        # Set the layout of the group box to the scroll area
        self.setLayout(QtWidgets.QGridLayout(self))
        self.layout().addWidget(self.scroll_area, 0, 0, 1, 2)

    def add_widget(self, widget, row, column, height=1, width=1):
        """Add a widget to the grid layout inside the scroll area.

        Parameters
        ----------
        widget : QtWidgets.QWidget
            The widget to add.
        row, column : int
            Where to place it in the grid.
        height, width : int, optional
            How many rows and columns it spans. Default 1.
        """
        self.content_layout.addWidget(widget, row, column, height, width)

    def remove_widget(self, widget):
        """Remove a widget from the grid layout inside the scroll area.

        Parameters
        ----------
        widget : QtWidgets.QWidget
            The widget to remove.
        """
        self.content_layout.removeWidget(widget)

    def remove_all_widgets(self, keep_labels=False):
        """Remove all widgets from the grid layout.

        Parameters
        ----------
        keep_labels : bool, optional
            If True, the QLabels are kept. Default False.
        """
        for i in reversed(range(self.content_layout.count())):
            widget = self.content_layout.itemAt(i).widget()
            if keep_labels and isinstance(widget, QtWidgets.QLabel):
                continue
            widget.setParent(None)
            widget.deleteLater()


class LogDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """QDoubleSpinBox with logarithmic step size."""

    def __init__(
        self, parent: QtWidgets.QWidget | None = None, factor: float = 1.2
    ) -> None:
        super().__init__(parent)
        self._factor = factor  # multiply/divide by this on each step

    def stepBy(self, steps: int) -> None:
        """Step the value multiplicatively, so the arrows move it by a factor
        rather than by a fixed amount.

        Parameters
        ----------
        steps : int
            Number of steps; negative steps divide instead of multiply.
        """
        if steps > 0:
            if self.value() <= 10 ** (-self.decimals()):
                self.setValue(2 * 10 ** (-self.decimals()))
            else:
                self.setValue(self.value() * (self._factor**steps))
        elif steps < 0:
            self.setValue(self.value() / (self._factor ** abs(steps)))


class RangeSlider(QtWidgets.QWidget):
    """Horizontal slider with two handles, spanning a value range.

    Qt has no two-handle slider, so this is a minimal one: a groove with
    a low and a high handle, painted to match the stylesheet of the plain
    ``QSlider``s used elsewhere in Picasso. Values are floats, so that the
    slider is not limited to the integer positions of ``QSlider``.

    ``valuesChanged`` is emitted whenever the pair changes, both on user
    interaction and on a programmatic ``setValues`` / ``setRange`` (like
    ``QSlider.setValue``); block the signals to update silently.
    """

    valuesChanged = QtCore.pyqtSignal(float, float)

    HANDLE_WIDTH = 10
    HANDLE_HEIGHT = 12
    GROOVE_HEIGHT = 4

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 1.0
        self._low = 0.0
        self._high = 1.0
        # Smallest allowed distance between the handles. 0 lets them meet;
        # consumers that quantize the values (e.g. to integers) set it to
        # the quantum so that low and high cannot collapse onto each other.
        self._min_gap = 0.0
        self._pressed_handle = None  # None, "low" or "high"
        self._last_handle = "low"  # the one the arrow keys move
        self._value_labels = ("Min", "Max")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._update_tooltip()

    # -- API ---------------------------------------------------------

    def setRange(self, minimum: float, maximum: float) -> None:
        """Set the ends of the track, re-clamping the current values.

        Parameters
        ----------
        minimum, maximum : float
            The new ends; ``maximum`` is raised to ``minimum`` if it is below
            it.
        """
        minimum = float(minimum)
        maximum = max(float(maximum), minimum)
        self._minimum = minimum
        self._maximum = maximum
        self.setValues(self._low, self._high)
        self.update()

    def range(self) -> tuple[float, float]:
        """The ends of the track, as ``(minimum, maximum)``."""
        return self._minimum, self._maximum

    def minimum(self) -> float:
        """The lower end of the track."""
        return self._minimum

    def maximum(self) -> float:
        """The upper end of the track."""
        return self._maximum

    def setValues(
        self, low: float, high: float, moved: str | None = None
    ) -> bool:
        """Set both handles, clamped into the track and kept ``_min_gap``
        apart.

        Parameters
        ----------
        low, high : float
            The requested handle positions.
        moved : str, optional
            Names the handle the user is dragging (``"low"`` or ``"high"``),
            which is the one that gives way when the two would cross. Default
            None.

        Returns
        -------
        changed : bool
            Whether anything changed; ``valuesChanged`` is emitted if so.
        """
        low = min(max(float(low), self._minimum), self._maximum)
        high = min(max(float(high), self._minimum), self._maximum)
        gap = min(self._min_gap, self._maximum - self._minimum)
        if high - low < gap:
            if moved == "low":
                low = high - gap
                if low < self._minimum:
                    low = self._minimum
                    high = low + gap
            else:
                high = low + gap
                if high > self._maximum:
                    high = self._maximum
                    low = high - gap
        if low == self._low and high == self._high:
            return False
        self._low = low
        self._high = high
        self._update_tooltip()
        self.update()
        self.valuesChanged.emit(low, high)
        return True

    def values(self) -> tuple[float, float]:
        """The handle positions, as ``(low, high)``."""
        return self._low, self._high

    def setMinimumGap(self, gap: float) -> None:
        """Set how far apart the two handles must stay.

        Parameters
        ----------
        gap : float
            Minimum distance, in track units; negative values are clipped to
            0.
        """
        self._min_gap = max(0.0, float(gap))
        self.setValues(self._low, self._high)

    def setValueLabels(self, low_label: str, high_label: str) -> None:
        """Name the two handles, for the tooltip.

        Parameters
        ----------
        low_label, high_label : str
            Names of the lower and upper handle.
        """
        self._value_labels = (low_label, high_label)
        self._update_tooltip()

    # -- Painting ----------------------------------------------------

    def sizeHint(self) -> QtCore.QSize:
        """The size the slider would like to have."""
        return QtCore.QSize(150, self.HANDLE_HEIGHT + 3)

    def minimumSizeHint(self) -> QtCore.QSize:
        """The smallest size at which both handles still fit."""
        return QtCore.QSize(4 * self.HANDLE_WIDTH, self.HANDLE_HEIGHT + 3)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        """Draw the groove, the selected span and the two handles.

        Parameters
        ----------
        _event : QtGui.QPaintEvent
            The Qt paint event; unused.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        enabled = self.isEnabled()
        groove_color = QtGui.QColor("#b0b0b0" if enabled else "#d8d8d8")
        handle_color = QtGui.QColor("#5a5a5a" if enabled else "#b8b8b8")
        mid_y = self.height() / 2
        radius = self.GROOVE_HEIGHT / 2
        groove = QtCore.QRectF(
            0.0,
            mid_y - radius,
            float(self.width()),
            float(self.GROOVE_HEIGHT),
        )
        painter.setBrush(groove_color)
        painter.drawRoundedRect(groove, radius, radius)
        # the selected span, so that the covered part of the range reads
        # at a glance
        x_low = self._value_to_x(self._low)
        x_high = self._value_to_x(self._high)
        painter.setBrush(handle_color)
        painter.drawRoundedRect(
            QtCore.QRectF(
                x_low,
                mid_y - radius,
                max(0.0, x_high - x_low),
                groove.height(),
            ),
            radius,
            radius,
        )
        for x in (x_low, x_high):
            painter.drawRoundedRect(
                QtCore.QRectF(
                    x - self.HANDLE_WIDTH / 2,
                    mid_y - self.HANDLE_HEIGHT / 2,
                    float(self.HANDLE_WIDTH),
                    float(self.HANDLE_HEIGHT),
                ),
                3.0,
                3.0,
            )
        painter.end()

    # -- Interaction -------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Grab the handle nearest the click and move it there.

        Parameters
        ----------
        event : QtGui.QMouseEvent
            The Qt mouse event; anything but a left click is passed on.
        """
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.position().x()
        self._pressed_handle = self._handle_at(x)
        self._last_handle = self._pressed_handle
        self._move_handle(self._pressed_handle, x)
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drag the grabbed handle.

        Parameters
        ----------
        event : QtGui.QMouseEvent
            The Qt mouse event; passed on when no handle is grabbed.
        """
        if self._pressed_handle is None:
            super().mouseMoveEvent(event)
            return
        self._move_handle(self._pressed_handle, event.position().x())
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Release the grabbed handle.

        Parameters
        ----------
        event : QtGui.QMouseEvent
            The Qt mouse event; passed on when no handle is grabbed.
        """
        if self._pressed_handle is None:
            super().mouseReleaseEvent(event)
            return
        self._pressed_handle = None
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Move the last-touched handle with the arrow keys.

        Parameters
        ----------
        event : QtGui.QKeyEvent
            The Qt key event; anything but the arrow keys is passed on.
        """
        key = event.key()
        if key in (
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Up,
        ):
            step = self._step()
            if key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Down):
                step = -step
            if self._last_handle == "high":
                self.setValues(self._low, self._high + step, moved="high")
            else:
                self.setValues(self._low + step, self._high, moved="low")
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Ignore the wheel, so that scrolling passes through to the parent.

        The slider sits under a scrollable movie view; scrolling over it
        should not silently change the contrast.

        Parameters
        ----------
        event : QtGui.QWheelEvent
            The Qt wheel event.
        """
        event.ignore()

    # -- Helpers -----------------------------------------------------

    def _step(self) -> float:
        span = self._maximum - self._minimum
        return max(self._min_gap, span / 100.0) if span else 0.0

    def _usable_width(self) -> float:
        return max(1.0, self.width() - self.HANDLE_WIDTH)

    def _value_to_x(self, value: float) -> float:
        """Pixel position of a handle's centre, inset by half a handle so
        that the handles stay inside the widget at both ends."""
        span = self._maximum - self._minimum
        fraction = (value - self._minimum) / span if span else 0.0
        fraction = min(max(fraction, 0.0), 1.0)
        return self.HANDLE_WIDTH / 2 + fraction * self._usable_width()

    def _x_to_value(self, x: float) -> float:
        fraction = (x - self.HANDLE_WIDTH / 2) / self._usable_width()
        fraction = min(max(fraction, 0.0), 1.0)
        return self._minimum + fraction * (self._maximum - self._minimum)

    def _handle_at(self, x: float) -> str:
        """The handle a click at ``x`` grabs: the one under the cursor, or
        else the one nearer to it (so clicking the bare groove moves the
        closer handle there)."""
        x_low = self._value_to_x(self._low)
        x_high = self._value_to_x(self._high)
        if x < x_low:
            return "low"
        if x > x_high:
            return "high"
        return "low" if (x - x_low) <= (x_high - x) else "high"

    def _move_handle(self, handle: str, x: float) -> None:
        value = self._x_to_value(x)
        if handle == "low":
            self.setValues(value, self._high, moved="low")
        else:
            self.setValues(self._low, value, moved="high")

    def _update_tooltip(self) -> None:
        low_label, high_label = self._value_labels
        self.setToolTip(
            f"{low_label}: {self._low:,.0f}, {high_label}: {self._high:,.0f}"
        )


class GenericPlotWindow(QtWidgets.QTabWidget):
    """Interface for displaying matplotlib plots in a separate
    window."""

    def __init__(self, window_title, app_name):
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvas,
            NavigationToolbar2QT,
        )

        super().__init__()
        self.setWindowTitle(window_title)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", f"{app_name}.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(1000, 500)
        self.figure = plt.Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        vbox.addWidget(self.toolbar)


class RemoveColumnsDialog(Dialog):
    """Allow the user to select columns to be removed from the locs
    DataFrame."""

    def __init__(
        self, window: QtWidgets.QMainWindow, columns: list[str]
    ) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Remove columns")
        self.setModal(True)
        vbox = QtWidgets.QVBoxLayout(self)
        self.setLayout(vbox)
        self.checks = {}
        for column in columns:
            check = QtWidgets.QCheckBox(column)
            check.setChecked(False)
            if column in REQUIRED_COLUMNS:
                check.setEnabled(False)
            vbox.addWidget(check)
            self.checks[column] = check
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
    def getParams(
        parent: QtWidgets.QMainWindow, columns: list[str]
    ) -> tuple[list[str], bool]:
        """Open the dialog and return the columns to be removed.

        Parameters
        ----------
        parent : QMainWindow
            Instance of the main window.
        columns : list of str
            List of column names in the locs DataFrame.

        Returns
        -------
        to_remove : list of str
            List of column names to be removed.
        accepted : bool
            True if the user clicked OK, False if the user clicked
            Cancel.
        """
        dialog = RemoveColumnsDialog(parent, columns)
        result = dialog.exec()
        to_remove = []
        for col in columns:
            if dialog.checks[col].isChecked():
                to_remove.append(col)
        return to_remove, result == QtWidgets.QDialog.DialogCode.Accepted


class HelpButton(QtWidgets.QToolButton):
    """A reusable ? button that opens a URL."""

    def __init__(
        self, url: str, parent=None, size: int | tuple[int, int] = 22
    ) -> None:
        super().__init__(parent)
        self.help_url = url
        self.setText("?")
        if isinstance(size, int):
            size = (size, size)
        self.setFixedSize(*size)
        self.setToolTip("Open documentation")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            """
            QToolButton {
                border: 1px solid palette(mid);
                border-radius: 11px;
                font-weight: bold;
                font-size: 12px;
                color: palette(button-text);
                background: palette(button);
            }
            QToolButton:hover {
                background: palette(highlight);
                color: palette(highlighted-text);
                border-color: palette(highlight);
            }
        """
        )
        self.clicked.connect(self._open_docs)

    def _open_docs(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.help_url))


def cancel_dialogs():
    """Closes all open dialogs (``ProgressDialog`` and ``StatusDialog``)
    in the GUI."""
    dialogs = [_ for _ in _dialogs]
    for dialog in dialogs:
        if isinstance(dialog, ProgressDialog):
            dialog.cancel()
        else:
            dialog.close()
    QtCore.QCoreApplication.instance().processEvents()  # just in case...


def install_excepthook(window) -> None:
    """Install a thread-safe excepthook that shows uncaught exceptions in a
    QMessageBox.

    Safe to call from QThread workers because the error signal is queued to
    the main thread by Qt's event loop.

    Parameters
    ----------
    window : QtWidgets.QWidget
        Parent of the message box.
    """

    class _ErrorSignaler(QtCore.QObject):
        error = QtCore.pyqtSignal(str)

    signaler = _ErrorSignaler()

    def _show_error(message: str) -> None:
        cancel_dialogs()
        QtWidgets.QMessageBox.critical(window, "An error occurred", message)

    signaler.error.connect(_show_error)

    def excepthook(type, value, tback):
        message = "".join(traceback.format_exception(type, value, tback))
        signaler.error.emit(message)
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook


def adjust_widget_size(
    widget: QtWidgets.QWidget,
    size_hint: QtCore.QSize,
    width_offset: int = 0,
    height_offset: int = 0,
) -> None:
    """Adjust the size of a QWidget based on its size hint. The user
    can specify the offsets to be added to the width and height of the
    size hint. The user can also specify whether to limit the width
    and height to the screen size.

    Parameters
    ----------
    widget : QtWidgets.QWidget
        The widget to be adjusted.
    size_hint : QtCore.QSize
        The size hint of the widget. Can be obtained with
        widget.sizeHint().
    width_offset : int, optional
        The offset to be added to the width of the size hint. Default is
        0.
    height_offset : int, optional
        The offset to be added to the height of the size hint. Default
        is 0.
    """
    intended_width = size_hint.width() + width_offset
    intended_height = size_hint.height() + height_offset
    # adjust to the screen size if necessary
    screen = QtWidgets.QApplication.primaryScreen()
    screen_height = 1000 if screen is None else screen.size().height()
    screen_width = 1000 if screen is None else screen.size().width()
    intended_width = min(intended_width, screen_width - 200)
    intended_height = min(intended_height, screen_height - 200)
    widget.resize(intended_width, intended_height)


def get_save_filename_ext_dialog(
    parent: QtWidgets.QWidget,
    caption: str = "",
    directory: str = "",
    filter: str = "",
    check_ext: str | list[str] = "",
) -> tuple[str, str]:
    """Custom getSaveFileName dialog that can check for the existence of
    files with other extensions (for example, if the user tries to save
    a .yaml file with the same name as an existing .hdf5 file, it will
    ask if the user wants to overwrite the .hdf5 file). The output is
    the same as for QtWidgets.QFileDialog.getSaveFileName.

    Parameters
    ----------
    parent : QWidget
        Parent widget for the dialog.
    caption : str, optional
        Dialog caption. Default is "".
    directory : str, optional
        Initial directory. Default is "".
    filter : str, optional
        File filter, e.g., "YAML files (*.yaml);;HDF5 files (*.hdf5)".
        Default is "".
    check_ext : str or list of str, optional
        Other extension(s) to be checked if they're available. Does not
        have to be a strict ".ext" format, can also include a suffix to
        the path, e.g., "_1.hdf5". If "", extensions are not checked,
        giving the standard getSaveFileName dialog behavior. Default is
        "".

    Returns
    -------
    selected_path : str
        Selected file path.
    selected_filter : str
        Selected file filter.
    """
    # first run the standard dialog to get the initial path and filter
    selected_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
        parent, caption, directory, filter
    )
    # check for the existence of files with other extensions and ask the
    # user if they want to overwrite them
    if selected_path and check_ext:
        paths_available = is_path_available(
            selected_path, check_ext=check_ext, parent=parent
        )
        if not all(paths_available):
            return "", ""
    # if the user selected a .yml file, change the extension to .yaml
    # for consistency
    if selected_path.endswith(".yml"):
        selected_path = selected_path[:-4] + ".yaml"
    return selected_path, selected_filter
