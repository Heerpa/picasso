"""
picasso.lib
~~~~~~~~~~~

Handy functions and classes.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import glob
import collections
import colorsys
import importlib
import os
import warnings
from copy import deepcopy
from typing import Any, TypeAlias, Literal, TYPE_CHECKING
from collections.abc import Callable
from asyncio import Future

import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.recfunctions import append_fields, drop_fields
from scipy import stats, optimize
from playsound3 import playsound
from tqdm import tqdm

from picasso import io

if TYPE_CHECKING:
    from PyQt6 import QtGui

# A global variable where we store all open progress and status dialogs.
# In case of an exception, we close them all,
# so that the GUI remains responsive.
_dialogs = []

# Min. time to use sound notification when ProcessDialog or
# StatusDialog is finished
SOUND_NOTIFICATION_DURATION = 60  # seconds

# Columns that are required for Picasso
REQUIRED_COLUMNS = ["frame", "x", "y", "z", "lpx", "lpy", "lpz"]

# Type alias
IntArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.integer[Any]]]
IntArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.integer[Any]]]
IntArray3D: TypeAlias = np.ndarray[
    tuple[int, int, int], np.dtype[np.integer[Any]]
]
FloatArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
FloatArray2D: TypeAlias = np.ndarray[
    tuple[int, int], np.dtype[np.floating[Any]]
]
FloatArray3D: TypeAlias = np.ndarray[
    tuple[int, int, int], np.dtype[np.floating[Any]]
]
FloatArray4D: TypeAlias = np.ndarray[
    tuple[int, int, int, int], np.dtype[np.floating[Any]]
]
SeriesOrFloatArray1D: TypeAlias = pd.Series | FloatArray1D
SeriesOrIntArray1D: TypeAlias = pd.Series | IntArray1D
BoolArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
BoolArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
Array3x3: TypeAlias = np.ndarray[
    tuple[Literal[3], Literal[3]], np.dtype[np.floating[Any]]
]


class _LazyQtModule:
    """Placeholder for a PyQt6 submodule that is imported on first
    attribute access, so that importing the surrounding module does not
    require PyQt6."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module = None

    def __getattr__(self, attr: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)


# Qt-dependent classes and functions live in picasso.lib_qt (which
# imports PyQt6) but stay accessible as lib.<name>; they are resolved
# lazily below so that importing picasso.lib does not require PyQt6.
_QT_NAMES = (
    "Dialog",
    "UserSettingsDialog",
    "MetadataDialog",
    "ProgressDialog",
    "StatusDialog",
    "ProgressType",
    "ScrollableGroupBox",
    "LogDoubleSpinBox",
    "GenericPlotWindow",
    "RemoveColumnsDialog",
    "HelpButton",
    "cancel_dialogs",
    "install_excepthook",
    "adjust_widget_size",
    "get_save_filename_ext_dialog",
)


def __getattr__(name: str) -> Any:
    if name in _QT_NAMES:
        from picasso import lib_qt

        value = getattr(lib_qt, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def normalize_frame_bounds(frame_bounds, n_frames):
    """Normalize ``frame_bounds`` to a list of concrete, inclusive,
    0-indexed ``(lo, hi)`` segments.

    Accepts either the legacy flat form ``(min, max)`` (where either bound
    may be None for an open end) or a list of such segments. ``None``
    bounds are resolved to ``0`` / ``n_frames``. Returns None when
    ``frame_bounds`` is None (i.e., all frames are used).

    Parameters
    ----------
    frame_bounds : tuple, list of tuples, or None
        A single ``(min, max)`` tuple, a list of such tuples, or None.
    n_frames : int
        Number of frames in the movie, used to resolve open upper bounds.

    Returns
    -------
    segments : list of tuple, or None
        List of ``(lo, hi)`` inclusive 0-indexed segments, or None.
    """
    if frame_bounds is None:
        return None
    # detect the legacy flat (min, max) form: the first element is a
    # scalar or None rather than a (lo, hi) segment
    first = frame_bounds[0]
    if first is None or np.isscalar(first):
        segments = [frame_bounds]
    else:
        segments = frame_bounds
    normalized = []
    for lo, hi in segments:
        lo = 0 if lo is None else lo
        hi = n_frames if hi is None else hi
        normalized.append((lo, hi))
    return normalized


def frame_in_bounds(frame_number, frame_bounds, n_frames):
    """Return True if ``frame_number`` falls within any segment of
    ``frame_bounds`` (or if ``frame_bounds`` is None, i.e., all frames are
    used). Bounds are inclusive. See ``normalize_frame_bounds``."""
    segments = normalize_frame_bounds(frame_bounds, n_frames)
    if segments is None:
        return True
    return any(lo <= frame_number <= hi for lo, hi in segments)


class MockProgress:
    """Class to mock a progress bar or dialog, allowing for calling
    the same methods but not displaying anything."""

    def __init__(self, *args, **kwargs):
        self.description_base = ""
        self._maximum = 0

    def init(self, *args, **kwargs):
        pass

    def set_value(self, *args, **kwargs):
        pass

    def setMaximum(self, maximum, *args, **kwargs):
        self._maximum = maximum

    def maximum(self):
        return self._maximum

    def update(self, *args, **kwargs):
        pass

    def closeEvent(self, *args, **kwargs):
        pass

    def zero_progress(self, description=None, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def setLabelText(self, *args, **kwargs):
        pass

    def play_sound_notification(self, *args, **kwargs):
        pass

    def get_iterator(self, start=0, end=100):
        return range(start, end)


class TqdmProgress:
    """Class to absorb calls to ProgressDialog but is used to display
    tqdm progress bar instead.

    Implements the same interface as ``ProgressDialog`` (see
    ``normalize_progress``): the bar is armed lazily on the first
    ``set_value`` call, using the description and maximum declared so
    far; ``zero_progress`` closes the current bar so that the next
    ``set_value`` starts a fresh one (a new phase)."""

    def __init__(self, *args, unit="it", **kwargs):
        self.description_base = (
            "" if "description" not in kwargs else kwargs["description"]
        )
        self.iterator = None
        self.unit = unit
        self._maximum = 0

    def init(self, *args, **kwargs):
        pass

    def set_value(self, value, *args, **kwargs):
        if self.iterator is None:
            self.iterator = tqdm(
                total=self._maximum or None,
                desc=self.description_base,
                unit=self.unit,
            )
        self.iterator.update(value - self.iterator.n)

    def setMaximum(self, maximum, *args, **kwargs):
        self._maximum = maximum
        if self.iterator is not None:
            self.iterator.total = maximum
            self.iterator.refresh()

    def maximum(self):
        return self._maximum

    def update(self, *args, **kwargs):
        pass

    def closeEvent(self, *args, **kwargs):
        pass

    def zero_progress(self, description=None, *args, **kwargs):
        if description:
            self.description_base = description
        self.close()

    def close(self, *args, **kwargs):
        if self.iterator is not None:
            self.iterator.close()
            self.iterator = None

    def setLabelText(self, *args, **kwargs):
        pass

    def play_sound_notification(self, *args, **kwargs):
        pass

    def get_iterator(self, start=0, end=100, unit="segment"):
        """Get an iterator for the progress bar."""
        self.close()
        iterator = tqdm(
            range(start, end),
            desc=self.description_base,
            unit=unit,
        )
        self.iterator = iterator
        return iterator


def normalize_progress(
    progress: Any,
    description: str = "",
    unit: str = "it",
) -> Any:
    """Normalize a public ``progress``/``callback`` argument to an
    object with the ``ProgressDialog`` interface (``set_value``,
    ``setMaximum``, ``zero_progress``, ...), so that callers can drive
    progress with plain method calls instead of checking which kind of
    tracker they hold (which would also pull in Qt in headless runs):

    * ``None`` -> ``MockProgress`` (reports nothing),
    * ``"console"`` -> ``TqdmProgress`` (tqdm bar in the console),
    * anything else (e.g. a ``lib.ProgressDialog``) is returned
      unchanged, as long as it exposes the interface.

    Parameters
    ----------
    progress : None, "console", or ProgressType
        The public progress argument.
    description : str, optional
        Initial description for a newly created ``TqdmProgress``.
        Default is "".
    unit : str, optional
        tqdm unit label for a newly created ``TqdmProgress``. Default
        is "it".

    Returns
    -------
    progress : ProgressType
        Object implementing the ``ProgressDialog`` interface.
    """
    if progress is None:
        return MockProgress()
    if isinstance(progress, str):
        if progress == "console":
            return TqdmProgress(description=description, unit=unit)
        raise ValueError(
            f"Invalid progress argument: {progress!r}. Must be None, "
            "'console', or an object with the ProgressDialog interface."
        )
    if not callable(getattr(progress, "set_value", None)):
        raise TypeError(
            f"Invalid progress argument: {progress!r}. Must be None, "
            "'console', or an object with the ProgressDialog interface "
            "(set_value, setMaximum, zero_progress, ...)."
        )
    return progress


class AutoDict(collections.defaultdict):
    """A defaultdict whose auto-generated values are defaultdicts
    itself. This allows for auto-generating nested values, e.g.
    a = AutoDict()
    a['foo']['bar']['carrot'] = 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(AutoDict, *args, **kwargs)


def deprecation_warning(message: str) -> None:
    """Display a deprecation warning message.

    Parameters
    ----------
    message : str
        The deprecation warning message to be displayed.
    """
    warnings.warn(message, DeprecationWarning, stacklevel=2)


def get_sound_notification_path() -> str | None:
    """Return the path to the sound notification file from the user
    settings file. If the file is not found or not specified, return
    None.

    Returns
    -------
    path : str or None
        Path to the sound notification file or None if not found or not
        specified.
    """
    settings = io.load_user_settings()
    if "Sound_notification" not in settings:  # add default settings (no sound)
        settings["Sound_notification"]["filename"] = None
        io.save_user_settings(settings)
    filename = settings["Sound_notification"]["filename"]
    sounds_dir = _sound_notification_dir()
    if filename is not None and os.path.isfile(
        os.path.join(sounds_dir, filename)
    ):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".mp3", ".wav"]:
            path = None
        else:
            path = os.path.join(sounds_dir, filename)
    else:
        path = None
    return path


def get_available_sound_notifications() -> list[str | None]:
    """Get a list of file names of the available sound notifications in
    the folder ``~/.picasso/notification_sounds``.

    Returns
    -------
    filenames : list of strs
        List of file names of the available sound notifications.
    """
    sounds_dir = _sound_notification_dir()
    filenames = [
        _
        for _ in os.listdir(sounds_dir)
        if os.path.isfile(os.path.join(sounds_dir, _))
        and os.path.splitext(_)[1].lower() in [".mp3", ".wav"]
    ]
    filenames = ["None"] + filenames
    return filenames


def set_sound_notification(action: QtGui.QAction) -> None:
    """Save the selected sound notification in the user settings
    file.

    Parameters
    ----------
    action : QtGui.QAction
        The action representing the selected sound notification.
    """
    settings = io.load_user_settings()
    selected_sound = action.objectName()  # file name with extension
    settings["Sound_notification"]["filename"] = selected_sound
    io.save_user_settings(settings)
    # play selected sound as a preview
    play_path = get_sound_notification_path()
    playsound(play_path, block=False) if play_path is not None else None


def _sound_notification_dir() -> str:
    """Return the path to the user sound notification folder
    (``~/.picasso/notification_sounds``)."""
    return io.notification_sounds_directory()


def open_sound_notifications_folder() -> None:
    """Open the user sound notification folder
    (``~/.picasso/notification_sounds``) in the system file browser."""
    from PyQt6 import QtCore, QtGui

    QtGui.QDesktopServices.openUrl(
        QtCore.QUrl.fromLocalFile(_sound_notification_dir())
    )


def get_from_metadata(
    info: list[dict] | dict,
    key: Any,
    default=None,
    *,
    raise_error: bool = False,
) -> Any:
    """Get a value from the localization metadata (list of dictionaries
    or a dictionary). Runs the search from the last to the first element
    of the input list. Returns default or raises an error if the key is
    not found.

    Parameters
    ----------
    info : list of dicts or dict
        Localization metadata.
    key : Any
        Key to be searched in the metadata.
    default : Any, optional
        Value to be returned if the key is not found. Default is None.
    raise_error : bool, optional
        If True, raises a KeyError if the key is not found. Default is
        False.

    Returns
    -------
    value : Any
        Value corresponding to the key in the metadata. If the key is
        not found, default is returned.
    """
    if isinstance(info, dict):
        if raise_error and key not in info:
            raise KeyError(f"Key '{key}' not found in metadata.")
        return info.get(key, default)
    elif isinstance(info, list):
        for inf in info[::-1]:
            if val := inf.get(key):
                return val
        if raise_error:
            raise KeyError(f"Key '{key}' not found in metadata.")
        return default
    else:
        raise ValueError("info must be a dict or a list of dicts.")


def extract_filter_steps(
    info: list[dict],
    current_columns,
) -> tuple[dict[str, list[float]], list[str], list[str]]:  # noqa: C901
    """Parse filter steps out of a Picasso Filter metadata list.

    Iterates ``info`` oldest -> newest. A dict is treated as a filter
    dict when its ``Generated by`` value contains ``"Filter"``. Numeric
    [min, max] ranges for columns present in ``current_columns`` are
    intersected; columns absent from the current data are reported as
    missing instead of being applied.

    Parameters
    ----------
    info : list of dicts
        Localization metadata loaded via ``io.load_info``.
    current_columns : iterable of str
        Columns available in the target localizations DataFrame.

    Returns
    -------
    ranges : dict[str, list[float]]
        Column -> [min, max] to apply.
    to_remove : list[str]
        Columns to drop (present in current data).
    missing : list[str]
        Columns referenced in metadata but absent from current data.
    """
    current = set(current_columns)
    ranges = {}
    to_remove_all = []
    missing = []

    for d in info:
        if not isinstance(d, dict):
            continue
        gen_by = get_from_metadata(d, "Generated by", default="")
        if "Filter" not in str(gen_by):
            continue
        for key, value in d.items():
            if key == "Generated by":
                continue
            if key == "Removed columns" and isinstance(value, (list, tuple)):
                to_remove_all.extend(value)
                continue
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(isinstance(v, (int, float)) for v in value)
            ):
                xmin, xmax = float(value[0]), float(value[1])
                if key not in current:
                    missing.append(key)
                    continue
                if key in ranges:
                    ranges[key][0] = max(ranges[key][0], xmin)
                    ranges[key][1] = min(ranges[key][1], xmax)
                else:
                    ranges[key] = [xmin, xmax]

    to_remove = [c for c in to_remove_all if c in current]
    for c in to_remove_all:
        if c not in current:
            missing.append(c)

    seen: set = set()
    missing_unique: list[str] = []
    for c in missing:
        if c not in seen:
            seen.add(c)
            missing_unique.append(c)

    return ranges, to_remove, missing_unique


def apply_filter_steps(
    locs: pd.DataFrame,
    info: list[dict],
) -> tuple[pd.DataFrame, dict[str, list[float]], list[str], list[str]]:
    """Apply Picasso Filter steps recorded in ``info`` to ``locs``.

    Thin wrapper around :func:`extract_filter_steps`: it parses the
    filter recipe out of the metadata, intersects each per-column
    [min, max] range against ``locs``, drops any "Removed columns",
    and reports columns that were referenced by the metadata but absent
    from ``locs``.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to filter.
    info : list of dicts
        Localization metadata loaded via ``io.load_info``.

    Returns
    -------
    filtered_locs : pd.DataFrame
        ``locs`` with the range filters and column removals applied.
    ranges : dict[str, list[float]]
        Column -> [min, max] that were applied.
    to_remove : list[str]
        Columns that were dropped.
    missing : list[str]
        Columns referenced in ``info`` but not present in ``locs``
        (skipped, not applied).
    """
    ranges, to_remove, missing = extract_filter_steps(info, locs.columns)
    for field, (xmin, xmax) in ranges.items():
        locs = locs[(locs[field] > xmin) & (locs[field] < xmax)]
    if to_remove:
        locs = locs.drop(columns=to_remove)
    return locs, ranges, to_remove, missing


def overwrite_metadata(
    info: list[dict] | dict, key: Any, value: Any
) -> list[dict] | dict:
    """Overwrite a value in the localization metadata (list of
    dictionaries or a dictionary). If the key does not exist an error
    is raised.

    Parameters
    ----------
    info : list of dicts or dict
        Localization metadata.
    key : Any
        Key to be overwritten or added in the metadata.
    value : Any
        Value to be set for the key.

    Returns
    -------
    updated_info : list of dicts or dict
        Metadata with the updated value.

    Raises
    ------
    KeyError
        If the key is not found in the metadata.
    """
    success = False
    if isinstance(info, dict):
        if key in info:
            info[key] = value
            success = True
    elif isinstance(info, list):
        for inf in info[::-1]:
            if key in inf:
                inf[key] = value
                success = True
                break
    if not success:
        raise KeyError(f"Key '{key}' not found in metadata.")
    return info


def get_colors(n_channels):
    """Create a list with rgb channels for each channel.

    Colors go from red to green, blue, pink and red again.

    Parameters
    ----------
    n_channels : int
        Number of locs channels

    Returns
    -------
    list
        Contains tuples with rgb channels
    """
    hues = np.arange(0, 1, 1 / n_channels)
    colors = [colorsys.hsv_to_rgb(_, 1, 1) for _ in hues]
    return colors


def is_hexadecimal(text):
    """Check if text represents a hexadecimal code for rgb, for
    example ``#ff02d4``.

    Parameters
    ----------
    text : str
        String to be checked.

    Returns
    -------
    bool
        True if text represents rgb, False otherwise.
    """
    allowed_characters = "0123456789abcdefABCDEF"
    if isinstance(text, str) and text[0] == "#" and len(text) == 7:
        n_valid = sum(char in allowed_characters for char in text[1:])
        if n_valid == 6:
            return True
    return False


def is_path_available(
    path: str, *, check_ext: str | list[str] = "", parent=None
) -> bool:
    """Check if a file or folder exists at the given path. Returns True
    if there is not such path. Returns False if the path already exists.
    Allows to easily change the extension of the path.

    Parameters
    ----------
    path : str
        Path to be checked.
    check_ext : str or list of str, optional
        Other extension(s) to be checked if they're available. Default
        is "".
    parent : QWidget, optional
        Parent widget for the error message box if raise_error is True.
        A message box will be displayed showing asking if the user wants
        to continue without the file or folder if the path does not
        exist.

    Returns
    -------
    paths_available : list of bools
        For each path generated with the new extension, True if the path
        is available, False if the path already exists.

    Raises
    ------
    ValueError
        If check_ext is not empty and does not start with a dot.
    """
    if check_ext:
        if isinstance(check_ext, str):
            check_ext = [check_ext]
        paths = [os.path.splitext(path)[0] + ext for ext in check_ext]
    else:
        paths = [path]
    paths_available = []
    for path in paths:
        if os.path.exists(path):
            if parent is not None:
                from PyQt6 import QtWidgets

                box = QtWidgets.QMessageBox(parent)
                box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                box.setWindowTitle("File or folder already exists")
                box.setText(
                    f"The path '{path}' already exists."
                    "\nDo you wish to overwrite it?"
                )
                box.setStandardButtons(
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No
                )
                result = box.exec()
                if result != QtWidgets.QMessageBox.StandardButton.Yes:
                    paths_available.append(False)
                else:
                    paths_available.append(True)
            else:
                paths_available.append(False)
        else:
            paths_available.append(True)
    return paths_available


@numba.njit
def find_local_minima(arr: FloatArray1D) -> IntArray1D:
    """Find positions of the local minima in a 1D numpy array.

    Parameters
    ----------
    arr : FloatArray1D
        1D array.

    Returns
    -------
    local_minima_indices : IntArray1D
        Indices of the local minima in the array.
    """
    # Compare each element with its neighbors
    local_minima_mask = (arr[1:-1] < arr[:-2]) & (arr[1:-1] < arr[2:])
    # Get the indices of local minima (adjust by +1 due to slicing)
    local_minima_indices = np.where(local_minima_mask)[0] + 1
    return local_minima_indices


def cumulative_exponential(
    x: FloatArray1D,
    a: float,
    t: float,
    c: float,
) -> FloatArray1D:
    """Used for binding kinetics estimation."""
    return a * (1 - np.exp(-(x / t))) + c


def fit_cum_exp(data: FloatArray1D) -> dict:
    """Fit a cumulative exponential function to data. Used for binding
    kinetics estimation.

    Parameters
    ----------
    data : FloatArray1D
        Input data to fit, shape (N,).

    Returns
    -------
    result : dict
        Contains the best fit parameters and the fitted data.
    """
    data.sort()
    n = len(data)
    y = np.arange(1, n + 1)
    data_min = data.min()
    data_max = data.max()
    p0 = [n, np.mean(data), data_min]
    bounds = ([0, data_min, 0], [np.inf, data_max, np.inf])
    popt, _ = optimize.curve_fit(
        cumulative_exponential, data, y, p0=p0, bounds=bounds
    )
    result = {
        "best_values": {"a": popt[0], "t": popt[1], "c": popt[2]},
        "data": data,
        "best_fit": cumulative_exponential(data, *popt),
    }
    return result


def estimate_kinetic_rate(data: FloatArray1D) -> float:
    """Find the mean dark/bright time by fitting to a cumulative
    exponential function.

    Parameters
    ----------
    data : FloatArray1D
        Input data to fit, shape (N,).

    Returns
    -------
    rate : float
        Mean dark/bright time from the fitted exponential function.
    """
    if len(data) > 2:
        if data.max() - data.min() == 0:
            rate = np.nanmean(data)
        else:
            result = fit_cum_exp(data)
            rate = result["best_values"]["t"]
    else:
        rate = np.nanmean(data)
    return rate


def plot_cumulative_exponential_fit(
    data: SeriesOrFloatArray1D,
    fit_result: dict,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a histogram for experimental data and the fitted cumulative
    exponential function. Used for binding kinetics fit display.

    Parameters
    ----------
    data : SeriesOrFloatArray1D
        Input data to fit, shape (N,). For example, bright or dark
        times.
    fit_result : dict
        Output of `fit_cum_exp` containing the best fit parameters and
        the fitted data.
    fig, ax : plt.Figure and plt.Axes, optional
        If given, the plot will be drawn on the given figure and axes.
        Otherwise, a new figure and axes will be created.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    else:
        ax.clear()

    # Bright
    a = fit_result["best_values"]["a"]
    t = fit_result["best_values"]["t"]
    c = fit_result["best_values"]["c"]

    ax.set_title(
        "Cumulative exponential\n"
        r"$Fit: {:.2f}\cdot(1-exp(-t/{:.2f}))+{:.2f}$".format(a, t, c)
    )
    data = data.copy()
    data.sort_values(inplace=True)
    y = np.arange(1, len(data) + 1)
    ax.semilogx(data, y, label="data")
    ax.semilogx(
        data,
        fit_result["best_fit"],
        label=f"fit ($\\bar \\tau = {t:.2f}$)",
    )
    ax.legend(loc="best")
    ax.set_xlabel("Duration (frames)")
    ax.set_ylabel("Counts")
    return fig


def plot_trace(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    fig: plt.Figure | None = None,
    include_photons: bool = True,
    return_trace: bool = False,
) -> (
    plt.Figure
    | tuple[
        plt.Figure,
        tuple[FloatArray1D, FloatArray1D, FloatArray1D]
        | tuple[FloatArray1D, FloatArray1D],
    ]
):
    """Plot the trace of a localization over time, showing the x and y
    positions and the spot size.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list[dict]
        Additional information for each localization.
    fig : plt.Figure, optional
        If given, the plot will be drawn on the given figure. Otherwise,
        a new figure will be created.
    include_photons : bool, optional
        If True, the photon count will also be plotted as well. Default
        is True.
    return_trace : bool, optional
        If True, the trace data will be returned as well. Default is
        False.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    trace_data : tuple of FloatArray1D, optional
        If return_trace is True, a tuple containing the x vector
        (frames), the y vector (localization ON/OFF) and the photon
        count vector (if include_photons is True) will be returned.
    """
    if fig is None:
        if include_photons:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(
                4, 1, figsize=(5, 5), constrained_layout=True, sharex=True
            )
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(5, 5), constrained_layout=True, sharex=True
            )
    else:
        fig.clear()
        if include_photons:
            ax1, ax2, ax3, ax4 = fig.subplots(4, sharex=True)
        else:
            ax1, ax2, ax3 = fig.subplots(3, sharex=True)

    n_frames = get_from_metadata(info, "Frames", raise_error=True)
    xvec = np.arange(n_frames)
    yvec = xvec[:] * 0
    yvec[locs["frame"]] = 1
    yvec_ph = xvec[:] * 0
    if "photons" in locs.columns:
        yvec_ph[locs["frame"]] = locs["photons"]
    else:
        yvec_ph = np.zeros_like(xvec)
    trace_data = (xvec, yvec, yvec_ph) if include_photons else (xvec, yvec)

    # frame vs x
    ax1.scatter(locs["frame"], locs["x"], s=2)
    ax1.set_title("X-pos vs frame")
    ax1.set_xlim(0, n_frames)
    ax1.set_ylabel("X-pos [Px]")

    # frame vs y
    ax2.scatter(locs["frame"], locs["y"], s=2)
    ax2.set_title("Y-pos vs frame")
    ax2.set_ylabel("Y-pos [Px]")

    # locs in time
    ax3.plot(xvec, yvec, linewidth=1)
    ax3.fill_between(xvec, 0, yvec, facecolor="red")
    ax3.set_title("Localizations")
    ax3.set_xlabel("Frames")
    ax3.set_ylabel("ON")
    ax3.set_yticks([0, 1])
    ax3.set_ylim([-0.1, 1.1])

    if include_photons:
        ax4.plot(xvec, yvec_ph, linewidth=1)
        ax4.set_title("Photons")
        ax4.set_xlabel("Frames")
        ax4.set_ylabel("Photons")
        ax4.set_ylim([0, yvec_ph.max() * 1.1])

    if return_trace:
        return fig, trace_data
    else:
        return fig


def calculate_optimal_bins(
    data: FloatArray1D | IntArray1D,
    max_n_bins: int | None = None,
    sample_size: int = 1_000_000,
) -> FloatArray1D:
    """Calculate the optimal bins for display, for example, in
    Picasso: Filter.

    Parameters
    ----------
    data : FloatArray1D | IntArray1D
        Data to be binned.
    max_n_bins : int | None, optional
        Maximum number of bins.
    sample_size : int, optional
        For large arrays, estimate the IQR from a random sample of this
        size instead of sorting the full array. min/max are still taken
        from the full data (cheap O(N) reductions). Set to a value >=
        ``len(data)`` to disable sampling. Default 1_000_000.

    Returns
    -------
    bins : FloatArray1D
        Bins for display.
    """
    data = np.asarray(data)  # positional indexing below; Series use labels
    n = len(data)
    if n == 0:
        return np.array([0.0, 1.0])
    if data.dtype.kind == "f":
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
    else:
        data_min = data.min()
        data_max = data.max()
    if n > sample_size:
        rng = np.random.default_rng(0)
        sample = data[rng.choice(n, sample_size, replace=False)]
    else:
        sample = data
    if sample.dtype.kind == "f":
        sample = sample[np.isfinite(sample)]
    if len(sample) == 0:
        return np.array([data_min - 1.0, data_max + 1.0])
    iqr = np.subtract(*np.percentile(sample, [75, 25]))
    if iqr == 0:
        return np.array([data[0] - 1.0, data[0] + 1.0])
    bin_size = 2 * iqr * n ** (-1 / 3)
    if data.dtype.kind in ("u", "i") and bin_size < 1:
        bin_size = 1
    bin_min = data_min - bin_size / 2
    try:
        n_bins = (data_max - bin_min) / bin_size
        n_bins = int(n_bins)
    except Exception:
        n_bins = 10
    if max_n_bins and n_bins > max_n_bins:
        n_bins = max_n_bins
    bins = np.linspace(bin_min, data_max, n_bins)
    return bins


@numba.njit(parallel=True, nogil=True)
def hist2d_numba(
    x: FloatArray1D,
    y: FloatArray1D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Fast 2D histogram with uniform bin edges.

    Non-finite points are skipped. Edge values (== x_max / == y_max) are
    folded into the last bin to match the inclusive-right behaviour of
    ``np.histogram2d``.

    Parameters
    ----------
    x, y : FloatArray1D
        Sample coordinates.
    x_min, x_max, y_min, y_max : float
        Outer edges of the histogram.
    nx, ny : int
        Number of bins along each axis.

    Returns
    -------
    counts : np.ndarray, shape (nx, ny), dtype int64
        Bin counts, indexed as counts[ix, iy].
    """
    n_threads = numba.get_num_threads()
    local = np.zeros((n_threads, nx, ny), dtype=np.int64)
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    n = x.shape[0]
    chunk = (n + n_threads - 1) // n_threads
    for t in numba.prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n:
            end = n
        for i in range(start, end):
            xi = x[i]
            yi = y[i]
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            ix = int((xi - x_min) / dx)
            iy = int((yi - y_min) / dy)
            if ix == nx:
                ix -= 1
            if iy == ny:
                iy -= 1
            if 0 <= ix < nx and 0 <= iy < ny:
                local[t, ix, iy] += 1
    return local.sum(axis=0)


def append_to_rec(
    rec_array: np.recarray,
    data: FloatArray1D | IntArray1D,
    name: str,
) -> np.recarray:
    """Append a new column to the existing np.recarray.

    Parameters
    ----------
    rec_array : np.recarray
        Recarray to which the new column is appended.
    data : FloatArray1D | IntArray1D
        1D data to be appended.
    name : str
        Name of the new column.

    Returns
    -------
    rec_array : np.recarray
        Recarray with the new column.
    """
    deprecation_warning(
        "Appending to recarrays is deprecated and will be removed in Picasso"
        " 1.0. Since 0.9.0, Picasso uses pandas DataFrames instead of"
        " recarrays. Simply use locs['new_column'] = data to add a new column"
        " to the DataFrame."
    )
    if hasattr(rec_array, name):
        rec_array = remove_from_rec(rec_array, name)
    rec_array = append_fields(
        rec_array,
        name,
        data,
        dtypes=data.dtype,
        usemask=False,
        asrecarray=True,
    )
    return rec_array


def merge_locs(
    locs_list: list[pd.DataFrame],
    increment_frames: bool | list[int] = True,
    increment_groups: bool | list[int] = True,
) -> pd.DataFrame:
    """Merge localization lists into one file. Can increment frames
    to avoid overlapping frames.

    Parameters
    ----------
    locs_list : list of pd.DataFrame's
        List of localization lists to be merged.
    increment_frames : bool or list, optional
        If True, increments frames of each localization list by the
        maximum frame number of the previous localization list. If a
        list is given, each element is an integer increment of the frame
        indices for each localization list. Default is True.
    increment_groups : bool or list, optional
        If True, increments group indices of each localization list by
        the maximum group number of the previous localization list. If a
        list is given, each element is an integer increment of the group
        indices for each localization list. Default is True.

    Returns
    -------
    locs : pd.DataFrame
        Merged localizations.
    """
    assert isinstance(
        increment_frames, (bool, list)
    ), "increment_frames must be a boolean or a list of integers."
    assert isinstance(
        increment_groups, (bool, list)
    ), "increment_groups must be a boolean or a list of integers."
    if isinstance(increment_frames, list):
        assert len(increment_frames) == len(locs_list), (
            "If increment_frames is a list, its length must be the same"
            " as locs_list."
        )
        assert all(isinstance(i, int) for i in increment_frames), (
            "If increment_frames is a list, all its elements must be "
            "integers."
        )
    if isinstance(increment_groups, list):
        assert len(increment_groups) == len(locs_list), (
            "If increment_groups is a list, its length must be the same"
            " as locs_list."
        )
        assert all(isinstance(i, int) for i in increment_groups), (
            "If increment_groups is a list, all its elements must be "
            "integers."
        )
    # convert boolean increments to lists of integers
    if increment_frames is True:
        increment_frames = np.cumsum(
            [0] + [locs["frame"].max() for locs in locs_list[:-1]]
        ).tolist()
    else:
        increment_frames = [0] * len(locs_list)
    if increment_groups is True:
        increment_groups = np.cumsum(
            [0] + [locs["group"].max() for locs in locs_list[:-1]]
        ).tolist()
    else:
        increment_groups = [0] * len(locs_list)
    return _merge_locs(locs_list, increment_frames, increment_groups)


def _merge_locs(
    locs_list: list[pd.DataFrame],
    increment_frames: list[int],
    increment_groups: list[int],
) -> pd.DataFrame:
    """Helper function for merge_locs. Assumes correct input types and
    values."""
    locs_list = locs_list.copy()
    for i, locs in enumerate(locs_list):
        locs["frame"] += increment_frames[i]
        if "group" in locs.columns:
            locs["group"] += increment_groups[i]
        locs_list[i] = locs
    locs = pd.concat(locs_list, ignore_index=True)
    locs.sort_values(by="frame", inplace=True)
    return locs


def append_group(
    locs: pd.DataFrame,
    group: int | SeriesOrIntArray1D,
) -> pd.DataFrame:
    """Assign the ``group`` column of ``locs``, preserving any existing
    grouping. If ``group`` already exists, its contents are moved to
    ``group_input`` to preserve information.

    The DataFrame is modified in place and returned for convenience.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to assign the group column to.
    group : int or array-like of int
        New group id(s). A scalar is broadcast to all localizations; an
        array-like must have the same length as ``locs`` and is assigned
        positionally.

    Returns
    -------
    locs : pd.DataFrame
        Localizations with the ``group`` column set. If a ``group``
        column was already present, its previous values are stored in
        the ``"group_input"`` column.
    """
    if "group" in locs.columns:
        locs["group_input"] = locs["group"].to_numpy()
    locs["group"] = group if np.isscalar(group) else np.asarray(group)
    return locs


def ensure_sanity(locs: pd.DataFrame, info: list[dict]) -> pd.DataFrame:
    """Ensure that localizations are within the image dimensions
    and have positive localization precisions and other parameters.

    v0.9.6: check that the info metadata contains the necessary
    information for processing: Width, Height, Pixelsize and Frames.
    Raises a KeyError if any of the required keys is missing.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Localization metadata.

    Returns
    -------
    locs : pd.DataFrame
        Localizations that pass the sanity checks.
    """
    locs = locs.copy()  # pandas SettingWithCopyWarning
    # no inf and nan:
    locs.replace([np.inf, -np.inf], np.nan, inplace=True)
    locs.dropna(axis=0, how="any", inplace=True)
    # other sanity checks:
    required_keys = ["Width", "Height", "Frames"]
    for key in required_keys:
        value = get_from_metadata(info, key)
        if value is None:
            raise KeyError(f"Metadata is missing required key: '{key}'")

    locs = locs[locs["x"] < get_from_metadata(info, "Width")]
    locs = locs[locs["y"] < get_from_metadata(info, "Height")]
    for attr in [
        "x",
        "y",
        "lpx",
        "lpy",
        "lpz",
        "photons",
        "ellipticity",
        "sx",
        "sy",
    ]:
        if attr in locs.columns:
            locs = locs[locs[attr] >= 0]
    return locs


def is_loc_at(x: float, y: float, locs: pd.DataFrame, r: float) -> BoolArray1D:
    """Check which localizations are within radius ``r`` from position
    ``(x, y)``.

    Parameters
    ----------
    x, y : float
        x and y-coordinate of the position.
    locs : pd.DataFrame
        Localizations.
    r : float
        Radius.

    Returns
    -------
    is_picked : BoolArray1D
        Boolean array - True if a localization is within radius r
        of position (x, y).
    """
    dx = locs["x"] - x
    dy = locs["y"] - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return is_picked.to_numpy()


def locs_at(x: float, y: float, locs: pd.DataFrame, r: float) -> pd.DataFrame:
    """Return localizations within radius ``r`` from the position
    ``(x, y)``.

    Parameters
    ----------
    x, y : float
        x and y-coordinate of the position.
    locs : pd.DataFrame
        Localizations.
    r : float
        Radius.

    Returns
    -------
    picked_locs : pd.DataFrame
        Localizations in the specified area.
    """
    is_picked = is_loc_at(x, y, locs, r)
    picked_locs = locs[is_picked]
    return picked_locs


@numba.jit(nopython=True, nogil=True, cache=True)
def is_loc_at_numba(
    x: float,
    y: float,
    locs_xy: FloatArray2D,
    r: float,
) -> BoolArray1D:
    """Numba implementation of ``locs_at``. Return the indices of
    localizations at the given coordinates within radius ``r``."""
    dx = locs_xy[0] - x
    dy = locs_xy[1] - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return is_picked


@numba.jit(nopython=True, nogil=True, cache=True)
def locs_at_numba(
    x: float,
    y: float,
    locs_xy: FloatArray2D,
    r: float,
) -> FloatArray2D:
    """Numba implementation of ``locs_at``. Return the localizations at
    the given coordinates within radius ``r``."""
    is_picked = is_loc_at_numba(x, y, locs_xy, r)
    return locs_xy[:, is_picked]


@numba.jit(nopython=True, nogil=True)
def rmsd_at_com(locs_xy: FloatArray2D) -> float:
    """Calculate the RMSD of the localizations at the center of mass
    (COM) of the localizations."""
    com_x = np.mean(locs_xy[0])
    com_y = np.mean(locs_xy[1])
    return np.sqrt(
        np.mean((locs_xy[0] - com_x) ** 2 + (locs_xy[1] - com_y) ** 2)
    )


@numba.jit(nopython=True)
def check_if_in_polygon(
    x: FloatArray1D,
    y: FloatArray1D,
    X: FloatArray1D,
    Y: FloatArray1D,
) -> BoolArray1D:
    """Check if points ``(x, y)`` are within the polygon defined by
    corners ``(X, Y)``. Uses the ray casting algorithm, see
    ``check_if_in_rectangle`` for details.

    Parameters
    ----------
    x, y : FloatArray1D
        x and y coordinates of points.
    X, Y : FloatArray1D
        x and y coordinates of polygon corners.

    Returns
    -------
    is_in_polygon : BoolArray1D
        Boolean array indicating which points are in the polygon.
    """
    n_locs = len(x)
    n_polygon = len(X)
    is_in_polygon = np.zeros(n_locs, dtype=np.bool_)

    for i in range(n_locs):
        count = 0
        for j in range(n_polygon):
            j_next = (j + 1) % n_polygon
            if ((Y[j] > y[i]) != (Y[j_next] > y[i])) and (
                (
                    x[i]
                    < X[j]
                    + (X[j_next] - X[j]) * (y[i] - Y[j]) / (Y[j_next] - Y[j])
                )
            ):
                count += 1
        if count % 2 == 1:
            is_in_polygon[i] = True

    return is_in_polygon


def locs_in_polygon(
    locs: pd.DataFrame,
    X: FloatArray1D,
    Y: FloatArray1D,
) -> pd.DataFrame:
    """Return localizations within the polygon defined by corners
    ``(X, Y)``.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    X, Y : FloatArray1D
        x and y-coordinates of polygon corners.

    Returns
    -------
    picked_locs : pd.DataFrame
        Localizations in polygon.
    """
    is_in_polygon = check_if_in_polygon(
        locs["x"].to_numpy(), locs["y"].to_numpy(), np.array(X), np.array(Y)
    )
    return locs[is_in_polygon]


@numba.jit(nopython=True)
def check_if_in_rectangle(
    x: FloatArray1D,
    y: FloatArray1D,
    X: FloatArray1D,
    Y: FloatArray1D,
) -> BoolArray1D:
    """Check if locs with coordinates (x, y) are in rectangle with
    corners (X, Y) by counting the number of rectangle sides which are
    hit by a ray originating from each loc to the right. If the number
    of hit rectangle sides is odd, then the loc is in the rectangle.

    Parameters
    ----------
    x, y : FloatArray1D
        x and y coordinates of points.
    X, Y : FloatArray1D
        x and y coordinates of polygon corners.

    Returns
    -------
    is_in_polygon : BoolArray1D
        Boolean array indicating if point is in polygon.
    """
    n_locs = len(x)
    ray_hits_rectangle_side = np.zeros((n_locs, 4))
    for i in range(4):
        # get two y coordinates of corner points forming one rectangle side
        y_corner_1 = Y[i]
        # take the first if we're at the last side:
        y_corner_2 = Y[0] if i == 3 else Y[i + 1]
        y_corners_min = min(y_corner_1, y_corner_2)
        y_corners_max = max(y_corner_1, y_corner_2)
        for j in range(n_locs):
            y_loc = y[j]
            # only if loc is on level of rectangle side, its ray can hit:
            if y_corners_min <= y_loc <= y_corners_max:
                x_corner_1 = X[i]
                # take the first if we're at the last side:
                x_corner_2 = X[0] if i == 3 else X[i + 1]
                # calculate intersection point of ray and side:
                m_inv = (x_corner_2 - x_corner_1) / (y_corner_2 - y_corner_1)
                x_intersect = m_inv * (y_loc - y_corner_1) + x_corner_1
                x_loc = x[j]
                if x_intersect >= x_loc:
                    # ray hits rectangle side on the right side
                    ray_hits_rectangle_side[j, i] = 1
    n_sides_hit = np.sum(ray_hits_rectangle_side, axis=1)
    is_in_rectangle = n_sides_hit % 2 == 1
    return is_in_rectangle


def locs_in_rectangle(
    locs: pd.DataFrame,
    X: FloatArray1D,
    Y: FloatArray1D,
) -> pd.DataFrame:
    """Return localizations within the rectangle defined by corners
    ``(X, Y)``.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    X, Y : FloatArray1D
        x and y coordinates of rectangle corners.

    Returns
    -------
    picked_locs : pd.DataFrame
        Localizations in rectangle.
    """
    is_in_rectangle = check_if_in_rectangle(
        locs["x"].to_numpy(), locs["y"].to_numpy(), np.array(X), np.array(Y)
    )
    picked_locs = locs[is_in_rectangle]
    return picked_locs


def minimize_shifts(
    shifts_x: FloatArray2D,
    shifts_y: FloatArray2D,
    shifts_z: FloatArray2D | None = None,
) -> tuple[FloatArray1D, FloatArray1D, FloatArray1D | None]:
    """Minimize shifts in x, y, and z directions. Used for drift
    correction.

    Parameters
    ----------
    shifts_x, shifts_y : FloatArray2D
        Shifts in x and y directions, shape (n_channels, n_channels).
    shifts_z : FloatArray2D, optional
        Shifts in z direction, shape (n_channels, n_channels). If None,
        only x and y shifts are minimized.

    Returns
    -------
    shift_y, shift_x : FloatArray1D
        Minimized shifts in y and x direction.
    shift_z : FloatArray1D, optional
        Minimized shifts in z direction if ``shifts_z`` is specified.
    """
    n_channels = shifts_x.shape[0]
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    n_dims = 2 if shifts_z is None else 3
    rij = np.zeros((n_pairs, n_dims))
    A = np.zeros((n_pairs, n_channels - 1))
    flag = 0
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            rij[flag, 0] = shifts_y[i, j]
            rij[flag, 1] = shifts_x[i, j]
            if n_dims == 3:
                rij[flag, 2] = shifts_z[i, j]
            A[flag, i:j] = 1
            flag += 1
    Dj = np.dot(np.linalg.pinv(A), rij)
    shift_y = np.insert(np.cumsum(Dj[:, 0]), 0, 0)
    shift_x = np.insert(np.cumsum(Dj[:, 1]), 0, 0)
    if n_dims == 2:
        return shift_y, shift_x
    else:
        shift_z = np.insert(np.cumsum(Dj[:, 2]), 0, 0)
        return shift_y, shift_x, shift_z


def n_futures_done(futures: list[Future]) -> int:
    """Return the number of finished futures, used in
    multiprocessing."""
    return sum([_.done() for _ in futures])


def remove_from_rec(rec_array: np.recarray, name: str) -> np.recarray:
    """Remove a column from the existing recarray.

    Parameters
    ----------
    rec_array : np.recarray
        Recarray from which the column is removed.
    name : str
        Name of the column to be removed.

    Returns
    -------
    rec_array : np.recarray
        Recarray without the column.
    """
    deprecation_warning(
        "Removing columns from recarrays is deprecated and will be removed in "
        " Picasso 1.0. Since 0.9.0, Picasso uses pandas DataFrames instead of"
        " recarrays. Simply use locs.drop('new_column', axis=1) to remove a"
        " column from the DataFrame."
    )
    rec_array = drop_fields(rec_array, name, usemask=False, asrecarray=True)
    return rec_array


def locs_glob_map(
    func: Callable[
        [pd.DataFrame, dict, str, Any], tuple[pd.DataFrame, list[dict]]
    ],
    pattern: str,
    args: list = [],
    kwargs: dict = {},
    extension: str = "",
) -> None:
    """Map a function to localization files, specified by the unix style
    path pattern.

    The function must take two arguments: ``locs`` and ``info``. It may
    take additional args and kwargs which are supplied to this map
    function. A new locs file will be saved if an extension is provided.
    In that case, the mapped function must return new locs and a new
    info dict.

    Parameters
    ----------
    func : Callable
        Function to be mapped to each locs file. It must take
        locs, info, path, and any additional args and kwargs.
    pattern : str
        Unix style path pattern to match locs files.
    args : list, optional
        Additional positional arguments to be passed to the function.
    kwargs : dict, optional
        Additional keyword arguments to be passed to the function.
    extension : str, optional
        If provided, the mapped function must return new locs and info
        dict, and a new locs file will be saved with this extension.
        If not provided, the function is expected to modify locs and
        info in place.
    """
    paths = glob.glob(pattern)
    for path in paths:
        locs, info = io.load_locs(path)
        result = func(locs, info, path, *args, **kwargs)
        if extension:
            base, ext = os.path.splitext(path)
            out_path = base + "_" + extension + ".hdf5"
            locs, info = result
            io.save_locs(out_path, locs, info)


def get_pick_polygon_corners(
    pick: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Return X and Y coordinates of a pick polygon.
    Return (None, None) if the pick is not a closed polygon.

    Parameters
    ----------
    pick : list of tuples
        List of tuples, each tuple contains x and y coordinates of a
        polygon corner.

    Returns
    -------
    X, Y : list of floats
        Lists of x and y coordinates of the polygon corners.
        Return (None, None) if the pick is not a closed polygon.
    """
    if len(pick) < 3 or pick[0] != pick[-1]:
        return None, None
    else:
        X = [_[0] for _ in pick]
        Y = [_[1] for _ in pick]
        return X, Y


def get_pick_rectangle_corners(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width: float,
) -> tuple[list[float], list[float]]:
    """Find the positions of corners of a rectangular pick.
    A rectangular pick is defined by:
        [(start_x, start_y), (end_x, end_y)]
    and its width. (all values in camera pixels).

    Parameters
    ----------
    start_x, start_y : float
        Starting point of the pick.
    end_x, end_y : float
        Ending point of the pick.
    width : float
        Width of the pick in camera pixels.

    Returns
    -------
    corners : tuple
        Contains corners' x and y coordinates in two lists.
    """
    if end_x == start_x:
        alpha = np.pi / 2
    else:
        alpha = np.arctan((end_y - start_y) / (end_x - start_x))
    dx = width * np.sin(alpha) / 2
    dy = width * np.cos(alpha) / 2
    x1 = float(start_x - dx)
    x2 = float(start_x + dx)
    x4 = float(end_x - dx)
    x3 = float(end_x + dx)
    y1 = float(start_y + dy)
    y2 = float(start_y - dy)
    y4 = float(end_y + dy)
    y3 = float(end_y - dy)
    corners = ([x1, x2, x3, x4], [y1, y2, y3, y4])
    return corners


def polygon_area(X: FloatArray1D, Y: FloatArray1D) -> float:
    """Find the area of a polygon defined by corners X and Y.

    Parameters
    ----------
    X, Y : FloatArray1D
        x-coordinates and y-coordinates of the polygon corners.

    Returns
    -------
    area : float
        Area of the polygon.
    """
    n_corners = len(X)
    area = 0
    for i in range(n_corners):
        j = (i + 1) % n_corners  # next corner
        area += X[i] * Y[j] - X[j] * Y[i]
    area = abs(area) / 2
    return area


def _pick_areas_polygon(
    picks: list[list[tuple[float, float]]],
) -> FloatArray1D:
    """Return pick areas for each polygonal pick in picks.

    Parameters
    ----------
    picks : list of lists of tuples
        List of picks, each pick is a list of (x, y) coordinates of the
        polygon corners.

    Returns
    -------
    areas : FloatArray1D
        Pick areas.
    """
    areas = []
    for i, pick in enumerate(picks):
        if len(pick) < 3 or pick[0] != pick[-1]:  # not a closed polygon
            continue
        X, Y = get_pick_polygon_corners(pick)
        areas.append(polygon_area(X, Y))
    areas = np.array(areas)
    areas = areas[areas > 0]  # remove open polygons
    return areas


def _pick_areas_rectangle(
    picks: list[list[tuple[float, float]]],
    w: float,
) -> FloatArray1D:
    """Return pick areas for each pick in picks.

    Parameters
    ----------
    picks : list
        List of picks, each pick is a list of coordinates of the
        rectangle corners.
    w : float
        Pick width.

    Returns
    -------
    areas : FloatArray1D
        Pick areas, same units as ``w``.
    """
    areas = np.zeros(len(picks))
    for i, pick in enumerate(picks):
        (xs, ys), (xe, ye) = pick
        areas[i] = w * np.sqrt((xe - xs) ** 2 + (ye - ys) ** 2)
    return areas


def pick_areas(
    picks: list[tuple],
    pick_shape: Literal["Circle", "Rectangle", "Polygon", "Square"],
    pick_size: float | None,
) -> FloatArray1D:
    """Get pick areas for each pick in picks.

    Parameters
    ----------
    picks : list of tuples
        Coordinates of picks in camera pixels.
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square"}
        Shape of picks.
    pick_size : float or None
        Size of picks in camera pixels. For circles - diameters. For
        rectangles - width. For squares - side length. For polygons -
        ignored.

    Returns
    -------
    areas : FloatArray1D
        Pick areas in camera pixels squared.
    """
    if pick_shape == "Circle":
        r = pick_size / 2
        # same area for all picks
        areas = np.pi * r**2 * np.ones(len(picks))
    elif pick_shape == "Rectangle":
        areas = _pick_areas_rectangle(picks, pick_size)
    elif pick_shape == "Polygon":
        areas = _pick_areas_polygon(picks)
    elif pick_shape == "Square":
        # same area for all picks
        areas = pick_size**2 * np.ones(len(picks))
    else:
        raise ValueError(f"Unknown pick shape: {pick_shape}")
    return areas


def permutation_test(
    arr1: FloatArray1D, arr2: FloatArray1D, iterations: int = 1000
) -> tuple[float, float, float]:
    """Perform a permutation test to compare two arrays. The test
    statistic is the Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    arr1, arr2 : FloatArray1D
        Arrays to be compared.
    iterations : int, optional
        Number of permutations to perform. Default is 1000.

    Returns
    -------
    obs_d : float
        Observed KS statistic.
    p_perm : float
        Permutation p-value.
    ks_pval : float
        KS test theoretical p-value.
    """
    combined = np.concatenate([arr1, arr2])
    n1 = len(arr1)

    # observe the real difference
    obs_d, ks_pval = stats.ks_2samp(arr1, arr2)

    # build null distribution by shuffling
    null_dist = []
    for _ in range(iterations):
        shuffled = np.random.permutation(combined)
        d_perm, _ = stats.ks_2samp(shuffled[:n1], shuffled[n1:])
        null_dist.append(d_perm)

    p_perm = np.sum(np.array(null_dist) >= obs_d) / iterations
    return obs_d, p_perm, ks_pval


def plot_subclustering_check(
    clustered_n_events: IntArray1D,
    sparse_n_events: IntArray1D,
    plot_path: str | list[str] = "",
    return_fig: bool = False,
    clustering_dist: float | None = None,
    sparse_dist: float | None = None,
) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
    """Plot the results of subclustering analysis, see
    ``picasso.clusterer.test_subclustering``.

    Parameters
    ----------
    clustered_n_events : IntArray1D
        Number of events for clustered molecules.
    sparse_n_eveents : IntArray1D
        Number of events for sparse molecules.
    plot_path : str or list of strs, optional
        If provided, the plot is saved to this path. If a list of
        strings is given, each is used to save a separate plot. Default
        is "".
    return_fig : bool, optional
        If True, the figure and axes are returned. Default is False.
    clustering_dist, sparse_dist : float, optional
        Clustering and sparse distances that are displayed in the
        legend. If None, distances are not displayed. Default is None.

    Returns
    -------
    fig, ax : (plt.Figure, plt.Axes) or (None, None)
        Figure and axes if ``return_fig`` is True, otherwise
        (None, None).
    """
    has_clustered = len(clustered_n_events) > 0
    has_sparse = len(sparse_n_events) > 0

    m_clustered = clustered_n_events.mean()
    m_sparse = sparse_n_events.mean()
    s_clustered = clustered_n_events.std()
    s_sparse = sparse_n_events.std()

    # create the plot
    fig, ax1 = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
    if has_clustered or has_sparse:
        all_events = np.concatenate((sparse_n_events, clustered_n_events))
        min_bin, max_bin = np.percentile(all_events, [2.5, 97.5])

    if has_clustered:
        vals, counts = np.unique(clustered_n_events, return_counts=True)
        if clustering_dist is not None:
            label = (
                f"Clustered (d < {clustering_dist:.1f} nm) "
                f"{m_clustered:.1f} +/- {s_clustered:.1f}"
            )
        else:
            label = f"Clustered {m_clustered:.1f} +/- {s_clustered:.1f}"
        ax1.bar(
            vals,
            counts,
            width=0.8,
            alpha=0.5,
            label=label,
            color="C0",
        )
        ax1.axvline(m_clustered, color="C0", linestyle="--")

    if has_sparse:
        vals, counts = np.unique(sparse_n_events, return_counts=True)
        if sparse_dist is not None:
            label = (
                f"Sparse (d > {sparse_dist:.1f} nm) "
                f"{m_sparse:.1f} +/- {s_sparse:.1f}"
            )
        else:
            label = f"Sparse {m_sparse:.1f} +/- {s_sparse:.1f}"
        ax1.bar(
            vals,
            counts,
            width=0.8,
            alpha=0.5,
            label=label,
            color="C1",
        )
        ax1.axvline(m_sparse, color="C1", linestyle="--")

    if has_clustered or has_sparse:
        ax1.set_xlabel("Number of events")
        ax1.set_ylabel("Counts")
        ax1.set_xlim(min_bin - 1, max_bin + 1)
        ax1.legend()

    if has_clustered and has_sparse:
        stat, p_perm, p = permutation_test(clustered_n_events, sparse_n_events)
        p_value_str = r"$p_{value}$"
        title = (
            f"KS test: stat={stat:.4f}\n"
            f"permutation {p_value_str}={p_perm:.4f}\n"
            f"theoretical {p_value_str}={p:.4f}"
        )
    elif has_clustered or has_sparse:
        title = (
            "Only one population found, no statistical test performed; "
            "adjust distance parameters."
        )
    else:
        title = (
            "No molecules found in either population, adjust distance"
            " parameters."
        )
    ax1.set_title(title, fontsize=10)
    if len(plot_path):
        if isinstance(plot_path, str):
            plot_path = [plot_path]
        for path in plot_path:
            fig.savefig(path, dpi=300)

    if return_fig:
        return fig, ax1
    else:
        plt.close(fig)
        return None, None


def plot_rel_sigma_check(
    mols: pd.DataFrame, info: list[dict], path: str
) -> None:
    """Plot the relative sigma of G5M molecules to inspect if lp values
    reflect the experimental sizes of localization clouds.

    Parameters
    ----------
    mols : pd.DataFrame
        Molecules to be plotted, output of ``picasso.g5m.g5m``.
    info : list of dicts
        Molecuels metadata.
    path : str
        Path to save the plot.
    """
    if "z" in mols.columns:
        # three plots, one for each dimension
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), constrained_layout=True)
        bins = calculate_optimal_bins(
            np.concatenate(
                (mols["rel_sigma_x"], mols["rel_sigma_y"], mols["rel_sigma_z"])
            )
        )
        for i, dim in enumerate(["x", "y", "z"]):
            ax = axes[i]
            ax.hist(
                mols[f"rel_sigma_{dim}"], bins=bins, color=f"C{i}", alpha=0.7
            )
            ax.set_xlabel(f"Relative sigma {dim}")
            ax.set_ylabel("Counts")
        fig.savefig(path, dpi=300)
        plt.close(fig)
    else:
        # only one plot
        fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
        bins = calculate_optimal_bins(mols["rel_sigma"])
        ax.hist(mols["rel_sigma"], bins=bins, color="C0", alpha=0.7)
        ax.set_xlabel("Relative sigma")
        ax.set_ylabel("Counts")
        fig.savefig(path, dpi=300)
        plt.close(fig)


def unfold_localizations_square(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    n_square: int = 10,
    spacing: int | float = 1,
):
    """Shift localizations onto a square grid (tile) based on their
    group indices. The localizations must contain a 'group' column.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be unfolded. Must contain a 'group' column.
    info : list of dicts
        Localization metadata.
    n_square : int, optional
        Number of groups per square side. Default is 10.
    spacing : int or float, optional
        Spacing between groups in camera pixels. Default is 1.

    Returns
    -------
    shifted_locs : pd.DataFrame
        Localizations shifted onto a square grid based on their group
        indices.
    updated_info : list of dicts
        Updated metadata with new FOV dimensions after unfolding.
    """
    assert (
        "group" in locs.columns
    ), "Localizations must contain a 'group' column."
    # ensure groups are consecutive integers starting from 0
    locs = locs.copy()  # pandas SettingWithCopyWarning
    updated_info = deepcopy(info)
    unique_groups = np.unique(locs["group"])
    group_mapping = {old: new for new, old in enumerate(unique_groups)}
    locs["group"] = locs["group"].map(group_mapping)

    # shift localizations to the middle of the FOV and by the COM
    # of each group
    cx = get_from_metadata(updated_info, "Width", raise_error=True) / 2
    cy = get_from_metadata(updated_info, "Height", raise_error=True) / 2
    for group_id in np.unique(locs["group"]):
        mask = locs["group"] == group_id
        mean_x = locs.loc[mask, "x"].mean()
        mean_y = locs.loc[mask, "y"].mean()
        locs.loc[mask, "x"] += cx - mean_x
        locs.loc[mask, "y"] += cy - mean_y

    # unfold onto grid
    locs["x"] += np.mod(locs["group"], n_square) * spacing
    locs["y"] += np.floor(locs["group"] / n_square) * spacing

    locs["x"] -= locs["x"].mean()
    locs["y"] -= locs["y"].mean()
    locs["x"] += np.absolute(locs["x"].min())
    locs["y"] += np.absolute(locs["y"].min())

    # Update FOV and clean up
    updated_info = overwrite_metadata(
        updated_info, "Width", int(np.ceil(locs["x"].max()))
    )
    updated_info = overwrite_metadata(
        updated_info, "Height", int(np.ceil(locs["y"].max()))
    )
    return locs, updated_info


def sync_groups(locs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Sync group indices across multiple localization lists. Can be
    used, for example, for removing clustered localizations after
    the cluster centers were filtered.

    Parameters
    ----------
    locs : list of pd.DataFrame
        List of localization lists to be synced. Each must contain a
        'group' column.

    Returns
    -------
    synced_locs : list of pd.DataFrame
        List of localization lists with synced group indices.
    """
    assert all(
        "group" in loc.columns for loc in locs
    ), "All localization lists must contain a 'group' column."
    unique_groups = [np.unique(loc["group"]) for loc in locs]
    common_groups = set(unique_groups[0]).intersection(*unique_groups)
    for i in range(len(locs)):
        mask = locs[i]["group"].isin(common_groups)
        locs[i] = locs[i][mask].reset_index(drop=True)
    return locs


# ---------------------------------------------------------------------------
# Affine (x, y) corrections chained onto any calibration
# ---------------------------------------------------------------------------
# A calibration - Gaussian astigmatism (YAML), cubic-spline PSF (HDF5) or a
# standalone affine calibration - may carry an ORDERED list of lateral affine
# corrections under AFFINE_TRANSFORMS_KEY. Each entry maps the coordinates of
# the movie being localized into a reference frame:
#
#   astigmatism : cylindrical-lens image -> reference (no-lens) image
#   chromatic   : this color channel     -> reference color channel
#
# They compose: a 3D two-color experiment applies the astigmatism correction
# and then the chromatic one, in the order they appear in the list.
#
# Single-channel only

AFFINE_TRANSFORMS_KEY = "Affine transforms"
# v0.11 wrote a single astigmatism transform under this key; still read.
LEGACY_AFFINE_TRANSFORM_KEY = "Affine transform"
AFFINE_TRANSFORM_TYPES = ("astigmatism", "chromatic")


def affine_transforms(calibration: dict | list | None) -> list[dict]:
    """The ordered affine-correction entries carried by a calibration.

    Parameters
    ----------
    calibration : dict or list or None
        Any calibration dictionary (Gaussian astigmatism, spline PSF or a
        standalone affine calibration), an already-extracted list of
        entries, or None.

    Returns
    -------
    transforms : list of dicts
        The entries in the order they must be applied. Empty if the
        calibration carries none. A calibration written before the ordered
        list existed (single ``"Affine transform"`` key) yields that one
        entry.
    """
    if calibration is None:
        return []
    if isinstance(calibration, list):
        return [t for t in calibration if t]
    if not isinstance(calibration, dict):
        # e.g. a calibration still given as a path; nothing to read
        return []
    transforms = calibration.get(AFFINE_TRANSFORMS_KEY)
    if transforms:
        return list(transforms)
    legacy = calibration.get(LEGACY_AFFINE_TRANSFORM_KEY)
    if legacy:
        return [legacy]
    return []


def affine_matrices(calibration: dict | list | None) -> list[np.ndarray]:
    """The ``(3, 3)`` homogeneous matrices of ``affine_transforms``, in the
    order they must be applied. Accepts entries or bare matrices."""
    matrices = []
    for transform in affine_transforms(calibration):
        if isinstance(transform, dict):
            transform = transform["Matrix"]
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Invalid affine transform of shape {matrix.shape}; "
                "expected a (3, 3) homogeneous matrix."
            )
        matrices.append(matrix)
    return matrices


def append_affine_transform(calibration: dict, entry: dict) -> dict:
    """Append an affine correction to ``calibration``'s ordered list.

    An existing entry of the same ``"Type"`` is replaced in place (keeping
    its position), so re-running a calibration updates it instead of
    stacking a second copy of the same correction. A legacy single-key
    transform is migrated into the list first.

    Parameters
    ----------
    calibration : dict
        Calibration to append to; modified in place. May be empty, which
        is how a standalone affine calibration file is started.
    entry : dict
        The transform entry, as built by
        ``picasso.localize.fit_affine_transform``. Must carry ``"Matrix"``
        and ``"Type"``.

    Returns
    -------
    calibration : dict
        The same dictionary, with the entry appended or replaced.
    """
    transforms = affine_transforms(calibration)
    calibration.pop(LEGACY_AFFINE_TRANSFORM_KEY, None)
    kind = entry.get("Type")
    for i, existing in enumerate(transforms):
        if existing.get("Type") == kind:
            transforms[i] = entry
            break
    else:
        transforms.append(entry)
    calibration[AFFINE_TRANSFORMS_KEY] = transforms
    return calibration


def apply_affine_transforms(
    locs: pd.DataFrame, calibration: dict | list | None
) -> pd.DataFrame:
    """Map ``locs`` x/y through a calibration's affine corrections.

    The transforms are applied in stored order, in camera-pixel
    coordinates. Returns ``locs`` unchanged (and untouched) when there is
    nothing to apply, so this is safe to call on every fit path.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with ``x`` and ``y`` columns.
    calibration : dict or list or None
        A calibration carrying affine corrections, a list of entries, or
        None.

    Returns
    -------
    locs : pd.DataFrame
        Localizations with corrected ``x`` and ``y``. A copy, if anything
        was applied.
    """
    matrices = affine_matrices(calibration)
    if not matrices or not len(locs):
        return locs
    locs = locs.copy()
    x = locs["x"].to_numpy(dtype=np.float64)
    y = locs["y"].to_numpy(dtype=np.float64)
    for matrix in matrices:
        x, y = (
            matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2],
            matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2],
        )
    locs["x"] = x.astype(np.float32)
    locs["y"] = y.astype(np.float32)
    return locs


def is_same_affine_transform(
    a: dict | list, b: dict | list, tol: float = 1e-9
) -> bool:
    """Whether two affine entries hold the same matrix (within ``tol``).

    Compared by matrix, not by identity or source path: the same correction
    saved twice under different names must count as one, since applying it
    twice would correct twice.
    """
    if isinstance(a, dict):
        a = a["Matrix"]
    if isinstance(b, dict):
        b = b["Matrix"]
    return bool(
        np.allclose(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
            atol=tol,
            rtol=0.0,
        )
    )


def drop_duplicate_affine_transforms(
    transforms: dict | list | None, applied: dict | list | None
) -> tuple[list[dict], list[dict]]:
    """Split ``transforms`` into the ones still to apply and the ones
    ``applied`` already covers.

    A calibration carries its own corrections and applies them itself, so an
    extra correction loaded alongside it - typically the very same file, or a
    standalone copy of the same transform - must not be applied a second
    time.

    Parameters
    ----------
    transforms : dict or list or None
        The extra corrections, as a calibration or a list of entries.
    applied : dict or list or None
        Corrections the fit applies on its own, i.e. those carried by its
        calibration.

    Returns
    -------
    new : list of dicts
        Entries of ``transforms`` not already in ``applied``, in order.
    duplicates : list of dicts
        Entries dropped because ``applied`` already covers them.
    """
    already = affine_transforms(applied)
    new, duplicates = [], []
    for transform in affine_transforms(transforms):
        if any(is_same_affine_transform(transform, a) for a in already):
            duplicates.append(transform)
        else:
            new.append(transform)
    return new, duplicates


def describe_affine_transforms(calibration: dict | list | None) -> list[str]:
    """One human-readable line per affine correction, for metadata and
    GUI labels, e.g. ``"astigmatism (25 bead pairs)"``."""
    described = []
    for transform in affine_transforms(calibration):
        kind = transform.get("Type", "affine")
        pairs = transform.get("Bead pairs")
        described.append(
            f"{kind} ({pairs} bead pairs)" if pairs else str(kind)
        )
    return described
