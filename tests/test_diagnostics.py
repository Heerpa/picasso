"""Test ``picasso.diagnostics``, the error plumbing of the one-click
(windowed, console-less) builds.

Simulates what PyInstaller does to a windowed process - setting
``sys.stdout`` and ``sys.stderr`` to None - and checks that Picasso
repairs the streams, that ``tqdm`` survives it, and that uncaught
exceptions (main thread, worker threads, unraisable) reach the log file
and the report callback instead of vanishing.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import sys
import threading

import pytest
from tqdm import tqdm

from picasso import diagnostics


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the log at ``tmp_path`` and restore all global state after."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows

    stdout, stderr = sys.stdout, sys.stderr
    hooks = (sys.excepthook, threading.excepthook, sys.unraisablehook)
    monkeypatch.setattr(diagnostics, "_log_file", None, raising=False)
    monkeypatch.setattr(diagnostics, "_streams_replaced", False)
    try:
        yield
    finally:
        file = diagnostics._log_file
        if file is not None:
            file.close()
        diagnostics._log_file = None
        diagnostics._streams_replaced = False
        sys.stdout, sys.stderr = stdout, stderr
        sys.excepthook, threading.excepthook, sys.unraisablehook = hooks


def _log_text() -> str:
    return open(diagnostics.log_path(), encoding="utf-8").read()


def test_ensure_std_streams_is_noop_with_a_console(isolated):
    assert diagnostics.ensure_std_streams() is False


def test_ensure_std_streams_replaces_none_streams(isolated, monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    assert diagnostics.ensure_std_streams() is True
    assert sys.stdout is not None and sys.stderr is not None

    print("hello from a windowed build")
    sys.stdout.flush()
    assert "hello from a windowed build" in _log_text()


def test_tqdm_works_without_a_console(isolated, monkeypatch):
    """tqdm writes to ``sys.stderr``; with None it raises AttributeError,
    which is how the one-click build used to fail silently."""
    monkeypatch.setattr(sys, "stderr", None)
    diagnostics.ensure_std_streams()

    for _ in tqdm(range(3), desc="progress"):
        pass

    assert "progress" in _log_text()


def test_excepthook_logs_and_reports(isolated):
    reported = []
    diagnostics.install_excepthooks(report=reported.append)

    try:
        raise ValueError("spline mode needs lpz")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    assert "spline mode needs lpz" in reported[0]
    assert "spline mode needs lpz" in _log_text()


def test_thread_exceptions_are_not_swallowed(isolated, monkeypatch):
    """Without ``threading.excepthook``, an exception in a worker thread
    is dropped entirely when ``sys.stderr`` is None."""
    monkeypatch.setattr(sys, "stderr", None)
    diagnostics.ensure_std_streams()

    reported = []
    diagnostics.install_excepthooks(report=reported.append)

    def boom():
        raise RuntimeError("worker failed")

    thread = threading.Thread(target=boom, name="worker")
    thread.start()
    thread.join()

    assert reported and "worker failed" in reported[0]
    assert "Exception in thread worker" in _log_text()


def test_report_errors_are_ignored(isolated):
    """A broken reporter (e.g. a dead Qt window) must not replace the
    error it was meant to show."""

    def report(message):
        raise RuntimeError("message box is gone")

    diagnostics.install_excepthooks(report=report)
    try:
        raise ValueError("original error")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    assert "original error" in _log_text()


def test_log_is_rotated(isolated, monkeypatch):
    monkeypatch.setattr(diagnostics, "MAX_LOG_BYTES", 10)
    diagnostics.log_message("first message, well over ten bytes")
    diagnostics._log_file.close()
    diagnostics._log_file = None

    diagnostics.log_message("second message")

    assert "first message" in open(diagnostics.log_path() + ".1").read()
    assert "second message" in _log_text()


# ---------------------------------------------------------------------------
# The GUI reporter
# ---------------------------------------------------------------------------


class _FakeBox:
    """Stand-in for ``QMessageBox`` that records instead of blocking."""

    shown: list[_FakeBox] = []

    class Icon:
        Critical = None

    def __init__(self, parent=None):
        self.text = self.detailed = ""

    def setIcon(self, icon):
        pass

    def setWindowTitle(self, title):
        pass

    def setText(self, text):
        self.text = text

    def setInformativeText(self, text):
        pass

    def setDetailedText(self, text):
        self.detailed = text

    def exec(self):
        _FakeBox.shown.append(self)


def test_worker_thread_error_reaches_the_gui(isolated, qapp, monkeypatch):
    """A QThread/worker failure must close the progress dialogs and show a
    message box - the installer build used to leave the dialog hanging."""
    from PyQt6 import QtWidgets

    from picasso import lib_qt

    monkeypatch.setattr(lib_qt.QtWidgets, "QMessageBox", _FakeBox)
    _FakeBox.shown.clear()
    cancelled = []
    monkeypatch.setattr(
        lib_qt, "cancel_dialogs", lambda: cancelled.append(True)
    )

    window = QtWidgets.QWidget()
    lib_qt.install_excepthook(window)

    def boom():
        raise RuntimeError("worker failed in a thread")

    thread = threading.Thread(target=boom)
    thread.start()
    thread.join()
    qapp.processEvents()  # deliver the queued error signal

    assert cancelled, "open progress dialogs were not closed"
    assert len(_FakeBox.shown) == 1
    box = _FakeBox.shown[0]
    assert "worker failed in a thread" in box.text
    assert "worker failed in a thread" in box.detailed
    window.deleteLater()
