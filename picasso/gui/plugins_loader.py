"""
picasso.gui.plugins_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discovery, loading and online management of user plugins from
``~/.picasso/plugins``.

A plugin is a single ``.py`` file defining a ``Plugin`` class with an
``__init__(self, window)`` that sets ``self.name`` (the target GUI app,
e.g. ``"render"``) and ``self.window``, plus an ``execute(self)`` method
that adds actions to ``window.plugin_menu``. See ``plugin_template.py``.

Loading is deliberately tolerant: a broken plugin prints a traceback and
is skipped so that it can never crash app startup.

The second half of this module implements an online plugin store for
browsing, installing, updating and uninstalling plugins from the
``picasso_plugins`` GitHub repository. The registry is a single
``index.json`` manifest at the repo root; plugins are plain ``.py``
files downloaded into the user plugins directory. Installed plugins and
their versions are tracked in a hidden ``.installed.json`` sidecar
there.

:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import traceback
import urllib.request

from PyQt6 import QtCore, QtGui, QtWidgets

from .. import io
from ..version import __version__ as PICASSO_VERSION


# =============================================================================
# Discovery and loading
# =============================================================================


def _discover_plugin_files() -> list[str]:
    """Return sorted ``.py`` files in the user plugins directory, skipping
    files whose name starts with ``_``."""
    directory = io.plugins_directory()
    files = sorted(glob.glob(os.path.join(directory, "*.py")))
    return [f for f in files if not os.path.basename(f).startswith("_")]


def _load_module_from_path(path: str):
    """Import a standalone ``.py`` file that is not part of any package."""
    name = "picasso_plugin_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plugins(window, app_name: str) -> list:
    """Discover, instantiate and execute plugins matching ``app_name``.

    Always sets ``window.plugins`` to the (possibly empty) list of
    ``Plugin`` instances whose ``name`` matches ``app_name`` (after a
    successful ``execute``), and returns that list. A failure in any single
    plugin is logged and the plugin skipped.

    Parameters
    ----------
    window
        The GUI main window; must expose ``plugin_menu``.
    app_name : str
        The GUI app name to filter by (e.g. ``"render"``).
    """
    plugins: list = []
    for path in _discover_plugin_files():
        try:
            module = _load_module_from_path(path)
            plugin = module.Plugin(window)
            if getattr(plugin, "name", None) == app_name:
                plugin.execute()
                plugins.append(plugin)
        except Exception:
            print(f"Failed to load plugin {path!r}:")
            traceback.print_exc()
    window.plugins = plugins
    return plugins


def execute_plugins(window) -> None:
    """Re-run ``execute`` on the plugins already loaded into ``window``.

    Used when the menu bar is rebuilt (e.g. "Remove all localizations") to
    re-add plugin actions to the new ``window.plugin_menu`` without
    re-discovering plugins from disk. Tolerant: a failing plugin is logged
    and skipped.
    """
    for plugin in getattr(window, "plugins", []):
        try:
            plugin.execute()
        except Exception:
            print(f"Failed to execute plugin {plugin!r}:")
            traceback.print_exc()


def add_plugins_menu_actions(window, app_name: str) -> None:
    """Append a separator plus 'Browse online plugins', 'Open plugins
    folder...' and 'Reload plugins' actions to ``window.plugin_menu``."""
    menu = window.plugin_menu
    menu.addSeparator()

    store_action = menu.addAction("Browse online plugins")
    store_action.triggered.connect(lambda: _open_store(window, app_name))

    open_action = menu.addAction("Open plugins folder...")
    open_action.triggered.connect(
        lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(io.plugins_directory())
        )
    )

    reload_action = menu.addAction("Reload plugins")
    reload_action.triggered.connect(lambda: reload_plugins(window, app_name))


def _open_store(window, app_name: str) -> None:
    """Open the online plugin store dialog for ``app_name``."""
    PluginStoreDialog(window, app_name).exec()


def reload_plugins(window, app_name: str) -> None:
    """Clear the plugins menu and re-load plugins from disk."""
    window.plugin_menu.clear()
    load_plugins(window, app_name)
    add_plugins_menu_actions(window, app_name)


# Backwards-compatible alias for the previous private name.
_reload = reload_plugins


# =============================================================================
# Online plugin store
# =============================================================================

# --- Registry location -------------------------------------------------------

REPO = "rafalkowalewski1/picasso_plugins"
BRANCH = "master"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
MANIFEST_URL = f"{RAW_BASE}/index.json"
REPO_URL = f"https://github.com/{REPO}"

_TIMEOUT = 15  # seconds, for every network request
_SIDECAR = ".installed.json"


# --- Version helpers ---------------------------------------------------------


def parse_version(value: str | None) -> tuple[int, ...]:
    """Parse a version string into a tuple of its numeric components.

    Non-numeric suffixes (e.g. the ``a0`` in ``0.11.0a0``) are split on so
    that ``0.11.0a0`` -> ``(0, 11, 0, 0)``. Good enough to order the simple
    versions plugins use; it is not a full PEP 440 implementation.
    """
    if not value:
        return ()
    return tuple(int(p) for p in re.findall(r"\d+", str(value)))


def compare_versions(a: str | None, b: str | None) -> int:
    """Return -1, 0 or 1 for ``a`` <, == or > ``b`` (zero-padded compare)."""
    ta, tb = parse_version(a), parse_version(b)
    n = max(len(ta), len(tb))
    ta = ta + (0,) * (n - len(ta))
    tb = tb + (0,) * (n - len(tb))
    return (ta > tb) - (ta < tb)


def is_compatible(entry: dict) -> bool:
    """Whether the running Picasso satisfies the plugin's minimum version."""
    minimum = entry.get("min_picasso_version")
    if not minimum:
        return True
    return compare_versions(PICASSO_VERSION, minimum) >= 0


# --- Installed-state sidecar -------------------------------------------------


def _sidecar_path() -> str:
    return os.path.join(io.plugins_directory(), _SIDECAR)


def load_state() -> dict:
    """Load the installed-state sidecar, tolerant of a missing/corrupt file."""
    path = _sidecar_path()
    state = {"plugins": {}, "trust_acknowledged": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            state["plugins"] = data.get("plugins", {}) or {}
            state["trust_acknowledged"] = bool(data.get("trust_acknowledged"))
    except (OSError, ValueError):
        pass
    return state


def save_state(state: dict) -> None:
    with open(_sidecar_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# --- Network + install operations -------------------------------------------


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "picasso"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def fetch_manifest() -> list[dict]:
    """Download and parse the registry manifest.

    Returns the list of plugin entries. Raises on any network/parse error so
    the caller can show a message and offer to retry.
    """
    data = json.loads(_get(MANIFEST_URL).decode("utf-8"))
    plugins = data.get("plugins", []) if isinstance(data, dict) else []
    return [p for p in plugins if isinstance(p, dict) and p.get("id")]


def _local_filename(entry: dict) -> str:
    """The on-disk ``.py`` name for an entry: the plugin id plus ``.py``."""
    return f"{entry['id']}.py"


def install(entry: dict, state: dict) -> None:
    """Download ``entry``'s ``.py`` into the plugins folder and record it."""
    content = _get(f"{RAW_BASE}/{entry['file']}")
    target = os.path.join(io.plugins_directory(), _local_filename(entry))
    with open(target, "wb") as f:
        f.write(content)
    state["plugins"][entry["id"]] = {
        "file": _local_filename(entry),
        "version": entry.get("version"),
        "app": entry.get("app"),
        "display_name": entry.get("display_name", entry["id"]),
    }
    save_state(state)


def uninstall(plugin_id: str, state: dict) -> None:
    """Delete the installed ``.py`` for ``plugin_id`` and forget it."""
    record = state["plugins"].get(plugin_id, {})
    filename = record.get("file", f"{plugin_id}.py")
    path = os.path.join(io.plugins_directory(), filename)
    try:
        os.remove(path)
    except OSError:
        pass
    state["plugins"].pop(plugin_id, None)
    save_state(state)


# --- Status model ------------------------------------------------------------

NOT_INSTALLED = "not_installed"
UP_TO_DATE = "up_to_date"
UPDATE_AVAILABLE = "update_available"
INCOMPATIBLE = "incompatible"
ORPHAN = "orphan"  # installed but no longer in the manifest


def status_for(entry: dict, state: dict) -> str:
    installed = state["plugins"].get(entry["id"])
    if entry.get("_orphan"):
        return ORPHAN
    if not is_compatible(entry):
        return INCOMPATIBLE
    if installed is None:
        return NOT_INSTALLED
    if compare_versions(entry.get("version"), installed.get("version")) > 0:
        return UPDATE_AVAILABLE
    return UP_TO_DATE


def merged_entries(manifest: list[dict], state: dict) -> list[dict]:
    """Manifest entries plus any installed plugins missing from the manifest.

    Orphans (installed locally but dropped from the registry) are surfaced so
    they can still be uninstalled; they are flagged with ``_orphan``.
    """
    by_id = {e["id"]: e for e in manifest}
    entries = list(manifest)
    for pid, rec in state["plugins"].items():
        if pid not in by_id:
            entries.append(
                {
                    "id": pid,
                    "display_name": rec.get("display_name", pid),
                    "app": rec.get("app"),
                    "description": "(no longer in the online registry)",
                    "version": rec.get("version"),
                    "_orphan": True,
                }
            )
    return entries


# --- Dialog ------------------------------------------------------------------

_STATUS_TEXT = {
    NOT_INSTALLED: "Not installed",
    UP_TO_DATE: "Installed",
    UPDATE_AVAILABLE: "Update available",
    INCOMPATIBLE: "Requires newer Picasso",
    ORPHAN: "Installed (unlisted)",
}


class PluginStoreDialog(QtWidgets.QDialog):
    """Browse and manage online plugins for a single Picasso app."""

    def __init__(self, window, app_name: str):
        super().__init__(window)
        self.window = window
        self.app_name = app_name
        self.state = load_state()
        self.manifest: list[dict] = []

        self.setWindowTitle("Online plugins")
        self.resize(760, 460)

        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.show_all = QtWidgets.QCheckBox(
            "Show plugins for all Picasso apps"
        )
        self.show_all.stateChanged.connect(self._populate)
        layout.addWidget(self.show_all)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Plugin", "App", "Description", "Status", "Action"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        repo_btn = QtWidgets.QPushButton("Open repository...")
        repo_btn.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(REPO_URL))
        )
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(repo_btn)
        buttons.addStretch()
        buttons.addWidget(refresh_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self._refresh()

    # -- data -----------------------------------------------------------------

    def _refresh(self) -> None:
        """(Re)download the manifest and rebuild the table."""
        self.state = load_state()
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            self.manifest = fetch_manifest()
            self.status_label.setText(
                f"Plugins available from {REPO}. Download only plugins you "
                "trust — they run arbitrary Python code on startup."
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.manifest = []
            self.status_label.setText(
                "Could not reach the online plugin registry "
                f"({exc}). Check your connection and press Refresh."
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._populate()

    def _visible_entries(self) -> list[dict]:
        entries = merged_entries(self.manifest, self.state)
        if not self.show_all.isChecked():
            entries = [
                e
                for e in entries
                if e.get("app") == self.app_name or e.get("_orphan")
            ]
        return sorted(
            entries, key=lambda e: e.get("display_name", e["id"]).lower()
        )

    def _populate(self) -> None:
        entries = self._visible_entries()
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            status = status_for(entry, self.state)

            name_item = QtWidgets.QTableWidgetItem(
                entry.get("display_name", entry["id"])
            )
            author = entry.get("author")
            version = entry.get("version")
            tip = []
            if author:
                tip.append(f"Author: {author}")
            if version:
                tip.append(f"Latest version: {version}")
            if tip:
                name_item.setToolTip("\n".join(tip))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(entry.get("app", ""))
            )
            self.table.setItem(
                row,
                2,
                QtWidgets.QTableWidgetItem(entry.get("description", "")),
            )

            status_text = _STATUS_TEXT.get(status, status)
            installed = self.state["plugins"].get(entry["id"])
            if status == UPDATE_AVAILABLE and installed:
                status_text += f" ({installed.get('version')} → {version})"
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(status_text))

            self.table.setCellWidget(
                row, 4, self._action_widget(entry, status)
            )
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )

    # -- per-row actions ------------------------------------------------------

    def _action_widget(self, entry: dict, status: str) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(container)
        box.setContentsMargins(2, 2, 2, 2)
        box.setSpacing(4)

        def add(label, slot):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(slot)
            box.addWidget(btn)
            return btn

        if status == NOT_INSTALLED:
            add("Install", lambda: self._install(entry))
        elif status == UPDATE_AVAILABLE:
            add("Update", lambda: self._install(entry))
            add("Uninstall", lambda: self._uninstall(entry))
        elif status in (UP_TO_DATE, ORPHAN):
            add("Uninstall", lambda: self._uninstall(entry))
        elif status == INCOMPATIBLE:
            label = QtWidgets.QLabel("—")
            label.setToolTip(
                "This plugin requires a newer version of Picasso "
                f"(>= {entry.get('min_picasso_version')})."
            )
            box.addWidget(label)
        box.addStretch()
        return container

    def _confirm_trust(self) -> bool:
        """Show the one-time trust warning before the first install."""
        if self.state.get("trust_acknowledged"):
            return True
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Install plugin?",
            "Plugins are Python files that run with full access to your "
            "computer every time the app starts. Only install plugins from "
            "sources you trust.\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        self.state["trust_acknowledged"] = True
        save_state(self.state)
        return True

    def _install(self, entry: dict) -> None:
        if not self._confirm_trust():
            return
        try:
            install(entry, self.state)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QtWidgets.QMessageBox.critical(
                self, "Installation failed", str(exc)
            )
            return
        self._reload_app()
        self._populate()

    def _uninstall(self, entry: dict) -> None:
        uninstall(entry["id"], self.state)
        self._reload_app()
        self._populate()

    def _reload_app(self) -> None:
        """Re-discover plugins in the host window so changes take effect."""
        reload_plugins(self.window, self.app_name)
