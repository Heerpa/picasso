"""
picasso.gui.plugins_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discovery, loading and online management of user plugins from
``~/.picasso/plugins``.

A plugin is a single ``.py`` file defining a ``Plugin`` class with an
``__init__(self, window)`` that sets ``self.name`` (the target GUI app,
e.g. ``"render"``) and ``self.window``, plus an ``execute(self)`` method
that adds actions to ``window.plugin_menu``. See ``plugin_template.py``.

Plugins are executed as ordinary Python inside the Picasso process, so
they cannot be sandboxed: a loaded plugin can do anything Picasso can.
The safeguards here therefore aim at *provenance and reviewability*
rather than containment:

* nothing is loaded until it has been explicitly enabled, so a file that
  merely appears in the plugins folder never runs (``is_enabled``);
* registry downloads are verified against a ``sha256`` pinned in the
  manifest, so the code behind a plugin cannot change without a reviewed
  manifest commit (``install``);
* ids and repository paths taken from the manifest are validated before
  they are used as file names or URLs (``is_safe_id``,
  ``is_safe_repo_path``);
* the store dialog can show a plugin's source before installing it and a
  diff of what changed before updating it.

Loading is deliberately tolerant: a broken plugin prints a traceback and
is skipped so that it can never crash app startup.

The second half of this module implements an online plugin store for
browsing, installing, updating and uninstalling plugins from the
``picasso_plugins`` GitHub repository. The registry is a single
``index.json`` manifest at the repo root; plugins are plain ``.py`` files
downloaded into the user plugins directory. Installed plugins, their
versions and which files are enabled are tracked in a hidden
``.installed.json`` sidecar there.

:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import difflib
import glob
import hashlib
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
# Safety helpers: identifiers, paths and integrity
# =============================================================================

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def is_safe_id(value) -> bool:
    """Whether ``value`` may be used as a plugin id and file name stem.

    Ids come from a remote manifest and are turned into file names inside
    the user plugins directory, so they must not be able to escape it:
    only ASCII letters, digits, ``_`` and ``-`` are allowed.
    """
    return isinstance(value, str) and bool(_ID_RE.match(value))


def is_safe_repo_path(value) -> bool:
    """Whether ``value`` is a safe relative path inside the registry repo.

    Rejects absolute paths, Windows separators, ``..`` segments and
    anything that is not a ``.py`` file, so a manifest entry cannot point
    the download at another repository or at a path outside it.
    """
    if not isinstance(value, str) or not value.endswith(".py"):
        return False
    if value.startswith("/") or "\\" in value or ":" in value:
        return False
    return all(part not in ("", ".", "..") for part in value.split("/"))


def is_safe_filename(value) -> bool:
    """Whether ``value`` is a bare ``.py`` file name with no directory part."""
    return (
        isinstance(value, str)
        and value.endswith(".py")
        and len(value) > len(".py")
        and os.path.basename(value) == value
    )


def is_valid_digest(value) -> bool:
    """Whether ``value`` looks like a hex-encoded SHA-256 digest."""
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def plugin_path(filename: str) -> str:
    """Absolute path of ``filename`` inside the user plugins directory.

    Raises ``ValueError`` if ``filename`` is not a bare ``.py`` name or if
    the resolved path would land outside the plugins directory.
    """
    if not is_safe_filename(filename):
        raise ValueError(f"Unsafe plugin file name: {filename!r}")
    directory = os.path.abspath(io.plugins_directory())
    path = os.path.abspath(os.path.join(directory, filename))
    if os.path.dirname(path) != directory:
        raise ValueError(
            f"Plugin file escapes the plugins folder: {filename!r}"
        )
    return path


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
    """Discover, instantiate and execute enabled plugins matching ``app_name``.

    Only files that have been explicitly enabled are executed; a ``.py``
    file that has merely been copied into the plugins folder is discovered
    but not run until the user enables it (via the plugin store dialog),
    because loading it means running it.

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

    Returns
    -------
    plugins : list
        The ``Plugin`` instances that matched ``app_name`` and executed
        successfully; also assigned to ``window.plugins``.
    """
    plugins: list = []
    state = load_state()
    for path in _discover_plugin_files():
        if not is_enabled(state, os.path.basename(path)):
            continue
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


def disabled_plugin_files(state: dict | None = None) -> list[str]:
    """File names present in the plugins folder that are not enabled."""
    state = load_state() if state is None else state
    return [
        os.path.basename(p)
        for p in _discover_plugin_files()
        if not is_enabled(state, os.path.basename(p))
    ]


def add_plugins_menu_actions(window, app_name: str) -> None:
    """Append a separator plus the plugin management actions to
    ``window.plugin_menu``.

    If plugin files are present but not enabled, a leading entry says so
    and opens the store dialog, where they can be reviewed and enabled.
    """
    menu = window.plugin_menu
    menu.addSeparator()

    pending = disabled_plugin_files()
    if pending:
        count = len(pending)
        noun = "file" if count == 1 else "files"
        pending_action = menu.addAction(
            f"{count} plugin {noun} found but not enabled..."
        )
        pending_action.triggered.connect(lambda: _open_store(window, app_name))

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

REPO = "jungmannlab/picasso_plugins"
BRANCH = "main"  # the registry's default branch
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
    """Load the installed-state sidecar, tolerant of a missing/corrupt file.

    The state holds three things: ``plugins`` (per-id records of what was
    installed from the registry), ``enabled`` (per-file-name flags deciding
    what may be loaded) and ``trust_acknowledged`` (the one-time warning).
    """
    path = _sidecar_path()
    state = {"plugins": {}, "enabled": {}, "trust_acknowledged": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            state["plugins"] = data.get("plugins", {}) or {}
            enabled = data.get("enabled", {}) or {}
            if isinstance(enabled, dict):
                state["enabled"] = {
                    k: bool(v)
                    for k, v in enabled.items()
                    if isinstance(k, str)
                }
            state["trust_acknowledged"] = bool(data.get("trust_acknowledged"))
    except (OSError, ValueError):
        pass
    return state


def save_state(state: dict) -> None:
    with open(_sidecar_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def is_enabled(state: dict, filename: str) -> bool:
    """Whether ``filename`` is allowed to be loaded and executed.

    Unknown files are *not* enabled: enabling is the explicit consent step,
    so a plugin only ever runs after the user has said so.
    """
    return bool(state.get("enabled", {}).get(filename))


def set_enabled(state: dict, filename: str, value: bool) -> None:
    """Record whether ``filename`` may be loaded, and save the sidecar."""
    state.setdefault("enabled", {})[filename] = bool(value)
    save_state(state)


# --- Network + install operations -------------------------------------------


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "picasso"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def _is_usable_entry(entry: dict) -> bool:
    """Whether a manifest entry is structurally safe to work with.

    Entries with an unsafe id or file path are dropped outright — they can
    never be legitimate, since both are used to build a local file name and
    a download URL. A missing or malformed ``sha256`` is *not* filtered here:
    such entries are kept and surfaced as ``UNVERIFIED`` so the user sees
    why they cannot be installed.
    """
    return is_safe_id(entry.get("id")) and is_safe_repo_path(entry.get("file"))


def fetch_manifest() -> list[dict]:
    """Download and parse the registry manifest.

    Returns the list of plugin entries whose id and file path pass
    validation. Raises on any network/parse error so the caller can show a
    message and offer to retry.
    """
    data = json.loads(_get(MANIFEST_URL).decode("utf-8"))
    plugins = data.get("plugins", []) if isinstance(data, dict) else []
    entries = []
    for entry in plugins:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        if not _is_usable_entry(entry):
            print(
                "Skipping plugin registry entry with an unsafe id or file "
                f"path: {entry.get('id')!r} -> {entry.get('file')!r}"
            )
            continue
        entries.append(entry)
    return entries


def _local_filename(entry: dict) -> str:
    """The on-disk ``.py`` name for an entry: the plugin id plus ``.py``."""
    if not is_safe_id(entry.get("id")):
        raise ValueError(f"Unsafe plugin id: {entry.get('id')!r}")
    return f"{entry['id']}.py"


def download_source(entry: dict) -> bytes:
    """Download an entry's ``.py`` and verify it against the pinned hash.

    Raises ``ValueError`` if the entry carries no valid ``sha256`` or if the
    downloaded bytes do not match it, so unverified code is never written to
    disk or shown as if it were the published plugin.
    """
    if not _is_usable_entry(entry):
        raise ValueError(f"Unsafe registry entry for {entry.get('id')!r}")
    expected = entry.get("sha256")
    if not is_valid_digest(expected):
        raise ValueError(
            f"The registry entry for {entry['id']!r} has no valid sha256 "
            "integrity hash, so it cannot be verified. Refusing to install."
        )
    content = _get(f"{RAW_BASE}/{entry['file']}")
    actual = sha256_bytes(content)
    if actual != expected.lower():
        raise ValueError(
            f"Integrity check failed for {entry['id']!r}: the downloaded file "
            f"does not match the hash pinned in the registry.\n\n"
            f"expected {expected.lower()}\ngot      {actual}\n\n"
            "The file may have been changed after it was published. Nothing "
            "was installed."
        )
    return content


def install(entry: dict, state: dict) -> None:
    """Download, verify and store ``entry``'s ``.py``, then enable it.

    Installing is an explicit user action, so the file is enabled on
    success. Raises before touching the disk if verification fails.
    """
    content = download_source(entry)
    filename = _local_filename(entry)
    target = plugin_path(filename)
    with open(target, "wb") as f:
        f.write(content)
    state["plugins"][entry["id"]] = {
        "file": filename,
        "version": entry.get("version"),
        "app": entry.get("app"),
        "display_name": entry.get("display_name", entry["id"]),
        "sha256": sha256_bytes(content),
    }
    state.setdefault("enabled", {})[filename] = True
    save_state(state)


def uninstall(plugin_id: str, state: dict) -> None:
    """Delete the installed ``.py`` for ``plugin_id`` and forget it."""
    record = state["plugins"].get(plugin_id, {})
    filename = record.get("file")
    if not is_safe_filename(filename):
        filename = f"{plugin_id}.py" if is_safe_id(plugin_id) else None
    if filename:
        try:
            os.remove(plugin_path(filename))
        except (OSError, ValueError):
            pass
        state.get("enabled", {}).pop(filename, None)
    state["plugins"].pop(plugin_id, None)
    save_state(state)


def read_local_source(filename: str) -> str:
    """Read an on-disk plugin file as text, replacing undecodable bytes."""
    path = plugin_path(filename)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def diff_sources(old: str, new: str, filename: str) -> str:
    """Unified diff between the installed and the incoming source."""
    lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{filename} (installed)",
            tofile=f"{filename} (update)",
        )
    )
    if not lines:
        return (
            "The new version is byte-for-byte identical to the installed one."
        )
    return "".join(lines)


# --- Status model ------------------------------------------------------------

NOT_INSTALLED = "not_installed"
UP_TO_DATE = "up_to_date"
UPDATE_AVAILABLE = "update_available"
INCOMPATIBLE = "incompatible"
UNVERIFIED = "unverified"  # registry entry without a usable sha256
UNMANAGED = "unmanaged"  # registry plugin whose file was copied in by hand
LOCAL_ONLY = "local_only"  # file in the folder with no registry entry
ORPHAN = "orphan"  # installed but no longer in the manifest


def status_for(entry: dict, state: dict) -> str:
    installed = state["plugins"].get(entry["id"])
    if entry.get("_orphan"):
        return ORPHAN
    if entry.get("_local"):
        return LOCAL_ONLY
    if not is_compatible(entry):
        return INCOMPATIBLE
    if not is_valid_digest(entry.get("sha256")):
        return UNVERIFIED
    if installed is None:
        try:
            exists = os.path.exists(plugin_path(_local_filename(entry)))
        except ValueError:
            exists = False
        return UNMANAGED if exists else NOT_INSTALLED
    if compare_versions(entry.get("version"), installed.get("version")) > 0:
        return UPDATE_AVAILABLE
    return UP_TO_DATE


def entry_filename(entry: dict) -> str | None:
    """The on-disk file name an entry maps to, or ``None`` if unsafe."""
    if entry.get("_local"):
        return entry.get("file")
    try:
        return _local_filename(entry)
    except ValueError:
        return None


def merged_entries(manifest: list[dict], state: dict) -> list[dict]:
    """Manifest entries, plus installed plugins and loose files not in it.

    Orphans (installed locally but dropped from the registry) are surfaced
    so they can still be uninstalled; they are flagged with ``_orphan``.
    Files sitting in the plugins folder that the registry knows nothing
    about are surfaced too — flagged with ``_local`` — so that manually
    shared plugins can be reviewed and enabled from the same place.
    """
    by_id = {e["id"]: e for e in manifest}
    entries = list(manifest)
    accounted = {
        name
        for name in (entry_filename(e) for e in manifest)
        if name is not None
    }
    for pid, rec in state["plugins"].items():
        if rec.get("file"):
            accounted.add(rec["file"])
        if pid not in by_id:
            entries.append(
                {
                    "id": pid,
                    "display_name": rec.get("display_name", pid),
                    "app": rec.get("app"),
                    "description": "(no longer in the online registry)",
                    "version": rec.get("version"),
                    "file": rec.get("file"),
                    "_orphan": True,
                }
            )
    for path in _discover_plugin_files():
        name = os.path.basename(path)
        if name in accounted:
            continue
        entries.append(
            {
                "id": os.path.splitext(name)[0],
                "display_name": name,
                "app": None,
                "description": (
                    "(local file, not from the online registry — review its "
                    "source before enabling it)"
                ),
                "file": name,
                "_local": True,
            }
        )
    return sorted(
        entries, key=lambda e: str(e.get("display_name", e["id"])).lower()
    )


# --- Dialogs -----------------------------------------------------------------

_STATUS_TEXT = {
    NOT_INSTALLED: "Not installed",
    UP_TO_DATE: "Installed",
    UPDATE_AVAILABLE: "Update available",
    INCOMPATIBLE: "Requires newer Picasso",
    UNVERIFIED: "No integrity hash",
    UNMANAGED: "On disk, not installed by Picasso",
    LOCAL_ONLY: "Local file",
    ORPHAN: "Installed (unlisted)",
}


class SourceViewDialog(QtWidgets.QDialog):
    """Read-only viewer for a plugin's source or an update's diff."""

    def __init__(self, parent, title: str, header: str, text: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 600)

        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel(header)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(label)

        view = QtWidgets.QPlainTextEdit()
        view.setReadOnly(True)
        view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        view.setFont(
            QtGui.QFontDatabase.systemFont(
                QtGui.QFontDatabase.SystemFont.FixedFont
            )
        )
        view.setPlainText(text)
        layout.addWidget(view)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)


class PluginStoreDialog(QtWidgets.QDialog):
    """Browse and manage online plugins for a single Picasso app."""

    def __init__(self, window, app_name: str):
        super().__init__(window)
        self.window = window
        self.app_name = app_name
        self.state = load_state()
        self.manifest: list[dict] = []

        self.setWindowTitle("Online plugins")
        self.resize(900, 500)

        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.show_all = QtWidgets.QCheckBox(
            "Show plugins for all Picasso apps"
        )
        self.show_all.stateChanged.connect(self._populate)
        layout.addWidget(self.show_all)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Plugin", "App", "Description", "Status", "Enabled", "Action"]
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
                f"Plugins available from {REPO}. Downloads are checked "
                "against the hash published in the registry, but a plugin "
                "still runs with full access to your computer once enabled — "
                "use 'View source' and only enable plugins you trust."
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
                if e.get("app") == self.app_name
                or e.get("_orphan")
                or e.get("_local")
            ]
        return entries

    def _populate(self) -> None:
        entries = self._visible_entries()
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            status = status_for(entry, self.state)

            name_item = QtWidgets.QTableWidgetItem(
                str(entry.get("display_name", entry["id"]))
            )
            author = entry.get("author")
            version = entry.get("version")
            tip = []
            if author:
                tip.append(f"Author: {author}")
            if version:
                tip.append(f"Latest version: {version}")
            if entry.get("sha256"):
                tip.append(f"sha256: {entry['sha256']}")
            if tip:
                name_item.setToolTip("\n".join(tip))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(entry.get("app") or "")
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
            status_item = QtWidgets.QTableWidgetItem(status_text)
            if status == UNVERIFIED:
                status_item.setToolTip(
                    "This registry entry does not publish a sha256 hash, so "
                    "the download cannot be verified. Installing is disabled."
                )
            self.table.setItem(row, 3, status_item)

            self.table.setCellWidget(row, 4, self._enabled_widget(entry))
            self.table.setCellWidget(
                row, 5, self._action_widget(entry, status)
            )
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )

    # -- enable / disable -----------------------------------------------------

    def _enabled_widget(self, entry: dict) -> QtWidgets.QWidget:
        """A checkbox controlling whether the file is loaded at startup.

        Only meaningful for files that exist on disk; otherwise a placeholder
        is shown.
        """
        container = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(container)
        box.setContentsMargins(2, 2, 2, 2)

        filename = entry_filename(entry)
        on_disk = False
        if filename:
            try:
                on_disk = os.path.exists(plugin_path(filename))
            except ValueError:
                on_disk = False
        if not on_disk:
            box.addWidget(QtWidgets.QLabel("—"))
            return container

        check = QtWidgets.QCheckBox()
        check.setChecked(is_enabled(self.state, filename))
        check.setToolTip(
            "Enabled plugins are imported and run every time the app starts."
        )
        check.clicked.connect(
            lambda checked, f=filename: self._toggle_enabled(f, checked)
        )
        box.addWidget(check)
        return container

    def _toggle_enabled(self, filename: str, enable: bool) -> None:
        if enable and not self._confirm_enable(filename):
            self._populate()
            return
        set_enabled(self.state, filename, enable)
        self._reload_app()
        self._populate()

    def _confirm_enable(self, filename: str) -> bool:
        """Confirm enabling a file that Picasso did not install itself.

        Registry installs are already confirmed at install time, so only
        untracked files ask again — those are the ones that arrived by hand.
        """
        tracked = any(
            rec.get("file") == filename
            for rec in self.state["plugins"].values()
        )
        if tracked:
            return True
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Enable plugin?",
            f"'{filename}' did not come from the online plugin registry, so "
            "its contents cannot be verified.\n\nEnabling it will run this "
            "Python file with full access to your computer every time the app "
            "starts. Only enable files you have reviewed or whose author you "
            "trust.\n\nEnable it?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

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
            add("View source", lambda: self._view_remote(entry))
        elif status == UNMANAGED:
            add("Install", lambda: self._install(entry))
            add("View source", lambda: self._view_local(entry))
        elif status == UPDATE_AVAILABLE:
            add("Update", lambda: self._install(entry))
            add("View changes", lambda: self._view_diff(entry))
            add("Uninstall", lambda: self._uninstall(entry))
        elif status in (UP_TO_DATE, ORPHAN):
            add("View source", lambda: self._view_local(entry))
            add("Uninstall", lambda: self._uninstall(entry))
        elif status == LOCAL_ONLY:
            add("View source", lambda: self._view_local(entry))
        elif status == UNVERIFIED:
            label = QtWidgets.QLabel("—")
            label.setToolTip(
                "This plugin cannot be installed until the registry "
                "publishes a sha256 hash for it."
            )
            box.addWidget(label)
        elif status == INCOMPATIBLE:
            label = QtWidgets.QLabel("—")
            label.setToolTip(
                "This plugin requires a newer version of Picasso "
                f"(>= {entry.get('min_picasso_version')})."
            )
            box.addWidget(label)
        box.addStretch()
        return container

    # -- source viewing -------------------------------------------------------

    def _view_remote(self, entry: dict) -> None:
        """Show the source that would be installed, hash-verified."""
        try:
            content = download_source(entry)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QtWidgets.QMessageBox.critical(
                self, "Could not show source", str(exc)
            )
            return
        self._show_source(
            f"Source of {entry.get('display_name', entry['id'])}",
            f"{RAW_BASE}/{entry['file']}\nsha256 {sha256_bytes(content)} "
            "(matches the registry)",
            content.decode("utf-8", errors="replace"),
        )

    def _view_local(self, entry: dict) -> None:
        """Show the source of the file currently on disk."""
        filename = entry_filename(entry)
        try:
            text = read_local_source(filename)
            path = plugin_path(filename)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(
                self, "Could not show source", str(exc)
            )
            return
        self._show_source(f"Source of {filename}", path, text)

    def _view_diff(self, entry: dict) -> None:
        """Show what an update would change in the installed file."""
        filename = entry_filename(entry)
        try:
            new = download_source(entry).decode("utf-8", errors="replace")
            old = read_local_source(filename)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QtWidgets.QMessageBox.critical(
                self, "Could not show changes", str(exc)
            )
            return
        installed = self.state["plugins"].get(entry["id"], {})
        self._show_source(
            f"Changes to {filename}",
            f"{installed.get('version')} → {entry.get('version')}",
            diff_sources(old, new, filename),
        )

    def _show_source(self, title: str, header: str, text: str) -> None:
        SourceViewDialog(self, title, header, text).exec()

    # -- install / uninstall --------------------------------------------------

    def _confirm_trust(self) -> bool:
        """Show the one-time trust warning before the first install."""
        if self.state.get("trust_acknowledged"):
            return True
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Install plugin?",
            "Plugins are Python files that run with full access to your "
            "computer every time the app starts. Downloads are verified "
            "against the hash published in the registry, but that only proves "
            "the file is the published one — not that it is safe. Only "
            "install plugins from sources you trust; 'View source' shows the "
            "code first.\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        self.state["trust_acknowledged"] = True
        save_state(self.state)
        return True

    def _confirm_overwrite(self, entry: dict, status: str) -> bool:
        """Confirm replacing a file Picasso did not install itself."""
        if status != UNMANAGED:
            return True
        filename = entry_filename(entry)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Replace existing file?",
            f"'{filename}' already exists in your plugins folder but was not "
            "installed by Picasso. Installing will overwrite it with the "
            "version from the registry.\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _install(self, entry: dict) -> None:
        status = status_for(entry, self.state)
        if not self._confirm_trust():
            return
        if not self._confirm_overwrite(entry, status):
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
