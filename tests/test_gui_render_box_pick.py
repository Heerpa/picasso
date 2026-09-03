"""The "Box" pick shape of ``picasso.gui.render``.

A box is dragged out rather than clicked into place, and unlike the
click-placed shapes it carries its own extent, so ``_pick_size`` is None
for it. These tests drive the real mouse handlers of a ``View`` to cover
the drag, the removal, the YAML round trip and the metadata, which is
where the None size would otherwise surface as a ``TypeError``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from PyQt6 import QtCore

from picasso import io, lib
from picasso.gui import render as gui_render, rotation

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0


def _locs(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "x": rng.uniform(0.0, WIDTH, size=n),
            "y": rng.uniform(0.0, HEIGHT, size=n),
            "lpx": np.full(n, 0.1),
            "lpy": np.full(n, 0.1),
            "photons": np.full(n, 1000.0),
        }
    )


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window holding one channel of uniform localizations."""
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(256, 256)
    return window


class _Event:
    """The parts of a Qt mouse event the pick handlers read."""

    def __init__(self, x, y, button=QtCore.Qt.MouseButton.LeftButton):
        self._pos = QtCore.QPoint(int(x), int(y))
        self._button = button

    def pos(self):
        return self._pos

    def button(self):
        return self._button

    def accept(self):
        pass

    def ignore(self):
        pass


def _drag(view, x0, y0, x1, y1):
    """Press, move and release, as the canvas would."""
    view.mousePressEvent(_Event(x0, y0))
    view.mouseMoveEvent(_Event(x1, y1))
    view.mouseReleaseEvent(_Event(x1, y1))


@pytest.fixture
def box_view(window):
    """The view of ``window``, in Pick mode with the Box shape."""
    view = window.view
    window.tools_settings_dialog.pick_shape.setCurrentText("Box")
    view._mode = "Pick"
    assert view._pick_shape == "Box"
    return view


class TestBoxPickTool:
    def test_box_is_registered_as_a_pick_shape(self):
        assert "Box" in lib.PICK_SHAPES
        assert "Box" in lib.PICK_SHAPES_WITHOUT_SIZE

    def test_shape_selector_offers_every_shape(self, window):
        combo = window.tools_settings_dialog.pick_shape
        offered = [combo.itemText(i) for i in range(combo.count())]
        assert offered == list(lib.PICK_SHAPES)

    def test_pick_size_is_none(self, box_view):
        assert box_view._pick_size is None

    def test_drag_creates_one_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        assert len(box_view._picks) == 1
        (x0, y0), (x1, y1) = box_view._picks[0]
        # corners come out ordered, whichever way the drag went
        assert x0 < x1 and y0 < y1

    def test_drag_upwards_gives_the_same_box(self, box_view):
        _drag(box_view, 120, 100, 40, 40)
        down = box_view._picks[0]
        box_view.clear_picks()
        _drag(box_view, 40, 40, 120, 100)
        assert box_view._picks[0] == pytest.approx(np.array(down))

    def test_box_spans_the_dragged_screen_region(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        (x0, y0), (x1, y1) = box_view._picks[0]
        assert (x0, y0) == pytest.approx(
            box_view.map_to_movie(QtCore.QPoint(40, 40))
        )
        assert (x1, y1) == pytest.approx(
            box_view.map_to_movie(QtCore.QPoint(120, 100))
        )

    def test_a_bare_click_creates_nothing(self, box_view):
        _drag(box_view, 60, 60, 60, 60)
        assert box_view._picks == []

    def test_drag_shorter_than_the_minimum_creates_nothing(self, box_view):
        short = gui_render.MIN_BOX_PICK_DRAG - 1
        _drag(box_view, 60, 60, 60 + short, 60 + short)
        assert box_view._picks == []

    def test_the_drag_overlay_is_cleared_on_release(self, box_view):
        box_view.mousePressEvent(_Event(40, 40))
        assert box_view._box_pick_ongoing
        box_view.mouseMoveEvent(_Event(120, 100))
        box_view.mouseReleaseEvent(_Event(120, 100))
        assert not box_view._box_pick_ongoing

    def test_right_click_inside_removes_the_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        inside = _Event(80, 70, QtCore.Qt.MouseButton.RightButton)
        box_view.mouseReleaseEvent(inside)
        assert box_view._picks == []

    def test_right_click_outside_keeps_the_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        outside = _Event(200, 200, QtCore.Qt.MouseButton.RightButton)
        box_view.mouseReleaseEvent(outside)
        assert len(box_view._picks) == 1

    def test_picked_locs_are_inside_the_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        (x0, y0), (x1, y1) = box_view._picks[0]
        picked = box_view.picked_locs(0)[0]
        assert len(picked) > 0
        assert (picked["x"] > x0).all() and (picked["x"] < x1).all()
        assert (picked["y"] > y0).all() and (picked["y"] < y1).all()

    def test_pick_bounds_match_the_drawn_box(self, box_view):
        # this is what "Move to pick" and the XY scatter frame on
        _drag(box_view, 40, 40, 120, 100)
        (x0, y0), (x1, y1) = box_view._picks[0]
        bounds = lib.pick_bounds(box_view._picks[0], "Box", None)
        assert bounds == pytest.approx((x0, x1, y0, y1))

    def test_pick_areas_are_per_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        _drag(box_view, 140, 140, 200, 160)
        areas = box_view.pick_areas()
        assert len(areas) == 2
        assert areas[0] > areas[1] > 0

    def test_saved_yaml_round_trips(self, box_view, tmp_path):
        _drag(box_view, 40, 40, 120, 100)
        _drag(box_view, 140, 140, 200, 160)
        path = str(tmp_path / "picks.yaml")
        box_view.save_picks(path)

        regions = yaml.full_load(open(path))
        assert regions["Shape"] == "Box"
        assert "Corners" in regions
        # a box has no global size to store
        assert not any("nm" in key for key in regions)

        saved = [np.array(pick) for pick in box_view._picks]
        box_view.clear_picks()
        box_view.load_picks(path)
        assert box_view._pick_shape == "Box"
        assert len(box_view._picks) == 2
        for loaded, original in zip(box_view._picks, saved):
            assert np.array(loaded) == pytest.approx(original)

    def test_pick_metadata_has_no_size_entry(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        pick_info = box_view._build_base_pick_info()
        assert pick_info["Pick Shape"] == "Box"
        assert pick_info["Number of picks"] == 1
        assert not any("Pick Diameter" in key for key in pick_info)
        # per-pick areas, unlike the single repeated value of a circle
        assert len(pick_info["Pick Areas (um^2)"]) == 1

    def test_index_blocks_are_not_built(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        assert box_view.get_index_blocks(0) is None

    def test_pick_similar_accepts_boxes(self, box_view, monkeypatch):
        _drag(box_view, 40, 40, 120, 100)
        _drag(box_view, 140, 40, 220, 100)
        monkeypatch.setattr(
            gui_render.View, "get_channel", lambda self, title: 0
        )
        warned = []
        monkeypatch.setattr(
            gui_render.QtWidgets.QMessageBox,
            "warning",
            lambda *a, **k: warned.append(a),
        )
        box_view.pick_similar()
        # the shape guard must not fire, and pick_size is None here
        assert warned == []
        assert len(box_view._picks) >= 2

    def test_the_scene_draws_with_a_none_pick_size(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        box_view.update_scene()
        assert box_view.qimage is not None

    def test_the_drag_overlay_paints(self, box_view):
        box_view.mousePressEvent(_Event(40, 40))
        box_view.mouseMoveEvent(_Event(120, 100))
        image = box_view.draw_box_pick_ongoing(box_view.qimage_no_picks.copy())
        assert image is not None


class TestPickRemovalAcrossShapes:
    """``remove_picks`` dispatches through ``lib.point_in_pick``."""

    def _polygon_view(self, window, polygons):
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Polygon")
        view._mode = "Pick"
        view._picks = polygons
        return view

    def test_polygon_click_removes_only_the_one_clicked(self, window):
        # this used to clear every pick: remove_picks had no polygon arm
        left = [(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0), (1.0, 1.0)]
        right = [
            (20.0, 20.0),
            (25.0, 20.0),
            (25.0, 25.0),
            (20.0, 25.0),
            (20.0, 20.0),
        ]
        view = self._polygon_view(window, [left, right])
        view.remove_picks((3.0, 3.0))
        assert len(view._picks) == 1
        assert list(view._picks[0]) == right

    def test_polygon_click_outside_removes_nothing(self, window):
        left = [(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0), (1.0, 1.0)]
        view = self._polygon_view(window, [left])
        view.remove_picks((10.0, 10.0))
        assert len(view._picks) == 1

    def test_vertical_rectangle_survives_a_click_elsewhere(self, window):
        # a perfectly vertical rectangle used to be deleted by any
        # right click, because its corner ray casting divided by zero
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Rectangle")
        window.tools_settings_dialog.pick_width.setValue(2.0 * PIXELSIZE)
        view._picks = [((5.0, 5.0), (5.0, 15.0))]
        view.remove_picks((20.0, 20.0))
        assert len(view._picks) == 1

    def test_vertical_rectangle_is_removed_from_inside(self, window):
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Rectangle")
        window.tools_settings_dialog.pick_width.setValue(2.0 * PIXELSIZE)
        view._picks = [((5.0, 5.0), (5.0, 15.0))]
        view.remove_picks((5.5, 10.0))
        assert view._picks == []


class TestRotationWindowShapes:
    """The 3D window frames and moves a pick of any shape."""

    class _Stub:
        """The attributes ``fit_in_view_rotated`` reads."""

        def __init__(self, pick, pick_shape, pick_size):
            self.pick = pick
            self.pick_shape = pick_shape
            self.pick_size = pick_size

    def _viewport(self, pick, shape, size):
        return rotation.ViewRotation.fit_in_view_rotated(
            self._Stub(pick, shape, size), get_viewport=True
        )

    def test_box_viewport(self):
        (y_min, x_min), (y_max, x_max) = self._viewport(
            ((1.0, 2.0), (5.0, 6.0)), "Box", None
        )
        assert (x_min, x_max, y_min, y_max) == (1.0, 5.0, 2.0, 6.0)

    def test_circle_viewport_is_unchanged(self):
        (y_min, x_min), (y_max, x_max) = self._viewport(
            (5.0, 5.0), "Circle", 2.0
        )
        assert (x_min, x_max, y_min, y_max) == (4.0, 6.0, 4.0, 6.0)

    def test_polygon_viewport_is_unchanged(self):
        polygon = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]
        (y_min, x_min), (y_max, x_max) = self._viewport(
            polygon, "Polygon", None
        )
        assert (x_min, x_max, y_min, y_max) == (0.0, 4.0, 0.0, 4.0)

    def test_no_pick_yet(self):
        assert self._viewport(None, None, None) is None


class TestFilterPicksAcrossShapes:
    """``filter_picks`` counts through ``picked_locs`` off the circular
    fast path."""

    def test_counts_match_picked_locs_for_boxes(self, window):
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Box")
        view._picks = [
            ((4.0, 4.0), (10.0, 10.0)),
            ((20.0, 20.0), (24.0, 22.0)),
        ]
        counts = view._count_locs_in_picks(0)
        expected = [len(_) for _ in view.picked_locs(0, add_group=False)]
        assert list(counts) == expected
        assert all(_ > 0 for _ in counts)

    def test_open_polygons_count_as_empty(self, window):
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Polygon")
        closed = [
            (4.0, 4.0),
            (10.0, 4.0),
            (10.0, 10.0),
            (4.0, 10.0),
            (4.0, 4.0),
        ]
        # picked_locs skips the open one, so the counts would otherwise
        # be misaligned with the picks
        view._picks = [closed, [(20.0, 20.0), (24.0, 20.0)], closed]
        counts = view._count_locs_in_picks(0)
        assert len(counts) == 3
        assert counts[1] == 0
        assert counts[0] > 0 and counts[0] == counts[2]
