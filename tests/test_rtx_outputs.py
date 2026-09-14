"""Regression coverage for OVRTX path-keyed color outputs."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from embodiedforge.viewers.rtx import RtxColorOutputMixin, _normalize_color_outputs


@pytest.mark.parametrize("key", ["LdrColor", "/Render/Vars/LdrColor"])
def test_color_alias_preserves_original_output(key):
    color = object()
    frame = SimpleNamespace(render_vars={key: color, "Depth": object()})
    products = {"product": SimpleNamespace(frames=[frame])}
    assert _normalize_color_outputs(products) is products
    assert frame.render_vars["LdrColor"] is color
    assert frame.render_vars[key] is color
    assert "Depth" in frame.render_vars


@pytest.mark.parametrize("async_rendering", [False, True])
@pytest.mark.parametrize("blit_fails", [False, True])
def test_path_keyed_color_reaches_window(monkeypatch, async_rendering, blit_fails):
    pixels = SimpleNamespace(
        device=SimpleNamespace(stream=SimpleNamespace(cuda_stream=7))
    )
    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(from_dlpack=lambda mapping, dtype: pixels, vec4ub=object()),
    )
    monkeypatch.setitem(
        sys.modules, "ovrtx", SimpleNamespace(Device=SimpleNamespace(CUDA=0))
    )
    color = MagicMock()
    interpolated_color = MagicMock()
    frame = SimpleNamespace(render_vars={"/Render/Vars/LdrColor": color})
    products = {
        "product": SimpleNamespace(
            frames=[
                SimpleNamespace(
                    render_vars={"/Render/Vars/LdrColor": interpolated_color}
                ),
                frame,
            ]
        )
    }
    renderer = MagicMock()
    renderer.step.return_value = products
    result = MagicMock()
    result.wait.return_value.fetch.return_value = products
    viewer = RtxColorOutputMixin()
    viewer._rtx = renderer
    viewer._should_close = False
    viewer._async = async_rendering
    viewer._render_result = result
    viewer._render_product_path = "product"
    viewer._window = SimpleNamespace(context=object())
    viewer._blit_to_window = MagicMock()
    viewer.fps = 60
    if blit_fails:
        viewer._blit_to_window.side_effect = RuntimeError("display failed")
        with pytest.raises(RuntimeError, match="display failed"):
            viewer._render_and_display()
    else:
        viewer._render_and_display()
    viewer._blit_to_window.assert_called_once_with(pixels)
    interpolated_color.map.assert_not_called()
    color.map.return_value.__enter__.return_value.unmap.assert_called_once_with(
        stream=7
    )
    if async_rendering:
        result.wait.assert_called_once()
        if blit_fails:
            renderer.step_async.assert_not_called()
        else:
            renderer.step_async.assert_called_once()
        renderer.step.assert_not_called()
    else:
        renderer.step.assert_called_once()
        renderer.step_async.assert_not_called()


@pytest.mark.parametrize("products", [{}, {"product": SimpleNamespace(frames=[])}])
def test_empty_render_is_not_ready(products):
    with pytest.raises(RuntimeError, match="no frames"):
        _normalize_color_outputs(products)


def test_missing_color_is_rejected_without_a_window(monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ovrtx", SimpleNamespace(Device=object()))
    viewer = RtxColorOutputMixin()
    viewer._rtx = MagicMock()
    viewer._rtx.step.return_value = {
        "product": SimpleNamespace(
            frames=[SimpleNamespace(render_vars={"Depth": object()})]
        )
    }
    viewer._should_close = False
    viewer._async = False
    viewer._window = None
    viewer._render_product_path = "product"
    viewer.fps = 60
    with pytest.raises(RuntimeError, match="no LdrColor.*Depth"):
        viewer._render_and_display()


def test_first_async_frame_is_available_for_headless_capture(monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ovrtx", SimpleNamespace(Device=object()))
    viewer = RtxColorOutputMixin()
    viewer._rtx = MagicMock()
    viewer._rtx.step.return_value = {
        "product": SimpleNamespace(
            frames=[SimpleNamespace(render_vars={"LdrColor": object()})]
        )
    }
    viewer._should_close = False
    viewer._async = True
    viewer._render_result = None
    viewer._window = None
    viewer._render_product_path = "product"
    viewer.fps = 60
    viewer._render_and_display()
    viewer._rtx.step.assert_called_once()
    viewer._rtx.step_async.assert_called_once()
    assert viewer._render_products is viewer._rtx.step.return_value


def test_screenshot_copies_last_presented_frame_not_pending_render(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "ovrtx", SimpleNamespace(Device=SimpleNamespace(CPU=0))
    )
    older = MagicMock()
    latest = MagicMock()
    source = np.full((2, 3, 4), 127, dtype=np.uint8)
    latest.map.return_value.__enter__.return_value = source
    viewer = RtxColorOutputMixin()
    viewer._render_product_path = "product"
    viewer._render_products = {
        "product": SimpleNamespace(
            frames=[
                SimpleNamespace(render_vars={"LdrColor": older}),
                SimpleNamespace(render_vars={"/Render/Vars/LdrColor": latest}),
            ]
        )
    }
    viewer._render_result = MagicMock()
    pixels = viewer._capture_screenshot_pixels()
    source[:] = 0  # The renderer can reuse its buffer after unmapping.
    assert np.all(pixels == 127)
    older.map.assert_not_called()
    viewer._render_result.wait.assert_not_called()


def test_wrong_render_product_does_not_silently_show_another_camera(monkeypatch):
    from embodiedforge.viewers.rtx import _latest_color_output

    products = {
        "other": SimpleNamespace(
            frames=[SimpleNamespace(render_vars={"LdrColor": object()})]
        )
    }
    with pytest.raises(RuntimeError, match="requested product expected"):
        _latest_color_output(products, "expected")
