"""Color-output compatibility for the pinned Newton 1.6.0rc1 viewer.

OVRTX 0.5 returns RenderVar paths; Newton expects the short ``LdrColor`` key.
Keep this adaptation local to our viewer, without modifying either installed SDK.
"""


def _normalize_color_outputs(products):
    frame_count = 0
    for product in products.values():
        for frame in product.frames:
            frame_count += 1
            outputs = frame.render_vars
            if "LdrColor" not in outputs and "/Render/Vars/LdrColor" in outputs:
                outputs["LdrColor"] = outputs["/Render/Vars/LdrColor"]
            if "LdrColor" not in outputs:
                raise RuntimeError(
                    f"RTX frame has no LdrColor output; received {list(outputs)}"
                )
    if not frame_count:
        raise RuntimeError("RTX renderer returned no frames")
    return products


def _latest_color_output(products, product_path):
    """OVRTX's final capture is the real frame, after any interpolated frames."""
    _normalize_color_outputs(products)
    if product_path not in products:
        raise RuntimeError(
            f"RTX renderer did not return requested product {product_path}"
        )
    frames = products[product_path].frames
    if not frames:
        raise RuntimeError(f"RTX renderer returned no frames for {product_path}")
    return frames[-1].render_vars["LdrColor"]


class RtxColorOutputMixin:
    """Adapt Newton's presentation loop while preserving async frame ownership.

    Uses private ViewerRTX hooks from the version checked by NewtonViewer.
    Recheck this integration when changing the pinned Newton version.
    """

    def _add_studio_lights(self):
        """Use broad neutral illumination with a soft key and opposite fill."""
        from pxr import Gf, UsdGeom, UsdLux

        dome = UsdLux.DomeLight.Define(self.stage, "/root/StudioAmbient")
        dome.CreateColorAttr(Gf.Vec3f(0.55, 0.68, 0.85))
        dome.CreateIntensityAttr(500)
        for name, rotation, color, intensity in (
            ("Key", (35, -30, 0), (1.0, 0.95, 0.87), 1200),
            ("Fill", (-25, 35, 0), (0.86, 0.92, 1.0), 700),
        ):
            path = f"/root/Studio{name}"
            transform = UsdGeom.Xform.Define(self.stage, path)
            transform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
            light = UsdLux.DistantLight.Define(self.stage, path + "/Light")
            light.CreateColorAttr(Gf.Vec3f(*color))
            light.CreateIntensityAttr(intensity)
            light.CreateAngleAttr(8 if name == "Key" else 15)
            # A fill without a second cast shadow keeps the contact cue readable.
            if name == "Fill":
                UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(False)

    def _render_and_display(self):
        import warp as wp
        from ovrtx import Device

        if self._rtx is None or self._should_close:
            return

        self._render_products = None
        if self._async and self._render_result is not None:
            self._render_products = self._render_result.wait().fetch()
        else:
            # Present a real first frame before reporting readiness. Later frames
            # retain Newton's asynchronous rendering/presentation pipeline.
            self._render_products = self._rtx.step(
                render_products={self._render_product_path}, delta_time=1.0 / self.fps
            )

        if self._render_products is None:
            raise RuntimeError("RTX renderer returned no frame products")
        color = _latest_color_output(self._render_products, self._render_product_path)
        if self._window is not None and self._window.context is not None:
            with color.map(device=Device.CUDA) as mapping:
                pixels = wp.from_dlpack(mapping, dtype=wp.vec4ub)
                try:
                    self._blit_to_window(pixels)
                finally:
                    mapping.unmap(stream=pixels.device.stream.cuda_stream)

        if self._async:
            self._render_result = self._rtx.step_async(
                render_products={self._render_product_path}, delta_time=1.0 / self.fps
            )

    def _capture_screenshot_pixels(self):
        import numpy as np
        from ovrtx import Device

        if self._render_products is None and self._render_result is not None:
            self._render_products = self._render_result.wait().fetch()
        if self._render_products is None:
            raise RuntimeError(
                "save_screenshot() requires at least one completed render frame"
            )
        color = _latest_color_output(self._render_products, self._render_product_path)
        with color.map(device=Device.CPU) as mapping:
            return np.array(np.from_dlpack(mapping), copy=True)
