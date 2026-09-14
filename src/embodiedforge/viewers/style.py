"""Display-only studio styling shared by point and robot viewers."""

from functools import lru_cache

import numpy as np

SKY_TOP = (0.12, 0.17, 0.24)
SKY_HORIZON = (0.36, 0.42, 0.49)
FLOOR_A = (0.32, 0.36, 0.41)
FLOOR_B = (0.33, 0.37, 0.42)


def floor_texture(size=128, width=None):
    """Seamless matte tiles: quiet surfaces with a narrow, antialiased joint."""
    width = size if width is None else width
    y, x = np.indices((size, width))
    tiles = (x >= width // 2) ^ (y >= size // 2)
    color = np.asarray([FLOOR_A, FLOOR_B])[tiles.astype(int)]
    # Sample pixel centers symmetrically on both sides of each wrapping seam.
    u, v = (x + 0.5) / width * 2, (y + 0.5) / size * 2
    distance = np.minimum(abs(u - np.round(u)), abs(v - np.round(v)))
    joint = np.clip((0.016 - distance) * min(size, width), 0, 1)
    color = color - joint[..., None] * 0.035
    return np.round(color * 255).astype(np.uint8)


def floor_period(model):
    """Two tiles per repeat: 0.5 m cells for small robots, 1 m for humanoids."""
    return 1.0 if model.stat.extent < 1.2 else 2.0


def texture_pixels(model, index):
    """Writable RGB(A) storage, including MuJoCo 3.2 point-viewer textures."""
    h, w = int(model.tex_height[index]), int(model.tex_width[index])
    channels = int(model.tex_nchannel[index]) if hasattr(model, "tex_nchannel") else 3
    storage = model.tex_data if hasattr(model, "tex_data") else model.tex_rgb
    start = model.tex_adr[index]
    return storage[start : start + h * w * channels].reshape(h, w, channels)


@lru_cache(maxsize=4)
def _sky_texture(size):
    """Let MuJoCo orient the gradient on all six cubemap faces correctly."""
    import mujoco

    def rgb(values):
        return " ".join(map(str, values))

    model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
      <texture type="skybox" builtin="gradient" width="{size}" height="{size}"
      rgb1="{rgb(SKY_TOP)}" rgb2="{rgb(SKY_HORIZON)}"/>
      </asset></mujoco>''')
    return texture_pixels(model, 0)[..., :3].copy()


def add_mujoco_assets(spec):
    """Add a studio sky and tiled floor to a renderer-owned MJCF specification."""
    import mujoco

    skies = [
        texture
        for texture in spec.textures
        if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX
    ]
    sky = skies[0] if skies else spec.add_texture(name="embodiedforge_studio_sky")
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2 = SKY_TOP, SKY_HORIZON
    sky.width, sky.height = 64, 64
    floor = spec.add_texture(name="embodiedforge_studio_tiles")
    floor.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor.rgb1, floor.rgb2 = FLOOR_A, FLOOR_B
    floor.width, floor.height = 128, 128
    material = spec.add_material(name="embodiedforge_studio_floor")
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = floor.name
    material.texuniform = True
    material.texrepeat = (1, 1)
    material.reflectance, material.shininess, material.specular = 0, 0.1, 0.1
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.material = material.name
            geom.rgba = (1, 1, 1, 1)


def style_mujoco_lighting(model):
    """Lighting fields also supported by the point viewer's MuJoCo 3.2 minimum."""
    model.vis.headlight.ambient[:] = (0.45, 0.48, 0.52)
    model.vis.headlight.diffuse[:] = (0.80, 0.80, 0.80)
    model.vis.headlight.specular[:] = (0.08, 0.08, 0.08)
    model.vis.headlight.active = 1
    model.vis.rgba.haze[:] = (*SKY_HORIZON, 1)
    model.vis.rgba.fog[:] = (*SKY_HORIZON, 1)
    model.vis.map.fogstart, model.vis.map.fogend = 5, 30
    model.vis.quality.shadowsize = 2048
    model.vis.quality.offsamples = 4
    for i in range(model.nlight):
        model.light_diffuse[i] = (0.75, 0.71, 0.65) if i == 0 else (0.45, 0.50, 0.58)
        model.light_specular[i] = (0.18, 0.18, 0.18)
        model.light_castshadow[i] = i == 0


def style_mujoco_model(model):
    """Change visual fields only, including sky/floor textures in compiled Go1 MJBs."""
    import mujoco

    style_mujoco_lighting(model)
    floor_materials = set(
        model.geom_matid[model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE]
    ) - {-1}
    floor_textures = set()
    robot_materials = set(
        model.geom_matid[model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE]
    ) - {-1}
    robot_textures = {
        int(t)
        for material in robot_materials
        for t in model.mat_texid[material]
        if t >= 0
    }
    for material in floor_materials:
        # Keep textures shared with robot surfaces intact.
        if np.any(
            (model.geom_matid == material)
            & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
        ):
            continue
        model.mat_rgba[material] = (1, 1, 1, 1)
        model.mat_specular[material] = 0.1
        model.mat_shininess[material] = 0.1
        model.mat_reflectance[material] = 0
        # MuJoCo's uniform mapping spans two world units per texture repeat.
        model.mat_texrepeat[material] = 2 / floor_period(model)
        texture = int(model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB])
        if texture >= 0 and texture not in robot_textures:
            floor_textures.add(texture)
    for i in range(model.ntex):
        data = texture_pixels(model, i)
        h, w, channels = data.shape
        if model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            if h == 6 * w and channels >= 3 and i not in robot_textures:
                data[..., :3] = _sky_texture(w)
        elif i in floor_textures and channels >= 3:
            data[..., :3] = floor_texture(h, w)
