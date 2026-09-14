"""Display-only studio styling shared by point and robot viewers."""

import numpy as np

SKY_TOP = (0.06, 0.08, 0.12)
SKY_HORIZON = (0.20, 0.25, 0.31)
FLOOR_A = (0.24, 0.28, 0.33)
FLOOR_B = (0.28, 0.32, 0.37)


def floor_texture(size=128):
    y, x = np.indices((size, size))
    tiles = (x >= size // 2) ^ (y >= size // 2)
    return (np.asarray([FLOOR_A, FLOOR_B])[tiles.astype(int)] * 255).astype(np.uint8)


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
    model.vis.rgba.haze[:] = (*SKY_HORIZON, 1)
    model.vis.rgba.fog[:] = (*SKY_HORIZON, 1)
    model.vis.map.fogstart, model.vis.map.fogend = 5, 30
    model.vis.quality.shadowsize = 2048
    model.vis.quality.offsamples = 4
    for i in range(model.nlight):
        model.light_diffuse[i] = (0.65, 0.62, 0.58) if i == 0 else (0.2, 0.23, 0.28)
        model.light_specular[i] = (0.1, 0.1, 0.1)
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
        model.mat_texrepeat[material] = (1, 1)
        texture = int(model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB])
        if texture >= 0 and texture not in robot_textures:
            floor_textures.add(texture)
    for i in range(model.ntex):
        h, w, channels = model.tex_height[i], model.tex_width[i], model.tex_nchannel[i]
        start = model.tex_adr[i]
        data = model.tex_data[start : start + h * w * channels].reshape(h, w, channels)
        if model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            if (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, i)
                != "embodiedforge_studio_sky"
            ):
                data[..., :3] = (255 * np.asarray(SKY_TOP)).astype(np.uint8)
        elif i in floor_textures and channels >= 3:
            y, x = np.indices((h, w))
            tile = (x >= w // 2) ^ (y >= h // 2)
            data[..., :3] = (
                np.asarray([FLOOR_A, FLOOR_B])[tile.astype(int)] * 255
            ).astype(np.uint8)
