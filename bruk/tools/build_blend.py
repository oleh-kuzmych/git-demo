"""Збирає bruk/blender/bruk_schemes.blend: 10 PBR-матеріалів (позначені як
Assets), сфери-прев'ю та майданчик 4x4 м. Опційно рендерить сфери.

Запуск у Blender (4.2+):
    blender -b -P bruk/tools/build_blend.py -- [--render] [--samples 64]
або з модулем bpy:  python build_blend.py --render
"""
import json
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BLEND = os.path.join(ROOT, "blender", "bruk_schemes.blend")
R = 0.5  # радіус сфери, м

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
RENDER = "--render" in argv
SAMPLES = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 64
ONLY = argv[argv.index("--only") + 1] if "--only" in argv else None

schemes = json.load(open(os.path.join(ROOT, "schemes.json"), encoding="utf-8"))["schemes"]
tex_meta = json.load(open(os.path.join(ROOT, "textures", "textures.json")))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.length_unit = "MILLIMETERS"


def make_material(s):
    sid = s["id"]
    tx, ty = (v / 1000 for v in tex_meta[sid]["texture_mm"])
    rng_m = tex_meta[sid]["displacement_range_mm"] / 1000
    mat = bpy.data.materials.new(f"Bruk_{sid}")
    mat.use_nodes = True
    try:
        mat.displacement_method = "BOTH"
    except Exception:
        mat.cycles.displacement_method = "BOTH"
    nt = mat.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links

    out = N.new("ShaderNodeOutputMaterial"); out.location = (900, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (550, 150)
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = N.new("ShaderNodeTexCoord"); tc.location = (-1100, 0)
    mp = N.new("ShaderNodeMapping"); mp.location = (-900, 0)
    mp.label = f"Масштаб: 1 UV = 1 м, текстура {tx*1000:.0f}x{ty*1000:.0f} мм"
    mp.inputs["Scale"].default_value = (1 / tx, 1 / ty, 1)
    L.new(tc.outputs["UV"], mp.inputs["Vector"])

    def img(kind, y, color=False):
        n = N.new("ShaderNodeTexImage"); n.location = (-600, y)
        ext = "png" if kind == "Displacement" else "jpg"
        path = os.path.join(ROOT, "textures", sid, f"{sid}_{kind}.{ext}")
        im = bpy.data.images.load(path, check_existing=True)
        im.colorspace_settings.name = "sRGB" if color else "Non-Color"
        n.image = im
        n.label = kind
        L.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    col = img("BaseColor", 400, True)
    ao = img("AO", 120)
    rough = img("Roughness", -160)
    nrm = img("Normal", -440)
    disp = img("Displacement", -720)

    mix = N.new("ShaderNodeMix"); mix.location = (-200, 350)
    mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.6
    L.new(col.outputs["Color"], mix.inputs["A"])
    L.new(ao.outputs["Color"], mix.inputs["B"])
    L.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    L.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    nm = N.new("ShaderNodeNormalMap"); nm.location = (-200, -440)
    L.new(nrm.outputs["Color"], nm.inputs["Color"])
    L.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    dn = N.new("ShaderNodeDisplacement"); dn.location = (550, -500)
    dn.inputs["Midlevel"].default_value = 0.0
    dn.inputs["Scale"].default_value = rng_m
    L.new(disp.outputs["Color"], dn.inputs["Height"])
    L.new(dn.outputs["Displacement"], out.inputs["Displacement"])

    mat["texture_size_mm"] = tex_meta[sid]["texture_mm"]
    mat["scheme"] = s["name"]
    try:
        mat.asset_mark()
        mat.asset_data.description = f"{s['name']}. Бруківка 90/120/180x120 мм, сіра. {s['note']}"
        mat.asset_data.tags.new("бруківка")
    except Exception as e:
        print("asset_mark:", e)
    return mat


def make_sphere(s, mat, x):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=256, ring_count=128, radius=R, location=(x, 0, R))
    ob = bpy.context.active_object
    ob.name = f"Сфера_{s['id']}"
    bpy.ops.object.shade_smooth()
    tx = tex_meta[s["id"]]["texture_mm"][0] / 1000
    k = max(1, round(2 * math.pi * R / tx))
    uw = k * tx  # окружність, кратна розміру текстури → без шва
    for loop in ob.data.uv_layers.active.data:
        loop.uv = (loop.uv[0] * uw, loop.uv[1] * math.pi * R)
    sub = ob.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 0
    sub.render_levels = 2
    ob.data.materials.append(mat)
    return ob


def make_plane(mat, size=4.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, -4, 0))
    ob = bpy.context.active_object
    ob.name = "Майданчик_4x4м"
    for loop in ob.data.uv_layers.active.data:
        loop.uv = (loop.uv[0] * size, loop.uv[1] * size)
    sub = ob.modifiers.new("Subdivision", "SUBSURF")
    sub.subdivision_type = "SIMPLE"
    sub.levels = 0
    sub.render_levels = 6
    ob.data.materials.append(mat)
    return ob


def setup_world():
    w = bpy.data.worlds.new("Courtyard")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    env = N.new("ShaderNodeTexEnvironment")
    hdr = bpy.utils.system_resource("DATAFILES", path="studiolights/world/courtyard.exr")
    if not hdr:  # bpy як модуль Python
        import glob
        hits = glob.glob(os.path.join(bpy.utils.resource_path("LOCAL"), "datafiles",
                                      "studiolights", "world", "courtyard.exr"))
        hdr = hits[0] if hits else ""
    env.image = bpy.data.images.load(hdr)
    env.image.pack()
    mapn = N.new("ShaderNodeMapping"); tc = N.new("ShaderNodeTexCoord")
    mapn.inputs["Rotation"].default_value[2] = math.radians(110)
    L.new(tc.outputs["Generated"], mapn.inputs["Vector"]); L.new(mapn.outputs["Vector"], env.inputs["Vector"])
    bg = N.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = 0.45
    L.new(env.outputs["Color"], bg.inputs["Color"])
    dark = N.new("ShaderNodeBackground"); dark.inputs["Color"].default_value = (0.022, 0.022, 0.023, 1)
    lp = N.new("ShaderNodeLightPath")
    mix = N.new("ShaderNodeMixShader")
    L.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    L.new(bg.outputs["Background"], mix.inputs[1]); L.new(dark.outputs["Background"], mix.inputs[2])
    out = N.new("ShaderNodeOutputWorld")
    L.new(mix.outputs["Shader"], out.inputs["Surface"])
    sun = bpy.data.lights.new("Сонце", "SUN"); sun.energy = 2.2; sun.angle = math.radians(3)
    so = bpy.data.objects.new("Сонце", sun); scene.collection.objects.link(so)
    so.rotation_euler = (math.radians(62), 0, math.radians(-35))


setup_world()
mats, spheres = {}, {}
for i, s in enumerate(schemes):
    mats[s["id"]] = make_material(s)
    spheres[s["id"]] = make_sphere(s, mats[s["id"]], i * 1.4)
make_plane(mats[schemes[0]["id"]])

cam_data = bpy.data.cameras.new("Камера"); cam_data.lens = 85
cam = bpy.data.objects.new("Камера", cam_data); scene.collection.objects.link(cam)
scene.camera = cam

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = 640
scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.quality = 90
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = -0.35


def frame_sphere(ob):
    x = ob.location.x
    d = 3.1
    el = math.radians(12)
    cam.location = (x + d * math.cos(el) * math.sin(math.radians(0)), -d * math.cos(el), R + d * math.sin(el))
    direction = (x - cam.location.x, 0 - cam.location.y, R - cam.location.z)
    import mathutils
    cam.rotation_euler = mathutils.Vector(direction).to_track_quat("-Z", "Y").to_euler()


def setup_viewports():
    """Файл відкривається одразу в Material Preview з видом на сфери."""
    import mathutils
    for sc in bpy.data.screens:
        for area in sc.areas:
            if area.type != "VIEW_3D":
                continue
            for sp in area.spaces:
                if sp.type != "VIEW_3D":
                    continue
                sp.shading.type = "MATERIAL"
                sp.clip_end = 1000
                r3d = sp.region_3d
                r3d.view_perspective = "PERSP"
                r3d.view_location = ((len(schemes) - 1) * 1.4 / 2, -1.5, 0.4)
                r3d.view_distance = 11
                r3d.view_rotation = mathutils.Euler((math.radians(68), 0, 0)).to_quaternion()


setup_viewports()
os.makedirs(os.path.dirname(BLEND), exist_ok=True)
frame_sphere(spheres[schemes[0]["id"]])
bpy.ops.wm.save_as_mainfile(filepath=BLEND, compress=True)
bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_as_mainfile(filepath=BLEND, compress=True)
print("saved", BLEND)

if RENDER:
    for s in schemes:
        if ONLY and s["id"] != ONLY:
            continue
        for sid, ob in spheres.items():
            ob.hide_render = sid != s["id"]
        bpy.data.objects["Майданчик_4x4м"].hide_render = True
        frame_sphere(spheres[s["id"]])
        scene.render.filepath = os.path.join(ROOT, "previews", f"{s['id']}_sphere.jpg")
        bpy.ops.render.render(write_still=True)
        print("rendered", s["id"], flush=True)
