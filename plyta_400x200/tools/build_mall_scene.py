"""Візуалізація: площа перед торговим центром, вимощена плитою 400x200 (колормікс).

Запуск:  blender -b -P plyta_400x200/tools/build_mall_scene.py -- [--samples 128] [--res 1920x1080] [--scheme 01_running_half]
Результат: blender/plyta_mall_scene.blend і previews/mall_render.jpg
"""
import json
import math
import os
import sys

import bpy
import mathutils

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
SAMPLES = int(opt("--samples", 128))
RX, RY = (int(v) for v in opt("--res", "1920x1080").split("x"))
SCHEME = opt("--scheme", "01_running_half")
TEX = SCHEME + "_XL" if os.path.isdir(os.path.join(ROOT, "textures", SCHEME + "_XL")) else SCHEME
OUT = opt("--out", os.path.join(ROOT, "previews", "mall_render.jpg"))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene
meta = json.load(open(os.path.join(ROOT, "textures", "textures.json")))[TEX]


# ------------------------------------------------------------------ матеріали
def principled(name, color, rough=0.5, metal=0.0, **kw):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    for k, v in kw.items():
        b.inputs[k].default_value = v
    return m


def emission(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def paving_material():
    tx, ty = (v / 1000 for v in meta["texture_mm"])
    m = bpy.data.materials.new(f"Plyta_{TEX}")
    m.use_nodes = True
    try:
        m.displacement_method = "BUMP"
    except Exception:
        m.cycles.displacement_method = "BUMP"
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    out = N["Material Output"]
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / tx, 1 / ty, 1)
    L.new(tc.outputs["UV"], mp.inputs["Vector"])

    def img(kind, color=False, ext="jpg"):
        n = N.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(os.path.join(ROOT, "textures", TEX, f"{TEX}_{kind}.{ext}"))
        n.image.colorspace_settings.name = "sRGB" if color else "Non-Color"
        L.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n
    col, ao, rough, nrm, disp = img("BaseColor", True), img("AO"), img("Roughness"), img("Normal"), img("Displacement8")
    mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.6
    L.new(col.outputs["Color"], mix.inputs["A"]); L.new(ao.outputs["Color"], mix.inputs["B"])
    L.new(mix.outputs["Result"], b.inputs["Base Color"])
    L.new(rough.outputs["Color"], b.inputs["Roughness"])
    nm = N.new("ShaderNodeNormalMap"); L.new(nrm.outputs["Color"], nm.inputs["Color"])
    L.new(nm.outputs["Normal"], b.inputs["Normal"])
    dn = N.new("ShaderNodeDisplacement")
    dn.inputs["Midlevel"].default_value = 0.0
    dn.inputs["Scale"].default_value = meta["displacement_range_mm"] / 1000
    L.new(disp.outputs["Color"], dn.inputs["Height"]); L.new(dn.outputs[0], out.inputs["Displacement"])
    return m


PAVE = paving_material()
GLASS = principled("Скло", (0.80, 0.88, 0.90), rough=0.02, **{"Transmission Weight": 1.0, "IOR": 1.5})
FRAME = principled("Анодований алюміній", (0.045, 0.048, 0.052), rough=0.32, metal=1.0)
PANEL = principled("Фасадна панель", (0.62, 0.63, 0.64), rough=0.38, metal=0.35)
PANEL2 = principled("Фасадна панель темна", (0.20, 0.21, 0.22), rough=0.4, metal=0.3)
WOOD = principled("Ламелі під дерево", (0.38, 0.20, 0.09), rough=0.55)
CONCRETE = principled("Бетон", (0.50, 0.49, 0.47), rough=0.85)
GRANITE = principled("Граніт бордюр", (0.22, 0.22, 0.23), rough=0.5)
INTERIOR = principled("Інтер'єр", (0.85, 0.83, 0.80), rough=0.6)
SHOP = [principled(f"Вітрина {i}", c, rough=0.5) for i, c in enumerate(((0.75, 0.25, 0.15), (0.15, 0.3, 0.55), (0.9, 0.85, 0.75), (0.2, 0.45, 0.3)))]
LIGHT = emission("Світло інтер'єру", (1.0, 0.86, 0.68), 18.0)
SIGN = emission("Вивіска", (1.0, 1.0, 1.0), 7.0)
LEAF = principled("Листя", (0.06, 0.16, 0.035), rough=0.7, **{"Subsurface Weight": 0.15})
BARK = principled("Кора", (0.12, 0.09, 0.07), rough=0.9)
SOIL = principled("Ґрунт", (0.09, 0.07, 0.05), rough=1.0)


def box(name, center, size, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat)
    return o


# ------------------------------------------------------------------ площа
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Площа_плита"
ground.scale = (220, 220, 1)
bpy.ops.object.transform_apply(scale=True)
for lp in ground.data.uv_layers.active.data:   # UV у метрах (+ випадковий зсув, щоб шов не на осі)
    lp.uv = (lp.uv[0] * 220 + 0.37, lp.uv[1] * 220 + 0.11)
ground.data.materials.append(PAVE)
ground.location = (0, -10, 0)

# ------------------------------------------------------------------ будівля ТРЦ
FY = 14.0                       # лінія фасаду
box("Обʼєм ТРЦ", (0, FY + 15, 6.0), (48, 30, 12), PANEL)
# вітрина першого поверху
box("Скло вітрини", (0, FY - 0.02, 2.3), (44, 0.03, 4.4), GLASS)
for i in range(-22, 23, 2):
    box(f"Стійка {i}", (i, FY - 0.08, 2.3), (0.09, 0.16, 4.6), FRAME)
box("Ригель низ", (0, FY - 0.08, 0.05), (44, 0.18, 0.1), FRAME)
box("Ригель верх", (0, FY - 0.08, 4.55), (44, 0.2, 0.2), FRAME)
box("Ригель середній", (0, FY - 0.08, 3.2), (44, 0.12, 0.06), FRAME)
# інтер'єр за склом
box("Підлога інтерʼєру", (0, FY + 6, 0.005), (44, 12, 0.01), INTERIOR)
box("Стеля інтерʼєру", (0, FY + 6, 4.45), (44, 12, 0.1), INTERIOR)
box("Задня стіна", (0, FY + 12, 2.2), (44, 0.2, 4.4), INTERIOR)
for i in range(-20, 21, 4):
    for j in (2.5, 6.5, 10.0):
        box(f"Світильник {i}_{j}", (i, FY + j, 4.38), (2.6, 0.18, 0.04), LIGHT)
for k, x in enumerate(range(-19, 20, 6)):
    box(f"Острів {k}", (x, FY + 6 + (k % 2) * 2, 0.6), (2.2, 1.0, 1.2), SHOP[k % 4])
    box(f"Стелаж {k}", (x + 1.5, FY + 11.5, 1.4), (3.5, 0.6, 2.8), SHOP[(k + 1) % 4])
# фасадні панелі верхньої частини
for i in range(-22, 22):
    for j in range(5):
        z = 4.75 + j * 1.45
        mat = PANEL2 if (i in (-1, 0)) else PANEL
        box(f"Панель {i}_{j}", (i + 0.5, FY - 0.05, z + 0.71), (0.98, 0.06, 1.42), mat)
for k in range(28):                   # дерев'яні ламелі праворуч
    box(f"Ламель {k}", (6.5 + k * 0.55, FY - 0.3, 8.3), (0.12, 0.45, 7.0), WOOD)
box("Парапет", (0, FY - 0.15, 12.1), (48, 0.4, 0.3), FRAME)
# навіс над входом і колони
box("Навіс", (0, FY - 2.5, 4.75), (16, 5.0, 0.35), FRAME)
for x in (-7.4, 7.4):
    box(f"Колона {x}", (x, FY - 4.6, 2.3), (0.3, 0.3, 4.6), FRAME)
for i in range(-7, 8, 2):
    for j in (1.2, 3.4):
        box(f"Даунлайт {i}_{j}", (i, FY - j, 4.56), (0.25, 0.25, 0.02), LIGHT)
# вивіска (без торгових марок)
bpy.ops.object.text_add(location=(-2.6, FY - 0.2, 6.6), rotation=(math.radians(90), 0, 0))
t = bpy.context.active_object
t.name = "Вивіска ТРЦ"
t.data.body = "ТРЦ"
t.data.size = 2.6
t.data.extrude = 0.08
t.data.materials.append(SIGN)

# ------------------------------------------------------------------ благоустрій
def tree(name, x, y, h=4.0, r=1.8):
    import random
    rnd = random.Random(name)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=h, location=(x, y, 0.55 + h / 2))
    tr = bpy.context.active_object; tr.name = f"{name} стовбур"; tr.data.materials.append(BARK)
    for k in range(4):                         # гілки
        ang = rnd.uniform(0, 2 * math.pi)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1.4, location=(x + 0.4 * math.cos(ang), y + 0.4 * math.sin(ang), 0.55 + h - 0.2),
                                            rotation=(math.radians(50) * math.sin(ang), math.radians(50) * math.cos(ang), 0))
        bpy.context.active_object.data.materials.append(BARK)
    tex = bpy.data.textures.new(f"{name}_tex", "CLOUDS"); tex.noise_scale = 0.18; tex.noise_depth = 3
    leaf = LEAF.copy(); leaf.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        0.05 + rnd.uniform(-0.01, 0.02), 0.15 + rnd.uniform(-0.03, 0.03), 0.03, 1)
    for k in range(22):                        # крона з багатьох кластерів листя
        u, v = rnd.uniform(0, 2 * math.pi), rnd.uniform(-0.6, 1.0)
        rr = r * rnd.uniform(0.35, 0.9)
        cx, cy, cz = x + rr * math.cos(u) * math.sqrt(1 - v * v * 0.5), y + rr * math.sin(u) * math.sqrt(1 - v * v * 0.5), 0.55 + h + v * r * 0.6
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=rnd.uniform(0.45, 0.8), location=(cx, cy, cz))
        c = bpy.context.active_object; c.name = f"{name} листя {k}"
        md = c.modifiers.new("Листя", "DISPLACE"); md.texture = tex; md.strength = 0.35
        c.data.materials.append(leaf)
        bpy.ops.object.shade_smooth()


def planter(name, x, y, w=3.2, d=1.4):
    box(f"{name} кадка", (x, y, 0.27), (w, d, 0.55), CONCRETE)
    box(f"{name} ґрунт", (x, y, 0.53), (w - 0.16, d - 0.16, 0.02), SOIL)
    box(f"{name} лавка", (x, y - d / 2 - 0.25, 0.45), (w * 0.85, 0.45, 0.06), WOOD)
    tree(name, x, y)


for k, (x, y) in enumerate(((-13, 3), (-13, -6), (13, 3), (13, -6), (4.5, -4))):
    planter(f"Озеленення {k}", x, y)
for x in range(-6, 7, 2):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.11, depth=0.9, location=(x, FY - 7.5, 0.45))
    bpy.context.active_object.data.materials.append(FRAME)
for k, (x, y) in enumerate(((-9, -2), (9, -2), (-20, -2), (20, -2))):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.07, depth=5.5, location=(x, y, 2.75))
    bpy.context.active_object.data.materials.append(FRAME)
    box(f"Ліхтар {k}", (x, y, 5.55), (0.5, 0.18, 0.1), FRAME)
box("Бордюр фасаду", (0, FY - 0.35, 0.03), (48, 0.3, 0.06), GRANITE)

# ------------------------------------------------------------------ світло, небо, камера
w = bpy.data.worlds.new("Небо"); scene.world = w
w.use_nodes = True
nt = w.node_tree; nt.nodes.clear()
sky = nt.nodes.new("ShaderNodeTexSky")
for st in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
    try:
        sky.sky_type = st
        break
    except Exception:
        continue
SUN_EL, SUN_AZ = 36, 205   # висота і азимут сонця, °
for attr, val in (("sun_elevation", math.radians(SUN_EL)), ("sun_rotation", math.radians(SUN_AZ))):
    if hasattr(sky, attr):
        setattr(sky, attr, val)
if hasattr(sky, "sun_disc"):
    sky.sun_disc = False
bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = 0.17
o = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(sky.outputs[0], bg.inputs[0]); nt.links.new(bg.outputs[0], o.inputs[0])
sun = bpy.data.lights.new("Сонце", "SUN"); sun.energy = 4.0; sun.angle = math.radians(0.8)
sun.color = (1.0, 0.92, 0.82)
so = bpy.data.objects.new("Сонце", sun); scene.collection.objects.link(so)
so.rotation_euler = mathutils.Euler((math.radians(90 - SUN_EL), 0, math.radians(SUN_AZ - 180)))

cam = bpy.data.objects.new("Камера", bpy.data.cameras.new("Камера"))
scene.collection.objects.link(cam); scene.camera = cam
cam.data.lens = 24
cam.data.sensor_width = 36
cam.location = (-8.5, -15.5, 1.65)
target = mathutils.Vector((1.5, FY, 1.9))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.shift_y = 0.0

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
scene.render.resolution_x, scene.render.resolution_y = RX, RY
scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.quality = 92
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = float(opt("--exposure", -0.9))

blend = os.path.join(ROOT, "blender", "plyta_mall_scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
print("saved", blend, "sky:", sky.sky_type)
if "--no-render" not in argv:
    scene.render.filepath = OUT
    bpy.ops.render.render(write_still=True)
    print("rendered", OUT)
