"""Фотореалістична візуалізація плити 400x200 (колормікс) — ракурс згори, як у
каталогах виробників: вирізи з декоративними злаками, бетонні вазони з дерев'яними
лавками, самшит у горщиках, ажурна тінь від крон, розмитий кущ на передньому плані.
Усе генерується процедурно (без сторонніх моделей).

Запуск:  blender -b -P plyta_400x200/tools/build_plaza_render.py -- [--samples 256] [--res 1080x1520] [--scheme 01_running_half]
Результат: blender/plyta_plaza.blend і previews/plaza_render.jpg
"""
import json
import math
import os
import random
import sys

import bpy
import bmesh  # noqa: E402 — доступний лише після bpy
import mathutils
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
SAMPLES = int(opt("--samples", 256))
RX, RY = (int(v) for v in opt("--res", "1080x1520").split("x"))
SCHEME = opt("--scheme", "01_running_half")
TEX = SCHEME + "_XL" if os.path.isdir(os.path.join(ROOT, "textures", SCHEME + "_XL")) else SCHEME
OUT = opt("--out", os.path.join(ROOT, "previews", "plaza_render.jpg"))
QUICK = "--quick" in argv          # менше листя/трави для тестів

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene
col_main = scene.collection
tmeta = json.load(open(os.path.join(ROOT, "textures", "textures.json")))[TEX]
R = random.Random(7)


# =================================================================== матеріали
def node_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree.nodes, m.node_tree.links, m.node_tree.nodes["Principled BSDF"]


def paving():
    tx, ty = (v / 1000 for v in tmeta["texture_mm"])
    m, N, L, b = node_mat(f"Plyta_{TEX}")
    try:
        m.displacement_method = "BUMP"
    except Exception:
        pass
    tc = N.new("ShaderNodeTexCoord"); mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / tx, 1 / ty, 1)
    L.new(tc.outputs["UV"], mp.inputs["Vector"])

    def img(kind, color=False, ext="jpg"):
        n = N.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(os.path.join(ROOT, "textures", TEX, f"{TEX}_{kind}.{ext}"))
        n.image.colorspace_settings.name = "sRGB" if color else "Non-Color"
        n.interpolation = "Cubic"
        L.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n
    col, ao, rough, nrm, disp = img("BaseColor", True), img("AO"), img("Roughness"), img("Normal"), img("Displacement8")
    mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.5
    L.new(col.outputs["Color"], mix.inputs["A"]); L.new(ao.outputs["Color"], mix.inputs["B"])
    L.new(mix.outputs["Result"], b.inputs["Base Color"])
    L.new(rough.outputs["Color"], b.inputs["Roughness"])
    nm = N.new("ShaderNodeNormalMap"); L.new(nrm.outputs["Color"], nm.inputs["Color"]); L.new(nm.outputs[0], b.inputs["Normal"])
    dn = N.new("ShaderNodeDisplacement"); dn.inputs["Midlevel"].default_value = 0.0
    dn.inputs["Scale"].default_value = tmeta["displacement_range_mm"] / 1000
    L.new(disp.outputs["Color"], dn.inputs["Height"]); L.new(dn.outputs[0], N["Material Output"].inputs["Displacement"])
    return m


def concrete(name, base=(0.47, 0.465, 0.45)):
    m, N, L, b = node_mat(name)
    tc = N.new("ShaderNodeTexCoord")
    nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 3.0; nz.inputs["Detail"].default_value = 8
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (base[0] * 0.86, base[1] * 0.86, base[2] * 0.86, 1)
    ramp.color_ramp.elements[1].color = (base[0] * 1.1, base[1] * 1.1, base[2] * 1.1, 1)
    L.new(nz.outputs["Fac"], ramp.inputs["Fac"]); L.new(ramp.outputs["Color"], b.inputs["Base Color"])
    fine = N.new("ShaderNodeTexNoise"); fine.inputs["Scale"].default_value = 180; fine.inputs["Detail"].default_value = 4
    L.new(tc.outputs["Object"], fine.inputs["Vector"])
    bump = N.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.12
    L.new(fine.outputs["Fac"], bump.inputs["Height"]); L.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = 0.82
    return m


def wood():
    m, N, L, b = node_mat("Дерево лавки")
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1, 18, 18)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    wv = N.new("ShaderNodeTexWave"); wv.wave_type = "BANDS"; wv.bands_direction = "X"
    wv.inputs["Scale"].default_value = 1.4; wv.inputs["Distortion"].default_value = 6; wv.inputs["Detail"].default_value = 4
    L.new(mp.outputs["Vector"], wv.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.33, 0.17, 0.07, 1)
    ramp.color_ramp.elements[1].color = (0.62, 0.38, 0.19, 1)
    L.new(wv.outputs["Fac"], ramp.inputs["Fac"]); L.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.55
    return m


def foliage(name, c1, c2, rough=0.45, transl=0.25):
    """Листя: колір варіюється між екземплярами (Object Info → Random), просвічує."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    N, L = m.node_tree.nodes, m.node_tree.links
    b = N["Principled BSDF"]
    oi = N.new("ShaderNodeObjectInfo")
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c1, 1); ramp.color_ramp.elements[1].color = (*c2, 1)
    L.new(oi.outputs["Random"], ramp.inputs["Fac"]); L.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = 0.6
    tr = N.new("ShaderNodeBsdfTranslucent"); L.new(ramp.outputs["Color"], tr.inputs["Color"])
    mix = N.new("ShaderNodeMixShader"); mix.inputs["Fac"].default_value = transl
    L.new(b.outputs[0], mix.inputs[1]); L.new(tr.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], N["Material Output"].inputs["Surface"])
    return m


def grass_mat(name, base, tip, plume=None):
    m = bpy.data.materials.new(name); m.use_nodes = True
    N, L = m.node_tree.nodes, m.node_tree.links
    b = N["Principled BSDF"]
    tc = N.new("ShaderNodeTexCoord"); sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*base, 1); ramp.color_ramp.elements[1].color = (*tip, 1)
    ramp.color_ramp.elements[0].position = 0.1
    L.new(sep.outputs["Z"], ramp.inputs["Fac"])
    oi = N.new("ShaderNodeObjectInfo")
    hsv = N.new("ShaderNodeHueSaturation")
    mr = N.new("ShaderNodeMapRange"); mr.inputs["To Min"].default_value = 0.8; mr.inputs["To Max"].default_value = 1.15
    L.new(oi.outputs["Random"], mr.inputs["Value"]); L.new(mr.outputs[0], hsv.inputs["Value"])
    L.new(ramp.outputs["Color"], hsv.inputs["Color"]); L.new(hsv.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.5
    tr = N.new("ShaderNodeBsdfTranslucent"); L.new(hsv.outputs["Color"], tr.inputs["Color"])
    mix = N.new("ShaderNodeMixShader"); mix.inputs["Fac"].default_value = 0.3
    L.new(b.outputs[0], mix.inputs[1]); L.new(tr.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], N["Material Output"].inputs["Surface"])
    return m


PAVE = paving()
CONC = concrete("Бетон вазонів")
CONC2 = concrete("Бетон сходів", (0.52, 0.515, 0.5))
POT = concrete("Бетон горщика", (0.42, 0.42, 0.41))
WOOD = wood()
SOIL, _, _, sb = node_mat("Мульча")[0], None, None, None
_m, _N, _L, _b = SOIL, SOIL.node_tree.nodes, SOIL.node_tree.links, SOIL.node_tree.nodes["Principled BSDF"]
_b.inputs["Base Color"].default_value = (0.05, 0.035, 0.022, 1); _b.inputs["Roughness"].default_value = 0.95
_nz = _N.new("ShaderNodeTexNoise"); _nz.inputs["Scale"].default_value = 60; _nz.inputs["Detail"].default_value = 6
_bp = _N.new("ShaderNodeBump"); _bp.inputs["Strength"].default_value = 0.6
_L.new(_nz.outputs["Fac"], _bp.inputs["Height"]); _L.new(_bp.outputs[0], _b.inputs["Normal"])
BARK = concrete("Кора", (0.075, 0.065, 0.055))
LEAF_TREE = foliage("Листя дерева", (0.07, 0.17, 0.035), (0.20, 0.32, 0.08))
LEAF_BOX = foliage("Листя самшиту", (0.04, 0.12, 0.02), (0.12, 0.25, 0.04), rough=0.35, transl=0.15)
LEAF_SHRUB = foliage("Листя куща", (0.05, 0.16, 0.04), (0.16, 0.30, 0.07), rough=0.3)
CAREX = grass_mat("Осока", (0.06, 0.13, 0.03), (0.22, 0.34, 0.09))
PENNI = grass_mat("Пенісетум", (0.07, 0.15, 0.04), (0.30, 0.38, 0.12))
PLUME = grass_mat("Колоски", (0.32, 0.27, 0.17), (0.60, 0.52, 0.38))


# =================================================================== геометрія
def box(name, c, s, mat, bevel=0.012):
    bpy.ops.mesh.primitive_cube_add(size=1, location=c)
    o = bpy.context.active_object; o.name = name; o.scale = s
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        bv = o.modifiers.new("Фаска", "BEVEL"); bv.width = bevel; bv.segments = 2
    o.data.materials.append(mat)
    return o


def world_uv(o):
    """UV = світові координати в метрах (масштаб плитки 1:1)."""
    mw = o.matrix_world
    uv = o.data.uv_layers.active or o.data.uv_layers.new()
    for poly in o.data.polygons:
        for li in poly.loop_indices:
            co = mw @ o.data.vertices[o.data.loops[li].vertex_index].co
            uv.data[li].uv = (co.x, co.y)


def paved_area(name, x0, x1, y0, y1, z, holes=()):
    bpy.ops.mesh.primitive_plane_add(size=1, location=((x0 + x1) / 2, (y0 + y1) / 2, z))
    o = bpy.context.active_object; o.name = name
    o.scale = (x1 - x0, y1 - y0, 1); bpy.ops.object.transform_apply(scale=True)
    for k, (hx0, hx1, hy0, hy1) in enumerate(holes):
        cut = box(f"cut{k}", ((hx0 + hx1) / 2, (hy0 + hy1) / 2, z), (hx1 - hx0, hy1 - hy0, 0.5), PAVE, bevel=0)
        md = o.modifiers.new("Виріз", "BOOLEAN"); md.object = cut; md.operation = "DIFFERENCE"; md.solver = "EXACT"
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=md.name)
        bpy.data.objects.remove(cut)
    o.data.materials.clear(); o.data.materials.append(PAVE)
    world_uv(o)
    return o


def bed(name, x0, x1, y0, y1):
    box(f"{name} мульча", ((x0 + x1) / 2, (y0 + y1) / 2, -0.05), (x1 - x0, y1 - y0, 0.1), SOIL, bevel=0)
    box(f"{name} край", ((x0 + x1) / 2, (y0 + y1) / 2, -0.06), (x1 - x0 + 0.002, y1 - y0 + 0.002, 0.11), CONC2, bevel=0)


# --- листок (меш): яйцеподібний, складений по центральній жилці
def leaf_mesh(name, L=1.0, W=0.45, fold=0.18, n=10):
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    base = bm.verts.new((0, 0, 0)); prev_l = prev_r = base; mids = [base]
    for i in range(1, n + 1):
        t = i / n
        w = W * (math.sin(math.pi * t) ** 0.75) * (1 - 0.25 * t) / 2
        y = L * t
        bend = -0.12 * L * t * t              # легкий вигин вниз
        mid = bm.verts.new((0, y, bend + 0.0))
        l = bm.verts.new((-w, y, bend + fold * w))
        r = bm.verts.new((w, y, bend + fold * w))
        bm.faces.new((mids[-1], prev_l, l, mid)) if prev_l is not mids[-1] else bm.faces.new((mids[-1], l, mid))
        bm.faces.new((mids[-1], mid, r, prev_r)) if prev_r is not mids[-1] else bm.faces.new((mids[-1], mid, r))
        mids.append(mid); prev_l, prev_r = l, r
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me)
    for p in me.polygons:
        p.use_smooth = True
    return o


def scatter_leaves(host, leaf, count, mat, size=1.0, size_rand=0.4, emit="FACE", show_host=False):
    leaf.data.materials.clear(); leaf.data.materials.append(mat)
    md = host.modifiers.new("Листя", "PARTICLE_SYSTEM")
    ps = host.particle_systems[-1]; st = ps.settings
    st.type = "EMITTER"; st.count = int(count * (0.25 if QUICK else 1)); st.frame_start = st.frame_end = 1; st.lifetime = 10000
    st.emit_from = emit; st.physics_type = "NO"; st.use_emit_random = True; st.use_even_distribution = True
    st.render_type = "OBJECT"; st.instance_object = leaf
    st.particle_size = size; st.size_random = size_rand
    st.use_rotations = True; st.rotation_mode = "NOR"; st.rotation_factor_random = 0.55
    st.phase_factor_random = 2.0
    ps.seed = R.randrange(1000)
    host.show_instancer_for_render = show_host
    host.show_instancer_for_viewport = show_host
    return ps


def ellipsoid(name, c, r, sub=3):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=sub, radius=1, location=c)
    o = bpy.context.active_object; o.name = name; o.scale = r
    bpy.ops.object.transform_apply(scale=True)
    return o


# --- кущик злаків: окремі леза, що розходяться дугою
def grass_clump_mesh(name, blades=90, h=0.45, spread=0.75, width=0.006, upright=False, seed=0):
    rnd = random.Random(seed)
    me = bpy.data.meshes.new(name); bm = bmesh.new(); seg = 6
    for _ in range(blades):
        a = rnd.uniform(0, 2 * math.pi); r0 = rnd.uniform(0, 0.035)
        tilt = rnd.uniform(0.05, spread) if not upright else rnd.uniform(0.0, spread * 0.4)
        Lb = h * rnd.uniform(0.55, 1.1)
        d = Vector((math.cos(a), math.sin(a), 0))
        side = Vector((-d.y, d.x, 0))
        prev = None
        for i in range(seg + 1):
            t = i / seg
            # дуга: вгору, далі назовні й донизу (гравітація)
            out = Lb * (math.sin(tilt) * t + 0.35 * tilt * t * t)
            up = Lb * (math.cos(tilt) * t - 0.45 * tilt * t * t)
            p = Vector((math.cos(a) * r0, math.sin(a) * r0, 0)) + d * out + Vector((0, 0, max(up, 0.02 * t)))
            w = width * (1 - t) ** 0.7 + 0.0004
            twist = side * w
            v1, v2 = bm.verts.new(p - twist), bm.verts.new(p + twist)
            if prev:
                bm.faces.new((prev[0], prev[1], v2, v1))
            prev = (v1, v2)
    bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


def plume_mesh(name, n=14, h=0.75, seed=0):
    """Колоски пенісетуму: тонкі стебла з пухнастими «вершками» (веретено)."""
    rnd = random.Random(seed)
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi); tilt = rnd.uniform(0.05, 0.45); Lb = h * rnd.uniform(0.85, 1.15)
        d = Vector((math.cos(a) * math.sin(tilt), math.sin(a) * math.sin(tilt), math.cos(tilt)))
        top = d * Lb
        # веретено: 8 граней по колу, 6 кілець
        rings = []
        for i in range(7):
            t = i / 6
            c = top - d * (0.16 * (1 - t))
            rad = 0.022 * math.sin(math.pi * min(1, t * 1.1)) + 0.002
            ortho1 = d.orthogonal().normalized(); ortho2 = d.cross(ortho1)
            rings.append([bm.verts.new(c + (ortho1 * math.cos(k * math.pi / 4) + ortho2 * math.sin(k * math.pi / 4)) * rad) for k in range(8)])
        for i in range(6):
            for k in range(8):
                bm.faces.new((rings[i][k], rings[i][(k + 1) % 8], rings[i + 1][(k + 1) % 8], rings[i + 1][k]))
    bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


def plant_grid(prefix, meshes, mats, x0, x1, y0, y1, z, step, jitter=0.4, scale=(0.85, 1.2)):
    n = 0
    y = y0 + step / 2
    while y < y1:
        x = x0 + step / 2 + (step / 2 if int((y - y0) / step) % 2 else 0)
        while x < x1:
            k = R.randrange(len(meshes))
            o = bpy.data.objects.new(f"{prefix}{n}", meshes[k])
            o.location = (x + R.uniform(-jitter, jitter) * step, y + R.uniform(-jitter, jitter) * step, z)
            o.rotation_euler = (0, 0, R.uniform(0, 2 * math.pi)); s = R.uniform(*scale); o.scale = (s, s, s)
            col_main.objects.link(o)
            if mats[k] and not o.data.materials:
                o.data.materials.append(mats[k])
            n += 1
            x += step
        y += step
    return n


# =================================================================== сцена
# нижній рівень площі з вирізами під газони (краї по швах плитки)
BED_A = (-1.2, 1.2, -1.2, 1.2)
BED_B = (-6.0, -3.2, -4.4, 0.8)
paved_area("Площа", -14, 14, -14, 3.4, 0.0, holes=(BED_A, BED_B))
bed("Газон центр", *BED_A)
bed("Газон ліворуч", *BED_B)
# підпірна стінка, тераса, сходи
paved_area("Тераса", -14, 14, 4.8, 14, 0.48)
box("Підпірна стінка", (5.5, 4.85, 0.24), (17, 0.1, 0.48), CONC2, bevel=0.005)
box("Підпірна стінка 2", (-9.6, 4.85, 0.24), (8.8, 0.1, 0.48), CONC2, bevel=0.005)
for i in range(4):   # сходи x −5.2…−3.2
    box(f"Сходинка {i}", (-4.2, 4.8 - 0.35 * (4 - i) + 0.175, 0.06 + 0.12 * i), (2.0, 0.35, 0.12 + 0.24 * i if False else 0.12), CONC2)
    box(f"Сходинка тіло {i}", (-4.2, 4.8 - 0.35 * (4 - i) + 0.175, 0.06 * i), (2.0, 0.35, 0.12 * i + 0.001), CONC2, bevel=0)
# бетонні вазони з лавками
for k, (x0, x1) in enumerate(((-2.8, 3.6), (4.8, 10.0))):
    box(f"Вазон {k}", ((x0 + x1) / 2, 4.1, 0.27), (x1 - x0, 1.4, 0.54), CONC)
    box(f"Вазон {k} ґрунт", ((x0 + x1) / 2, 4.1, 0.535), (x1 - x0 - 0.14, 1.26, 0.02), SOIL, bevel=0)
    bx0, bx1 = x0 + 0.6, x1 - 0.9
    box(f"Лавка {k} основа", ((bx0 + bx1) / 2, 3.17, 0.2), (bx1 - bx0, 0.45, 0.4), CONC)
    for s in range(7):
        box(f"Лавка {k} рейка {s}", ((bx0 + bx1) / 2, 2.97 + s * 0.065, 0.42), (bx1 - bx0 + 0.04, 0.055, 0.04), WOOD, bevel=0.004)
    # злаки з колосками у вазоні
    pm = [grass_clump_mesh(f"Пенісетум {k}_{v}", blades=110, h=0.55, spread=0.6, width=0.004, seed=k * 10 + v) for v in range(3)]
    for me in pm:
        me.materials.append(PENNI)
    pl = [plume_mesh(f"Колоски {k}_{v}", n=16, h=0.8, seed=k * 20 + v) for v in range(3)]
    for me in pl:
        me.materials.append(PLUME)
    plant_grid(f"Злак {k}_", pm, [None] * 3, x0 + 0.1, x1 - 0.1, 3.5, 4.75, 0.545, 0.38)
    plant_grid(f"Колосок {k}_", pl, [None] * 3, x0 + 0.1, x1 - 0.1, 3.5, 4.75, 0.545, 0.42)

# осока в газонах
carex = [grass_clump_mesh(f"Осока {v}", blades=120, h=0.38, spread=1.05, width=0.0045, seed=100 + v) for v in range(4)]
for me in carex:
    me.materials.append(CAREX)
plant_grid("Осока A", carex, [None] * 4, *BED_A, -0.03, 0.34)
plant_grid("Осока B", carex, [None] * 4, *BED_B, -0.03, 0.36)

# самшит у горщиках
box_leaf = leaf_mesh("Листок самшиту", L=0.022, W=0.013, fold=0.15)
for k, (x, y) in enumerate(((3.95, 3.15), (10.35, 3.15), (-2.3, 2.95))):
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.26, depth=0.5, location=(x, y, 0.25))
    pot = bpy.context.active_object; pot.name = f"Горщик {k}"
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="DESELECT"); bpy.ops.object.mode_set(mode="OBJECT")
    for v in pot.data.vertices:
        if v.co.z < 0:
            v.co.x *= 0.8; v.co.y *= 0.8
    bv = pot.modifiers.new("Фаска", "BEVEL"); bv.width = 0.01; bv.segments = 2
    pot.data.materials.append(POT)
    ball = ellipsoid(f"Самшит {k}", (x, y, 0.5 + 0.3), (0.33, 0.33, 0.31), sub=4)
    dm = ball.modifiers.new("Нерівність", "DISPLACE"); tx_ = bpy.data.textures.new(f"box{k}", "CLOUDS"); tx_.noise_scale = 0.12
    dm.texture = tx_; dm.strength = 0.03
    ball.data.materials.append(foliage(f"Самшит тінь {k}", (0.015, 0.05, 0.01), (0.02, 0.06, 0.012)))
    scatter_leaves(ball, box_leaf, 9000, LEAF_BOX, size=1.0, size_rand=0.35, show_host=True)

# центральне багатостовбурне дерево
tree_leaf = leaf_mesh("Листок дерева", L=0.085, W=0.042, fold=0.2)
crowns = []
for k in range(5):
    a = k * 2 * math.pi / 5 + R.uniform(-0.3, 0.3)
    cu = bpy.data.curves.new(f"Стовбур {k}", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.045 - k * 0.004
    cu.bevel_resolution = 3
    sp = cu.splines.new("BEZIER"); sp.bezier_points.add(2)
    pts = [Vector((0.08 * math.cos(a), 0.08 * math.sin(a), 0)), Vector((0.45 * math.cos(a), 0.45 * math.sin(a), 1.6)),
           Vector((0.85 * math.cos(a), 0.85 * math.sin(a), 2.4 + R.uniform(-0.25, 0.25)))]
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = p; bp.handle_left_type = bp.handle_right_type = "AUTO"
    sp.bezier_points[2].radius = 0.45
    ob = bpy.data.objects.new(f"Стовбур {k}", cu); col_main.objects.link(ob); cu.materials.append(BARK)
    crowns.append(ellipsoid(f"Крона {k}", (0.9 * math.cos(a), 0.9 * math.sin(a), 2.75), (0.75, 0.75, 0.5)))
for k, c in enumerate(crowns):
    scatter_leaves(c, tree_leaf, 1700, LEAF_TREE, size=1.0, size_rand=0.45, emit="VOLUME")

# кущ на передньому плані (розмитий) і кущ у лівому газоні
shrub_leaf = leaf_mesh("Листок куща", L=0.11, W=0.05, fold=0.25)
for k, (c, r) in enumerate((((0.9, -3.95, 0.75), (0.75, 0.65, 0.65)), ((1.6, -3.4, 0.5), (0.55, 0.55, 0.5)),
                            ((-3.9, -1.6, 0.7), (0.7, 0.9, 0.7)), ((-4.5, -0.1, 0.8), (0.8, 0.8, 0.75)))):
    h = ellipsoid(f"Кущ {k}", c, r)
    scatter_leaves(h, shrub_leaf, 2200, LEAF_SHRUB, size=1.0, size_rand=0.45, emit="VOLUME")
    for b in range(6):     # гілки
        a = R.uniform(0, 2 * math.pi)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=c[2] * 1.6, location=(c[0] + 0.2 * math.cos(a), c[1] + 0.2 * math.sin(a), c[2] * 0.6),
                                            rotation=(0.4 * math.sin(a), 0.4 * math.cos(a), 0))
        bpy.context.active_object.data.materials.append(BARK)

# крона за кадром — дає ажурну тінь на плитку
SUN_EL, SUN_AZ = 42, 115            # висота і азимут сонця (° від осі +X)
sun_dir = Vector((math.cos(math.radians(SUN_EL)) * math.cos(math.radians(SUN_AZ)),
                  math.cos(math.radians(SUN_EL)) * math.sin(math.radians(SUN_AZ)), math.sin(math.radians(SUN_EL))))
for k, (p, t) in enumerate((((2.4, 0.2, 0), 5.5),)):          # одна сусідня крона: пляма тіні збоку, решта — сонце
    c = Vector(p) + sun_dir * t
    h = ellipsoid(f"Тіньова крона {k}", c, (1.7, 1.4, 0.55))
    scatter_leaves(h, tree_leaf, 2600, LEAF_TREE, size=1.6, size_rand=0.5, emit="VOLUME")
    h.visible_camera = False
    for ch in [h]:
        ch.visible_glossy = False

# =================================================================== світло і камера
w = bpy.data.worlds.new("Небо"); scene.world = w; w.use_nodes = True
nt = w.node_tree; nt.nodes.clear()
sky = nt.nodes.new("ShaderNodeTexSky")
for st in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
    try:
        sky.sky_type = st; break
    except Exception:
        continue
if hasattr(sky, "sun_elevation"):
    sky.sun_elevation = math.radians(SUN_EL)
if hasattr(sky, "sun_rotation"):
    sky.sun_rotation = math.radians(SUN_AZ - 90)
if hasattr(sky, "sun_disc"):
    sky.sun_disc = False
bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = 0.16
o = nt.nodes.new("ShaderNodeOutputWorld")
hs = nt.nodes.new("ShaderNodeHueSaturation"); hs.inputs["Saturation"].default_value = 0.45   # нейтральніші тіні
nt.links.new(sky.outputs[0], hs.inputs["Color"]); nt.links.new(hs.outputs[0], bg.inputs[0]); nt.links.new(bg.outputs[0], o.inputs[0])
sun = bpy.data.lights.new("Сонце", "SUN"); sun.energy = 4.6; sun.angle = math.radians(0.3); sun.color = (1.0, 0.91, 0.78)
so = bpy.data.objects.new("Сонце", sun); col_main.objects.link(so)
so.rotation_euler = (-sun_dir).to_track_quat("-Z", "Y").to_euler()

target = Vector((0.9, -1.6, 0.0))
cam = bpy.data.objects.new("Камера", bpy.data.cameras.new("Камера")); col_main.objects.link(cam); scene.camera = cam
cam.location = target + Vector((-2.4, -3.7, 5.6))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.lens = 55; cam.data.sensor_width = 36; cam.data.sensor_fit = "AUTO"
cam.data.dof.use_dof = True
cam.data.dof.focus_distance = (target - cam.location).length
cam.data.dof.aperture_fstop = 2.0

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
scene.render.resolution_x, scene.render.resolution_y = RX, RY
scene.render.image_settings.file_format = "JPEG"; scene.render.image_settings.quality = 93
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = float(opt("--exposure", -0.85))

blend = os.path.join(ROOT, "blender", "plyta_plaza.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
print("saved", blend)
if "--no-render" not in argv:
    scene.render.filepath = OUT
    bpy.ops.render.render(write_still=True)
    print("rendered", OUT)
