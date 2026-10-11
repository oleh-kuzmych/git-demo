"""Збирає bruk/artifact/bruk_schemes.html: шаблон + дані схем + прев'ю сфер (base64)."""
import base64
import io
import json
import os

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data = json.load(open(f"{ROOT}/schemes.json", encoding="utf-8"))
data["textures"] = json.load(open(f"{ROOT}/textures/textures.json"))
data["previews"] = {}
for s in data["schemes"]:
    p = f"{ROOT}/previews/{s['id']}_sphere.jpg"
    if os.path.exists(p):
        im = Image.open(p).convert("RGB").resize((360, 360), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=84)
        data["previews"][s["id"]] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
tpl = open(f"{ROOT}/artifact/template.html", encoding="utf-8").read()
out = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
open(f"{ROOT}/artifact/bruk_schemes.html", "w", encoding="utf-8").write(out)
print("ok", len(out) // 1024, "KB, previews:", len(data["previews"]))
