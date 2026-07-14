"""
Lightweight WebGL viewer for panoramas and point clouds.

Examples:
    python tools/viewer.py --panos ../data/zind/sample_tour/000/panos
    python tools/viewer.py --panos path/to/pano.jpg
    python tools/viewer.py --pointcloud results/pointclouds/zind_merge_8panos.ply
    python tools/viewer.py --panos path/to/panos --pointcloud path/to/cloud.npy

The script starts a local HTTP server and prints a browser URL. Panorama images
are rendered as an interactive perspective view of an equirectangular image.
Point clouds are loaded once in Python, optionally downsampled, then shown in a
simple orbit viewer.
"""
import argparse
import base64
import io
import json
import mimetypes
import struct
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
POINT_EXTS = {".ply", ".pcd", ".xyz", ".txt", ".npy"}


def discover_panos(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Panorama path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported panorama image type '{path.suffix}'. Supported: {sorted(IMAGE_EXTS)}")
        return [path]
    panos = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not panos:
        raise ValueError(f"No panorama images found in {path}. Supported extensions: {sorted(IMAGE_EXTS)}")
    return panos


def load_pointcloud(path, max_points=250000):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Point cloud path does not exist: {path}")
    if path.suffix.lower() not in POINT_EXTS:
        raise ValueError(f"Unsupported point cloud type '{path.suffix}'. Supported: {sorted(POINT_EXTS)}")

    if path.suffix.lower() == ".npy":
        points, colors = _load_npy(path)
    elif path.suffix.lower() == ".ply":
        points, colors = _load_ply(path)
    elif path.suffix.lower() == ".pcd":
        points, colors = _load_pcd(path)
    else:
        points, colors = _load_xyz(path)

    points, colors = _clean_cloud(points, colors)
    if len(points) == 0:
        raise ValueError(f"Point cloud contains no finite xyz points: {path}")

    if len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx] if colors is not None else None

    return _cloud_payload(path, points, colors)


def _load_npy(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(".npy point clouds must have shape Nx3, Nx4, Nx6, or Nx7")
    points = arr[:, :3].astype(np.float32)
    colors = arr[:, 3:6] if arr.shape[1] >= 6 else None
    return points, colors


def _load_xyz(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 3:
        raise ValueError(".xyz/.txt point clouds must contain at least x y z columns")
    points = arr[:, :3].astype(np.float32)
    colors = arr[:, 3:6] if arr.shape[1] >= 6 else None
    return points, colors


def _load_ply(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: missing end_header")
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1] == "end_header":
                break
        data_offset = f.tell()
        raw = f.read()

    fmt = None
    vertex_count = None
    props = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[:1] == ["format"]:
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"] and parts[1] != "list":
            props.append((parts[2], parts[1]))

    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format '{fmt}'. Supported: ascii, binary_little_endian")
    if vertex_count is None or not props:
        raise ValueError("Invalid PLY: missing vertex element/properties")

    names = [p[0] for p in props]
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex properties must include x, y, z")

    if fmt == "ascii":
        arr = np.loadtxt(io.StringIO(raw.decode("ascii", errors="replace")), max_rows=vertex_count)
        if arr.ndim == 1:
            arr = arr[None, :]
        get = lambda name: arr[:, names.index(name)]
    else:
        dtype = np.dtype([(name, _ply_dtype(kind)) for name, kind in props])
        arr = np.frombuffer(raw, dtype=dtype, count=vertex_count)
        get = lambda name: arr[name]

    points = np.stack([get("x"), get("y"), get("z")], axis=1).astype(np.float32)
    color_names = _first_color_triplet(names)
    colors = None
    if color_names:
        colors = np.stack([get(color_names[0]), get(color_names[1]), get(color_names[2])], axis=1)
    return points, colors


def _ply_dtype(kind):
    mapping = {
        "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
        "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
        "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    }
    if kind not in mapping:
        raise ValueError(f"Unsupported PLY property type '{kind}'")
    return mapping[kind]


def _load_pcd(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PCD: missing DATA line")
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text.lower().startswith("data"):
                break
        raw = f.read()

    meta = {}
    for line in header:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        meta[parts[0].upper()] = parts[1:]

    fields = meta.get("FIELDS")
    if not fields or not {"x", "y", "z"}.issubset(fields):
        raise ValueError("PCD fields must include x y z")
    points_n = int(meta.get("POINTS", meta.get("WIDTH", ["0"]))[0])
    data_kind = meta["DATA"][0].lower()

    if data_kind == "ascii":
        arr = np.loadtxt(io.StringIO(raw.decode("ascii", errors="replace")), comments="#")
        if arr.ndim == 1:
            arr = arr[None, :]
        get = lambda name: arr[:, fields.index(name)]
    elif data_kind == "binary":
        sizes = [int(x) for x in meta["SIZE"]]
        types = meta["TYPE"]
        counts = [int(x) for x in meta.get("COUNT", ["1"] * len(fields))]
        dtype_fields = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            dtype_fields.append((name, _pcd_dtype(size, typ), (count,)) if count > 1 else (name, _pcd_dtype(size, typ)))
        dtype = np.dtype(dtype_fields)
        arr = np.frombuffer(raw, dtype=dtype, count=points_n)
        get = lambda name: arr[name]
    else:
        raise ValueError(f"Unsupported PCD DATA mode '{data_kind}'. Supported: ascii, binary")

    points = np.stack([get("x"), get("y"), get("z")], axis=1).astype(np.float32)
    colors = None
    color_names = _first_color_triplet(fields)
    if color_names:
        colors = np.stack([get(color_names[0]), get(color_names[1]), get(color_names[2])], axis=1)
    elif "rgb" in fields:
        colors = _decode_packed_rgb(get("rgb"))
    return points, colors


def _pcd_dtype(size, typ):
    if typ == "F" and size == 4:
        return "<f4"
    if typ == "F" and size == 8:
        return "<f8"
    if typ == "U":
        return {1: "u1", 2: "<u2", 4: "<u4"}[size]
    if typ == "I":
        return {1: "i1", 2: "<i2", 4: "<i4"}[size]
    raise ValueError(f"Unsupported PCD field type {typ}{size}")


def _first_color_triplet(names):
    for triplet in (("red", "green", "blue"), ("r", "g", "b")):
        if all(c in names for c in triplet):
            return triplet
    return None


def _decode_packed_rgb(values):
    if np.issubdtype(values.dtype, np.floating):
        packed = values.astype("<f4").view("<u4")
    else:
        packed = values.astype(np.uint32)
    r = (packed >> 16) & 255
    g = (packed >> 8) & 255
    b = packed & 255
    return np.stack([r, g, b], axis=1)


def _clean_cloud(points, colors):
    points = np.asarray(points, dtype=np.float32)
    ok = np.isfinite(points).all(axis=1)
    points = points[ok]
    if colors is not None:
        colors = np.asarray(colors)[ok]
        if colors.dtype.kind == "f" and np.nanmax(colors) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return points, colors


def _cloud_payload(path, points, colors):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.linalg.norm(maxs - mins) / 2.0)
    if radius <= 1e-6:
        radius = 1.0

    point_bytes = np.ascontiguousarray(points.astype("<f4")).tobytes()
    color_bytes = np.ascontiguousarray(colors.astype("u1")).tobytes() if colors is not None else b""
    return {
        "name": path.name,
        "count": int(len(points)),
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "center": center.tolist(),
        "radius": radius,
        "points_b64": base64.b64encode(point_bytes).decode("ascii"),
        "colors_b64": base64.b64encode(color_bytes).decode("ascii") if colors is not None else "",
        "has_color": colors is not None,
    }


def make_handler(panos, cloud):
    class ViewerHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_html(panos, cloud))
            elif parsed.path == "/manifest.json":
                self._send_json({"panos": [{"name": p.name, "url": f"/pano/{i}"} for i, p in enumerate(panos)]})
            elif parsed.path == "/cloud.json":
                if cloud is None:
                    self.send_error(404, "No point cloud was provided")
                else:
                    self._send_json(cloud)
            elif parsed.path.startswith("/pano/"):
                self._send_pano(parsed.path)
            else:
                self.send_error(404, "Not found")

        def _send_pano(self, path):
            try:
                idx = int(path.rsplit("/", 1)[-1])
                pano_path = panos[idx]
            except (ValueError, IndexError):
                self.send_error(404, "Panorama index not found")
                return
            mime = mimetypes.guess_type(pano_path)[0] or "application/octet-stream"
            data = pano_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html):
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ViewerHandler


def render_html(panos, cloud):
    initial_mode = "pano" if panos else "cloud"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sparse Panorama Viewer</title>
<style>
html, body {{ margin: 0; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: system-ui, sans-serif; }}
#gl {{ width: 100vw; height: 100vh; display: block; }}
#bar {{ position: fixed; left: 12px; top: 12px; right: 12px; display: flex; gap: 8px; align-items: center; pointer-events: none; }}
#bar > * {{ pointer-events: auto; }}
button, select {{ background: rgba(30, 32, 36, 0.92); color: #eee; border: 1px solid #555; border-radius: 6px; padding: 7px 10px; }}
button.active {{ border-color: #f2c14e; color: #f2c14e; }}
#status {{ margin-left: auto; background: rgba(0, 0, 0, 0.58); padding: 7px 10px; border-radius: 6px; font-size: 13px; max-width: 55vw; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
#hint {{ position: fixed; left: 12px; bottom: 12px; background: rgba(0, 0, 0, 0.58); padding: 7px 10px; border-radius: 6px; font-size: 13px; }}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="bar">
  <button id="panoBtn">Panorama</button>
  <button id="cloudBtn">Point Cloud</button>
  <select id="panoSelect"></select>
  <span id="status"></span>
</div>
<div id="hint"></div>
<script>
const HAS_PANOS = {str(bool(panos)).lower()};
const HAS_CLOUD = {str(cloud is not None).lower()};
let mode = "{initial_mode}";
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl", {{ antialias: true }});
const statusEl = document.getElementById("status");
const hintEl = document.getElementById("hint");
const panoBtn = document.getElementById("panoBtn");
const cloudBtn = document.getElementById("cloudBtn");
const panoSelect = document.getElementById("panoSelect");

if (!gl) {{
  statusEl.textContent = "WebGL is not available in this browser.";
  throw new Error("WebGL unavailable");
}}

let panoProgram, cloudProgram, quadBuffer, cloudBuffers = null, cloudData = null;
let panoTexture = null, panos = [];
const MAX_PITCH = Math.PI * 0.5 - 0.01;
let yaw = 0, pitch = 0, fov = 75;
let orbitYaw = 0.7, orbitPitch = -0.35, distance = 4, panX = 0, panY = 0;
let dragging = false, lastX = 0, lastY = 0, dragButton = 0;

function compile(type, src) {{
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}}

function program(vs, fs) {{
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}}

function initPrograms() {{
  panoProgram = program(`
attribute vec2 a_pos;
varying vec2 v_pos;
void main() {{
  v_pos = a_pos;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}}`, `
precision highp float;
varying vec2 v_pos;
uniform sampler2D u_tex;
uniform vec2 u_resolution;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_fov;
const float PI = 3.141592653589793;
mat3 rotY(float a) {{ float c=cos(a), s=sin(a); return mat3(c,0.0,-s, 0.0,1.0,0.0, s,0.0,c); }}
mat3 rotX(float a) {{ float c=cos(a), s=sin(a); return mat3(1.0,0.0,0.0, 0.0,c,s, 0.0,-s,c); }}
void main() {{
  float aspect = u_resolution.x / u_resolution.y;
  float t = tan(radians(u_fov) * 0.5);
  vec3 ray = normalize(vec3(v_pos.x * aspect * t, v_pos.y * t, 1.0));
  ray = rotY(u_yaw) * rotX(u_pitch) * ray;
  float lon = atan(ray.x, ray.z);
  float lat = asin(clamp(ray.y, -1.0, 1.0));
  vec2 uv = vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI);
  gl_FragColor = texture2D(u_tex, uv);
}}`);

  cloudProgram = program(`
attribute vec3 a_xyz;
attribute vec3 a_rgb;
uniform mat4 u_mvp;
uniform float u_pointSize;
varying vec3 v_rgb;
void main() {{
  v_rgb = a_rgb;
  gl_Position = u_mvp * vec4(a_xyz, 1.0);
  gl_PointSize = u_pointSize;
}}`, `
precision mediump float;
varying vec3 v_rgb;
void main() {{
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(v_rgb, 1.0);
}}`);

  quadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
}}

async function loadManifest() {{
  if (!HAS_PANOS) return;
  const res = await fetch("/manifest.json");
  panos = (await res.json()).panos;
  panoSelect.innerHTML = "";
  for (const [i, p] of panos.entries()) {{
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${{i + 1}} / ${{panos.length}}  ${{p.name}}`;
    panoSelect.appendChild(opt);
  }}
  await loadPano(0);
}}

function loadPano(i) {{
  return new Promise((resolve, reject) => {{
    const img = new Image();
    img.onload = () => {{
      const source = panoTextureSource(img);
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      const err = gl.getError();
      if (err !== gl.NO_ERROR) {{
        gl.deleteTexture(tex);
        reject(new Error(`Could not upload panorama texture. WebGL error ${{err}}. Try a smaller image.`));
        return;
      }}
      if (panoTexture) gl.deleteTexture(panoTexture);
      panoTexture = tex;
      const resized = source !== img ? `  resized to ${{source.width}}x${{source.height}} for WebGL` : "";
      statusEl.textContent = `${{panos[i].name}}${{resized}}`;
      resolve();
    }};
    img.onerror = reject;
    img.src = panos[i].url;
  }});
}}

function panoTextureSource(img) {{
  const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  const largest = Math.max(img.naturalWidth, img.naturalHeight);
  if (largest <= maxTex) return img;
  const scale = maxTex / largest;
  const canvas2d = document.createElement("canvas");
  canvas2d.width = Math.max(1, Math.floor(img.naturalWidth * scale));
  canvas2d.height = Math.max(1, Math.floor(img.naturalHeight * scale));
  const ctx = canvas2d.getContext("2d");
  ctx.drawImage(img, 0, 0, canvas2d.width, canvas2d.height);
  return canvas2d;
}}

async function loadCloud() {{
  if (!HAS_CLOUD) return;
  const res = await fetch("/cloud.json");
  cloudData = await res.json();
  const points = b64ToFloat32(cloudData.points_b64);
  const colors = cloudData.has_color ? b64ToUint8(cloudData.colors_b64) : null;
  const rgb = new Float32Array(cloudData.count * 3);
  if (colors) {{
    for (let i = 0; i < colors.length; i++) rgb[i] = colors[i] / 255.0;
  }} else {{
    for (let i = 0; i < cloudData.count; i++) {{
      rgb[i*3] = 0.95; rgb[i*3+1] = 0.82; rgb[i*3+2] = 0.38;
    }}
  }}
  cloudBuffers = {{
    xyz: gl.createBuffer(),
    rgb: gl.createBuffer(),
  }};
  gl.bindBuffer(gl.ARRAY_BUFFER, cloudBuffers.xyz);
  gl.bufferData(gl.ARRAY_BUFFER, points, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloudBuffers.rgb);
  gl.bufferData(gl.ARRAY_BUFFER, rgb, gl.STATIC_DRAW);
  distance = Math.max(cloudData.radius * 2.6, 0.1);
}}

function b64ToFloat32(b64) {{
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}

function b64ToUint8(b64) {{
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}}

function resize() {{
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {{
    canvas.width = w; canvas.height = h;
  }}
  gl.viewport(0, 0, canvas.width, canvas.height);
}}

function draw() {{
  resize();
  gl.clearColor(0.06, 0.06, 0.065, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  panoBtn.classList.toggle("active", mode === "pano");
  cloudBtn.classList.toggle("active", mode === "cloud");
  panoBtn.style.display = HAS_PANOS ? "" : "none";
  cloudBtn.style.display = HAS_CLOUD ? "" : "none";
  panoSelect.style.display = HAS_PANOS && mode === "pano" && panos.length > 1 ? "" : "none";

  if (mode === "pano" && panoTexture) drawPano();
  if (mode === "cloud" && cloudBuffers) drawCloud();
  requestAnimationFrame(draw);
}}

function drawPano() {{
  hintEl.textContent = "Drag to look around. Mouse wheel changes field of view.";
  gl.disable(gl.DEPTH_TEST);
  gl.useProgram(panoProgram);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  const aPos = gl.getAttribLocation(panoProgram, "a_pos");
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, panoTexture);
  gl.uniform1i(gl.getUniformLocation(panoProgram, "u_tex"), 0);
  gl.uniform2f(gl.getUniformLocation(panoProgram, "u_resolution"), canvas.width, canvas.height);
  gl.uniform1f(gl.getUniformLocation(panoProgram, "u_yaw"), yaw);
  gl.uniform1f(gl.getUniformLocation(panoProgram, "u_pitch"), pitch);
  gl.uniform1f(gl.getUniformLocation(panoProgram, "u_fov"), fov);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}}

function drawCloud() {{
  hintEl.textContent = "Left drag rotates. Right/middle drag pans. Mouse wheel zooms.";
  statusEl.textContent = `${{cloudData.name}}  (${{cloudData.count.toLocaleString()}} points)`;
  gl.enable(gl.DEPTH_TEST);
  gl.useProgram(cloudProgram);
  const mvp = cloudMvp();
  gl.uniformMatrix4fv(gl.getUniformLocation(cloudProgram, "u_mvp"), false, mvp);
  gl.uniform1f(gl.getUniformLocation(cloudProgram, "u_pointSize"), Math.max(1.5, Math.min(5.0, 900.0 / distance)));
  bindAttrib(cloudProgram, "a_xyz", cloudBuffers.xyz, 3);
  bindAttrib(cloudProgram, "a_rgb", cloudBuffers.rgb, 3);
  gl.drawArrays(gl.POINTS, 0, cloudData.count);
}}

function bindAttrib(prog, name, buffer, size) {{
  const loc = gl.getAttribLocation(prog, name);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
}}

function cloudMvp() {{
  const aspect = canvas.width / Math.max(canvas.height, 1);
  const proj = perspective(55 * Math.PI / 180, aspect, 0.01, Math.max(distance + cloudData.radius * 5, 100));
  const eye = [
    Math.sin(orbitYaw) * Math.cos(orbitPitch) * distance,
    Math.sin(orbitPitch) * distance,
    Math.cos(orbitYaw) * Math.cos(orbitPitch) * distance,
  ];
  const center = cloudData.center;
  const view = lookAt(
    [eye[0] + center[0] + panX, eye[1] + center[1] + panY, eye[2] + center[2]],
    [center[0] + panX, center[1] + panY, center[2]],
    [0, 1, 0]
  );
  return multiply(proj, view);
}}

function perspective(fovy, aspect, near, far) {{
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([
    f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0
  ]);
}}

function lookAt(eye, center, up) {{
  const z = norm([eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]]);
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
  ]);
}}

function multiply(a, b) {{
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {{
    out[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  }}
  return out;
}}

function norm(v) {{ const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0]/l, v[1]/l, v[2]/l]; }}
function cross(a,b) {{ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }}
function dot(a,b) {{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }}

canvas.addEventListener("mousedown", e => {{
  dragging = true; lastX = e.clientX; lastY = e.clientY; dragButton = e.button;
}});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (mode === "pano") {{
    yaw -= dx * 0.004;
    pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch + dy * 0.004));
  }} else {{
    if (dragButton === 2 || dragButton === 1 || e.shiftKey) {{
      panX -= dx * distance * 0.0015;
      panY += dy * distance * 0.0015;
    }} else {{
      orbitYaw -= dx * 0.006;
      orbitPitch = Math.max(-1.5, Math.min(1.5, orbitPitch + dy * 0.006));
    }}
  }}
}});
canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  if (mode === "pano") fov = Math.max(35, Math.min(110, fov + Math.sign(e.deltaY) * 4));
  else distance *= Math.exp(Math.sign(e.deltaY) * 0.12);
}}, {{ passive: false }});
canvas.addEventListener("contextmenu", e => e.preventDefault());

panoBtn.onclick = () => {{ if (HAS_PANOS) mode = "pano"; }};
cloudBtn.onclick = () => {{ if (HAS_CLOUD) mode = "cloud"; }};
panoSelect.onchange = () => loadPano(Number(panoSelect.value));
window.addEventListener("keydown", e => {{
  if (mode === "pano" && panos.length > 1 && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {{
    const step = e.key === "ArrowRight" ? 1 : -1;
    const next = (Number(panoSelect.value) + step + panos.length) % panos.length;
    panoSelect.value = next;
    loadPano(next);
  }}
}});

initPrograms();
Promise.all([loadManifest(), loadCloud()]).then(draw).catch(err => {{
  statusEl.textContent = err.message;
  console.error(err);
}});
</script>
</body>
</html>"""


def parse_args():
    ap = argparse.ArgumentParser(description="Interactive browser viewer for equirectangular panoramas and point clouds.")
    ap.add_argument("--panos", help="Panorama image file or folder of equirectangular images.")
    ap.add_argument("--pointcloud", help="Point cloud file (.ply, .pcd, .xyz, .txt, .npy).")
    ap.add_argument("--host", default="127.0.0.1", help="HTTP host to bind. Default: 127.0.0.1")
    ap.add_argument("--port", type=int, default=8765, help="HTTP port. Default: 8765")
    ap.add_argument("--max_points", type=int, default=5000000, help="Maximum points sent to the browser.")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.panos and not args.pointcloud:
        raise SystemExit("Provide --panos and/or --pointcloud. Example: python tools/viewer.py --panos path/to/panos")

    try:
        panos = discover_panos(args.panos) if args.panos else []
        cloud = load_pointcloud(args.pointcloud, args.max_points) if args.pointcloud else None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"viewer error: {exc}") from exc

    handler = make_handler(panos, cloud)
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        raise SystemExit(f"viewer error: could not bind {args.host}:{args.port}: {exc}") from exc

    url = f"http://{args.host}:{args.port}/"
    print(f"Serving viewer at {url}")
    if panos:
        print(f"  panoramas: {len(panos)}")
    if cloud:
        print(f"  point cloud: {cloud['name']} ({cloud['count']:,} points)")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
