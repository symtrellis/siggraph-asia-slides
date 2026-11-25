#!/usr/bin/env python3
"""Re-export the presentation meshes with the original SymTRELLIS CUDA exporter."""

import argparse
import ast
import copy
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT.parent / "symtrellis_fast_forward_video"
REPO = ROOT.parent / "SymTRELLIS"
sys.path[:0] = [str(REPO), str(REPO / "third_party/trellis2")]

import cumesh
from inference.trellis2 import trelli2_mesh_to_glb
from trellis2.modules.sparse import SparseTensor
from trellis2.representations.mesh import Mesh


OLD_EXAMPLES = ["00_turboblade", "01_fountain", "02_icosahedral", "06_gear"]
NEW_EXAMPLES = ["03_silvercup", "04_lotus", "07_icosahedral", "08_gear",
                "09_omniwheel", "10_artifact", "11_blade", "12_omniwheel"]
NAMES = [f"{name}_{variant}" for name in OLD_EXAMPLES
         for variant in ("vanilla", "symtrellis")]
NAMES += [f"{name}_symtrellis" for name in NEW_EXAMPLES]
GEOMETRY_TARGET = 25_000_000
SMOOTHING = {"crease_degrees": 75.0, "iterations": 3, "blend": 0.35,
             "neighbor_angle_degrees": 35.0}


def retrieve_normal_filter():
    # Use the same CUDA corner-normal computation as the original animation.
    # Load only the function, not the original script's cache-writing setup.
    source = VIDEO / "blender/compute_scene_01_normals_gpu.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "compute")
    namespace = {"torch": torch, "np": np, "math": math, "DEVICE": "cuda"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["compute"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def unpack_glb(data):
    magic, version, length = struct.unpack_from("<4sII", data)
    assert magic == b"glTF" and version == 2 and length == len(data)
    size, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    metadata = json.loads(data[20:20 + size])
    bin_size, bin_kind = struct.unpack_from("<II", data, 20 + size)
    assert bin_kind == 0x004E4942
    return metadata, memoryview(data)[28 + size:28 + size + bin_size]


def view_bytes(metadata, binary, index):
    view = metadata["bufferViews"][index]
    assert view.get("buffer", 0) == 0
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]]


def accessor_array(metadata, binary, index):
    a = metadata["accessors"][index]
    view = metadata["bufferViews"][a["bufferView"]]
    dtype = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[a["componentType"]]
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    dtype = np.dtype(dtype)
    return np.ndarray((a["count"], components), dtype=dtype, buffer=binary,
                      offset=view.get("byteOffset", 0) + a.get("byteOffset", 0),
                      strides=(view.get("byteStride", dtype.itemsize * components),
                               dtype.itemsize)).copy()


def mesh_arrays(metadata, binary):
    primitive = metadata["meshes"][0]["primitives"][0]
    vertices = accessor_array(metadata, binary, primitive["attributes"]["POSITION"])
    faces = accessor_array(metadata, binary, primitive["indices"]).reshape(-1, 3)
    return vertices, faces


def repair_normals_and_split(vertices, faces, uv, normal_filter):
    v = torch.as_tensor(vertices, dtype=torch.float32, device="cuda")
    f = torch.as_tensor(faces.astype(np.int64), device="cuda")
    corners, report = normal_filter(v, f, SMOOTHING["crease_degrees"], SMOOTHING)
    normals = torch.as_tensor(corners, device="cuda").contiguous()
    bits = normals.view(torch.int32).to(torch.int64)
    keys = torch.cat([f.reshape(-1, 1), bits], dim=1)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    source_index = unique[:, 0].cpu().numpy()
    out_normals = unique[:, 1:].to(torch.int32).contiguous().view(torch.float32).cpu().numpy()
    out_faces = inverse.reshape(-1, 3).cpu().numpy().astype("<u4")
    out_vertices = vertices[source_index]
    out_uv = uv[source_index]
    assert np.array_equal(out_vertices[out_faces], vertices[faces])
    assert np.array_equal(out_uv[out_faces], uv[faces])
    assert np.isfinite(out_normals).all()
    return out_vertices, out_faces, out_uv, out_normals, report


def compress_texture(data, color):
    original = Image.open(io.BytesIO(data)).convert("RGB")
    quality = 94 if color else 96
    output = io.BytesIO()
    original.save(output, format="JPEG", quality=quality, subsampling=0, optimize=True)
    encoded = output.getvalue()
    reconstructed = Image.open(io.BytesIO(encoded)).convert("RGB")
    error = np.asarray(original).astype(np.float32) - np.asarray(reconstructed).astype(np.float32)
    rmse = np.sqrt(np.mean(error * error, axis=(0, 1)))
    report = {"resolution": list(original.size), "quality": quality,
              "original_bytes": len(data), "compressed_bytes": len(encoded),
              "channel_rmse_8bit": rmse.round(4).tolist()}
    return encoded, report


def compact_glb(raw_glb, normal_filter):
    original, binary = unpack_glb(raw_glb)
    p = original["meshes"][0]["primitives"][0]
    vertices, faces = mesh_arrays(original, binary)
    uv = accessor_array(original, binary, p["attributes"]["TEXCOORD_0"])
    vertices, faces, uv, normals, normal_report = repair_normals_and_split(vertices, faces, uv, normal_filter)
    assert uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6
    metadata = copy.deepcopy(original)
    metadata["bufferViews"], metadata["accessors"] = [], []
    output = bytearray()

    def add_view(data, target=None, stride=None):
        output.extend(b"\0" * (-len(output) % 4))
        view = {"buffer": 0, "byteOffset": len(output), "byteLength": len(data)}
        if target:
            view["target"] = target
        if stride:
            view["byteStride"] = stride
        output.extend(data)
        metadata["bufferViews"].append(view)
        return len(metadata["bufferViews"]) - 1

    def add_accessor(array, component_type, kind, target, normalized=False, bounds=False, stride=None):
        if stride is None:
            data = np.ascontiguousarray(array).tobytes()
        else:
            size = array.shape[1] * array.dtype.itemsize
            packed = np.zeros((len(array), stride), dtype=np.uint8)
            packed[:, :size] = np.ascontiguousarray(array).view(np.uint8).reshape(len(array), size)
            data = packed.tobytes()
        a = {"bufferView": add_view(data, target, stride),
             "componentType": component_type, "count": len(array), "type": kind}
        if normalized:
            a["normalized"] = True
        if bounds:
            a["min"] = array.min(axis=0).tolist()
            a["max"] = array.max(axis=0).tolist()
        metadata["accessors"].append(a)
        return len(metadata["accessors"]) - 1

    primitive = metadata["meshes"][0]["primitives"][0]
    attrs = {"POSITION": add_accessor(vertices.astype("<f4"), 5126, "VEC3", 34962, bounds=True)}
    # Preserve full-precision positions; lossless 32-bit indices and extremely
    # fine UV/normal quantization are preferable to further geometric decimation.
    full_geometry_bytes = len(vertices) * 32 + faces.size * 4
    use_uv16 = full_geometry_bytes > GEOMETRY_TARGET
    uv_bytes = 4 if use_uv16 else 8
    use_normal16 = len(vertices) * (24 + uv_bytes) + faces.size * 4 > GEOMETRY_TARGET
    if use_normal16:
        normals16 = np.round(np.clip(normals, -1, 1) * 32767).astype("<i2")
        attrs["NORMAL"] = add_accessor(normals16, 5122, "VEC3", 34962, normalized=True, stride=8)
        for key in ("extensionsUsed", "extensionsRequired"):
            metadata.setdefault(key, [])
            if "KHR_mesh_quantization" not in metadata[key]:
                metadata[key].append("KHR_mesh_quantization")
    else:
        attrs["NORMAL"] = add_accessor(normals.astype("<f4"), 5126, "VEC3", 34962)
    if use_uv16:
        uv16 = np.round(np.clip(uv, 0, 1) * 65535).astype("<u2")
        attrs["TEXCOORD_0"] = add_accessor(uv16, 5123, "VEC2", 34962, normalized=True)
    else:
        attrs["TEXCOORD_0"] = add_accessor(uv.astype("<f4"), 5126, "VEC2", 34962)
    primitive["attributes"] = attrs
    primitive["indices"] = add_accessor(faces.reshape(-1, 1), 5125, "SCALAR", 34963)
    geometry_bytes = len(output)

    material = metadata["materials"][0]["pbrMetallicRoughness"]
    base_source = metadata["textures"][material["baseColorTexture"]["index"]]["source"]
    texture_reports = []
    for index, image in enumerate(metadata["images"]):
        old_image = original["images"][index]
        data = view_bytes(original, binary, old_image["bufferView"])
        encoded, report = compress_texture(data, index == base_source)
        image.update(bufferView=add_view(encoded), mimeType="image/jpeg")
        image.pop("uri", None)
        texture_reports.append(report)

    output.extend(b"\0" * (-len(output) % 4))
    metadata["buffers"] = [{"byteLength": len(output)}]
    js = json.dumps(metadata, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    glb = (struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(output))
           + struct.pack("<II", len(js), 0x4E4F534A) + js
           + struct.pack("<II", len(output), 0x004E4942) + output)
    report = {"triangle_count": len(faces), "vertex_count": len(vertices),
              "geometry_bytes": geometry_bytes, "texture_bytes": sum(i["compressed_bytes"] for i in texture_reports),
              "bytes": len(glb), "uv_16bit": use_uv16, "normal_16bit": use_normal16,
              "textures": texture_reports, "normal_report": normal_report,
              "output_sha256": sha256(glb)}
    return glb, report, vertices, faces


def sample_surface(vertices, faces, count=65536):
    v = torch.as_tensor(vertices, dtype=torch.float32, device="cuda")
    f = torch.as_tensor(faces.astype(np.int64), device="cuda")
    triangles = v[f]
    area = torch.linalg.vector_norm(torch.cross(triangles[:, 1] - triangles[:, 0],
                                                triangles[:, 2] - triangles[:, 0], dim=1), dim=1)
    generator = torch.Generator(device="cuda").manual_seed(42)
    ix = torch.multinomial(area, count, replacement=True, generator=generator)
    uv = torch.rand((count, 2), device="cuda", generator=generator)
    uv = torch.where(uv.sum(dim=1, keepdim=True) > 1, 1 - uv, uv)
    triangles = triangles[ix]
    return triangles[:, 0] + uv[:, :1]*(triangles[:, 1]-triangles[:, 0]) + uv[:, 1:]*(triangles[:, 2]-triangles[:, 0])


def compare_geometry(source, vertices, faces):
    metadata, binary = unpack_glb(source)
    source_vertices, source_faces = mesh_arrays(metadata, binary)
    diagonal = float(np.linalg.norm(source_vertices.max(0) - source_vertices.min(0)))

    def measure(a_v, a_f, b_v, b_f):
        bvh = cumesh.cuBVH(b_v.astype(np.float32), b_f.astype(np.int32))
        distances, _, _ = bvh.unsigned_distance(sample_surface(a_v, a_f))
        distances = distances.cpu().numpy() / diagonal
        return dict(rms=float(np.sqrt(np.mean(distances**2))),
                    p95=float(np.percentile(distances, 95)),
                    p99=float(np.percentile(distances, 99)),
                    maximum=float(distances.max()))

    return {"original_triangles": len(source_faces),
            "source_to_compressed": measure(source_vertices, source_faces, vertices, faces),
            "compressed_to_source": measure(vertices, faces, source_vertices, source_faces),
            "distance_unit": "fraction of original bounding-box diagonal"}


def export(name, output, normal_filter, target, texture_size):
    begin = time.monotonic()
    source_dir = VIDEO / "shapes" / name
    source = (source_dir / "model.glb").read_bytes()
    config = json.loads((source_dir / "config.json").read_text())
    params = config["export"]["params"]
    raw = torch.load(source_dir / "shape_raw_mesh.pt", map_location="cpu", weights_only=True)
    pbr = torch.load(source_dir / "pbr_voxel.pt", map_location="cpu", weights_only=True)
    mesh = Mesh(vertices=raw["vertices"], faces=raw["faces"])
    pbr_voxel = SparseTensor(coords=pbr["coords"], feats=pbr["feats"])
    print(f"{name}: source {len(source)/1e6:.2f} MB; target {target:,} triangles", flush=True)
    exported = trelli2_mesh_to_glb(
        shape_mesh=mesh, res=int(config["shape"]["result"]["oVoxelGridSize"]),
        device=torch.device("cuda"), texture_size=texture_size, pbr_voxel=pbr_voxel,
        remesh=params["remesh"], decimation_target=target,
        remesh_band=params["remeshBand"], remesh_project=params["remeshProject"],
        report=lambda progress: print(f"  {name} {progress['stage']}", flush=True),
    )
    raw_glb = exported.export(file_type="glb")
    del exported, mesh, pbr_voxel, raw, pbr
    gc.collect()
    torch.cuda.empty_cache()
    glb, report, vertices, faces = compact_glb(raw_glb, normal_filter)
    report["geometry_comparison"] = compare_geometry(source, vertices, faces)
    report.update(name=name, source_sha256=sha256(source), source_bytes=len(source),
                  face_decimation_target=target, texture_size=texture_size,
                  elapsed_seconds=round(time.monotonic()-begin, 2))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    # Validate the entire GLB container before writing the candidate output.
    meta, binary = unpack_glb(glb)
    out_v, out_f = mesh_arrays(meta, binary)
    assert np.array_equal(out_v, vertices) and np.array_equal(out_f, faces)
    assert report["geometry_bytes"] <= GEOMETRY_TARGET, "geometry exceeds target"
    assert all(i["p99"] < 0.0015 for i in
               (report["geometry_comparison"]["source_to_compressed"],
                report["geometry_comparison"]["compressed_to_source"])), "geometry changed too much"
    (output / f"{name}.glb").write_bytes(glb)
    (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument("--faces", type=int, default=650_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    print("GPU:", torch.cuda.get_device_name(), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    normal_filter = retrieve_normal_filter()
    failures = []
    for name in args.names:
        try:
            export(name, args.output, normal_filter, args.faces, args.texture_size)
        except Exception:
            failures.append(name)
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f"Export failed: {failures}")


if __name__ == "__main__":
    main()
