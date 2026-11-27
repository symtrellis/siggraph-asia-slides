"""在 RTX 4090 上计算陀螺三个旋转姿态共用的 32³ 表面占用网格。"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "docs/assets/models"
N = 32
LOWER = -1.04
SIDE = 2.08
CELL = SIDE / N


def triangle_box_overlap(triangles, centers, half):
    """三角形与候选轴对齐体素的精确分离轴测试。"""
    relative = triangles - centers[:, None, :]
    edges = relative.roll(-1, 1) - relative
    normal = torch.cross(edges[:, 0], edges[:, 1], dim=1)
    basis = torch.eye(3, device=triangles.device)
    axes = [normal]
    for edge in range(3):
        for axis in range(3):
            axes.append(torch.cross(edges[:, edge], basis[axis].expand_as(edges[:, edge]), dim=1))
    valid = torch.ones(len(triangles), dtype=torch.bool, device=triangles.device)
    for axis in axes:
        projection = (relative * axis[:, None, :]).sum(-1)
        radius = axis.abs().sum(-1) * half
        valid &= (projection.amin(1) <= radius + 1e-8) & (projection.amax(1) >= -radius - 1e-8)
    return valid


def rasterize(triangles):
    # 所有姿态共享 LOWER、CELL 和 N，绝不根据各姿态的包围盒重新归一化。
    lower = triangles.amin((0, 1))
    upper = triangles.amax((0, 1))
    assert (lower >= LOWER).all() and (upper <= LOWER + SIDE).all(), "网格未包含整个陀螺"
    lo = torch.floor((triangles.amin(1) - LOWER) / CELL).long().clamp(0, N - 1)
    hi = torch.floor((triangles.amax(1) - LOWER) / CELL).long().clamp(0, N - 1)
    contained = (lo == hi).all(1)
    occupied = torch.zeros(N**3, dtype=torch.bool, device=triangles.device)

    def flatten(coords):
        return coords[:, 0] * N * N + coords[:, 1] * N + coords[:, 2]

    occupied[flatten(lo[contained])] = True
    remaining = triangles[~contained]
    lo, hi = lo[~contained], hi[~contained]
    # 枚举每个跨格三角形实际可能接触的体素，不构造三角形 × 全部体素的矩阵。
    for start in range(0, len(remaining), 32768):
        batch = remaining[start:start + 32768]
        low = lo[start:start + len(batch)]
        high = hi[start:start + len(batch)]
        widths = high - low + 1
        counts = widths.prod(1)
        faces = torch.repeat_interleave(torch.arange(len(batch), device=batch.device), counts)
        offsets = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        local = torch.arange(len(faces), device=batch.device) - offsets
        sizes = widths[faces]
        dz = local % sizes[:, 2]
        dy = (local // sizes[:, 2]) % sizes[:, 1]
        dx = local // (sizes[:, 1] * sizes[:, 2])
        candidates = low[faces] + torch.stack((dx, dy, dz), dim=1)
        centers = LOWER + (candidates.float() + 0.5) * CELL
        hits = triangle_box_overlap(batch[faces], centers, CELL / 2)
        occupied[flatten(candidates[hits])] = True

    ids = occupied.nonzero().flatten()
    coords = torch.stack((ids // (N*N), (ids // N) % N, ids % N), dim=1)
    return coords.cpu().tolist()


def main():
    assert torch.cuda.is_available(), "必须使用 CUDA，不允许 CPU 回退。"
    gpu = next(i for i in range(torch.cuda.device_count()) if "4090" in torch.cuda.get_device_name(i))
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    entry = json.loads((MODELS / "rotation_axes.json").read_text())["models"]["symtrellis"]
    source = MODELS / entry["source"]
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    mesh = np.fromfile(source, dtype=dtype, offset=84)
    triangles = torch.tensor(mesh["vertices"].copy(), device=device)
    axis = torch.tensor(entry["axis_direction"], device=device)
    axis = axis / axis.norm()
    cross = torch.cross(axis, torch.tensor([0., 0., 1.], device=device), dim=0)
    x, y, z = cross.unbind()
    zero = torch.zeros((), device=device)
    k = torch.stack((torch.stack((zero, -z, y)), torch.stack((z, zero, -x)), torch.stack((-y, x, zero))))
    alignment = torch.eye(3, device=device) + k + k @ k / (1 + axis[2])
    triangles = (triangles - torch.tensor(entry["axis_point_at_z0"], device=device)) @ alignment.T
    angle = 2 * math.pi / 3
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]], device=device)
    source_voxels = rasterize(triangles)
    target_voxels = rasterize(triangles @ rotation.T)
    angle_240 = 4 * math.pi / 3
    rotation_240 = torch.tensor([[math.cos(angle_240), -math.sin(angle_240), 0], [math.sin(angle_240), math.cos(angle_240), 0], [0, 0, 1]], device=device)
    rotated_240_voxels = rasterize(triangles @ rotation_240.T)
    data = {
        "resolution": N,
        "lower": [LOWER] * 3,
        "side": SIDE,
        "cell_size": CELL,
        "rotation_degrees": 120,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(gpu),
        "source": source_voxels,
        "rotated": target_voxels,
        "rotated_240": rotated_240_voxels,
    }
    path = MODELS / "mapper-voxels-32.json"
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"GPU: {data['device']}; 0°: {len(source_voxels)}; 120°: {len(target_voxels)}; 240°: {len(rotated_240_voxels)}")
    print(f"共用坐标系: [{LOWER}, {LOWER + SIDE}]³; 单元边长: {CELL}; {path}")


if __name__ == "__main__":
    main()
