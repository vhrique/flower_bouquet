"""Crop the trained splat to the bouquet and export web formats.

Usage: python crop_splat.py <splat_full.ply> <bouquet_points.ply> <out_dir>
Both inputs are in the aligned frame (Z up, vase base at origin, bouquet height = 1).
Writes bouquet.ply (full SH) and bouquet.splat (compact, 32 bytes per splat).
"""
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def nearest_dist(q, ref):
    """Distance from each query point to its nearest reference point (GPU)."""
    import torch
    ref_t = torch.tensor(ref, device="cuda", dtype=torch.float32)
    out = [torch.cdist(torch.tensor(c, device="cuda", dtype=torch.float32), ref_t).min(1).values.cpu().numpy()
           for c in np.array_split(q, max(1, len(q) // 2048))]
    return np.concatenate(out) if out else np.zeros(0)


def main(splat_path, points_path, out):
    v = PlyData.read(splat_path)["vertex"].data
    pts = PlyData.read(points_path)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)

    # Cylinder around the vase, sized from the dense bouquet points.
    b = np.stack([pts["x"], pts["y"], pts["z"]], 1)
    radius = np.quantile(np.linalg.norm(b[:, :2], axis=1), 0.995) * 1.08
    r = np.linalg.norm(xyz[:, :2], axis=1)
    keep = (r < radius) & (xyz[:, 2] > -0.01) & (xyz[:, 2] < 1.06)

    opacity = 1 / (1 + np.exp(-v["opacity"]))
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1))
    keep &= opacity > 0.03
    keep &= scales.max(1) < 0.08  # drop big background blobs that leak into the cylinder
    # Tablecloth: only a small disc under the vase.
    keep &= (xyz[:, 2] > 0.03) | (r < 0.3)
    # Above the vase, drop haze not near a reconstructed bouquet surface.
    # (The glass vase itself is absent from the dense points, so it is exempt.)
    upper = keep & (xyz[:, 2] > 0.45)
    keep[upper] = nearest_dist(xyz[upper], b) < 0.06
    print(f"kept {keep.sum()} / {len(v)} splats, crop radius {radius:.3f}")

    v = v[keep]
    PlyData([PlyElement.describe(v, "vertex")]).write(os.path.join(out, "bouquet.ply"))

    # Compact .splat: pos f32x3, scale f32x3, rgba u8x4, rot u8x4 (w,x,y,z), sorted by importance.
    xyz, opacity, scales = xyz[keep], opacity[keep], scales[keep]
    rgb = 0.5 + C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    order = np.argsort(-(scales.prod(1) * opacity))
    rec = np.zeros(len(v), dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    rec["pos"], rec["scale"] = xyz[order], scales[order]
    rec["rgba"] = np.clip(np.c_[rgb, opacity][order] * 255, 0, 255).astype(np.uint8)
    rec["rot"] = np.clip(q[order] * 128 + 128, 0, 255).astype(np.uint8)
    rec.tofile(os.path.join(out, "bouquet.splat"))
    for f in ("bouquet.ply", "bouquet.splat"):
        print(f, f"{os.path.getsize(os.path.join(out, f)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
