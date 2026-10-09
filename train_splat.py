"""Train a 3D Gaussian Splat from the COLMAP reconstruction with gsplat.

Usage: python train_splat.py <colmap_work_dir> <align.json> <out_dir> [iters] [downscale]
Cameras are moved into the upright, height-normalized frame from align.json,
so the resulting splat is Z-up with the vase base at the origin.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import DefaultStrategy, rasterization
from PIL import Image
from plyfile import PlyData, PlyElement

SH_DEGREE = 3


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def load_scene(work, align, downscale):
    sp = os.path.join(work, "dense", "sparse_txt")
    cam = [l.split() for l in open(os.path.join(sp, "cameras.txt")) if not l.startswith("#")][0]
    W, H, fx, fy, cx, cy = int(cam[2]), int(cam[3]), *map(float, cam[4:8])
    w, h = round(W / downscale), round(H / downscale)
    sx, sy = w / W, h / H
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float32)

    Rw, a = np.array(align["R"]), np.array(align["origin"])
    zoff, s = np.array([0, 0, align["zmin"]]), align["height"]

    lines = [l for l in open(os.path.join(sp, "images.txt")) if not l.startswith("#")]
    viewmats, images = [], []
    for line in lines[::2]:
        v = line.split()
        R, t = qvec2rot(list(map(float, v[1:5]))), np.array(list(map(float, v[5:8])))
        R2 = R @ Rw.T
        t2 = (R2 @ zoff + R @ a + t) / s
        M = np.eye(4, dtype=np.float32)
        M[:3, :3], M[:3, 3] = R2, t2
        viewmats.append(M)
        im = Image.open(os.path.join(work, "dense", "images", v[9])).convert("RGB").resize((w, h), Image.LANCZOS)
        images.append(np.asarray(im))

    pts, cols = [], []
    for l in open(os.path.join(sp, "points3D.txt")):
        if l.startswith("#"):
            continue
        v = l.split()
        pts.append(list(map(float, v[1:4])))
        cols.append(list(map(int, v[4:7])))
    pts = ((np.array(pts) - a) @ Rw.T - zoff) / s
    return K, w, h, np.stack(viewmats), np.stack(images), pts.astype(np.float32), np.array(cols, np.float32) / 255


def ssim(x, y):
    """x, y: [B,3,H,W] in [0,1]."""
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    f = lambda z: F.conv2d(z, win, padding=5, groups=3)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean()


def knn_mean_dist(p, k=3):
    out = []
    for chunk in torch.split(p, 4096):
        d = torch.cdist(chunk, p)
        out.append(d.topk(k + 1, largest=False).values[:, 1:].mean(1))
    return torch.cat(out)


def save_ply(path, params):
    m = params["means"].detach().cpu().numpy()
    sh0 = params["sh0"].detach().cpu().numpy().reshape(len(m), -1)
    shN = params["shN"].detach().transpose(1, 2).cpu().numpy().reshape(len(m), -1)
    op = params["opacities"].detach().cpu().numpy()[:, None]
    sc = params["scales"].detach().cpu().numpy()
    q = F.normalize(params["quats"].detach(), dim=-1).cpu().numpy()
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(shN.shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([m, np.zeros_like(m), sh0, shN, op, sc, q], 1).astype(np.float32)
    arr = np.empty(len(m), dtype=[(n, "f4") for n in names])
    for i, n in enumerate(names):
        arr[n] = data[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def main(work, align_path, out, iters=30000, downscale=2.0):
    os.makedirs(out, exist_ok=True)
    dev = "cuda"
    K, w, h, viewmats, images, pts, cols = load_scene(work, json.load(open(align_path)), downscale)
    n_img = len(images)
    print(f"{n_img} images at {w}x{h}, {len(pts)} init points")
    K = torch.from_numpy(K).to(dev)
    viewmats = torch.from_numpy(viewmats).to(dev)
    images = torch.from_numpy(images).to(dev)  # uint8 on GPU

    centers = torch.linalg.inv(viewmats)[:, :3, 3]
    scene_scale = (centers - centers.mean(0)).norm(dim=1).max().item() * 1.1

    p = torch.from_numpy(pts).to(dev)
    C0 = 0.28209479177387814
    sh0 = ((torch.from_numpy(cols).to(dev) - 0.5) / C0)[:, None, :]
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(p),
        "scales": torch.nn.Parameter(torch.log(knn_mean_dist(p).clamp_min(1e-7))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.rand(len(p), 4, device=dev)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((len(p),), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(torch.zeros(len(p), (SH_DEGREE + 1) ** 2 - 1, 3, device=dev)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
            for k, lr in lrs.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / iters))

    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)

    t0 = time.time()
    for step in range(iters):
        i = np.random.randint(n_img)
        gt = images[i].float()[None] / 255
        sh_deg = min(step // 1000, SH_DEGREE)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        render, alpha, info = rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors,
            viewmats[i:i + 1], K[None], w, h, sh_degree=sh_deg, packed=False)
        strategy.step_pre_backward(params, opts, state, step, info)
        l1 = (render - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(render.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, opts, state, step, info, packed=False)

        if step % 500 == 0 or step == iters - 1:
            psnr = -10 * math.log10(((render - gt) ** 2).mean().item())
            print(f"step {step:6d}  loss {loss.item():.4f}  psnr {psnr:5.2f}  "
                  f"gaussians {len(params['means']):8d}  {time.time() - t0:6.0f}s", flush=True)
        if step in (iters // 2,):
            save_ply(os.path.join(out, "splat_full.ply"), params)

    save_ply(os.path.join(out, "splat_full.ply"), params)
    # Render a few training views for inspection.
    with torch.no_grad():
        for i in (0, n_img // 4, n_img // 2):
            colors = torch.cat([params["sh0"], params["shN"]], 1)
            r, _, _ = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                    torch.sigmoid(params["opacities"]), colors,
                                    viewmats[i:i + 1], K[None], w, h, sh_degree=SH_DEGREE)
            pair = torch.cat([images[i].float() / 255, r[0].clamp(0, 1)], 1)
            Image.fromarray((pair.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out, f"check_{i:02d}.jpg"))
    print("done")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 30000, float(a[5]) if len(a) > 5 else 2.0)
