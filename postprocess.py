"""Isolate the bouquet from the COLMAP dense cloud and mesh it.

Usage: python postprocess.py <colmap_work_dir> <out_dir>
Expects <work>/sparse_txt (model_converter TXT export) and <work>/dense/fused.ply.
"""
import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh


def read_cameras(images_txt):
    """Return camera centers and viewing directions (world frame)."""
    centers, dirs, ups = [], [], []
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#")]
    for line in lines[::2]:
        v = line.split()
        qw, qx, qy, qz, tx, ty, tz = map(float, v[1:8])
        R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
        t = np.array([tx, ty, tz])
        centers.append(-R.T @ t)
        dirs.append(R.T @ [0, 0, 1])
        ups.append(R.T @ [0, -1, 0])
    return np.array(centers), np.array(dirs), np.array(ups)


def closest_point_to_rays(c, d):
    """Least-squares point nearest to all camera optical axes."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for ci, di in zip(c, d):
        P = np.eye(3) - np.outer(di, di)
        A += P
        b += P @ ci
    return np.linalg.solve(A, b)


def main(work, out):
    os.makedirs(out, exist_ok=True)
    centers, dirs, ups = read_cameras(os.path.join(work, "sparse_txt", "images.txt"))
    target = closest_point_to_rays(centers, dirs)
    orbit_r = np.median(np.linalg.norm(centers - target, axis=1))
    cam_up = ups.mean(0)
    cam_up /= np.linalg.norm(cam_up)

    pcd = o3d.io.read_point_cloud(os.path.join(work, "dense", "fused.ply"))
    print(f"fused points: {len(pcd.points)}, orbit radius {orbit_r:.3f}")
    pts = np.asarray(pcd.points)

    # Table plane: largest RANSAC plane near the target whose normal agrees with camera "up".
    near = np.linalg.norm(pts - target, axis=1) < 1.2 * orbit_r
    sub = pcd.select_by_index(np.where(near)[0])
    up, plane_d = None, None
    for _ in range(6):
        model, inl = sub.segment_plane(0.01 * orbit_r, 3, 2000)
        n = np.array(model[:3])
        if abs(n @ cam_up) > 0.8:
            up = n if n @ cam_up > 0 else -n
            plane_d = model[3] if n @ cam_up > 0 else -model[3]
            break
        sub = sub.select_by_index(inl, invert=True)
    if up is None:
        raise RuntimeError("table plane not found")
    print("table normal", up, "agreement", up @ cam_up)

    # Height above table and horizontal distance to the vase axis.
    h = pts @ up + plane_d
    axis_pt = target - (target @ up + plane_d) * up
    rel = pts - axis_pt
    radial = np.linalg.norm(rel - np.outer(rel @ up, up), axis=1)
    keep = (h > 0.008 * orbit_r) & (h < 1.5 * orbit_r) & (radial < 0.6 * orbit_r)
    obj = pcd.select_by_index(np.where(keep)[0])

    # Keep the connected cluster closest to the vase axis.
    labels = np.array(obj.cluster_dbscan(eps=0.02 * orbit_r, min_points=10))
    opts = np.asarray(obj.points)
    orel = opts - axis_pt
    orad = np.linalg.norm(orel - np.outer(orel @ up, up), axis=1)
    best, best_score = None, np.inf
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        if m.sum() < 500:
            continue
        score = np.median(orad[m])
        if score < best_score:
            best, best_score = lab, score
    obj = obj.select_by_index(np.where(labels == best)[0])
    obj, _ = obj.remove_statistical_outlier(20, 2.0)
    print("bouquet points:", len(obj.points))

    # Re-orient so the table normal is +Z and the vase base sits at the origin, scaled so height = 1.
    z = up
    x = np.cross([0, 1, 0] if abs(z[1]) < 0.9 else [1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rw = np.stack([x, y, z])
    P = (np.asarray(obj.points) - axis_pt) @ Rw.T
    zmin = P[:, 2].min()
    P[:, 2] -= zmin
    height = P[:, 2].max()
    P /= height
    # aligned = (Rw @ (world - origin) - [0, 0, zmin]) / height
    with open(os.path.join(out, "align.json"), "w") as f:
        json.dump({"R": Rw.tolist(), "origin": axis_pt.tolist(), "zmin": float(zmin),
                   "height": float(height)}, f, indent=1)
    obj.points = o3d.utility.Vector3dVector(P)
    obj.normals = o3d.utility.Vector3dVector(np.asarray(obj.normals) @ Rw.T)
    o3d.io.write_point_cloud(os.path.join(out, "bouquet_points.ply"), obj)

    # Poisson surface, trimmed where point support is weak.
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(obj, depth=10)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.06))
    mesh = mesh.crop(obj.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    tri_clusters, counts, _ = mesh.cluster_connected_triangles()
    tri_clusters, counts = np.asarray(tri_clusters), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[tri_clusters] < 0.01 * counts.max())
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    print("mesh:", len(mesh.vertices), "verts,", len(mesh.triangles), "tris")
    o3d.io.write_triangle_mesh(os.path.join(out, "bouquet_mesh.ply"), mesh)

    to_trimesh(mesh, linear=False).export(os.path.join(out, "bouquet.obj"))
    to_trimesh(mesh, linear=True).export(os.path.join(out, "bouquet.glb"))

    # Lighter copy for the web viewer.
    web = mesh.simplify_quadric_decimation(min(len(mesh.triangles), 300_000))
    to_trimesh(web, linear=True).export(os.path.join(out, "bouquet_web.glb"))
    print("web mesh:", len(web.triangles), "tris")


def to_trimesh(mesh, linear):
    """Open3D mesh -> trimesh with RGBA vertex colors.

    Photo colors are sRGB; glTF defines COLOR_0 as linear, so convert for .glb output.
    """
    c = np.clip(np.asarray(mesh.vertex_colors), 0, 1)
    if linear:
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    c = np.round(c * 255).astype(np.uint8)
    return trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                           vertex_colors=np.c_[c, np.full(len(c), 255, np.uint8)], process=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
