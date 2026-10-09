# Flower bouquet in 3D

A flower bouquet reconstructed in 3D from 87 phone photos taken in a loop around it,
rendered in the browser as a Gaussian Splat.

**View it:** https://vhrique.github.io/flower_bouquet/

## Pipeline

1. **Prepare photos:** apply EXIF rotation, downscale to 2000 px (`prep_images.py`).
2. **Structure-from-Motion (COLMAP):** SIFT features, exhaustive matching, incremental mapping
   to recover the camera poses (87/87 registered, 0.82 px mean reprojection error).
3. **Multi-view stereo (COLMAP):** undistort, PatchMatch stereo, fusion into a dense point cloud.
4. **Isolate the bouquet** (`postprocess.py`): find the table plane with RANSAC and the vase axis
   from the cameras' optical axes, keep the cluster standing there, and save the transform
   that makes it upright and normalized (`align.json`). This also produces a Poisson mesh.
5. **Gaussian Splatting** (`train_splat.py`): train 30k steps with
   [gsplat](https://github.com/nerfstudio-project/gsplat) on the COLMAP poses, in the aligned frame.
6. **Crop and export** (`crop_splat.py`): keep the bouquet and a bit of tablecloth, remove
   background haze, and write a compact `.splat` for the web.
7. **Viewer** (`site/`): [Spark](https://sparkjs.dev) on three.js.

## Reproducing

```bash
python prep_images.py images work/images 2000

cd work
colmap feature_extractor --database_path db.db --image_path images \
    --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL
colmap exhaustive_matcher --database_path db.db
mkdir -p sparse && colmap mapper --database_path db.db --image_path images --output_path sparse
colmap model_converter --input_path sparse/0 --output_path sparse_txt --output_type TXT
colmap image_undistorter --image_path images --input_path sparse/0 --output_path dense \
    --output_type COLMAP --max_image_size 2000
colmap patch_match_stereo --workspace_path dense --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path dense --input_type geometric --output_path dense/fused.ply
colmap model_converter --input_path dense/sparse --output_path dense/sparse_txt --output_type TXT
cd ..

python postprocess.py work out                      # needs open3d, trimesh
python train_splat.py work out/align.json out/splat 30000 1.5   # needs torch (CUDA), gsplat
python crop_splat.py out/splat/splat_full.ply out/bouquet_points.ply out/splat
cp out/splat/bouquet.splat site/
```

Tested with COLMAP 4.2.1 (CUDA), PyTorch 2.4.1 + CUDA 12.4, and gsplat 1.5.3 on an RTX 3060 Laptop GPU.
Dense stereo takes about 75 minutes and splat training about 70 minutes.
