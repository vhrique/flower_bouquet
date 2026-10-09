"""Apply EXIF orientation and downscale photos for reconstruction."""
import glob, os, sys
from PIL import Image, ImageOps

src, dst, max_dim = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000
os.makedirs(dst, exist_ok=True)
for f in sorted(glob.glob(os.path.join(src, "*.jpg"))):
    im = ImageOps.exif_transpose(Image.open(f))
    im.thumbnail((max_dim, max_dim), Image.LANCZOS)
    im.save(os.path.join(dst, os.path.basename(f)), quality=95)
print("done", im.size)
