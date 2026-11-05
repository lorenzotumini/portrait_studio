"""Executed only by the external CorridorKey Python environment."""
import os
import runpy
import sys
from pathlib import Path


def main():
    repo, root = (Path(value).resolve() for value in sys.argv[1:3])
    strength = sys.argv[3]
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    sys.path.insert(0, str(repo))
    sys.argv = [str(repo / "infer.py"), "--input", str(root / "frames"),
                "--output_dir", str(root / "output"), "--outputs", "cutout",
                "--output_format", "exr", "--exr_dtype", "float",
                "--hann_chunk", "1", "--hann_stride", "1", "--low_vram",
                "--temp_dir", str(root / "features"), "--despill_strength", strength]
    runpy.run_path(str(repo / "infer.py"), run_name="__main__")
    import cv2
    import numpy as np
    from corridorkey.color_utils import linear_to_srgb
    files = list((root / "output").rglob("cutout_*.exr"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one cutout EXR, found {len(files)}")
    bgra = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise RuntimeError("Cannot decode CorridorKey's RGBA EXR; check OpenCV EXR support.")
    # Official EXR cutouts contain straight scene-linear RGB and coverage alpha.
    rgba = bgra[..., [2, 1, 0, 3]].astype(np.float32)
    rgba[..., :3] = linear_to_srgb(rgba[..., :3])
    np.save(root / "cutout.npy", np.clip(rgba, 0, 1), allow_pickle=False)


if __name__ == "__main__":
    main()
