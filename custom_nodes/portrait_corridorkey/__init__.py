"""Optional, isolated CorridorKey 2.0 still-photo experiment."""
from pathlib import Path
import os
import subprocess
import tempfile

import numpy as np
import cv2
from PIL import Image
import torch


class PortraitCorridorKeyTest:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "repository_path": ("STRING", {"default": "C:/AI/CorridorKey2.0"}),
            "python_executable": ("STRING", {"default": "C:/AI/CorridorKey2.0/venv/Scripts/python.exe"}),
            "despill_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_long_edge": ("INT", {"default": 1440, "min": 256, "max": 8192, "step": 16}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("foreground", "alpha", "transparent_cutout", "checker_preview")
    FUNCTION = "key"
    CATEGORY = "Portrait/Experimental"
    DESCRIPTION = "Run CorridorKey in a separate environment. Resizes to the test limit; processes each photo independently. Requires a separately installed official repository and weights."

    def key(self, image, repository_path, python_executable, despill_strength, max_long_edge, low_vram=True):
        import comfy.model_management as mm
        repo = Path(repository_path).expanduser().resolve()
        # Preserve virtualenv executable symlinks: resolving them bypasses the
        # environment's pyvenv.cfg and launches the base Python instead.
        python = Path(os.path.abspath(os.path.expanduser(python_executable)))
        if not (repo / "infer.py").is_file():
            raise ValueError("CorridorKey: repository_path must contain the official infer.py.")
        if not python.is_file():
            raise ValueError("CorridorKey: select the Python executable in its separate environment.")
        # Release ComfyUI models before the external CUDA process starts.
        mm.unload_all_models()
        mm.soft_empty_cache()
        results = []
        for frame in image:
            mm.throw_exception_if_processing_interrupted()
            with tempfile.TemporaryDirectory(prefix="portrait_corridorkey_") as tmp:
                root = Path(tmp)
                frames = root / "frames"
                frames.mkdir()
                pixels = (frame.detach().cpu().numpy()[..., :3].clip(0, 1) * 255).round().astype(np.uint8)
                photo = Image.fromarray(pixels)
                if max_long_edge:
                    photo.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
                photo.save(frames / "frame_000000.png")
                command = [str(python), str(Path(__file__).with_name("runner.py")),
                           str(repo), str(root), str(despill_strength), "1" if low_vram else "0"]
                env = os.environ.copy()
                env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                # Do not leak ComfyUI's Python search path into the isolated env.
                env.pop("PYTHONPATH", None)
                env.pop("PYTHONHOME", None)
                log = root / "inference.log"
                with log.open("w", encoding="utf-8") as output:
                    process = subprocess.Popen(command, cwd=repo, env=env, stdout=output,
                                               stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    try:
                        while True:
                            try:
                                process.wait(timeout=0.5)
                                break
                            except subprocess.TimeoutExpired:
                                mm.throw_exception_if_processing_interrupted()
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait()
                if process.returncode:
                    detail = log.read_text(encoding="utf-8", errors="replace")[-12000:]
                    raise RuntimeError("CorridorKey failed. Check its environment, weights, CUDA and ffmpeg.\n" + detail)
                rgba = np.load(root / "cutout.npy", allow_pickle=False)
                if rgba.shape != (photo.height, photo.width, 4) or not np.isfinite(rgba).all():
                    raise RuntimeError("CorridorKey returned invalid RGBA data.")
                results.append(torch.from_numpy(rgba.copy()).float().clamp(0, 1))
        rgba = torch.stack(results)
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        y, x = torch.meshgrid(torch.arange(rgba.shape[1]), torch.arange(rgba.shape[2]), indexing="ij")
        checker = (0.25 + 0.5 * ((x // 24 + y // 24) % 2)).float()[None, ..., None]
        preview = rgb * alpha[..., None] + checker * (1 - alpha[..., None])
        return rgb, alpha, rgba, preview


class PortraitCorridorKeyBatch(PortraitCorridorKeyTest):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["max_long_edge"] = ("INT", {"default": 0, "min": 0, "max": 32768, "step": 1})
        inputs["required"]["low_vram"] = ("BOOLEAN", {"default": False})
        return inputs

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("foreground", "alpha", "transparent_cutout")
    DESCRIPTION = "CorridorKey still-image matte; 0 keeps native resolution. Output is restored to framing size after any inference downscale."

    def key(self, image, repository_path, python_executable, despill_strength, max_long_edge, low_vram):
        rgb, alpha, rgba, _ = super().key(image, repository_path, python_executable, despill_strength, max_long_edge, low_vram)
        if rgba.shape[1:3] != image.shape[1:3]:
            # Restore the framing size for the shared composite and exposure stages.
            rgba = torch.nn.functional.interpolate(rgba.permute(0, 3, 1, 2), size=image.shape[1:3], mode="bilinear", align_corners=False).permute(0, 2, 3, 1).clamp(0, 1)
        return rgba[..., :3], rgba[..., 3], rgba


class PortraitCorridorSubjectMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"alpha": ("MASK",), "subject_mask": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "filter"
    CATEGORY = "Portrait/Experimental"

    def filter(self, alpha, subject_mask):
        if alpha.shape[-2:] != subject_mask.shape[-2:]:
            raise ValueError("CorridorKey: subject mask and alpha dimensions differ.")
        output = []
        for index, matte in enumerate(alpha):
            h, w = matte.shape
            scale = min(1.0, 2048 / max(h, w))
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            a = cv2.resize(matte.detach().cpu().numpy(), size, interpolation=cv2.INTER_AREA)
            coarse = cv2.resize(subject_mask[min(index, len(subject_mask) - 1)].detach().cpu().numpy(), size, interpolation=cv2.INTER_AREA)
            count, labels, stats, _ = cv2.connectedComponentsWithStats((coarse > 0.5).astype(np.uint8), 8)
            if count <= 1:
                raise ValueError("CorridorKey: subject mask is empty; cannot select the subject safely.")
            subject = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
            radius = max(4, round(max(size) * 0.02))
            subject = cv2.dilate(subject, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)))
            count, components = cv2.connectedComponents((a > 0.05).astype(np.uint8), 8)
            keep = np.unique(components[subject > 0]); keep = keep[keep != 0]
            if not len(keep):
                raise ValueError("CorridorKey: subject mask does not overlap the predicted foreground.")
            selected = np.isin(components, keep).astype(np.uint8)
            selected = cv2.dilate(selected, np.ones((5, 5), np.uint8))
            selected = cv2.resize(selected, (w, h), interpolation=cv2.INTER_NEAREST)
            output.append(matte * torch.from_numpy(selected).to(device=matte.device, dtype=matte.dtype))
        return (torch.stack(output),)


NODE_CLASS_MAPPINGS = {"PortraitCorridorKeyTest": PortraitCorridorKeyTest, "PortraitCorridorKeyBatch": PortraitCorridorKeyBatch, "PortraitCorridorSubjectMask": PortraitCorridorSubjectMask}
NODE_DISPLAY_NAME_MAPPINGS = {"PortraitCorridorKeyTest": "Portrait: CorridorKey 2.0 Test", "PortraitCorridorKeyBatch": "Portrait: CorridorKey 2.0 Batch", "PortraitCorridorSubjectMask": "Portrait: CorridorKey Subject Mask"}
