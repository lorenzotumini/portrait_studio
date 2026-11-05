"""Optional, isolated CorridorKey 2.0 still-photo experiment."""
from pathlib import Path
import os
import subprocess
import tempfile

import numpy as np
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

    def key(self, image, repository_path, python_executable, despill_strength, max_long_edge):
        import comfy.model_management as mm
        repo = Path(repository_path).expanduser().resolve()
        python = Path(python_executable).expanduser().resolve()
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
                photo.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
                photo.save(frames / "frame_000000.png")
                command = [str(python), str(Path(__file__).with_name("runner.py")),
                           str(repo), str(root), str(despill_strength)]
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


NODE_CLASS_MAPPINGS = {"PortraitCorridorKeyTest": PortraitCorridorKeyTest}
NODE_DISPLAY_NAME_MAPPINGS = {"PortraitCorridorKeyTest": "Portrait: CorridorKey 2.0 Test"}
