# CorridorKey for difficult green-screen photos

CorridorKey is an optional matting model for photos where the standard green-screen workflow does not give a good result. It uses the same framing, backgrounds and output options as the standard batch workflow.

## Setup

1. Complete the main Portrait Studio installation, including BiRefNet and `portrait_tools`.
2. Stop ComfyUI, copy `custom_nodes/portrait_corridorkey` from this project into `ComfyUI/custom_nodes/`, and restart ComfyUI.
3. Install the [official CorridorKey 2.0 repository](https://github.com/nikopueringer/CorridorKey2.0#setup) and its CUDA-enabled PyTorch and other dependencies in a **separate Python environment**. Follow its setup instructions and run `python scripts/download_weights.py` to download the weights. Do not install CorridorKey's dependencies into ComfyUI's environment.
4. Ensure `ffmpeg` is on the PATH used by ComfyUI. The model requires a supported NVIDIA GPU and CUDA-enabled PyTorch. Allow approximately 5 GB for weights, plus space for the environment and temporary files.

## Use in a batch

Start ComfyUI on the GPU you want to use, then select CorridorKey with `--model corridor`:

```bash
python portrait_batch.py --portraits /path/to/photos --backgrounds /path/to/backgrounds \
  --output /path/to/results --model corridor \
  --corridor-repo /path/to/CorridorKey2.0 \
  --corridor-python /path/to/CorridorKey2.0/.venv/bin/python --cutout
```

Use the actual path to the repository containing `infer.py` and the Python executable in its separate environment. On Windows, the executable will typically be `venv\Scripts\python.exe`. These paths are on the ComfyUI server; alternatively set `CORRIDORKEY_REPO` and `CORRIDORKEY_PYTHON` where the batch script runs. If using multiple ComfyUI servers, each needs the node, environment, weights and the same paths. The normal batch options, including `--comfy-url`, aspect ratio, reference and `--cutout`, still apply.

## Memory and quality

**16 GB VRAM is recommended** for large images. On an 8 GB GPU, add `--corridor-low-vram`; if memory is still insufficient, also set `--corridor-max-long-edge 4800`, reducing the value further if needed. The default limit of `0` uses the framed image's native size. Smaller inference images are resized back to the framed output size, but lost detail cannot be recovered. Actual memory needs vary with image size and other GPU use.

Use `--corridor-despill` to adjust CorridorKey's own green-spill correction from `0` to `1` (default `0.5`). The standard `--spill-*` options do not apply to CorridorKey. Inspect hair and green clothing: the subject filter removes disconnected objects but can also remove isolated fine details, and objects connected to the subject may remain. Use a separate output folder when comparing models because existing cutouts are skipped.

The official model and weights are not bundled; see the [CorridorKey license](https://github.com/nikopueringer/CorridorKey2.0/blob/master/LICENSE) for their terms.
