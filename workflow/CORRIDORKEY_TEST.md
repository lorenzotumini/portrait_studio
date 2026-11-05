# CorridorKey 2.0: independent still-photo test

Open `portrait_corridorkey_test.json` in ComfyUI. The master workflows and batch CLI are unchanged. This experiment needs only ComfyUI's built-in nodes and the new `portrait_corridorkey` package; LayerStyle, VNCCS and KJNodes are not used.

## Install once

1. Copy the entire `custom_nodes/portrait_corridorkey` folder into `ComfyUI/custom_nodes/`, then restart ComfyUI. This is separate from `portrait_tools`.
2. Install the [official CorridorKey 2.0 repository](https://github.com/nikopueringer/CorridorKey2.0) in a **separate Python environment**, following its current setup instructions. Download its weights with `python scripts/download_weights.py` from that environment. Do not install its dependencies into ComfyUI's environment.
3. Ensure `ffmpeg` is on the PATH inherited by ComfyUI. The official runner checks for it even for EXR output. The documented hardware is an NVIDIA RTX 30-series or newer with CUDA-enabled PyTorch. Allow approximately 5 GB for model downloads, plus environment and temporary storage.
4. On the test node, set `repository_path` to the folder containing `infer.py`, and `python_executable` to that installation's Python executable. Defaults assume `C:/AI/CorridorKey2.0` and its `venv/Scripts/python.exe`; change both to match your installation. Linux environments typically use `venv/bin/python`.

## Run and inspect

Choose a green-screen photo using Load Image, then queue the workflow. Start with the default 1440-pixel long-edge limit. Smaller images are not enlarged. Outputs stay at the test resolution; increasing the limit can substantially increase memory use.

The graph previews foreground colors, the alpha matte, and a checkerboard composite. It saves a straight RGBA PNG under `ComfyUI/output/CorridorKey_test/`. White alpha means subject; black means background. The alpha output is coverage, not the inverted mask emitted by ComfyUI's Load Image node. The API JSON is also provided for direct queueing, independently of `portrait_batch.py`.

Try despill strengths 0, 0.5, and 1 on the same photo. Inspect hair, translucent edges, skin, green clothing and flags. Compare the transparent PNG against the master workflow's cutout for the same input at comparable resolution. No exposure matching or other spill cleanup runs in this test.

## Implementation and limits

The custom node launches the official `infer.py` in a separate process for each photo, using one-frame windows (`--hann_chunk 1 --hann_stride 1 --low_vram`). Unrelated photos never share temporal state. ComfyUI models are unloaded first to make room on the GPU; the external process releases its models on exit. This initial adapter reloads weights each time, so it favors isolation over throughput.

Input is resized and written as an 8-bit sRGB PNG. The official runner exports a float EXR cutout, which its own environment decodes. Straight scene-linear foreground RGB is converted with CorridorKey's `linear_to_srgb`; alpha stays unchanged. Float RGBA is passed back to ComfyUI without video encoding or a second despill pass. PNG output is SDR and clips values outside [0, 1]. Temporary inputs, EXRs, and intermediate features are removed after each run. Errors include the tail of the runner log. ComfyUI cancellation terminates the runner.

The official repository and weights are **not bundled**. Review its [license](https://github.com/nikopueringer/CorridorKey2.0/blob/master/LICENSE) before distribution or commercial software integration. This adapter does not change those terms.

Structural and mocked bridge tests do not establish image quality, GPU compatibility, or successful model inference. A real run requires the external environment, weights and GPU.
