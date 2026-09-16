# Portrait Studio

This folder is the distributable batch tool for the final Portrait Master workflow. ComfyUI remains the image-processing engine; [`portrait_batch.py`](portrait_batch.py) submits the approved API workflow and saves the results locally.

The current workflow source is [`workflow/portrait_master_pipeline_v3.json`](workflow/portrait_master_pipeline_v3.json). [`workflow/portrait_master_pipeline_v3_api.json`](workflow/portrait_master_pipeline_v3_api.json) is the execution format used by the batch script. The UI workflow in `ComfyUI/user/default/workflows` is retained as ComfyUI's working copy; the older root-level exports were moved to `archive/`.

## Install on a ComfyUI machine

1. Install ComfyUI and its normal Python environment. A Windows NVIDIA Portable install is the easiest dedicated workstation option. For a shared service, use a Linux or Windows NVIDIA machine and keep it on the private team network.
2. Copy both directories from `custom_nodes/` into the ComfyUI `custom_nodes/` directory:

   ```text
   portrait_tools/
   comfyui_layerstyle/
   ```

3. In the ComfyUI Python environment, install the LayerStyle dependencies once:

   ```bash
   python -m pip install -r custom_nodes/comfyui_layerstyle/requirements.txt
   ```

   Restart ComfyUI after installing or updating custom nodes.

   The repository keeps LayerStyle's runtime code and required resources; its optional documentation screenshots and example workflows are left out of Git.

4. Put the model files listed in [`MODELS.md`](MODELS.md) in the matching ComfyUI `models/` folders. The model binaries are not duplicated here because they are large and may have separate redistribution terms.

5. Copy the UI workflow into ComfyUI's workflow folder if it is not already there, then open it once in ComfyUI to confirm that all nodes and models resolve:

   ```text
   ComfyUI/user/default/workflows/portrait_master_pipeline_v3.json
   ```

## Run a batch

Start ComfyUI first, for example:

```bash
python main.py --listen 127.0.0.1 --port 8188
```

If the model directory is kept outside the ComfyUI checkout, add `--models-directory /path/to/models` to that command.

Then, from this project folder, run the batch tool. It accepts a portrait folder and one or more background files or folders:

```bash
python portrait_batch.py \
  --portraits /path/to/original_portraits \
  --backgrounds /path/to/backgrounds \
  --output /path/to/portrait_results
```

The script processes every portrait/background combination sequentially and writes files such as `portrait_name__background_name.png` to the output folder. Add multiple backgrounds by listing them after `--backgrounds`, or by passing a folder containing them. An optional reference portrait can be supplied with `--reference`; when omitted, the portrait itself is used as the reference. Existing results are preserved with a numeric suffix instead of being overwritten.

The defaults are defined near the top of `portrait_batch.py` and can also be overridden for a run. For example:

```bash
python portrait_batch.py --portraits ./originals --backgrounds ./blue.png ./gray.png \
  --output ./results --reference ./reference.png \
  --aspect-width 1 --aspect-height 1 --no-background-upscale \
  --spill-strength 0.75
```

Useful environment variables:

```text
COMFY_URL=http://127.0.0.1:8188
```

The batch tool uses only Python's standard library. It does not download models or make outbound internet requests. Apple Silicon can run the same workflow through a manual ComfyUI installation, but the current LayerStyle VITMatte path is configured for CUDA and falls back to CPU; an NVIDIA machine will be substantially faster.

## Updating the workflow

Make and test graph changes in ComfyUI, export the API workflow, and replace both files in `workflow/`. Keep the node IDs used by `portrait_batch.py` (`120`, `125`, `156`, `141:25`, `160:25`, `180`, `203`, and `46`) stable, or update the corresponding mappings in the script together with the workflow.

Review the licenses of ComfyUI, LayerStyle, and every model before redistributing this folder outside the team.
