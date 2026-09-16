# Model files

Place these files under the ComfyUI installation. Paths are relative to its `models/` directory.

| Path | Used by |
| --- | --- |
| `background_removal/birefnet-portrait.safetensors` | Background removal and portrait matte generation |
| `upscale_models/RealESRGAN_x2plus.pth` | Optional portrait/background upscaling |
| `vitmatte/config.json` | LayerStyle VITMatte configuration |
| `vitmatte/preprocessor_config.json` | LayerStyle VITMatte preprocessor configuration |
| `vitmatte/model.safetensors` or the model file expected by the installed LayerStyle version | LayerStyle VITMatte edge refinement |

The workflow contains neutral placeholder names for its three image inputs. The batch script uploads the portrait and background automatically; pass `--reference` when you want a separate reference image, otherwise the portrait is reused as the reference.

Do not copy model binaries into this project unless the team has confirmed their licenses and storage requirements.
