# Model files

Download these files into ComfyUI's configured `models/` directory. Paths below are relative to that directory.

| Save as | Download | Used by |
| --- | --- | --- |
| `background_removal/birefnet-portrait.safetensors` | [BiRefNet portrait](https://huggingface.co/1038lab/BiRefNet/blob/main/BiRefNet-portrait.safetensors) — rename to the filename shown here | Background removal |
| `upscale_models/RealESRGAN_x2plus.pth` | [Real-ESRGAN v0.2.1](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1) | Upscaling; install it for the bundled UI workflow |
| `vitmatte/config.json` | [Configuration](https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/config.json?download=true) | VITMatte |
| `vitmatte/preprocessor_config.json` | [Preprocessor configuration](https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/preprocessor_config.json?download=true) | VITMatte |
| `vitmatte/model.safetensors` | [VITMatte weights](https://huggingface.co/hustvl/vitmatte-small-composition-1k/blob/main/model.safetensors) | Hair/edge refinement |

Keep all three VITMatte files together in `models/vitmatte/`. Restart ComfyUI after copying the files.
