#!/usr/bin/env python3
"""Run the Portrait Studio workflow for every portrait/background pair."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# These are the defaults used by the current final workflow. Change them here
# when the batch tool should use a different house style for every run.
DEFAULTS = {
    "aspect_width": 4,
    "aspect_height": 5,
    "portrait_upscale": True,
    "background_upscale": True,
    "exposure_strength": 0.8,
    "contrast_strength": 0.2,
    "max_brightening_stops": 0.5,
    "max_darkening_stops": 2.0,
    "spill_mode": "auto",
    "spill_strength": 0.91,
    "spill_edge_width": 35,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKFLOW = PROJECT_DIR / "workflow" / "portrait_master_pipeline_v3_api.json"


def comfy_request(base_url: str, path: str, data: bytes | None = None, method: str = "GET", headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(base_url + path, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(request, timeout=60 if path == "/prompt" else 30) as response:
        return response.read()


def describe_http_error(error: urllib.error.HTTPError) -> str:
    detail = error.read().decode("utf-8", "replace").strip()
    return f"ComfyUI HTTP {error.code}: {detail or error.reason}"


def upload_image(base_url: str, path: Path) -> str:
    content = path.read_bytes()
    boundary = "----portraitstudio" + secrets.token_hex(12)
    filename = path.name.replace("\r", "").replace("\n", "")
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    try:
        result = json.loads(comfy_request(base_url, "/upload/image", body, "POST", {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }))
    except urllib.error.HTTPError as error:
        raise ValueError(describe_http_error(error)) from error
    return "/".join(part for part in (result.get("subfolder", ""), result["name"]) if part)


def load_workflow(path: Path) -> dict:
    workflow = json.loads(path.read_text())
    required = {"120", "125", "156", "180", "203", "46", "141:25", "160:25", "147"}
    missing = required.difference(workflow)
    if missing:
        raise ValueError(f"Workflow is missing the nodes used by this script: {', '.join(sorted(missing))}")
    return workflow


def set_widget(workflow: dict, node_id: str, name: str, value) -> None:
    node = workflow[node_id]
    if name not in node["inputs"]:
        raise ValueError(f"Workflow node {node_id} has no input named {name!r}.")
    node["inputs"][name] = value


def queue_workflow(base_url: str, workflow: dict) -> dict:
    try:
        response = json.loads(comfy_request(base_url, "/prompt", json.dumps({"prompt": workflow}).encode(), "POST", {
            "Content-Type": "application/json",
        }))
    except urllib.error.HTTPError as error:
        raise ValueError(describe_http_error(error)) from error
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise ValueError(response.get("error", "ComfyUI did not return a prompt id."))

    deadline = time.monotonic() + 30 * 60
    while time.monotonic() < deadline:
        try:
            history = json.loads(comfy_request(base_url, "/history/" + urllib.parse.quote(prompt_id)))[prompt_id]
        except (urllib.error.HTTPError, KeyError):
            time.sleep(0.5)
            continue
        status = history.get("status", {})
        if status.get("status_str") == "success":
            images = history.get("outputs", {}).get("147", {}).get("images", [])
            if not images:
                raise ValueError("Workflow completed without a final image.")
            return images[0]
        if status.get("status_str") in {"error", "failed"}:
            messages = status.get("messages", [])
            detail = f": {messages[-1]}" if messages else ""
            raise ValueError(f"ComfyUI failed to execute the workflow{detail}")
        time.sleep(1)
    raise TimeoutError("ComfyUI did not finish within 30 minutes.")


def image_files(value: Path) -> list[Path]:
    if value.is_file():
        return [value] if value.suffix.lower() in IMAGE_EXTENSIONS else []
    if value.is_dir():
        return sorted(path for path in value.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(value)


def collect_backgrounds(values: list[str]) -> list[Path]:
    result = []
    for value in values:
        result.extend(image_files(Path(value).expanduser()))
    unique = {path.resolve(): path for path in result}
    if not unique:
        raise ValueError("No background images were found.")
    return sorted(unique.values())


def output_name(portrait: Path, background: Path, output_dir: Path) -> Path:
    clean = lambda value: re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "image"
    target = output_dir / f"{clean(portrait.stem)}__{clean(background.stem)}.png"
    if not target.exists():
        return target
    index = 2
    while True:
        candidate = output_dir / f"{clean(portrait.stem)}__{clean(background.stem)}_{index}.png"
        if not candidate.exists():
            return candidate
        index += 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portraits", required=True, help="Folder (or image file) containing the original portraits.")
    parser.add_argument("--backgrounds", required=True, nargs="+", help="One or more background files or folders.")
    parser.add_argument("--output", default="portrait_results", help="Output folder; defaults to ./portrait_results.")
    parser.add_argument("--reference", help="Optional reference portrait. If omitted, the workflow's configured reference is used.")
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW), help=argparse.SUPPRESS)
    parser.add_argument("--comfy-url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    parser.add_argument("--aspect-width", type=int, default=DEFAULTS["aspect_width"])
    parser.add_argument("--aspect-height", type=int, default=DEFAULTS["aspect_height"])
    parser.add_argument("--exposure-strength", type=float, default=DEFAULTS["exposure_strength"])
    parser.add_argument("--contrast-strength", type=float, default=DEFAULTS["contrast_strength"])
    parser.add_argument("--max-brightening-stops", type=float, default=DEFAULTS["max_brightening_stops"])
    parser.add_argument("--max-darkening-stops", type=float, default=DEFAULTS["max_darkening_stops"])
    parser.add_argument("--spill-mode", choices=("auto", "off", "force green"), default=DEFAULTS["spill_mode"])
    parser.add_argument("--spill-strength", type=float, default=DEFAULTS["spill_strength"])
    parser.add_argument("--spill-edge-width", type=int, default=DEFAULTS["spill_edge_width"])
    parser.add_argument("--no-portrait-upscale", action="store_false", dest="portrait_upscale", default=DEFAULTS["portrait_upscale"])
    parser.add_argument("--no-background-upscale", action="store_false", dest="background_upscale", default=DEFAULTS["background_upscale"])
    parser.add_argument("--center-subject", action="store_true", default=True)
    parser.add_argument("--no-center-subject", action="store_false", dest="center_subject")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    base_url = args.comfy_url.rstrip("/")
    portraits = image_files(Path(args.portraits).expanduser())
    backgrounds = collect_backgrounds(args.backgrounds)
    if not portraits:
        raise ValueError("No portrait images were found.")
    reference = Path(args.reference).expanduser() if args.reference else None
    if reference is not None and not reference.is_file():
        raise FileNotFoundError(reference)
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(portraits)} portrait(s) × {len(backgrounds)} background(s)")

    for portrait_index, portrait in enumerate(portraits, 1):
        for background_index, background in enumerate(backgrounds, 1):
            label = f"[{portrait_index}/{len(portraits)} × {background_index}/{len(backgrounds)}] {portrait.name} + {background.name}"
            print(label, flush=True)
            workflow = load_workflow(Path(args.workflow).expanduser())
            portrait_input = upload_image(base_url, portrait)
            background_input = upload_image(base_url, background)
            set_widget(workflow, "120", "image", portrait_input)
            set_widget(workflow, "125", "image", background_input)
            if reference is not None:
                set_widget(workflow, "180", "image", upload_image(base_url, reference))
            else:
                set_widget(workflow, "180", "image", portrait_input)
            set_widget(workflow, "156", "aspect_width", args.aspect_width)
            set_widget(workflow, "156", "aspect_height", args.aspect_height)
            set_widget(workflow, "156", "center_subject", args.center_subject)
            set_widget(workflow, "203", "mode", args.spill_mode)
            set_widget(workflow, "203", "strength", args.spill_strength)
            set_widget(workflow, "203", "edge_width", args.spill_edge_width)
            set_widget(workflow, "46", "exposure_strength", args.exposure_strength)
            set_widget(workflow, "46", "contrast_strength", args.contrast_strength)
            set_widget(workflow, "46", "max_brightening_stops", args.max_brightening_stops)
            set_widget(workflow, "46", "max_darkening_stops", args.max_darkening_stops)
            set_widget(workflow, "141:25", "value", args.portrait_upscale)
            set_widget(workflow, "160:25", "value", args.background_upscale)
            image = queue_workflow(base_url, workflow)
            payload = comfy_request(base_url, "/view?" + urllib.parse.urlencode(image))
            target = output_name(portrait, background, output_dir)
            target.write_bytes(payload)
            print(f"  -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, TimeoutError, urllib.error.URLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
