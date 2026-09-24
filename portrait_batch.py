#!/usr/bin/env python3
"""Run the Portrait Studio workflow for every portrait/background pair."""  # noqa: EXE001

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# These are the defaults used by the current final workflow. Change them here
# when the batch tool should use a different house style for every run.
DEFAULTS = {
    "aspect_width": 4,
    "aspect_height": 5,
    "portrait_upscale": False,
    "background_upscale": False,
    "exposure_strength": 0.0, # 0.8
    "contrast_strength": 0.0, # 0.2
    "max_brightening_stops": 0.0, # 0.5,
    "max_darkening_stops": 0.0, # 2.0,
    "spill_mode": "auto",
    "spill_strength": 0.9,
    "spill_edge_width": 35,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# RAW formats ComfyUI cannot read. They are decoded to high-quality 8-bit sRGB
# JPEG before upload (see convert_raw); ComfyUI flattens every image to 8-bit RGB anyway,
# so the conversion is lossless with respect to what the workflow consumes.
RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".dng", ".erf", ".mrw", ".nef", ".nrf", ".nrw",
    ".orf", ".pef", ".raf", ".raw", ".rwl", ".rw2", ".sr2", ".x3f",
}

INPUT_EXTENSIONS = IMAGE_EXTENSIONS | RAW_EXTENSIONS
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


_raw_lock = threading.Lock()
_magick_path: str | None = None
_magick_checked = False


def _magick_binary() -> str | None:
    global _magick_path, _magick_checked
    if not _magick_checked:
        _magick_path = shutil.which("magick") or shutil.which("convert")
        _magick_checked = True
    return _magick_path


def convert_raw(path: Path, cache_dir: Path) -> Path:
    """Decode a RAW file into a cached high-quality 8-bit sRGB JPEG.

    ComfyUI's LoadImage flattens every image to 8-bit RGB, and a JPEG at
    quality 92 is visually lossless for the workflow's purposes. JPEG keeps
    uploads far below ComfyUI's 100 MiB request limit, which full-resolution
    PNGs of large RAWs can exceed. Decodes use the camera's embedded white
    balance ("as shot"). The JPEG is cached in cache_dir and reused until
    the source RAW is modified.
    """
    with _raw_lock:
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = f"{path.stem}.jpg" if path.parent.resolve() == cache_dir.resolve() \
            else f"{path.parent.name}_{path.stem}.jpg"
        dest = cache_dir / name
        if dest.exists() and dest.stat().st_mtime >= path.stat().st_mtime:
            return dest
        binary = _magick_binary()
        if binary is None:
            raise RuntimeError(
                f"Cannot decode RAW file {path.name}: ImageMagick (magick) with libraw "
                "support is required on the machine running this script.")
        tmp = dest.with_name(dest.name + f".part.{os.getpid()}")
        try:
            # "jpeg:" forces the output format: the temp file's extension is not .jpg.
            command = [
                binary, str(path), "-auto-orient", "-colorspace", "sRGB",
                "-quality", "92", "-sampling-factor", "4:2:0", f"jpeg:{tmp}",
            ]
            proc = subprocess.run(command, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"RAW conversion failed for {path.name}: {proc.stderr.strip()[:400]}")
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"Converted RAW {path.name} -> {dest.name} ({dest.stat().st_size // (1024 * 1024)} MiB)", flush=True)
    return dest


def load_workflow(path: Path, require_cutout: bool = False) -> dict:
    workflow = json.loads(path.read_text())
    required = {"120", "125", "156", "180", "203", "46", "141:25", "160:25", "147"}
    if require_cutout:
        # Node 207 is a PreviewImage fed by node 46 (the color-corrected, transparent-background cutout).
        required.add("207")
    missing = required.difference(workflow)
    if missing:
        raise ValueError(f"Workflow is missing the nodes used by this script: {', '.join(sorted(missing))}")
    return workflow


def set_widget(workflow: dict, node_id: str, name: str, value) -> None:
    node = workflow[node_id]
    if name not in node["inputs"]:
        raise ValueError(f"Workflow node {node_id} has no input named {name!r}.")
    node["inputs"][name] = value


def queue_workflow(base_url: str, workflow: dict, want_cutout: bool = False) -> dict:
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
            outputs = history.get("outputs", {})
            images = outputs.get("147", {}).get("images", [])
            if not images:
                raise ValueError("Workflow completed without a final image.")
            result = {"final": images[0]}
            if want_cutout:
                cutout_images = outputs.get("207", {}).get("images", [])
                if not cutout_images:
                    raise ValueError("Workflow completed without a cutout image.")
                result["cutout"] = cutout_images[0]
            return result
        if status.get("status_str") in {"error", "failed"}:
            messages = status.get("messages", [])
            detail = f": {messages[-1]}" if messages else ""
            raise ValueError(f"ComfyUI failed to execute the workflow{detail}")
        time.sleep(1)
    raise TimeoutError("ComfyUI did not finish within 30 minutes.")


def image_files(value: Path) -> list[Path]:
    if value.is_file():
        return [value] if value.suffix.lower() in INPUT_EXTENSIONS else []
    if value.is_dir():
        return sorted(path for path in value.iterdir() if path.is_file() and path.suffix.lower() in INPUT_EXTENSIONS)
    raise FileNotFoundError(value)


def collect_backgrounds(values: list[str]) -> list[Path]:
    result = []
    for value in values:
        result.extend(image_files(Path(value).expanduser()))
    unique = {path.resolve(): path for path in result}
    if not unique:
        raise ValueError("No background images were found.")
    return sorted(unique.values())


def clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "image"


def output_name(portrait: Path, background: Path, output_dir: Path,
                reserved: set[Path] | None = None) -> Path:
    target = output_dir / f"{clean_name(portrait.stem)}__{clean_name(background.stem)}.png"
    if not target.exists() and (reserved is None or target not in reserved):
        return target
    index = 2
    while True:
        candidate = output_dir / f"{clean_name(portrait.stem)}__{clean_name(background.stem)}_{index}.png"
        if not candidate.exists() and (reserved is None or candidate not in reserved):
            return candidate
        index += 1


def plan_output_names(jobs: list[tuple[Path, Path]], output_dir: Path) -> dict[tuple[Path, Path], Path]:
    """Reserve unique output paths before workers start writing files."""
    reserved: set[Path] = set()
    targets: dict[tuple[Path, Path], Path] = {}
    for portrait, background in jobs:
        target = output_name(portrait, background, output_dir, reserved)
        reserved.add(target)
        targets[(portrait, background)] = target
    return targets


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portraits", required=True, help="Folder (or image file) containing the original portraits.")
    parser.add_argument("--backgrounds", required=True, nargs="+", help="One or more background files or folders.")
    parser.add_argument("--output", default="portrait_results", help="Output folder; defaults to ./portrait_results.")
    parser.add_argument("--reference", help="Optional reference portrait. If omitted, the workflow's configured reference is used.")
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW), help=argparse.SUPPRESS)
    parser.add_argument("--comfy-url", nargs="+", default=[os.environ.get("COMFY_URL", "http://127.0.0.1:8188")],
                        help="One or more ComfyUI URLs (one per GPU). Jobs are processed in parallel, "
                             "one worker per instance. Defaults to $COMFY_URL or http://127.0.0.1:8188.")
    parser.add_argument("--aspect-width", type=int, default=DEFAULTS["aspect_width"])
    parser.add_argument("--aspect-height", type=int, default=DEFAULTS["aspect_height"])
    parser.add_argument("--exposure-strength", type=float, default=DEFAULTS["exposure_strength"])
    parser.add_argument("--contrast-strength", type=float, default=DEFAULTS["contrast_strength"])
    parser.add_argument("--max-brightening-stops", type=float, default=DEFAULTS["max_brightening_stops"])
    parser.add_argument("--max-darkening-stops", type=float, default=DEFAULTS["max_darkening_stops"])
    parser.add_argument("--spill-mode", choices=("auto", "off", "force green"), default=DEFAULTS["spill_mode"])
    parser.add_argument("--spill-strength", type=float, default=DEFAULTS["spill_strength"])
    parser.add_argument("--spill-edge-width", type=int, default=DEFAULTS["spill_edge_width"])
    parser.add_argument("--cutout", action="store_true", default=False,
                        help="Also save each portrait without a background (PNG with transparent alpha) "
                             "as <portrait>__cutout.png. The cutout uses the same matte and color "
                             "correction as the composited result but not its sharpen/grain.")
    parser.add_argument("--raw-cache-dir", default="~/.cache/portrait_studio/raw",
                        help="Cache directory for RAW inputs decoded to JPEG before upload "
                             "(ComfyUI cannot read RAW files). Defaults to %(default)s.")
    parser.add_argument("--no-portrait-upscale", action="store_false", dest="portrait_upscale", default=DEFAULTS["portrait_upscale"])
    parser.add_argument("--no-background-upscale", action="store_false", dest="background_upscale", default=DEFAULTS["background_upscale"])
    parser.add_argument("--center-subject", action="store_true", default=True)
    parser.add_argument("--no-center-subject", action="store_false", dest="center_subject")
    return parser.parse_args()


def process_pair(base_url: str, workflow_path: Path, portrait: Path, background: Path,
                 reference: Path | None, args: argparse.Namespace, target: Path,
                 cutout_target: Path | None = None) -> Path:
    workflow = load_workflow(workflow_path, require_cutout=args.cutout)
    raw_cache = Path(args.raw_cache_dir).expanduser()

    def stage(path: Path) -> Path:
        if path.suffix.lower() in RAW_EXTENSIONS:
            return convert_raw(path, raw_cache)
        return path

    portrait = stage(portrait)
    background = stage(background)
    if reference is not None:
        reference = stage(reference)
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
    result = queue_workflow(base_url, workflow, want_cutout=args.cutout)
    payload = comfy_request(base_url, "/view?" + urllib.parse.urlencode(result["final"]))
    target.write_bytes(payload)
    if cutout_target is not None and not cutout_target.exists():
        # The cutout is identical for every background of the same portrait, so each
        # portrait saves it once; workers that race on the same file write identical
        # bytes, and the atomic replace keeps the file whole either way.
        cutout_payload = comfy_request(base_url, "/view?" + urllib.parse.urlencode(result["cutout"]))
        tmp_path = cutout_target.with_name(cutout_target.name + ".tmp")
        tmp_path.write_bytes(cutout_payload)
        os.replace(tmp_path, cutout_target)
    return target


def main() -> int:
    args = arguments()
    base_urls = []
    for url in args.comfy_url:
        normalized_url = url.rstrip("/")
        if not normalized_url:
            raise ValueError("ComfyUI URLs must not be empty.")
        if normalized_url in base_urls:
            raise ValueError(f"Duplicate ComfyUI URL: {url}")
        base_urls.append(normalized_url)
    portraits = image_files(Path(args.portraits).expanduser())
    backgrounds = collect_backgrounds(args.backgrounds)
    if not portraits:
        raise ValueError("No portrait images were found.")
    reference = Path(args.reference).expanduser() if args.reference else None
    if reference is not None and not reference.is_file():
        raise FileNotFoundError(reference)
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(portraits)} portrait(s) × {len(backgrounds)} background(s) "
          f"across {len(base_urls)} instance(s): {', '.join(base_urls)}")

    # Shared work queue: each GPU worker grabs the next unfinished pair, so the
    # load self-balances even if individual jobs run for different times.
    jobs = [(portrait, background) for portrait in portraits for background in backgrounds]
    output_targets = plan_output_names(jobs, output_dir)
    cutout_targets = {portrait: output_dir / f"{clean_name(portrait.stem)}__cutout.png"
                      for portrait in portraits} if args.cutout else None
    pending_jobs = deque(jobs)
    jobs_lock = threading.Lock()
    errors: list[str] = []

    def take_job() -> tuple[Path, Path] | None:
        with jobs_lock:
            if not pending_jobs:
                return None
            return pending_jobs.popleft()

    def worker(base_url: str) -> None:
        while True:
            job = take_job()
            if job is None:
                return
            portrait, background = job
            print(f"[{base_url}] {portrait.name} + {background.name}", flush=True)
            try:
                target = process_pair(base_url, Path(args.workflow).expanduser(), portrait,
                                      background, reference, args, output_targets[job],
                                      cutout_targets[portrait] if cutout_targets else None)
            except (TimeoutError, urllib.error.URLError) as error:
                # A transport failure usually means this instance cannot make
                # progress. Remove it from rotation and let another instance
                # retry the current job instead of consuming the whole queue.
                with jobs_lock:
                    pending_jobs.appendleft(job)
                print(f"Warning: [{base_url}] unavailable; requeued {portrait.name} + {background.name}: {error}",
                      file=sys.stderr, flush=True)
                return
            except Exception as error:  # noqa: BLE001 - one failed job must not stop the batch
                message = f"[{base_url}] {portrait.name} + {background.name}: {error}"
                with jobs_lock:
                    errors.append(message)
                print(f"Error: {message}", file=sys.stderr, flush=True)
                continue
            print(f"  -> {target}", flush=True)

    futures = []
    with ThreadPoolExecutor(max_workers=len(base_urls), thread_name_prefix="comfy") as pool:
        for base_url in base_urls:
            futures.append((base_url, pool.submit(worker, base_url)))

    # Exceptions outside the per-job handler must not disappear inside a
    # Future. They indicate that a worker stopped unexpectedly.
    for base_url, future in futures:
        try:
            future.result()
        except Exception as error:  # noqa: BLE001 - last-resort report for a crashed worker
            message = f"[{base_url}] worker stopped unexpectedly: {type(error).__name__}: {error}"
            with jobs_lock:
                errors.append(message)
            print(f"Error: {message}", file=sys.stderr, flush=True)

    with jobs_lock:
        remaining_jobs = list(pending_jobs)
    if remaining_jobs:
        for portrait, background in remaining_jobs:
            errors.append(f"{portrait.name} + {background.name}: no available ComfyUI instance")
        print(f"Error: {len(remaining_jobs)} job(s) were not processed because no ComfyUI instance was available.",
              file=sys.stderr, flush=True)

    if errors:
        print(f"Finished with {len(errors)} failed job(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, TimeoutError, urllib.error.URLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
