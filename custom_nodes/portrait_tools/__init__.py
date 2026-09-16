import logging
import math

import cv2
import numpy as np
import torch


def srgb_to_linear(rgb):
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(rgb):
    return torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb.clamp_min(0).pow(1 / 2.4) - 0.055).clamp(0, 1)


def face_levels(image, mask, detector):
    height, width = image.shape[:2]
    if tuple(mask.shape) != (height, width):
        raise ValueError("Portrait exposure: image and foreground mask must have matching dimensions.")
    scale = min(1.0, 1024 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = cv2.resize(image[..., :3].detach().cpu().numpy(), size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(mask.detach().cpu().numpy(), size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(48, 48))
    faces = [box for box in faces if alpha[box[1]:box[1] + box[3], box[0]:box[0] + box[2]].mean() > 0.75]
    if len(faces) != 1:
        return None
    x, y, w, h = faces[0]
    yy, xx = np.ogrid[:size[1], :size[0]]
    oval = ((xx - x - w * 0.5) / (w * 0.33)) ** 2 + ((yy - y - h * 0.54) / (h * 0.38)) ** 2 < 1
    pixels = rgb[oval & (alpha > 0.98)]
    if len(pixels) < 64:
        return None
    linear = np.where(pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.quantile(np.maximum(luminance, 1e-6), [0.2, 0.5, 0.8])


class PortraitMatteFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "center_subject": ("BOOLEAN", {"default": True}),
            "side_margin": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.25, "step": 0.01}),
        }, "optional": {
            "aspect_width": ("INT", {"default": 1, "min": 1, "max": 32}),
            "aspect_height": ("INT", {"default": 1, "min": 1, "max": 32}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "frame"
    CATEGORY = "image/portrait"
    DESCRIPTION = "Center the largest foreground silhouette horizontally without resizing or cropping the photo height. Set the width:height ratio with aspect_width and aspect_height. Wide subjects expand the canvas while preserving that ratio. Disable centering for an image-centered crop/pad."

    def frame(self, image, mask, center_subject, side_margin, aspect_width=1, aspect_height=1):
        batch, height, width, channels = image.shape
        if tuple(mask.shape[-2:]) != (height, width):
            raise ValueError("Portrait framing: image and foreground mask must have matching dimensions.")
        plans = []
        output_height = height
        for b in range(batch):
            center = width / 2
            if center_subject:
                alpha = mask[min(b, len(mask) - 1)].detach().cpu().numpy()
                scale = min(1.0, 1024 / max(height, width))
                small = cv2.resize(alpha, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
                count, _, stats, _ = cv2.connectedComponentsWithStats((small > 0.5).astype(np.uint8), connectivity=8)
                if count > 1:
                    x, _, w, _, _ = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
                    center = (x + w / 2) * width / small.shape[1]
                    required_width = math.ceil(w * width / small.shape[1] + 2 * side_margin * height)
                    output_height = max(output_height, math.ceil(required_width * aspect_height / aspect_width))
                else:
                    output_height = max(output_height, math.ceil(width * aspect_height / aspect_width))
                    logging.warning("Portrait framing: empty matte; keeping the full image.")
            plans.append(center)
        output_width = max(1, round(output_height * aspect_width / aspect_height))
        result = image.new_zeros((batch, output_height, output_width, channels))
        top = (output_height - height) // 2
        for b, center in enumerate(plans):
            left = round(center - output_width / 2) if center_subject else ((width - output_width) // 2 if width >= output_width else -((output_width - width) // 2))
            start, end = max(0, left), min(width, left + output_width)
            result[b, top:top + height, start - left:end - left] = image[b, :, start:end]
        return (result,)


class PortraitExposureMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "reference_image": ("IMAGE",),
            "reference_mask": ("MASK",),
            "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            "contrast_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_exposure_stops": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.25}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match"
    CATEGORY = "image/portrait"
    DESCRIPTION = "Match facial exposure and optionally contrast to one reference portrait in linear RGB, preserving channel ratios and alpha. No face or multiple faces in the source leaves it unchanged. Clipped detail cannot be recovered."

    def match(self, image, mask, reference_image, reference_mask, strength, contrast_strength, max_exposure_stops):
        return match_reference(image, mask, reference_image, reference_mask, strength, strength * contrast_strength, strength * max_exposure_stops, strength * max_exposure_stops)


class PortraitExposureMatchV3:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "reference_image": ("IMAGE",),
            "reference_mask": ("MASK",),
            "exposure_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            "contrast_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_brightening_stops": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.25}),
            "max_darkening_stops": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.25}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match"
    CATEGORY = "image/portrait"
    DESCRIPTION = "Match facial exposure and contrast independently. Brightening/darkening limits cap the exposure shift after strength; contrast pivots around the facial median. Zero exposure strength keeps that median without matching reference brightness. Both strengths at zero bypass correction. Reference selection is manual."

    def match(self, image, mask, reference_image, reference_mask, exposure_strength, contrast_strength, max_brightening_stops, max_darkening_stops):
        return match_reference(image, mask, reference_image, reference_mask, exposure_strength, contrast_strength, max_brightening_stops, max_darkening_stops)


def match_reference(image, mask, reference_image, reference_mask, exposure_strength, contrast_strength, max_brightening_stops, max_darkening_stops):
    if exposure_strength == 0 and contrast_strength == 0:
        return (image,)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    ref = face_levels(reference_image[0], reference_mask[0], detector)
    if ref is None:
        raise ValueError("Portrait exposure: choose a reference with one clear frontal face and a matching foreground mask.")
    results = []
    for b, frame in enumerate(image):
        source = face_levels(frame, mask[min(b, len(mask) - 1)], detector)
        if source is None:
            logging.warning("Portrait exposure skipped: use a portrait with one clear frontal face.")
            results.append(frame)
            continue
        ev = float(np.clip(exposure_strength * np.log2(ref[1] / source[1]), -max_darkening_stops, max_brightening_stops))
        spread = math.log(source[2] / source[0])
        contrast = float(np.clip(math.log(ref[2] / ref[0]) / spread, 0.7, 1.5)) if spread > 0.05 else 1.0
        slope = 1 + contrast_strength * (contrast - 1)
        rgb = frame[..., :3]
        linear = srgb_to_linear(rgb)
        luminance = linear @ linear.new_tensor([0.2126, 0.7152, 0.0722])
        target = float(source[1]) * 2 ** ev * (luminance.clamp_min(1e-6) / float(source[1])).pow(slope)
        gain = target / luminance.clamp_min(1e-6)
        gain = torch.minimum(gain, linear.amax(dim=-1).clamp_min(1e-6).reciprocal())
        corrected = linear * gain.unsqueeze(-1)
        corrected = linear_to_srgb(corrected)
        if frame.shape[-1] > 3:
            corrected = torch.cat((corrected, frame[..., 3:]), dim=-1)
        results.append(corrected)
    return (torch.stack(results),)


def has_green_screen(image, mask):
    height, width = image.shape[:2]
    scale = min(1.0, 1024 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = cv2.resize(image[..., :3].detach().cpu().numpy(), size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(mask.detach().cpu().numpy(), size, interpolation=cv2.INTER_AREA)
    background = (alpha < 0.02) & (rgb.max(axis=-1) > 0.06)
    if background.sum() < 0.03 * alpha.size:
        return False
    other = np.maximum(rgb[..., 0], rgb[..., 2])
    green = (rgb[..., 1] > 0.12) & (rgb[..., 1] > 1.3 * other) & (rgb[..., 1] - other > 0.04)
    if green[background].mean() < 0.75:
        return False
    hues = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0][background & green]
    low, middle, high = np.quantile(hues, [0.25, 0.5, 0.75])
    return bool(75 <= middle <= 155 and high - low <= 20)


class PortraitGreenSpill:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "original_image": ("IMAGE",),
            "mask": ("MASK",),
            "mode": (["auto", "off", "force green"],),
            "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
            "edge_width": ("INT", {"default": 32, "min": 1, "max": 256, "tooltip": "Width in input pixels inside the opaque foreground; soft matte pixels are also included."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "affected_area")
    FUNCTION = "despill"
    CATEGORY = "image/portrait"
    DESCRIPTION = "Reduce excess green near the matte boundary while preserving linear luminance and alpha. Auto requires a dominant, consistent green background in the original photo; it bypasses uncertain cases. The affected-area mask shows correction blend strength. Genuine green at an edge may also be affected."

    def despill(self, image, original_image, mask, mode, strength, edge_width):
        if mode == "off" or strength == 0:
            return (image, image.new_zeros(image.shape[:3]))
        if mode not in ("auto", "force green"):
            raise ValueError("Portrait spill: mode must be auto, off, or force green.")
        if image.shape[1:3] != original_image.shape[1:3] or image.shape[1:3] != mask.shape[-2:]:
            raise ValueError("Portrait spill: foreground, original photo and mask must have matching dimensions.")
        images, areas = [], []
        for b, frame in enumerate(image):
            alpha = mask[min(b, len(mask) - 1)]
            original = original_image[min(b, len(original_image) - 1)]
            if mode == "auto" and not has_green_screen(original, alpha):
                images.append(frame)
                areas.append(frame.new_zeros(frame.shape[:2]))
                continue
            distance = cv2.distanceTransform((alpha.detach().cpu().numpy() >= 0.98).astype(np.uint8), cv2.DIST_L2, 5)
            weight = torch.as_tensor(np.clip(1 - distance / edge_width, 0, 1), device=frame.device, dtype=frame.dtype)
            weight = weight * (alpha.to(device=frame.device) > 1 / 255) * strength
            linear = srgb_to_linear(frame[..., :3])
            excess = (linear[..., 1] - torch.maximum(linear[..., 0], linear[..., 2])).clamp_min(0)
            amount = excess * weight
            # Neutralize excess green without darkening the edge or changing the red-blue difference.
            corrected = linear + amount.unsqueeze(-1) * linear.new_tensor([0.7152, -0.2848, 0.7152])
            corrected = linear_to_srgb(corrected)
            corrected = torch.where((amount > 0).unsqueeze(-1), corrected, frame[..., :3])
            if frame.shape[-1] > 3:
                corrected = torch.cat((corrected, frame[..., 3:]), dim=-1)
            images.append(corrected)
            areas.append(weight * (excess > 0))
        return (torch.stack(images), torch.stack(areas))


NODE_CLASS_MAPPINGS = {
    "PortraitMatteFrame": PortraitMatteFrame,
    "PortraitExposureMatch": PortraitExposureMatch,
    "PortraitExposureMatchV3": PortraitExposureMatchV3,
    "PortraitGreenSpill": PortraitGreenSpill,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PortraitMatteFrame": "Portrait: Frame from Matte",
    "PortraitExposureMatch": "Portrait: Match Reference Exposure",
    "PortraitExposureMatchV3": "Portrait: Match Reference Exposure V3",
    "PortraitGreenSpill": "Portrait: Green Spill Cleanup",
}
