import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("portrait_tools", Path(__file__).with_name("__init__.py"))
portrait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portrait)


class GreenSpillTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros(1, 128, 128)
        self.mask[:, 16:112, 16:112] = 1
        self.foreground = torch.full((1, 128, 128, 3), 0.15)
        self.foreground[:, 16:112, 16:25] = torch.tensor([0.1, 0.3, 0.12])
        self.foreground[:, 55:75, 55:75] = torch.tensor([0.05, 0.6, 0.1])
        self.original = torch.empty_like(self.foreground)
        self.original[:] = torch.tensor([0.12, 0.45, 0.15])
        self.original[self.mask.bool()] = self.foreground[self.mask.bool()]
        self.node = portrait.PortraitGreenSpill()

    def test_auto_detects_screen_but_not_green_clothing_on_gray(self):
        self.assertTrue(portrait.has_green_screen(self.original[0], self.mask[0]))
        gray = self.original.clone()
        gray[~self.mask.bool()] = 0.3
        self.assertFalse(portrait.has_green_screen(gray[0], self.mask[0]))
        out, area = self.node.despill(self.foreground, gray, self.mask, "auto", 1, 12)
        self.assertTrue(torch.equal(out, self.foreground))
        self.assertEqual(area.count_nonzero().item(), 0)

    def test_auto_skips_mixed_background_and_missing_background(self):
        mixed = self.original.clone()
        mixed[:, :, :64] = torch.tensor([0.1, 0.2, 0.6])
        self.assertFalse(portrait.has_green_screen(mixed[0], self.mask[0]))
        self.assertFalse(portrait.has_green_screen(self.original[0], torch.ones_like(self.mask[0])))

    def test_spill_reduction_preserves_luminance_and_core(self):
        out, area = self.node.despill(self.foreground, self.original, self.mask, "auto", 1, 12)
        before, after = portrait.srgb_to_linear(self.foreground), portrait.srgb_to_linear(out)
        weights = torch.tensor([0.2126, 0.7152, 0.0722])
        self.assertTrue(torch.allclose(before @ weights, after @ weights, atol=1e-6))
        self.assertLess((after[..., 1] - after[..., 0])[area > 0].mean(), (before[..., 1] - before[..., 0])[area > 0].mean())
        self.assertTrue(torch.equal(out[:, 55:75, 55:75], self.foreground[:, 55:75, 55:75]))
        self.assertTrue(torch.equal(out[area == 0], self.foreground[area == 0]))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_off_and_zero_strength_are_exact_bypasses(self):
        for mode, strength in [("off", 1), ("auto", 0)]:
            out, area = self.node.despill(self.foreground, self.original, self.mask, mode, strength, 12)
            self.assertIs(out, self.foreground)
            self.assertEqual(area.count_nonzero().item(), 0)

    def test_force_green_works_without_screen_detection(self):
        out, area = self.node.despill(self.foreground, torch.zeros_like(self.original), self.mask, "force green", 1, 12)
        self.assertGreater(area.count_nonzero().item(), 0)
        self.assertFalse(torch.equal(out, self.foreground))

    def test_alpha_is_preserved_and_soft_edges_are_included(self):
        alpha = self.mask.clone()
        alpha[:, 16:112, 16:25] = 0.5
        rgba = torch.cat((self.foreground, alpha.unsqueeze(-1)), dim=-1)
        out, area = self.node.despill(rgba, self.original, alpha, "force green", 1, 12)
        self.assertTrue(torch.equal(out[..., 3], alpha))
        self.assertGreater(area[:, 16:112, 16:25].min().item(), 0)

    def test_auto_decides_per_image_in_batch(self):
        gray = self.original.clone()
        gray[~self.mask.bool()] = 0.3
        batch = self.foreground.repeat(2, 1, 1, 1)
        out, area = self.node.despill(batch, torch.cat((self.original, gray)), self.mask, "auto", 1, 12)
        self.assertGreater(area[0].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(out[1], batch[1]))

    def test_mismatched_mask_reports_error(self):
        with self.assertRaisesRegex(ValueError, "matching dimensions"):
            self.node.despill(self.foreground, self.original, self.mask[:, :64], "auto", 1, 12)


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.linear = torch.tensor([0.025, 0.05, 0.1, 0.2, 0.4]).view(1, 1, 5, 1).expand(-1, -1, -1, 3)
        self.image = portrait.linear_to_srgb(self.linear)
        self.mask = torch.ones(1, 1, 5)
        self.source = np.array([0.05, 0.1, 0.2])
        self.node = portrait.PortraitExposureMatchV3()

    def correct(self, reference, exposure=1, contrast=0, bright=0.5, dark=2):
        with patch.object(portrait, "face_levels", side_effect=[np.array(reference), self.source]):
            return self.node.match(self.image, self.mask, self.image, self.mask, exposure, contrast, bright, dark)[0]

    def test_brightening_limit_applies_after_strength(self):
        out = self.correct([0.2, 0.4, 0.8], exposure=0.8, bright=0.5)
        self.assertTrue(torch.allclose(portrait.srgb_to_linear(out), self.linear * 2 ** 0.5, atol=1e-6))

    def test_darkening_limit_is_independent(self):
        out = self.correct([0.005, 0.01, 0.02], dark=1.5)
        self.assertTrue(torch.allclose(portrait.srgb_to_linear(out), self.linear * 2 ** -1.5, atol=1e-6))

    def test_contrast_without_reference_brightness_matching(self):
        out = self.correct([0.025, 0.2, 0.8], exposure=0, contrast=1)
        linear = portrait.srgb_to_linear(out)
        self.assertTrue(torch.allclose(linear[:, :, 2], self.linear[:, :, 2], atol=1e-6))
        self.assertLess(linear[0, 0, 0, 0], self.linear[0, 0, 0, 0])
        self.assertGreater(linear[0, 0, 4, 0], self.linear[0, 0, 4, 0])

    def test_zero_strengths_bypass_even_invalid_reference(self):
        out = self.node.match(self.image, self.mask, self.image, self.mask, 0, 0, 0.5, 2)[0]
        self.assertIs(out, self.image)

    def test_unrecognized_source_is_unchanged(self):
        with patch.object(portrait, "face_levels", side_effect=[self.source, None]):
            out = self.node.match(self.image, self.mask, self.image, self.mask, 1, 1, 0.5, 2)[0]
        self.assertTrue(torch.equal(out, self.image))

    def test_legacy_settings_map_to_same_v3_transform(self):
        ref = np.array([0.08, 0.3, 0.7])
        with patch.object(portrait, "face_levels", side_effect=[ref, self.source, ref, self.source]):
            old = portrait.PortraitExposureMatch().match(self.image, self.mask, self.image, self.mask, 0.8, 0.25, 2)[0]
            new = self.node.match(self.image, self.mask, self.image, self.mask, 0.8, 0.2, 1.6, 1.6)[0]
        self.assertTrue(torch.equal(old, new))


if __name__ == "__main__":
    unittest.main()
