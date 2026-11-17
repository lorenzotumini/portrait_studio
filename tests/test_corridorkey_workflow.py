"""Dependency-free checks of the optional graph and external runner contract."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


class CorridorKeyWorkflowTests(unittest.TestCase):
    def test_batch_graph_preserves_framing_and_uses_corridor_alpha_consistently(self):
        api = json.loads((ROOT / "workflow/portrait_corridorkey_pipeline_api.json").read_text())
        self.assertEqual(api["214"]["inputs"]["image"], ["156", 0])
        self.assertEqual(api["156"]["inputs"]["mask"], ["184", 0])
        self.assertEqual(api["46"]["inputs"]["image"], ["216", 0])
        self.assertEqual(api["46"]["inputs"]["mask"], ["217", 0])
        self.assertEqual(api["52"]["inputs"]["layer_mask"], ["217", 0])
        self.assertEqual(api["215"]["inputs"]["mask"], ["217", 0])
        self.assertEqual(api["217"]["inputs"], {"alpha": ["214", 1], "subject_mask": ["34", 0]})
        self.assertEqual(api["216"]["inputs"], {"image": ["214", 0], "alpha": ["215", 0]})
        self.assertEqual(api["207"]["inputs"]["images"], ["46", 0])
        for node in api.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], api)
        self.assertNotIn("157", api)
        self.assertNotIn("208", api)

    def test_subject_filter_removes_disconnected_objects_and_preserves_edges(self):
        try:
            import torch
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("requires ComfyUI's torch, numpy and OpenCV")
        spec = importlib.util.spec_from_file_location("corridor_node_test", ROOT / "custom_nodes/portrait_corridorkey/__init__.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        alpha = torch.zeros((1, 200, 200))
        alpha[:, 70:160, 70:130] = 1
        alpha[:, 69, 90:110] = 0.1
        alpha[:, 5:12, 5:40] = 1
        subject = torch.zeros_like(alpha)
        subject[:, 80:150, 80:120] = 1
        filtered, = module.PortraitCorridorSubjectMask().filter(alpha, subject)
        self.assertEqual(filtered[0, 8, 10].item(), 0)
        self.assertEqual(filtered[0, 100, 100].item(), 1)
        self.assertAlmostEqual(filtered[0, 69, 100].item(), 0.1)
        with self.assertRaisesRegex(ValueError, "subject mask is empty"):
            module.PortraitCorridorSubjectMask().filter(alpha, torch.zeros_like(subject))

    def test_graph_links_and_api_match(self):
        graph = json.loads((ROOT / "workflow/portrait_corridorkey_test.json").read_text())
        nodes = {node["id"]: node for node in graph["nodes"]}
        for link_id, source, slot, target, target_slot, kind in graph["links"]:
            output = nodes[source]["outputs"][slot]
            input_ = nodes[target]["inputs"][target_slot]
            self.assertEqual(output["type"], kind)
            self.assertEqual(input_["type"], kind)
            self.assertIn(link_id, output["links"])
            self.assertEqual(input_["link"], link_id)
        api = json.loads((ROOT / "workflow/portrait_corridorkey_test_api.json").read_text())
        self.assertEqual(api["3"]["inputs"]["images"], ["2", 2])
        self.assertEqual(nodes[2]["widgets_values"], [api["2"]["inputs"][key] for key in
                         ("repository_path", "python_executable", "despill_strength", "max_long_edge")])

    def test_runner_uses_single_frame_exr_and_preserves_separate_alpha(self):
        spec = importlib.util.spec_from_file_location("corridorkey_test_runner", ROOT / "custom_nodes/portrait_corridorkey/runner.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        cv2, np, colors = MagicMock(), MagicMock(), MagicMock()
        bgra, rgba = MagicMock(), MagicMock()
        bgra.ndim, bgra.shape = 3, (8, 12, 4)
        bgra.__getitem__.return_value.astype.return_value = rgba
        cv2.imread.return_value = bgra
        modules = {"cv2": cv2, "numpy": np, "corridorkey": MagicMock(), "corridorkey.color_utils": colors}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "output").mkdir()
            (root / "output/cutout_000000.exr").touch()
            original_path = sys.path[:]
            try:
                with patch.object(sys, "argv", ["runner.py", tmp, tmp, "0.75"]), patch.dict(sys.modules, modules), patch.object(runner.runpy, "run_path") as run:
                    runner.main()
                    args = sys.argv
                    for flag, value in [("--hann_chunk", "1"), ("--hann_stride", "1"),
                                        ("--output_format", "exr"), ("--outputs", "cutout"),
                                        ("--despill_strength", "0.75")]:
                        self.assertEqual(args[args.index(flag) + 1], value)
                    self.assertIn("--low_vram", args)
                    run.assert_called_once_with(str(root / "infer.py"), run_name="__main__")
                    colors.linear_to_srgb.assert_called_once_with(rgba.__getitem__.return_value)
                    # Only RGB is transfer-encoded; alpha is not transformed or multiplied.
                    self.assertEqual(rgba.__setitem__.call_args.args[0], (Ellipsis, slice(None, 3)))
                    self.assertEqual(np.save.call_args.args[0], root / "cutout.npy")
                with patch.object(sys, "argv", ["runner.py", tmp, tmp, "0.5", "0"]), patch.dict(sys.modules, modules), patch.object(runner.runpy, "run_path"):
                    runner.main()
                    self.assertNotIn("--low_vram", sys.argv)
            finally:
                sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
