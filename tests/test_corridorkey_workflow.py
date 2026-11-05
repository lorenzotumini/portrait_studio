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
            finally:
                sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
