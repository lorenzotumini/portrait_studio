import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import portrait_batch as batch


class InputConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"

    def fake_convert(self, command, **kwargs):
        Path(command[-1].removeprefix("jpeg:")).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0, stderr="")

    def test_discovers_heic_and_heif_case_insensitively(self):
        for name in ("one.HEIC", "two.heif", "three.jpg", "skip.txt"):
            (self.root / name).touch()
        self.assertEqual(len(batch.image_files(self.root)), 3)
        self.assertEqual(batch.image_files(self.root / "one.HEIC"), [self.root / "one.HEIC"])

    def test_conversion_is_cached_and_invalidated_when_source_changes(self):
        source = self.root / "photo.HEIC"
        source.write_bytes(b"source")
        with patch.object(batch, "_magick_binary", return_value="magick"), \
                patch.object(batch.subprocess, "run", side_effect=self.fake_convert) as run:
            first = batch.convert_input_image(source, self.cache)
            self.assertEqual(batch.convert_input_image(source, self.cache), first)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertEqual(command[1], str(source.resolve()) + "[0]")
            self.assertIn("-auto-orient", command)
            self.assertIn(str(batch.PROJECT_DIR / "assets" / "sRGB.icc"), command)
            source.write_bytes(b"updated source")
            self.assertNotEqual(batch.convert_input_image(source, self.cache), first)

    def test_same_stem_formats_and_paths_do_not_share_cache_entries(self):
        sources = [self.root / "photo.heic", self.root / "photo.rw2",
                   self.root / "other" / "photo.heic"]
        for source in sources:
            source.parent.mkdir(exist_ok=True)
            source.write_bytes(b"source")
        with patch.object(batch, "_magick_binary", return_value="magick"), \
                patch.object(batch.subprocess, "run", side_effect=self.fake_convert):
            self.assertEqual(len({batch.convert_input_image(p, self.cache) for p in sources}), 3)

    def test_decode_failure_leaves_no_partial_cache(self):
        source = self.root / "broken.heic"
        source.touch()

        def fail(command, **kwargs):
            self.fake_convert(command)
            return subprocess.CompletedProcess(command, 1, stderr="unsupported codec")

        with patch.object(batch, "_magick_binary", return_value="magick"), \
                patch.object(batch.subprocess, "run", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "HEIC/HEIF conversion failed.*libheif"):
                batch.convert_input_image(source, self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_heif_is_staged_for_portrait_background_and_reference(self):
        args = argparse.Namespace(**batch.DEFAULTS, raw_cache_dir=str(self.cache),
                                  center_subject=True, cutout=False)
        inputs = [self.root / "portrait.HEIC", self.root / "background.heif", self.root / "ref.heic"]
        with patch.object(batch, "convert_input_image", side_effect=lambda p, _: p.with_suffix(".jpg")) as convert, \
                patch.object(batch, "upload_image", side_effect=lambda _, p: p.name) as upload, \
                patch.object(batch, "queue_workflow", return_value={"final": {"filename": "out.png"}}), \
                patch.object(batch, "comfy_request", return_value=b"result"):
            output = self.root / "result.png"
            batch.process_pair("http://unused", batch.DEFAULT_WORKFLOW, *inputs, args, output)
            self.assertEqual([call.args[0] for call in convert.call_args_list], inputs)
            self.assertTrue(all(call.args[1].suffix == ".jpg" for call in upload.call_args_list))
            self.assertEqual(output.read_bytes(), b"result")




class GreenScreenWorkflowTests(unittest.TestCase):
    def test_ui_and_api_route_rgb_through_lazy_gate_and_restore_vitmatte_alpha(self):
        import json
        ui = json.loads((batch.PROJECT_DIR / "workflow/portrait_master_pipeline_v3.json").read_text())
        api = batch.load_workflow(batch.DEFAULT_WORKFLOW)
        nodes = {n["id"]: n for n in ui["nodes"]}
        links = {link[0]: link for link in ui["links"]}
        self.assertEqual(api["208"]["inputs"]["image"], ["101", 0])
        self.assertEqual(api["211"]["inputs"]["image"], ["208", 0])
        self.assertEqual(api["213"]["inputs"]["keyed_image"], ["211", 0])
        self.assertEqual(api["213"]["inputs"]["mode"], "auto")
        self.assertEqual(api["212"]["inputs"], {"image": ["213", 0], "alpha": ["100", 0]})
        self.assertEqual(api["46"]["inputs"]["image"], ["212", 0])
        self.assertEqual(api["209"]["inputs"]["mask"], ["208", 1])
        for lid, source, output, target, input_, _ in links.values():
            self.assertIn(lid, nodes[source]["outputs"][output]["links"])
            self.assertEqual(nodes[target]["inputs"][input_]["link"], lid)
        self.assertEqual(nodes[213]["widgets_values"], ["auto"])

    def test_batch_spill_mode_controls_vnccs_gate_and_existing_despill(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for mode in ("auto", "off", "force green"):
                args = argparse.Namespace(**{**batch.DEFAULTS, "spill_mode": mode},
                                          raw_cache_dir=str(root), center_subject=True, cutout=False)
                with patch.object(batch, "upload_image", return_value="input.png"), \
                        patch.object(batch, "queue_workflow", return_value={"final": {"filename": "out.png"}}) as queue, \
                        patch.object(batch, "comfy_request", return_value=b"result"):
                    batch.process_pair("http://unused", batch.DEFAULT_WORKFLOW, root / "portrait.png",
                                       root / "bg.png", None, args, root / "out.png")
                workflow = queue.call_args.args[1]
                self.assertEqual(workflow["203"]["inputs"]["mode"], mode)
                self.assertEqual(workflow["213"]["inputs"]["mode"], mode)


if __name__ == "__main__":
    unittest.main()
