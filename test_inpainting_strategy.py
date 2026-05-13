import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from detectors.base import TextRegion
from inpainting.strategy import (
    apply_bubble_fill_fast_path,
    boxes_from_mask,
    composite_masked,
    crop_box,
    crop_windows_from_text_regions,
    run_inpaint_crop,
)
from inpainting.lama_manga import LamaMangaInpainter


class TestInpaintingStrategy(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is not available")
    def test_boxes_from_mask_detects_separate_components(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[1:3, 1:4] = 255
        mask[7:10, 8:11] = 255

        boxes = boxes_from_mask(mask)

        self.assertEqual(boxes, [(1, 1, 4, 3), (8, 7, 11, 10)])

    @unittest.skipIf(np is None, "numpy is not available")
    def test_crop_box_expands_with_margin_and_clamps(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:3, 1:3] = 255

        crop_image, crop_mask, crop_bounds = crop_box(image, mask, (1, 1, 3, 3), 4)

        self.assertEqual(crop_bounds, (0, 0, 7, 7))
        self.assertEqual(crop_image.shape[:2], (7, 7))
        self.assertEqual(crop_mask.shape, (7, 7))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_crop_windows_from_text_regions_enlarges_and_merges(self):
        text_regions = [
            TextRegion(bbox=(10, 10, 18, 18), reading_order=0),
            TextRegion(bbox=(18, 12, 26, 20), reading_order=1),
            TextRegion(bbox=(70, 70, 78, 78), reading_order=2),
        ]

        windows = crop_windows_from_text_regions(text_regions, (100, 100, 3))

        self.assertEqual(len(windows), 2)
        self.assertLessEqual(windows[0][0], 10)
        self.assertLessEqual(windows[0][1], 10)
        self.assertGreaterEqual(windows[0][2], 26)
        self.assertGreaterEqual(windows[0][3], 20)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_composite_masked_only_changes_masked_pixels(self):
        base_image = np.zeros((5, 5, 3), dtype=np.uint8)
        patch_image = np.zeros((5, 5, 3), dtype=np.uint8) + 200
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2:4, 1:3] = 255

        composite_masked(base_image, patch_image, mask, 0, 0)

        self.assertTrue(np.all(base_image[2:4, 1:3] == 200))
        self.assertTrue(np.all(base_image[0:2, :] == 0))
        self.assertTrue(np.all(base_image[:, 3:] == 0))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_supplied_crop_windows_are_used_instead_of_raw_mask_boxes(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[12:14, 12:14] = 255
        forward_shapes = []

        def fake_forward(crop_image, crop_mask, crop_bubble):
            forward_shapes.append(crop_image.shape)
            return crop_image.copy()

        run_inpaint_crop(
            fake_forward,
            image,
            mask,
            bubble_mask=None,
            crop_trigger_size=800,
            crop_margin=0,
            resize_limit=200,
            pad_mod=1,
            crop_windows=[(20, 20, 30, 30)],
        )

        self.assertEqual(forward_shapes, [(42, 42, 3)])

    @unittest.skipIf(np is None, "numpy is not available")
    def test_crop_window_with_empty_mask_is_skipped_then_residual_mask_runs(self):
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        forward_shapes = []

        def fake_forward(crop_image, crop_mask, crop_bubble):
            forward_shapes.append(crop_image.shape)
            return crop_image.copy()

        run_inpaint_crop(
            fake_forward,
            image,
            mask,
            bubble_mask=None,
            crop_trigger_size=800,
            crop_margin=0,
            resize_limit=200,
            pad_mod=1,
            crop_windows=[(20, 20, 25, 25)],
        )

        self.assertEqual(forward_shapes, [(2, 2, 3)])

    @unittest.skipIf(np is None, "numpy is not available")
    def test_apply_bubble_fill_fast_path_fills_component_inside_bubble(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8) + 200
        text_mask = np.zeros((20, 20), dtype=np.uint8)
        text_mask[8:12, 8:12] = 255
        bubble_mask = np.zeros((20, 20), dtype=np.uint8)
        bubble_mask[4:16, 4:16] = 255

        filled_image, remaining_mask, stats = apply_bubble_fill_fast_path(
            image,
            text_mask,
            bubble_mask,
        )

        self.assertEqual(stats["filled_components"], 1)
        self.assertEqual(stats["remaining_pixels"], 0)
        self.assertTrue(np.all(filled_image[8:12, 8:12] == 200))
        self.assertEqual(int(np.count_nonzero(remaining_mask)), 0)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_apply_bubble_fill_fast_path_leaves_component_outside_bubble_for_lama(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8) + 180
        text_mask = np.zeros((20, 20), dtype=np.uint8)
        text_mask[2:6, 2:6] = 255
        bubble_mask = np.zeros((20, 20), dtype=np.uint8)
        bubble_mask[10:18, 10:18] = 255

        filled_image, remaining_mask, stats = apply_bubble_fill_fast_path(
            image,
            text_mask,
            bubble_mask,
        )

        self.assertEqual(stats["filled_components"], 0)
        self.assertEqual(int(np.count_nonzero(remaining_mask)), int(np.count_nonzero(text_mask)))
        self.assertTrue(np.array_equal(filled_image, image))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_apply_bubble_fill_fast_path_uses_median_sampled_color(self):
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        image[4:20, 4:20] = (180, 180, 180)
        text_mask = np.zeros((24, 24), dtype=np.uint8)
        text_mask[10:14, 10:14] = 255
        bubble_mask = np.zeros((24, 24), dtype=np.uint8)
        bubble_mask[4:20, 4:20] = 255

        filled_image, remaining_mask, stats = apply_bubble_fill_fast_path(
            image,
            text_mask,
            bubble_mask,
        )

        self.assertEqual(stats["remaining_pixels"], 0)
        self.assertTrue(np.all(filled_image[10:14, 10:14] == 180))
        self.assertEqual(int(np.count_nonzero(remaining_mask)), 0)

    @unittest.skipIf(np is None, "numpy is not available")
    def test_run_model_on_patch_short_circuits_when_fast_fill_clears_mask(self):
        class ShortCircuitInpainter(LamaMangaInpainter):
            def load(self):
                raise AssertionError("load should not be called when bubble fill clears the mask")

        image = np.zeros((20, 20, 3), dtype=np.uint8) + 210
        text_mask = np.zeros((20, 20), dtype=np.uint8)
        text_mask[8:12, 8:12] = 255
        bubble_mask = np.zeros((20, 20), dtype=np.uint8)
        bubble_mask[4:16, 4:16] = 255

        output = ShortCircuitInpainter()._run_model_on_patch(
            image,
            text_mask,
            bubble_mask,
        )

        self.assertTrue(np.all(output[8:12, 8:12] == 210))

    @unittest.skipIf(np is None, "numpy is not available")
    def test_run_inpaint_crop_passes_crop_local_bubble_mask(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[12:16, 12:16] = 255
        bubble_mask = np.zeros((40, 40), dtype=np.uint8)
        bubble_mask[8:20, 8:20] = 255
        recorded_bubble_masks = []

        def fake_forward(crop_image, crop_mask, crop_bubble):
            recorded_bubble_masks.append(None if crop_bubble is None else crop_bubble.copy())
            return crop_image.copy()

        run_inpaint_crop(
            fake_forward,
            image,
            mask,
            bubble_mask=bubble_mask,
            crop_trigger_size=800,
            crop_margin=0,
            resize_limit=200,
            pad_mod=1,
            crop_windows=[(10, 10, 18, 18)],
        )

        self.assertEqual(len(recorded_bubble_masks), 1)
        self.assertEqual(recorded_bubble_masks[0].shape, (34, 34))
        self.assertGreater(int(np.count_nonzero(recorded_bubble_masks[0])), 0)


if __name__ == "__main__":
    unittest.main()
