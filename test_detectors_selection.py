import unittest

from detectors.base import TextRegion
from detectors.selection import (
    dedupe_text_regions_koharu_style,
    is_near_duplicate_bbox,
    overlap_over_area,
    sort_manga_reading_order,
)


class TestSelectionGeometry(unittest.TestCase):
    def test_overlap_over_area_returns_both_directions(self):
        overlap_a, overlap_b = overlap_over_area((0, 0, 10, 10), (5, 0, 15, 10))

        self.assertAlmostEqual(overlap_a, 0.5)
        self.assertAlmostEqual(overlap_b, 0.5)

    def test_high_containment_duplicate_detected_even_with_lower_iou(self):
        self.assertTrue(
            is_near_duplicate_bbox(
                (10, 10, 50, 50),
                (14, 14, 30, 30),
                iou_threshold=0.45,
                overlap_threshold=0.80,
                containment_threshold=0.70,
            )
        )


class TestTextRegionDedupe(unittest.TestCase):
    IMAGE_SHAPE = (100, 120, 3)

    def test_strict_overlap_duplicate_removed(self):
        regions = [
            TextRegion(bbox=(10, 10, 30, 30), confidence=0.70, detector="pp_doclayout_v3"),
            TextRegion(bbox=(11, 11, 29, 29), confidence=0.72, detector="pp_doclayout_v3"),
        ]

        merged = dedupe_text_regions_koharu_style(regions, image_shape=self.IMAGE_SHAPE)

        self.assertEqual(len(merged), 1)

    def test_pp_large_block_and_comic_specific_block_keeps_comic(self):
        merged = dedupe_text_regions_koharu_style(
            [
                TextRegion(bbox=(10, 10, 40, 32), confidence=0.55, detector="pp_doclayout_v3"),
                TextRegion(bbox=(14, 12, 30, 28), confidence=0.95, detector="comic_text_detector"),
            ],
            image_shape=self.IMAGE_SHAPE,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, (14, 12, 30, 28))
        self.assertEqual(merged[0].detector, "comic_text_detector")

    def test_pp_only_outside_region_is_preserved(self):
        regions = [
            TextRegion(bbox=(10, 10, 40, 30), confidence=0.55, detector="pp_doclayout_v3"),
            TextRegion(bbox=(70, 15, 90, 28), confidence=0.60, detector="pp_doclayout_v3"),
        ]

        merged = dedupe_text_regions_koharu_style(regions, image_shape=self.IMAGE_SHAPE)

        self.assertEqual(len(merged), 2)
        self.assertIn((70, 15, 90, 28), [region.bbox for region in merged])

    def test_adjacent_non_overlapping_regions_are_not_merged(self):
        merged = dedupe_text_regions_koharu_style(
            [
                TextRegion(bbox=(10, 10, 30, 24), confidence=0.8, detector="comic_text_detector"),
                TextRegion(bbox=(32, 10, 52, 24), confidence=0.8, detector="comic_text_detector"),
            ],
            image_shape=self.IMAGE_SHAPE,
        )

        self.assertEqual(len(merged), 2)

    def test_sort_manga_reading_order_is_deterministic(self):
        regions = [
            TextRegion(bbox=(55, 10, 80, 22), reading_order=5),
            TextRegion(bbox=(10, 10, 35, 22), reading_order=9),
            TextRegion(bbox=(12, 34, 30, 48), reading_order=1),
            TextRegion(bbox=(60, 34, 82, 48), reading_order=0),
        ]

        first = sort_manga_reading_order(regions, order="ltr")
        second = sort_manga_reading_order(regions, order="ltr")

        self.assertEqual([region.bbox for region in first], [region.bbox for region in second])
        self.assertEqual(
            [region.bbox for region in first],
            [(10, 10, 35, 22), (55, 10, 80, 22), (12, 34, 30, 48), (60, 34, 82, 48)],
        )


if __name__ == "__main__":
    unittest.main()
