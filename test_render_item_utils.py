import unittest

from detectors.base import BubbleRegion, TextRegion
from render_item_utils import (
    build_outside_text_blocks,
    consolidate_render_items,
    dedupe_ocr_items_by_text_and_geometry,
)


IMAGE_SHAPE = (140, 180, 3)


def make_text_region(bbox, *, detector, confidence=0.9, reading_order=0):
    return TextRegion(
        bbox=bbox,
        confidence=confidence,
        detector=detector,
        reading_order=reading_order,
    )


def make_bubble_item(bubble_region, text_regions):
    return {
        "kind": "bubble",
        "bubble_region": bubble_region,
        "text_regions": list(text_regions),
        "coords": bubble_region.bbox,
        "render_bbox": bubble_region.bbox,
        "inpaint_bbox": bubble_region.bbox,
        "inpaint_bboxes": [bubble_region.bbox],
        "ocr_bbox": bubble_region.bbox,
        "ocr_crop": None,
        "reading_order": min(
            [region.reading_order for region in text_regions if region.reading_order is not None],
            default=None,
        ),
        "inpaint_fallback_used": not bool(text_regions),
        "fallback_reason": None if text_regions else "no_matched_text_regions",
        "huge_region_skipped": False,
    }


class TestOutsideTextBlockSelection(unittest.TestCase):
    def test_comic_region_inside_pp_block_attaches_instead_of_becoming_fragment(self):
        pp_regions = [
            make_text_region((100, 18, 136, 34), detector="pp_doclayout_v3", reading_order=4),
        ]
        comic_regions = [
            make_text_region((106, 20, 118, 28), detector="comic_text_detector", reading_order=4),
            make_text_region((119, 20, 130, 29), detector="comic_text_detector", reading_order=5),
        ]
        bubbles = [BubbleRegion(bbox=(10, 10, 40, 40), score=0.9)]

        blocks, stats = build_outside_text_blocks(
            pp_regions,
            comic_regions,
            bubbles,
            IMAGE_SHAPE,
        )

        self.assertEqual(stats["pp_outside_blocks"], 1)
        self.assertEqual(stats["comic_fallback_outside_blocks"], 0)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["outside_source"], "pp")
        self.assertEqual(len(blocks[0]["text_regions"]), 2)
        self.assertEqual(blocks[0]["container_bbox"], (100, 18, 136, 34))

    def test_tiny_comic_fragment_without_pp_block_is_dropped(self):
        blocks, stats = build_outside_text_blocks(
            [],
            [make_text_region((90, 40, 98, 46), detector="comic_text_detector", confidence=0.95)],
            [],
            IMAGE_SHAPE,
        )

        self.assertEqual(blocks, [])
        self.assertEqual(stats["comic_fallback_outside_blocks"], 0)

    def test_pp_block_mostly_inside_bubble_is_not_used_as_outside_text(self):
        pp_regions = [
            make_text_region((14, 14, 28, 24), detector="pp_doclayout_v3"),
            make_text_region((102, 18, 134, 32), detector="pp_doclayout_v3", reading_order=3),
        ]
        bubbles = [BubbleRegion(bbox=(10, 10, 40, 40), score=0.9)]

        blocks, _ = build_outside_text_blocks(pp_regions, [], bubbles, IMAGE_SHAPE)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["container_bbox"], (102, 18, 134, 32))

    def test_duplicate_pp_blocks_collapse_by_overlap(self):
        pp_regions = [
            make_text_region((100, 18, 136, 34), detector="pp_doclayout_v3", confidence=0.7),
            make_text_region((101, 18, 135, 34), detector="pp_doclayout_v3", confidence=0.8),
        ]

        blocks, stats = build_outside_text_blocks(pp_regions, [], [], IMAGE_SHAPE)

        self.assertEqual(stats["pp_outside_blocks"], 1)
        self.assertEqual(len(blocks), 1)

    def test_synthetic_three_bubbles_two_outside_blocks_produces_about_five_items(self):
        bubbles = [
            BubbleRegion(bbox=(10, 10, 40, 40), score=0.95),
            BubbleRegion(bbox=(48, 10, 78, 40), score=0.95),
            BubbleRegion(bbox=(18, 56, 46, 88), score=0.95),
        ]
        pp_regions = [
            make_text_region((14, 14, 28, 24), detector="pp_doclayout_v3", reading_order=0),
            make_text_region((50, 14, 68, 24), detector="pp_doclayout_v3", reading_order=1),
            make_text_region((22, 60, 38, 72), detector="pp_doclayout_v3", reading_order=2),
            make_text_region((100, 18, 136, 34), detector="pp_doclayout_v3", reading_order=3),
            make_text_region((102, 18, 136, 34), detector="pp_doclayout_v3", reading_order=4),
            make_text_region((98, 70, 138, 88), detector="pp_doclayout_v3", reading_order=5),
            make_text_region((20, 12, 30, 20), detector="pp_doclayout_v3", reading_order=6),
        ]
        comic_regions = [
            make_text_region((15, 15, 20, 22), detector="comic_text_detector", reading_order=0),
            make_text_region((21, 15, 28, 23), detector="comic_text_detector", reading_order=0),
            make_text_region((52, 15, 58, 22), detector="comic_text_detector", reading_order=1),
            make_text_region((59, 15, 68, 23), detector="comic_text_detector", reading_order=1),
            make_text_region((23, 61, 29, 69), detector="comic_text_detector", reading_order=2),
            make_text_region((30, 61, 38, 70), detector="comic_text_detector", reading_order=2),
            make_text_region((106, 20, 118, 28), detector="comic_text_detector", reading_order=3),
            make_text_region((119, 20, 130, 29), detector="comic_text_detector", reading_order=3),
            make_text_region((104, 22, 132, 30), detector="comic_text_detector", reading_order=3),
            make_text_region((104, 72, 116, 80), detector="comic_text_detector", reading_order=4),
            make_text_region((117, 72, 132, 82), detector="comic_text_detector", reading_order=4),
            make_text_region((105, 72, 133, 83), detector="comic_text_detector", reading_order=4),
            make_text_region((145, 20, 152, 25), detector="comic_text_detector", reading_order=5),
            make_text_region((146, 21, 151, 24), detector="comic_text_detector", reading_order=5),
            make_text_region((12, 12, 18, 18), detector="comic_text_detector", reading_order=0),
            make_text_region((50, 12, 56, 18), detector="comic_text_detector", reading_order=1),
        ]

        bubble_items = [
            make_bubble_item(
                bubble,
                [region for region in comic_regions if region.bbox[0] < bubble.bbox[2] and region.bbox[2] > bubble.bbox[0] and region.bbox[1] < bubble.bbox[3] and region.bbox[3] > bubble.bbox[1]],
            )
            for bubble in bubbles
        ]
        outside_blocks, stats = build_outside_text_blocks(pp_regions, comic_regions, bubbles, IMAGE_SHAPE)
        outside_items = [
            {
                "kind": "outside_text",
                "outside_source": block["outside_source"],
                "text_region": block["text_region"],
                "text_regions": list(block["text_regions"]),
                "coords": block["container_bbox"],
                "container_bbox": block["container_bbox"],
                "render_bbox": block["container_bbox"],
                "inpaint_bbox": block["container_bbox"],
                "inpaint_bboxes": [block["container_bbox"], block["ocr_bbox"]],
                "ocr_bbox": block["ocr_bbox"],
                "ocr_crop": None,
                "reading_order": block["reading_order"],
                "inpaint_fallback_used": False,
                "huge_region_skipped": False,
            }
            for block in outside_blocks
        ]

        final_items = consolidate_render_items(bubble_items + outside_items, IMAGE_SHAPE)

        self.assertEqual(stats["pp_outside_blocks"], 2)
        self.assertEqual(stats["comic_fallback_outside_blocks"], 0)
        self.assertEqual(len([item for item in final_items if item["kind"] == "bubble"]), 3)
        self.assertEqual(len([item for item in final_items if item["kind"] == "outside_text"]), 2)
        self.assertEqual(len(final_items), 5)


class TestOcrItemDedupe(unittest.TestCase):
    def test_overlapping_repeated_ocr_text_is_deduped(self):
        bubble = BubbleRegion(bbox=(10, 10, 40, 40), score=0.9)
        items = [
            make_bubble_item(bubble, [make_text_region((14, 14, 30, 26), detector="comic_text_detector")]),
            make_bubble_item(BubbleRegion(bbox=(11, 11, 39, 39), score=0.8), [make_text_region((15, 15, 29, 25), detector="comic_text_detector")]),
        ]
        texts = [
            "HE APPEARED OUT OF NOWHERE",
            "HE APPEARED OUT OF NOWHERE",
        ]

        filtered_items, filtered_texts = dedupe_ocr_items_by_text_and_geometry(items, texts)

        self.assertEqual(len(filtered_items), 1)
        self.assertEqual(filtered_texts, ["HE APPEARED OUT OF NOWHERE"])

    def test_same_text_in_different_locations_is_preserved(self):
        items = [
            {
                "kind": "outside_text",
                "render_bbox": (100, 18, 136, 34),
                "ocr_bbox": (106, 20, 130, 29),
                "coords": (100, 18, 136, 34),
                "text_regions": [make_text_region((106, 20, 130, 29), detector="comic_text_detector")],
            },
            {
                "kind": "outside_text",
                "render_bbox": (98, 70, 138, 88),
                "ocr_bbox": (104, 72, 132, 82),
                "coords": (98, 70, 138, 88),
                "text_regions": [make_text_region((104, 72, 132, 82), detector="comic_text_detector")],
            },
        ]
        texts = ["NO", "NO"]

        filtered_items, filtered_texts = dedupe_ocr_items_by_text_and_geometry(items, texts)

        self.assertEqual(len(filtered_items), 2)
        self.assertEqual(filtered_texts, texts)


if __name__ == "__main__":
    unittest.main()
