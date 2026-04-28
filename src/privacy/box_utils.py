"""Bounding-box utilities for the privacy gate."""
from __future__ import annotations

from PIL import Image, ImageDraw


def iou(box1: list[int], box2: list[int]) -> float:
    """Intersection-over-Union for two [xmin, ymin, xmax, ymax] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes: list[list[int]], iou_threshold: float = 0.3) -> list[list[int]]:
    """
    Iteratively merge overlapping boxes (union) until stable.

    Unlike standard NMS (which picks the highest-scoring box and suppresses
    others), we take the union so that adjacent sensitive regions are covered
    by a single inpainting pass, reducing both redundancy and computation.
    """
    if not boxes:
        return []
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        result: list[list[int]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = list(merged[i])
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if iou(cur, merged[j]) > iou_threshold:
                    cur = [
                        min(cur[0], merged[j][0]),
                        min(cur[1], merged[j][1]),
                        max(cur[2], merged[j][2]),
                        max(cur[3], merged[j][3]),
                    ]
                    used[j] = True
                    changed = True
            result.append(cur)
        merged = result
    return merged


def expand_box(box: list[int], image_w: int, image_h: int, padding: int = 8) -> list[int]:
    """Expand box by `padding` pixels on each side, clamped to image bounds."""
    return [
        max(0, box[0] - padding),
        max(0, box[1] - padding),
        min(image_w, box[2] + padding),
        min(image_h, box[3] + padding),
    ]


def calculate_box_area(box: list[int]) -> int:
    """Area of [xmin, ymin, xmax, ymax] box; 0 for invalid input."""
    if box is None or len(box) != 4:
        return 0
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def draw_boxes_on_image(image: Image.Image, boxes: list[list[int]]) -> Image.Image:
    """Return a copy of `image` with red bounding boxes drawn over detected regions."""
    vis = image.copy()
    if not boxes:
        return vis
    draw = ImageDraw.Draw(vis)
    for box in boxes:
        draw.rectangle(box, outline=(220, 50, 50), width=3)
    return vis
