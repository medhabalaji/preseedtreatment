from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VisionResult:
    sprout_count: int
    total_white_area: float
    overlay_path: str
    treated_sprout_count: int = 0
    treated_white_area: float = 0.0
    untreated_sprout_count: int = 0
    untreated_white_area: float = 0.0


def analyze_image(
    image_path,
    overlay_path,
    *,
    l_threshold=200,
    min_contour_area=35,
    max_contour_area_ratio=0.025,
    blur_kernel=(5, 5),
):
    """Detect bright chickpea radicals in left/right batches and save an annotated overlay."""
    image_path = Path(image_path)
    overlay_path = Path(overlay_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Unable to decode image. Send a valid JPEG frame.")

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_channel, _, _ = cv2.split(lab_image)
    _, saturation, value = cv2.split(hsv_image)
    hue = hsv_image[:, :, 0]

    blurred = cv2.GaussianBlur(l_channel, blur_kernel, 0)
    _, bright_mask = cv2.threshold(blurred, l_threshold, 255, cv2.THRESH_BINARY)
    low_saturation_mask = cv2.inRange(saturation, 0, 110)
    visible_mask = cv2.inRange(value, 100, 255)
    white_mask = cv2.bitwise_and(bright_mask, low_saturation_mask)
    white_mask = cv2.bitwise_and(white_mask, visible_mask)

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    seed_body_mask, seed_proximity_mask = build_seed_masks(hue, saturation, value)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    max_contour_area = image_area * max_contour_area_ratio
    sprout_contours = [
        contour
        for contour in contours
        if is_germination_contour(
            contour,
            image.shape,
            min_contour_area,
            max_contour_area,
            seed_body_mask,
            seed_proximity_mask,
        )
    ]

    overlay = image.copy()
    filtered_mask = np.zeros_like(white_mask)
    midpoint = image.shape[1] // 2
    treated_sprouts = 0
    untreated_sprouts = 0

    for contour in sprout_contours:
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
        x, y, width, height = cv2.boundingRect(contour)
        center_x = x + (width / 2)
        if center_x < midpoint:
            treated_sprouts += 1
        else:
            untreated_sprouts += 1
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (45, 106, 79), 2)

    total_white_area = float(cv2.countNonZero(filtered_mask))
    treated_white_area = float(cv2.countNonZero(filtered_mask[:, :midpoint]))
    untreated_white_area = float(cv2.countNonZero(filtered_mask[:, midpoint:]))

    cv2.line(overlay, (midpoint, 0), (midpoint, image.shape[0]), (45, 106, 79), 3)
    draw_label(overlay, "Treated chickpea", (18, 34))
    draw_label(overlay, "Control chickpea", (midpoint + 18, 34))

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)

    return VisionResult(
        sprout_count=len(sprout_contours),
        total_white_area=round(total_white_area, 2),
        overlay_path=str(overlay_path),
        treated_sprout_count=treated_sprouts,
        treated_white_area=round(treated_white_area, 2),
        untreated_sprout_count=untreated_sprouts,
        untreated_white_area=round(untreated_white_area, 2),
    )


def build_seed_masks(hue, saturation, value):
    raw_seed_mask = (
        (hue >= 4)
        & (hue <= 34)
        & (saturation >= 55)
        & (value >= 70)
    ).astype(np.uint8) * 255
    seed_kernel = np.ones((5, 5), np.uint8)
    raw_seed_mask = cv2.morphologyEx(raw_seed_mask, cv2.MORPH_OPEN, seed_kernel, iterations=1)
    raw_seed_mask = cv2.morphologyEx(raw_seed_mask, cv2.MORPH_CLOSE, seed_kernel, iterations=2)

    seed_body_mask = np.zeros_like(raw_seed_mask)
    contours, _ = cv2.findContours(raw_seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = raw_seed_mask.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        extent = area / max(width * height, 1)
        if area < 120 or area > image_height * image_width * 0.025:
            continue
        if y < image_height * 0.18 or y + height > image_height * 0.95:
            continue
        if x < image_width * 0.05 or x + width > image_width * 0.95:
            continue
        if aspect_ratio > 2.4 or extent < 0.28:
            continue
        cv2.drawContours(seed_body_mask, [contour], -1, 255, thickness=cv2.FILLED)

    proximity_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
    seed_proximity_mask = cv2.dilate(seed_body_mask, proximity_kernel, iterations=1)
    return seed_body_mask, seed_proximity_mask


def is_germination_contour(contour, image_shape, min_area, max_area, seed_body_mask, seed_proximity_mask):
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False

    x, y, width, height = cv2.boundingRect(contour)
    image_height, image_width = image_shape[:2]
    if y < image_height * 0.12 or y + height > image_height * 0.97:
        return False
    if x < image_width * 0.03 or x + width > image_width * 0.97:
        return False

    has_seeds = cv2.countNonZero(seed_body_mask) > 0
    if has_seeds:
        if not contour_touches_seed_proximity(x, y, width, height, seed_proximity_mask):
            return False
        if contour_overlaps_seed_body(contour, x, y, width, height, seed_body_mask):
            return False

    longest_side = max(width, height)
    shortest_side = max(min(width, height), 1)
    aspect_ratio = longest_side / shortest_side
    extent = area / max(width * height, 1)

    if longest_side < 6:
        return False
    if aspect_ratio < 1.1:
        return False
    if extent > 0.85:
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False
    thinness = (perimeter * perimeter) / max(area, 1)
    return thinness >= 8


def contour_touches_seed_proximity(x, y, width, height, seed_proximity_mask):
    padding = 8
    y1 = max(y - padding, 0)
    y2 = min(y + height + padding, seed_proximity_mask.shape[0])
    x1 = max(x - padding, 0)
    x2 = min(x + width + padding, seed_proximity_mask.shape[1])
    return cv2.countNonZero(seed_proximity_mask[y1:y2, x1:x2]) > 0


def contour_overlaps_seed_body(contour, x, y, width, height, seed_body_mask):
    contour_mask = np.zeros((height, width), dtype=np.uint8)
    shifted_contour = contour - np.array([[[x, y]]])
    cv2.drawContours(contour_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
    seed_crop = seed_body_mask[y : y + height, x : x + width]
    overlap = cv2.countNonZero(cv2.bitwise_and(contour_mask, seed_crop))
    contour_pixels = cv2.countNonZero(contour_mask)
    if contour_pixels == 0:
        return False
    return (overlap / contour_pixels) > 0.22


def draw_label(image, text, origin):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        image,
        (x - 8, y - height - 10),
        (x + width + 8, y + baseline + 8),
        (255, 255, 255),
        thickness=cv2.FILLED,
    )
    cv2.putText(image, text, (x, y), font, font_scale, (45, 106, 79), thickness, cv2.LINE_AA)
