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
    l_threshold=240,
    min_contour_area=35,
    max_contour_area_ratio=0.08,
    blur_kernel=(5, 5),
):
    """Detect bright chickpea radicals in left/right batches and save an annotated overlay."""
    image_path = Path(image_path)
    overlay_path = Path(overlay_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Unable to decode image. Send a valid JPEG frame.")

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab_image)

    blurred = cv2.GaussianBlur(l_channel, blur_kernel, 0)
    _, white_mask = cv2.threshold(blurred, l_threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    max_contour_area = image_area * max_contour_area_ratio
    sprout_contours = [
        contour
        for contour in contours
        if min_contour_area <= cv2.contourArea(contour) <= max_contour_area
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
