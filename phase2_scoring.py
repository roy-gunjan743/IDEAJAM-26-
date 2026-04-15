# phase2_scoring.py

import math

def calculate_best_regions(img, detections):
    h, w, _ = img.shape

    scored_boxes = []

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class"]

        area = (x2 - x1) * (y2 - y1)
        area_weight = area / (w * h)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        center_dist = ((cx - w//2)**2 + (cy - h//2)**2)**0.5
        max_dist = ((w//2)**2 + (h//2)**2)**0.5

        center_weight = 1 - (center_dist / max_dist)

        if cls_id == 0:
            type_weight = 1.0
        else:
            type_weight = 0.5

        score = (type_weight * 0.5) + (area_weight * 0.3) + (center_weight * 0.2)

        scored_boxes.append((score, (x1, y1, x2, y2)))

    # Sort by score
    scored_boxes.sort(reverse=True, key=lambda x: x[0])

    # Return top 2 regions
    top_boxes = [box for _, box in scored_boxes[:2]]

    if len(top_boxes) == 0:
        h, w, _ = img.shape
        # fallback: center region
        fallback_box = (
            int(w * 0.4),
            int(h * 0.4),
            int(w * 0.6),
            int(h * 0.6)
        )
        return [fallback_box]

    return top_boxes