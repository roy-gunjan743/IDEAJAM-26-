# phase4_embedding.py

import cv2
import numpy as np
import random

def embed_watermark(image, signal, box):
    x1, y1, x2, y2 = box

    # Convert to grayscale (for processing only)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract region
    region = gray[y1:y2, x1:x2]
    region = np.float32(region)

    # Apply DCT
    dct = cv2.dct(region)

    # -----------------------------
    # EMBEDDING (SAFE + INVISIBLE)
    # -----------------------------
    idx = 0
    rows, cols = dct.shape

    random.seed(42)

    # Smaller mid-frequency area (balanced)
    positions = [(i, j) for i in range(6, 12) for j in range(6, 12)]
    random.shuffle(positions)

    for (i, j) in positions:
        if idx >= len(signal) // 4:   # limit embedding → avoid distortion
            break

        # Very low strength (invisible)
        strength = 1.0 + abs(dct[i][j]) * 0.003
        dct[i][j] += signal[idx] * strength

        idx += 1

    # -----------------------------
    # RECONSTRUCT
    # -----------------------------
    idct = cv2.idct(dct)
    idct = np.clip(idct, 0, 255)

    # -----------------------------
    # APPLY BACK TO COLOR IMAGE
    # -----------------------------
    watermarked = image.copy()

    alpha = 0.97  # strong blending → preserves original look

    for c in range(3):  # apply to all RGB channels
        watermarked[y1:y2, x1:x2, c] = (
            alpha * image[y1:y2, x1:x2, c] +
            (1 - alpha) * idct
        )

    return watermarked.astype(np.uint8)