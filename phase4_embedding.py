# phase4_embedding.py

import cv2
import numpy as np

def embed_watermark(image, signal, box):
    x1, y1, x2, y2 = box

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract important region
    region = gray[y1:y2, x1:x2]

    # Convert to float32
    region = np.float32(region)

    # Apply DCT
    dct = cv2.dct(region)

    # Embed watermark signal
    idx = 0
    rows, cols = dct.shape

    for i in range(10, rows):
        for j in range(10, cols):
            if idx >= len(signal):
                break

            # Modify frequency slightly
            strength = 2 + abs(dct[i][j]) * 0.01
            dct[i][j] += signal[idx] * strength
            idx += 1

        if idx >= len(signal):
            break

    # Convert back using IDCT
    idct = cv2.idct(dct)

    # Replace region
    watermarked = gray.copy()
    watermarked[y1:y2, x1:x2] = idct

    return watermarked