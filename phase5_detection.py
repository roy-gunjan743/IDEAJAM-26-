# phase5_detection.py

import cv2
import numpy as np

def extract_watermark(image, box, length):
    x1, y1, x2, y2 = box

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract region
    region = gray[y1:y2, x1:x2]
    region = np.float32(region)

    # Apply DCT
    dct = cv2.dct(region)

    extracted = []
    idx = 0

    rows, cols = dct.shape

    for i in range(10, rows):
        for j in range(10, cols):
            if idx >= length:
                break

            value = dct[i][j]

            # Detect sign
            if value > 0:
                extracted.append(1)
            else:
                extracted.append(-1)

            idx += 1

        if idx >= length:
            break

    return np.array(extracted)