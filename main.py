# main.py

import cv2
import numpy as np
from phase1_detection import detect_objects
from phase2_scoring import calculate_best_regions
from phase3_encoding import generate_watermark_signal
from phase4_embedding import embed_watermark
from phase5_detection import extract_watermark
from attack_simulator import crop, compress, blur
# -----------------------------
# STEP 1: Load image + detect
# -----------------------------
image_path = "input.jpg"

img, detections = detect_objects(image_path)

# -----------------------------
# STEP 2: Select best region
# -----------------------------
top_boxes = calculate_best_regions(img, detections)

# Draw result
for box in top_boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Show result
cv2.imshow("Best Region Selected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------
# STEP 3: Generate watermark
# -----------------------------
secret_key = "my_secret_key_123"
signal = generate_watermark_signal(secret_key)

print("\nWatermark Signal Ready!")
print("Length:", len(signal))
# -----------------------------
# STEP 4: Embed watermark
# -----------------------------
watermarked_img = img.copy()

for box in top_boxes:
    watermarked_img = embed_watermark(watermarked_img, signal, box)

# Save output
cv2.imwrite("watermarked.jpg", watermarked_img)

# Show result
cv2.imshow("Watermarked Image", watermarked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# -----------------------------
# STEP 5: Extract watermark
# -----------------------------
test_img = cv2.imread("watermarked.jpg")

# Extract from first region (or combine later)
all_extracted = []

for box in top_boxes:
    ext = extract_watermark(test_img, box, len(signal))
    all_extracted.append(ext)

# Average signals
extracted_signal = np.mean(all_extracted, axis=0)

# Compare original vs extracted
def calculate_similarity(original, extracted):
    return np.dot(original, extracted) / (np.linalg.norm(original) * np.linalg.norm(extracted))

similarity = calculate_similarity(signal, extracted_signal)

print("\nDetection Result:")
print("Correlation:", similarity)

if similarity > 0.2:
    print("✅ Watermark DETECTED")
else:
    print("❌ Not detected")

test_img = crop(watermarked_img)
# or compress / blur