# phase1_detection.py

from ultralytics import YOLO
import cv2

# Load YOLO model (only once)
model = YOLO("yolov8n.pt", task="detect")
def detect_objects_from_img(img):
    results = model(img)

    detections = []

    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(classes[i])

            detections.append({
                "box": (x1, y1, x2, y2),
                "class": cls_id
            })

    return img, detections

def detect_objects(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    detections = []

    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(classes[i])

            detections.append({
                "box": (x1, y1, x2, y2),
                "class": cls_id
            })
    
    

    return img, detections