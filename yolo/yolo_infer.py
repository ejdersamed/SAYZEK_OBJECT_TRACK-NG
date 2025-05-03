"""
from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            if conf >= self.conf_threshold:
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
        return detections
"""
from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.allowed_classes = [0, 2]  # sadece person (0) ve car (2)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)
            if conf >= self.conf_threshold and cls in self.allowed_classes:
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])

        return detections
