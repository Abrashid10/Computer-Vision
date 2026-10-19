import torch
import cv2
import numpy as np


class YOLOv3Detector:
    """
    YOLOv3 detector using PyTorch Hub.
    Returns detections as [x1, y1, x2, y2, confidence]
    """
    def __init__(self, conf_threshold=0.5, classes=None):
        """
        conf_threshold: minimum confidence to keep detection
        classes:        list of class names to track (None = all)
                        e.g. ['person', 'car', 'truck']
        """
        print("Loading YOLOv3...")
        self.model = torch.hub.load(
            'ultralytics/yolov3',
            'yolov3',
            pretrained=True
        )
        self.model.eval()
        self.conf_threshold = conf_threshold
        self.classes = classes
        print("YOLOv3 loaded!")

    def detect(self, frame):
        """
        Run detection on one frame.

        frame: BGR image (from cv2.imread or VideoCapture)
        Returns: numpy array (N, 5) [x1,y1,x2,y2,confidence]
        """
        # YOLOv3 expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(rgb_frame)

        # Parse results
        detections = []
        for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
            # Filter by confidence
            if conf < self.conf_threshold:
                continue

            # Filter by class if specified
            if self.classes is not None:
                class_name = results.names[int(cls_id)]
                if class_name not in self.classes:
                    continue

            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, conf])

        if len(detections) == 0:
            return np.empty((0, 5))

        return np.array(detections, dtype=np.float32)
