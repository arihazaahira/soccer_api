import cv2
import numpy as np
import os
from ultralytics import YOLO
from django.conf import settings
from typing import Optional

class ExtractKeypointsService:
    def __init__(self):
        # Utilise YOLOv8 Nano pour une vitesse maximale
        self.model = YOLO('yolov8n-pose.pt') 

    def extract(self, video_path: str) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        keypoints_sequence = []
        frame_idx = 0
        frame_step = 5  # ANALYSE 1 IMAGE SUR 5 (Accélère par 5x)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sauter les images pour éviter le timeout
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue
            
            # Redimensionnement léger pour accélérer l'inférence
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * (640 / w))))

            # Inférence YOLO (uniquement une personne)
            results = self.model.predict(frame, conf=0.5, verbose=False)
            
            found_person = False
            for r in results:
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    # Extraction des 17 points clés COCO
                    kp = r.keypoints.data[0].cpu().numpy() # (17, 3)
                    keypoints_sequence.append(kp)
                    found_person = True
                    break 
            
            if not found_person:
                keypoints_sequence.append(np.zeros((17, 3), dtype=np.float32))
            
            frame_idx += 1
        
        cap.release()
        
        if not keypoints_sequence:
            return None
            
        return np.array(keypoints_sequence, dtype=np.float32)