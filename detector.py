"""Модуль для детекции"""

import cv2
import torch
from ultralytics import YOLO

class PeopleDetector:
    """Класс детекции"""    
    def __init__(self, model_name='yolo11s', device=None):
        """Инициализация детектора"""
        self.model_name = model_name        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device            
        print(f"Устройство: {self.device}")
        self.model = self._load_model()        
    
    def _load_model(self):
        """Загрузка модели"""
        try:
            model = YOLO(f'{self.model_name}.pt')
            if self.device == 'cuda':
                model = model.cuda()
            else:
                model = model.cpu()
                
            print(f"Модель {self.model_name} загружена")
            return model
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки: {e}")
        
    def detect(self, image, confidence_threshold=0.5):
        """Детектирование людей"""
        with torch.no_grad():
            results = self.model(image, conf=confidence_threshold, verbose=False)[0]
        
        people_detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confidences, classes):
                if int(cls) == 0:  # 0 = person
                    x1, y1, x2, y2 = map(int, box)
                    people_detections.append([x1, y1, x2, y2, float(conf)])
                
        return people_detections    
    
    def draw_detections(self, image, detections):
        """Отрисовка bounding boxes"""
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2, confidence = det
            color = (0, 255, 0)  
            thickness = 2
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            label = f"Person: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, text_thickness
            )
            cv2.rectangle(result, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            cv2.putText(result, label, 
                       (x1, y1 - 5),
                       font, font_scale, 
                       (0, 0, 0), text_thickness)
            
        return result
