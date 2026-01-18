#Модуль для детекции людей с использованием YOLOv5

import torch
import numpy as np
import cv2

class PeopleDetector:

    def __init__(self, model_name='yolov5s', device=None):

        self.model_name = model_name
        
        # Определение устройства
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Используемое устройство: {self.device}")
        
        # Загрузка модели
        self.model = self._load_model()
              
    def _load_model(self):
    
        try:
            model = torch.hub.load('ultralytics/yolov5', 
                                  self.model_name, 
                                  pretrained=True,
                                  verbose=False)
            model.to(self.device)
            model.eval()
            print(f"Модель {self.model_name} успешно загружена")
            return model
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")
    
    
    def detect(self, image, confidence_threshold=0.5):
    
        # Конвертация BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Детекция
        with torch.no_grad():
            results = self.model(image_rgb)
        
        # Извлечение результатов для класса 'person' (класс 0 в COCO)
        detections = results.xyxy[0].cpu().numpy()
        people_detections = []
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf >= confidence_threshold:  # 0 = person
                people_detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
                
        return people_detections
        
    def draw_detections(self, image, detections):
    
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2, confidence = det
            
            # Отрисовка bounding box
            color = (0, 255, 0)  # Зеленый
            thickness = 2
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Отрисовка подписи
            label = f"Person: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 2
            
            # Фон для текста
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, text_thickness
            )
            cv2.rectangle(result, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            
            # Текст
            cv2.putText(result, label, 
                       (x1, y1 - 5),
                       font, font_scale, 
                       (0, 0, 0), text_thickness)
            
        return result