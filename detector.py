"""
Модуль для детекции людей с использованием YOLOv11 от Ultralytics.
"""

import cv2
import torch
from ultralytics import YOLO


class PeopleDetector:
    """
    Класс для детекции людей на изображениях с использованием YOLOv11.
    """
    
    def __init__(self, model_name='yolo11s', device=None):
        """
        Инициализация детектора.
        
        Args:
            model_name (str): Название модели YOLOv11
            device (str): Устройство для вычислений ('cpu', 'cuda', или None для авто)
        """
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
        """Загрузка предобученной модели YOLOv11."""
        try:
            # YOLOv11 доступен в ultralytics с версии 8.0.196
            model = YOLO(f'{self.model_name}.pt')
            
            # Перемещение модели на выбранное устройство
            if self.device == 'cuda':
                model = model.cuda()
            else:
                model = model.cpu()
                
            print(f"Модель {self.model_name} успешно загружена")
            return model
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")
    
    
    def detect(self, image, confidence_threshold=0.5):
        """
        Детектирование людей на изображении.
        
        Args:
            image (numpy.ndarray): Входное изображение в формате BGR
            confidence_threshold (float): Порог уверенности для детекции
            
        Returns:
            list: Список детекций, каждая в формате [x1, y1, x2, y2, confidence]
        """
        # YOLOv11 автоматически конвертирует BGR to RGB
        with torch.no_grad():
            results = self.model(image, conf=confidence_threshold, verbose=False)[0]
        
        # Извлечение результатов для класса 'person' (класс 0 в COCO)
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
        """
        Отрисовка bounding boxes на изображении.
        
        Args:
            image (numpy.ndarray): Исходное изображение
            detections (list): Список детекций
            
        Returns:
            numpy.ndarray: Изображение с отрисованными детекциями
        """
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