#Модуль обработки видео

import cv2
import time
from tqdm import tqdm


class VideoProcessor:
    
    def __init__(self, detector):

        self.detector = detector
        
    
    def process_video(self, input_path, output_path, confidence_threshold=0.5):

        # Открытие видеофайла
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {input_path}")
        
        # Получение параметров видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Инициализация VideoWriter для сохранения
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Параметры видео:")
        print(f"  Разрешение: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Всего кадров: {total_frames}")
        print("-" * 60)
        
        # Статистика
        total_people = 0
        max_people_in_frame = 0
        start_time = time.time()
        
        # Обработка кадров
        for frame_num in tqdm(range(total_frames), desc="Обработка кадров"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекция людей
            detections = self.detector.detect(frame, confidence_threshold)
            
            # Отрисовка детекций
            frame_with_detections = self.detector.draw_detections(frame, detections)
            
            # Сохранение кадра
            out.write(frame_with_detections)
            
            # Обновление статистики
            people_count = len(detections)
            total_people += people_count
            max_people_in_frame = max(max_people_in_frame, people_count)
        
        # Освобождение ресурсов
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Расчет статистики
        processing_time = time.time() - start_time
        
        return {
            'total_frames': total_frames,
            'total_people': total_people,
            'avg_people_per_frame': total_people / total_frames if total_frames > 0 else 0,
            'max_people_in_frame': max_people_in_frame,
            'processing_time': processing_time,
            'processing_fps': total_frames / processing_time if processing_time > 0 else 0
        }