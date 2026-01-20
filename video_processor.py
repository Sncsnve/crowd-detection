"""Модуль обработки видео"""

import cv2
import time
from tqdm import tqdm

class VideoProcessor:
    
    def __init__(self, detector):
        self.detector = detector   
    def process_video(self, input_path, output_path, confidence_threshold=0.5):
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {input_path}")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        total_people = 0
        max_people_in_frame = 0
        start_time = time.time()

        for frame_num in tqdm(range(total_frames), desc="Обработка кадров"):
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.detector.detect(frame, confidence_threshold)
            frame_with_detections = self.detector.draw_detections(frame, detections)
            out.write(frame_with_detections)
            people_count = len(detections)
            total_people += people_count
            max_people_in_frame = max(max_people_in_frame, people_count)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        processing_time = time.time() - start_time
        
        return {
            'total_frames': total_frames,
            'total_people': total_people,
            'avg_people_per_frame': total_people / total_frames if total_frames > 0 else 0,
            'max_people_in_frame': max_people_in_frame,
            'processing_time': processing_time,
            'processing_fps': total_frames / processing_time if processing_time > 0 else 0
        }