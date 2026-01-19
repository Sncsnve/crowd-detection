"""
Точка входа для детекции людей на видео.
Программа загружает видео, детектирует людей, отрисовывает bounding boxes
и сохраняет результат.
"""

import argparse
import sys
from pathlib import Path
from video_processor import VideoProcessor
from detector import PeopleDetector


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Детекция людей на видео')
    parser.add_argument('--input', type=str, default='crowd.mp4',
                       help='Путь к входному видеофайлу')
    parser.add_argument('--output', type=str, default='output/crowd_output.mp4',
                       help='Путь для сохранения результата')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Порог уверенности для детекции (0-1)')
    parser.add_argument('--model', type=str, default='yolo11s',
                       choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
                       help='Модель YOLOv11 для использования')
    return parser.parse_args()


def main():
    """Основная функция программы."""
    args = parse_args()
    
    # Проверка существования входного файла
    if not Path(args.input).exists():
        print(f"Ошибка: файл {args.input} не найден.")
        print("Убедитесь, что crowd.mp4 находится в текущей директории.")
        sys.exit(1)
    
    # Создание папки для результата
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ДЕТЕКЦИЯ ЛЮДЕЙ НА ВИДЕО")
    print("=" * 60)
    print(f"Входной файл: {args.input}")
    print(f"Выходной файл: {args.output}")
    print(f"Модель: {args.model}")
    print(f"Порог уверенности: {args.confidence}")
    print("-" * 60)
    
    try:
        # Инициализация детектора
        print("Загрузка модели YOLOv11...")
        detector = PeopleDetector(model_name=args.model)
        
        # Обработка видео
        processor = VideoProcessor(detector)
        
        print("Обработка видео начата...")
        stats = processor.process_video(
            input_path=args.input,
            output_path=args.output,
            confidence_threshold=args.confidence
        )
        
        # Вывод статистики
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ОБРАБОТКИ")
        print("=" * 60)
        print(f"Обработано кадров: {stats['total_frames']}")
        print(f"Всего обнаружено людей: {stats['total_people']}")
        print(f"Среднее людей на кадр: {stats['avg_people_per_frame']:.2f}")
        print(f"Максимум людей в кадре: {stats['max_people_in_frame']}")
        print(f"FPS обработки: {stats['processing_fps']:.2f}")
        print(f"Время обработки: {stats['processing_time']:.2f} сек")
        print("\nРезультат сохранён в:", args.output)
        
    except Exception as e:
        print(f"\nОшибка при выполнении: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()