from PIL import Image, ImageDraw
from ultralytics import YOLO
import os
import logging

# Настройки логирования
logger = logging.getLogger(__name__)

# Папка для временных файлов
BASE_DIR = "documents"
os.makedirs(BASE_DIR, exist_ok=True)

# Загрузка модели переломов
MODEL_PATH = "perelom_yolo11x.pt"
model = YOLO(MODEL_PATH)

def delete_file(file_path):
    """Удаляет указанный файл, если он существует."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Файл {file_path} успешно удалён.")
    except Exception as e:
        logger.error(f"Ошибка при удалении файла {file_path}: {e}")

def process_fracture_image(image_path, user_id):
    """Обработка изображения для поиска переломов."""
    annotated_image_path = f"{BASE_DIR}/{user_id}_annotated_fracture.jpg"
    try:
        # Загрузка изображения
        image = Image.open(image_path)

        # Прогоняем изображение через модель
        results = model(image)

        # Проверяем, есть ли предсказания
        if results and results[0].boxes:
            # Получаем боксы и классы
            boxes = results[0].boxes.xyxy.numpy()  # Координаты боксов (x1, y1, x2, y2)
            confidences = results[0].boxes.conf.numpy()  # Уверенность
            labels = results[0].boxes.cls.numpy()  # Классы

            # Аннотируем изображение
            draw = ImageDraw.Draw(image)

            result_message = "Обнаружены переломы или трещины:\n"

            for i in range(len(boxes)):
                box = boxes[i]
                confidence = confidences[i]
                label = labels[i]
                class_name = model.names[label]  # Получаем имя класса по индексу

                # Рисуем прямоугольник для хитбокса
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
                draw.text((box[0], box[1]), f"{class_name} {confidence:.2f}", fill="red")

                # Добавляем в результат строку о найденном переломе
                result_message += f"{class_name}: {confidence * 100:.2f}%\n"

            # Сохраняем аннотированное изображение
            image.save(annotated_image_path)

            return result_message, annotated_image_path
        else:
            # Если модель ничего не нашла
            return (
                "На изображении не найдено переломов.\nПроверьте, возможно, имеются смещения.",
                None
            )
    except Exception as e:
        logger.error(f"Ошибка при обработке снимка перелома: {e}")
        return "Произошла ошибка при обработке снимка перелома.", None
    finally:
        # Удаляем оригинал после обработки
        delete_file(image_path)

def clean_up_files(*file_paths):
    """Удаляет указанные файлы."""
    for file_path in file_paths:
        delete_file(file_path)
