from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import logging

# Настройки логирования
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели
MODEL_PATH = "pneumonia_yolo11x.pt"
model = YOLO(MODEL_PATH)


def process_pneumonia_image(image_path, user_id):
    try:
        # Загрузка изображения
        image = Image.open(image_path)

        # Прогоняем изображение через модель
        results = model.predict(image)

        # Проверка наличия предсказаний
        if results and hasattr(results[0], 'probs') and results[0].probs is not None:
            probs = results[0].probs
            class_names = list(model.names.values())
            top_class_idx = probs.top1  # Индекс класса с максимальной уверенностью
            top_class_name = class_names[top_class_idx]  # Имя класса
            confidence = probs.top1conf * 100  # Уверенность в процентах

            # Создаём аннотированное изображение
            annotated_image_path = f"documents/{user_id}_annotated_pneumonia.jpg"
            draw = ImageDraw.Draw(image)

            # Настраиваем шрифт
            font_path = "arial.ttf"  # Убедитесь, что этот шрифт доступен на вашей системе
            try:
                font = ImageFont.truetype(font_path, size=40)  # Увеличиваем шрифт
            except IOError:
                font = ImageFont.load_default()  # Используем шрифт по умолчанию, если заданный отсутствует

            # Логика для вывода результата в зависимости от уверенности
            if confidence < 95:
                # Выводим оба класса, если уверенность ниже 95%
                result_message = f"Результаты:\n{class_names[0]} {probs.top1conf:.2f}, {class_names[1]} {probs.top5conf[1]:.2f}"
                text = f"{class_names[0]}: {probs.top1conf * 100:.2f}%\n{class_names[1]}: {probs.top5conf[1] * 100:.2f}%"
            else:
                # Если уверенность выше 95%, выводим только один класс
                result_message = f"{top_class_name} ({confidence:.2f}%)"
                text = f"{top_class_name}: {confidence:.2f}%"

            # Добавляем текст на изображение (цвет текста белый)
            draw.text((10, 10), text, fill="white", font=font)
            image.save(annotated_image_path)

            return result_message, annotated_image_path
        else:
            # Если предсказаний нет
            return "Модель не смогла найти объектов на изображении.", ""
    except Exception as e:
        logger.error(f"Ошибка при обработке снимка легких: {e}")
        return "Произошла ошибка при обработке снимка легких.", ""
