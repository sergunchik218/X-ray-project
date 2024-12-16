import os
import logging
import telebot
from telebot import types
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Настройки логирования
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
user_selection = {}

# Папка для сохранения файлов
BASE_DIR = "documents"
os.makedirs(BASE_DIR, exist_ok=True)

# Инициализация бота
TOKEN = os.getenv('TELEGRAM_TOKEN')  # Получаем токен из .env
bot = telebot.TeleBot(TOKEN)

# Загружаем модель YOLO
MODEL_PATH = os.getenv('MODEL_PATH', 'pneumonia_yolo11x.pt')  # Путь к модели YOLO
model = YOLO(MODEL_PATH)


@bot.message_handler(commands=['start'])
def start(message):
    # Клавиатура для выбора
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    button1 = types.InlineKeyboardButton("Снимок легких", callback_data="pneumonia")
    button2 = types.InlineKeyboardButton("Перелом", callback_data="fracture")
    keyboard.add(button1, button2)

    bot.reply_to(message, "Привет! Выберите, что вы хотите проанализировать:", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: True)
def button(call):
    user_selection[call.from_user.id] = call.data
    bot.edit_message_text(
        text=f"Вы выбрали: {'Снимок легких' if call.data == 'pneumonia' else 'Перелом'}. Загрузите изображение для анализа.",
        chat_id=call.message.chat.id,
        message_id=call.message.message_id
    )


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    user_id = message.from_user.id
    if user_id not in user_selection:
        bot.reply_to(message, "Сначала выберите категорию анализа с помощью меню.")
        return

    try:
        # Получаем информацию о файле фотографии
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        local_path = os.path.join(BASE_DIR, f"{user_id}_original.jpg")

        # Сохраняем файл локально
        with open(local_path, "wb") as new_file:
            new_file.write(downloaded_file)
        logger.info(f"Изображение сохранено по пути: {local_path}")

        # Загрузка и обработка изображения с помощью модели YOLO
        image = Image.open(local_path)
        results = model.predict(image)  # Выполняем предсказание

        # Проверка наличия предсказаний
        if results and hasattr(results[0], 'probs') and results[0].probs is not None:
            probs = results[0].probs
            class_names = list(model.names.values())

            # Получаем индекс и уверенность для наиболее вероятного класса
            top_class_idx = probs.top1  # Используем top1 для индекса класса с наибольшей вероятностью
            top_class_name = class_names[top_class_idx]
            confidence = probs.top1conf * 100  # Используем top1conf для уверенности

            # Начинаем формировать строку ответа
            response_message = ""

            if confidence >= 95:
                response_message = f"Предсказанный класс: {top_class_name} ({confidence:.2f}%)"
            else:
                response_message = "Результаты анализа:\n"
                for idx, prob in enumerate(probs):
                    class_name = class_names[idx]
                    prob_percentage = prob * 100
                    response_message += f"{class_name}: {prob_percentage:.2f}%\n"

            # Создаем аннотированное изображение
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("arial.ttf", size=30) if os.name == 'nt' else ImageFont.load_default()
            text = f"{top_class_name} ({confidence:.2f}%)"

            # Вместо textsize() используем textbbox()
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            draw.rectangle(((0, 0), (text_width + 10, text_height + 10)), fill="black")
            draw.text((5, 5), text, fill="white", font=font)

            # Сохраняем аннотированное изображение
            annotated_image_path = os.path.join(BASE_DIR, f"{user_id}_annotated.jpg")
            image.save(annotated_image_path)

            # Отправляем аннотированное изображение
            with open(annotated_image_path, "rb") as f:
                bot.send_photo(message.chat.id, f, caption="Анализ завершен. Смотрите результат.")

            # Отправляем сообщение с результатами
            bot.reply_to(message, response_message)
        else:
            bot.reply_to(message, "Модель не смогла найти объектов на изображении.")

        # Добавление кнопки "/start"
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button_start = types.KeyboardButton("/start")
        keyboard.add(button_start)
        bot.send_message(message.chat.id, "Нажмите кнопку START ниже, чтобы начать заново.", reply_markup=keyboard)

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        bot.reply_to(message, "Произошла ошибка при обработке изображения. Попробуйте снова.")


@bot.message_handler(content_types=["document"])
def handle_document(message):
    try:
        # Получаем информацию о файле документа
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        local_path = os.path.join(BASE_DIR, message.document.file_name)

        # Сохраняем файл локально
        with open(local_path, "wb") as new_file:
            new_file.write(downloaded_file)
        logger.info(f"Документ сохранен по пути: {local_path}")
        bot.reply_to(message, "Документ успешно сохранен.")
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        bot.reply_to(message, "Произошла ошибка при обработке документа. Попробуйте снова.")


def main():
    """Запуск Telegram-бота."""
    bot.polling(none_stop=True)


if __name__ == "__main__":
    main()
