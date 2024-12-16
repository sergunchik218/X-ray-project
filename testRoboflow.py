import os
import logging
import telebot
from telebot import types
from roboflow import Roboflow
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

# Инициализация клиента для инференса через Roboflow
API_KEY = os.getenv('ROBOFLOW_API_KEY')  # Получаем API ключ из .env
rf = Roboflow(api_key=API_KEY)
PROJECT_NAME = os.getenv('PROJECT_NAME')  # Получаем название проекта из .env
PROJECT_VERSION = int(os.getenv('PROJECT_VERSION'))  # Получаем версию модели
project = rf.workspace().project(PROJECT_NAME)
model = project.version(PROJECT_VERSION).model


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

        # Выполняем инференс с использованием модели Roboflow
        prediction = model.predict(local_path)
        predictions = prediction.json()

        logger.info(f"Результаты инференса: {predictions}")

        # Обрабатываем результат: например, выбираем класс с наибольшей уверенностью
        if predictions['predictions']:
            top_prediction = predictions['predictions'][0]  # Получаем первый элемент в списке
            label = top_prediction['predictions'][0]['class']  # Класс с наибольшей уверенностью
            confidence = top_prediction['predictions'][0]['confidence']  # Уверенность

            # Отправляем результат пользователю
            response_message = f"Предсказанный класс: {label}\nУверенность: {confidence:.2f}"
            bot.reply_to(message, response_message)

            # Визуализация (сохранение или отсылка аннотированного изображения)
            prediction_image_path = os.path.join(BASE_DIR, f"{user_id}_prediction.jpg")
            prediction.save(prediction_image_path)

            # Отправляем аннотированное изображение обратно пользователю
            with open(prediction_image_path, "rb") as f:
                bot.send_photo(message.chat.id, f, caption="Анализ завершен. Смотрите результат.")
        else:
            bot.reply_to(message, "Модель не смогла найти ничего на изображении.")

        # Добавляем кнопку "/start"
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button_start = types.KeyboardButton("/start")
        keyboard.add(button_start)
        bot.send_message(message.chat.id, "Нажмите кнопку ниже, чтобы начать заново.", reply_markup=keyboard)

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
