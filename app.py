import os
import logging
import telebot
from telebot import types
from dotenv import load_dotenv
from pneumonia_model import process_pneumonia_image
from fracture_model import process_fracture_image

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

        # Обработка изображения
        if user_selection[user_id] == "pneumonia":
            # Обработка снимка легких
            result_message, annotated_image_path = process_pneumonia_image(local_path, user_id)
        else:
            # Обработка перелома
            result_message, annotated_image_path = process_fracture_image(local_path, user_id)

        # Отправляем результат сообщения
        bot.reply_to(message, result_message)

        # Если есть аннотированное изображение, отправляем его
        if annotated_image_path:
            with open(annotated_image_path, "rb") as f:
                bot.send_photo(message.chat.id, f, caption="Анализ завершен. Смотрите результат.")
            # Удаляем аннотированный файл
            if os.path.exists(annotated_image_path):
                os.remove(annotated_image_path)

        # Удаляем оригинальный файл
        if os.path.exists(local_path):
            os.remove(local_path)

        # Добавляем кнопку "/start"
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button_start = types.KeyboardButton("/start")
        keyboard.add(button_start)
        bot.send_message(message.chat.id, "Нажмите кнопку ниже, чтобы начать заново.", reply_markup=keyboard)

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        bot.reply_to(message, "Произошла ошибка при обработке изображения. Попробуйте снова.")
        # Удаляем оригинальный файл в случае ошибки
        if os.path.exists(local_path):
            os.remove(local_path)


def main():
    """Запуск Telegram-бота."""
    bot.polling(none_stop=True)


if __name__ == "__main__":
    main()
