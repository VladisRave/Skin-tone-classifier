# Stage 1: Используем Python 3.11 базовый образ
FROM python:3.11

# Устанавливаем необходимые зависимости
RUN apt-get update && apt-get install -y \
    unzip \
    libgtk-3-dev \
    libnotify-dev \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    curl \
    build-essential \
    pkg-config \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем gdown для скачивания файлов с Google Диска
RUN pip install gdown

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Клонируем репозиторий SkinToneClassifier
RUN git clone https://github.com/ChenglongMa/SkinToneClassifier.git && \
    cd SkinToneClassifier && \
    pip install skin-tone-classifier --upgrade

# Копируем Python-скрипт
COPY fitzpatrick.py /app/fitzpatrick.py

# Точка входа для запуска программы
ENTRYPOINT ["python", "fitzpatrick.py"]
