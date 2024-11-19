# Skin Tone Classifier

## Описание

Этот проект разработан для анализа оттенков кожи на изображениях и определения их типа по [шкале Фицпатрика](https://ru.wikipedia.org/wiki/Шкала_Фитцпатрика) и [шкале Лушана](https://ru.wikipedia.org/wiki/Цветная_шкала_Лушана). Программа использует математические методы для классификации оттенков кожи, а также предоставляет подробные отчеты с результатами анализа. 

Проект можно использовать как для анализа изображений индивидуально, так и для пакетной обработки большого количества файлов.

## Возможности

1. **Определение типа Фицпатрика**: Классификация оттенков кожи по 6 фототипам.
2. **Сопоставление с шкалой фон Лушана**: Определение наиболее подходящего оттенка из 36 предустановленных.
3. **Анализ изображений**: Поддержка пакетной обработки изображений для массового анализа.
4. **Вывод результатов**: Сохранение результатов в формате CSV для последующего анализа.

## Установка и запуск

### Предустановки

Перед запуском проекта убедитесь, что у вас установлены следующие инструменты:

- **Docker** (опционально для упрощённого запуска)
- **Python 3.11+**
- **Необходимые библиотеки** указаны в файле `requirements.txt`.

### Установка через Docker

Если вы хотите использовать Docker, выполните следующие шаги:

1. Соберите Docker-образ:
   ```bash
   docker build -t skin-tone-classifier .
   ```

2. Запустите контейнер:
   ```bash
   docker run -it --rm skin-tone-classifier
   ```

3. Контейнер автоматически выполнит все шаги, включая загрузку изображений, анализ и сохранение результатов.

**Примечание:** Установка Docker-образа может занять до 500 секунд из-за загрузки и установки всех необходимых зависимостей.

### Установка через Conda (для локальной версии)

1. Создайте окружение с помощью файла `environment.yaml`:
   ```bash
   conda env create -f environment.yaml
   ```

2. Активируйте окружение:
   ```bash
   conda activate skin-tone-classifier
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. После этого запустите проект:
   ```bash
   python fitzpatrick.py
   ```

## Основные файлы проекта

### `fitzpatrick.py`

Этот файл содержит основной код для анализа изображений и классификации оттенков кожи:

- Выполняет конвертацию цветов.
- Анализирует изображения и классифицирует оттенки кожи по шкале Фицпатрика и фон Лушана.
- Сохраняет результаты в формате CSV.

### `Dockerfile`

Этот файл содержит инструкции для создания Docker-образа:

- Установка всех необходимых библиотек.
- Загрузка изображений.
- Поддержка работы через сетевые папки.

### `environment.yaml`

Этот файл содержит конфигурацию для создания виртуальной среды через Conda. Он гарантирует, что все необходимые зависимости будут установлены и настроены корректно.

## Рабочий процесс

1. **Обработка изображений**: 
   Для анализа изображений их необходимо поместить в папку `extracted_files/images_for_fitzpatrick`(по умолчанию в нее загружаются файлы с [Google-диска](https://drive.google.com/drive/folders/1K0JRk908tDp7xwV_5sjq7FWdNBvZdQKl?usp=sharing).

2. **Результаты работы**: 
   После выполнения алгоритма, результаты сохраняются в папке `extracted_files/results`. Это включает два файла:
   
   - **extended_results.csv** — подробные результаты по каждому изображению:
     - `photo`: путь к изображению.
     - `skin_tone_hex`: цвет кожи в формате HEX.
     - `von_lus_index`: индекс по шкале фон Лушана.
     - `fitzpatrick_index`: индекс по шкале Фицпатрика.

   - **results.csv** — краткий отчет:
     - `file`: путь к файлу.
     - `image type`: тип изображения.
     - `dominant 1, dominant 2`: доминирующие оттенки.
     - `accuracy`: точность классификации (от 0 до 100).

3. **Добавление собственных изображений**:
   Если вы хотите добавить кастомные изображения для анализа, просто поместите их в папку `extracted_files/images_for_fitzpatrick`. Алгоритм автоматически обработает их при следующем запуске.

## Дополнительная информация

- Программа способна определять фототипы кожи на основе шкалы Фицпатрика и шкалы фон Лушана, хотя на практике фототипы I, V и VI были редкими.
- Для работы используется тулкит [SkinToneClassifier](https://github.com/ChenglongMa/SkinToneClassifier.git), который применяет математические модели для поиска наиболее близких оттенков кожи.

### Примечания

1. **Обработка архивов**: Если архив с изображениями (`images_for_fitzpatrick.rar`) недоступен, вы можете использовать скрипт `fitzpatrick_without_url.py`, который будет работать с уже загруженными изображениями.

2. **Сетевые папки**: Docker-образ автоматически подключается к сетевой папке для скачивания архива с изображениями. Если такого подключения нет, просто используйте скрипт `fitzpatrick_without_url.py` для работы с локальными файлами.

## Пример использования

1. Запуск Docker-контейнера:
   ```bash
   docker run -it --rm skin-tone-classifier
   ```

2. Результаты будут сохранены в файл:
   ```bash
   /app/SkinToneClassifier/extracted_files/results/extended_results.csv
   ```

---

**Примечание**: Это проект, который позволяет классифицировать оттенки кожи с использованием продвинутых алгоритмов. Для максимальной точности убедитесь, что изображения соответствуют стандартам качества (разрешение, освещение и т.д.).
