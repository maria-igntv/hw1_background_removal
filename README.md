# Background Removal in Real Time

## Выбранное решение

[MediaPipe Image Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter) + модель **Selfie Multiclass** (`selfie_multiclass_256x256.tflite`) — классы: фон, волосы, кожа тела, кожа лица, одежда, прочее; передний план = `1 − P(фон)`.

## Описание

CNN 256×256, float32. Маска после ресайза под кадр, композит на зелёный фон. Тяжелее, чем лёгкий `selfie_segmenter.tflite`.

При первом запуске модель качается через `curl` в файл `selfie_multiclass_256x256.tflite` рядом с `main.py`. Если не вышло — скачайте вручную (см. сообщение скрипта).

## Запуск

```bash
pip install -r requirements.txt
python main.py путь/к/видео.mp4
```

Веб-камера (индекс `0`, окно превью, выход **Esc**):

```bash
python main.py --camera 0
```

Запись с камеры в файл:

```bash
python main.py --camera 0 -o out.mp4
```

Разрешение камеры (по умолчанию 640×480):

```bash
python main.py -c 0 --width 1280 --height 720
```

Результат по умолчанию: рядом файл `имя_nobg.mp4`. Свой путь вывода:

```bash
python main.py путь/к/видео.mp4 -o result.mp4
```

В консоли — средний FPS по сегментации и композитингу на кадр. Звук в выход не переносится (только картинка).

Python 3.10–3.12, если `mediapipe` не ставится на 3.13.

## Результаты

После прогона заполните: разрешение исходника, FPS из последней строки консоли, модель CPU, кратко про качество маски.

## Демо

Ссылка на видео: _вставьте_
