# Background Removal in Real Time

## Выбранное решение

**MediaPipe Image Segmenter** + модель **Selfie Multiclass 256x256**

Официальная страница: [MediaPipe Image Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)

 [Модель](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite) `selfie_multiclass_256x256.tflite`

Это решение выбрано потому, что оно оптимизировано для CPU, имеет малый размер модели (~1.7 MB), обеспечивает высокую скорость (>20 FPS на CPU) и предоставляет multiclass сегментацию для более точных границ
## Описание архитектуры

(CNN) с входным разрешением 256×256, формат float32, TensorFlow Lite с XNNPACK delegate.
Модель возвращает 6 масок. Маска человека вычисляется как `person = 1.0 - P(background)`. После получения маски она масштабируется до размера исходного кадра и применяется альфа-композитинг: `output = frame × mask + background × (1 - mask)`.

При первом запуске модель качается через `curl` в файл `selfie_multiclass_256x256.tflite` рядом с `main.py`. Если не работает, скачайте вручную:

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

## Результаты

Устройство: Apple M4 Pro (CPU)
ОС: macOS
Модель: selfie_multiclass_256x256.tflite (CPU, XNNPACK delegate)

**Тест 1: Видеофайл**
* Разрешение: 480×852
* Кадров: 128
* FPS: 24.4
* Latency: 41.0 ms/кадр

**Тест 2: Веб-камера**
* Разрешение: 640×480
* FPS: 36.2

**Тест 3: Веб-камера**
* Разрешение: 1280×720
* FPS: 30.1

Требования выполнены: минимум 10 FPS (есть 24.4), рекомендуемые 15 FPS (есть 24.4), разрешение ≥320×240 (есть 480×852), работает на CPU.

### Ограничения
Звук не копируется (только видео)

## Демо

[Ссылка на видео](https://drive.google.com/file/d/1Apre0MdoRW7DXzDWEkA4lRPxceVg2BRn/view?usp=sharing) 
