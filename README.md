# Нейросетевое шумоподавление речи в медиафайлах

**Студент:** Волчек Константин Павлович, группа СДП-ИИ-221  
**Дипломный проект**

Система очищает речь от фоновых шумов с помощью нейронной сети U-Net, работающей на STFT-спектрограммах.  
На вход подаётся зашумлённый WAV-файл, на выходе — очищенная запись.

- **Архитектура** — U-Net (энкодер-декодер со skip-connections)
- **Представление сигнала** — STFT (Short-Time Fourier Transform), маска 0..1
- **Метрика** — SI-SNR (Scale-Invariant Signal-to-Noise Ratio, дБ)

## Требования

- Python 3.10+
- pip
- GPU рекомендуется (без GPU работает корректно, но медленнее)

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/<your-username>/speech-denoising-diplom.git
cd speech-denoising-diplom
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Подготовить данные

Скачать датасеты и указать пути в `config.py` (`CLEAN_DIR`, `NOISE_DIR`):

- **LibriSpeech train-clean-100** — чистая речь (~6.3 GB, ~28 000 файлов `.flac`)  
  https://www.openslr.org/12
- **UrbanSound8K** — городские шумы (8 732 файла `.wav`, 10 классов)  
  https://urbansounddataset.weebly.com/urbansound8k.html

Либо запустить автоматическую загрузку:

```bash
python main.py --mode download
```

> При загрузке из LibriSpeech берутся только первые 5 000 файлов — этого достаточно для обучения.

### 4. Получить веса модели

**Вариант А — скачать готовые веса (рекомендуется):**

Скачать `best_model.pt` и положить в папку `checkpoints/`:  
*(ссылка добавляется после публикации)*

**Вариант Б — обучить самостоятельно:**

```bash
python main.py --mode train
```

Обучение занимает ~2–4 часа на GPU (T4), до 50 эпох с early stopping.

### 5. Запустить

**Очистить один WAV-файл:**

```bash
python main.py --mode denoise --input noisy_audio.wav
```

**Очистить с указанием пути к результату:**

```bash
python main.py --mode denoise --input noisy_audio.wav --output clean_audio.wav
```

**Оценить качество модели:**

```bash
python main.py --mode evaluate
```

**Экспортировать модель (PyTorch + ONNX):**

```bash
python main.py --mode export
```

## Структура проекта

```
speech-denoising-diploma/
├── main.py              # Точка входа — CLI и оркестрация
├── model.py             # Архитектура U-Net
├── preprocessing.py     # Датасет, DataLoader, STFT-утилиты
├── train.py             # Функция потерь, SI-SNR, цикл обучения
├── inference.py         # Инференс, оценка качества, экспорт модели
├── visualization.py     # Графики обучения и сравнение спектрограмм
├── config.py            # Все гиперпараметры и пути
├── requirements.txt     # Зависимости
│
├── checkpoints/         # Веса модели (не включены в репозиторий)
│   └── .gitkeep
│
├── data/                # Датасеты (не включены в репозиторий)
│   ├── clean_speech/
│   └── noise/
│
├── results/             # Графики и обработанные файлы
│   └── .gitkeep
│
└── notebooks/
    └── speech_denoising_diploma.ipynb  # Оригинальный Google Colab ноутбук
```

## Архитектура

Базовая модель: **U-Net** (энкодер-декодерная свёрточная сеть со skip-connections).

**Принцип работы:**
1. Зашумлённый WAV → STFT → лог-спектрограмма `[1, 257, 251]`
2. U-Net предсказывает **маску** (значения 0..1 для каждого частотно-временного бина)
3. `Очищенный спектр = Зашумлённый × Маска`
4. ISTFT → очищенный WAV

**Параметры STFT:** `n_fft=512`, `hop_length=128`, `win_length=512` → 257 частотных бинов, 251 временной фрейм на сегмент 2 сек.

**Энкодер** (4 уровня + bottleneck): `32 → 64 → 128 → 256 → 512` каналов, MaxPool2d ↓2×.  
**Декодер** (4 уровня): ConvTranspose2d ↑2× + конкатенация со skip-connection.  
**Финальный слой:** Conv2d 1×1 + Sigmoid → маска.

Всего параметров: ~7.7 млн.

## Обучение

| Параметр | Значение |
|---|---|
| Оптимизатор | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 1e-3 → 1e-5 (Cosine Annealing) |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Эпох | до 50 (early stopping, patience=10) |
| Функция потерь | 0.8 × L1 + 0.2 × MSE |
| SNR при микшировании | 0–20 дБ (случайный) |

Шумные пары генерируются **на лету** (on-the-fly augmentation) — случайное смешивание чистой речи и шума при каждом обращении к элементу датасета.

## Результаты

| Входной SNR | Улучшение SI-SNR |
|---|---|
| 0 дБ  | ~ +X.XX дБ |
| 5 дБ  | ~ +X.XX дБ |
| 10 дБ | ~ +X.XX дБ |
| 15 дБ | ~ +X.XX дБ |
| 20 дБ | ~ +X.XX дБ |

> Значения заполняются после обучения финальной модели.

## Данные

- [LibriSpeech ASR corpus](https://www.openslr.org/12) — чистая речь на основе аудиокниг Project Gutenberg
- [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) — городские шумы (автомобили, стройка, музыка, толпа и др.)
