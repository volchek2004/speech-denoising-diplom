"""
preprocessing.py — Датасет, загрузчики данных и STFT-утилиты.

Класс SpeechDenoiseDataset выполняет:
1. Загружает файл чистой речи.
2. Загружает случайный файл шума.
3. Смешивает их в заданном соотношении SNR (on-the-fly augmentation).
4. Вычисляет STFT обоих сигналов.
5. Возвращает (зашумлённая_спектрограмма, чистая_спектрограмма).
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa


class SpeechDenoiseDataset(Dataset):
    """
    Датасет для обучения нейросети шумоподавления.

    Каждый элемент — пара (зашумлённый спектр, чистый спектр).
    Шум добавляется случайно при каждом обращении к элементу,
    что эффективно увеличивает размер датасета.
    """

    def __init__(self, clean_files, noise_files, cfg, augment=True):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.cfg = cfg
        self.augment = augment
        self.sr = cfg['sample_rate']
        self.seg_len = cfg['segment_samples']

    def __len__(self):
        return len(self.clean_files)

    def load_and_resample(self, path):
        """Загружает аудио и приводит к нужной частоте дискретизации."""
        try:
            audio, sr = librosa.load(path, sr=self.sr, mono=True)
        except Exception:
            # Если файл повреждён — возвращаем тишину
            audio = np.zeros(self.seg_len, dtype=np.float32)
        return audio.astype(np.float32)

    def crop_or_pad(self, audio, length):
        """Обрезает или дополняет нулями до нужной длины."""
        if len(audio) >= length:
            # Случайный старт (аугментация) или с начала
            if self.augment:
                start = random.randint(0, len(audio) - length)
            else:
                start = 0
            return audio[start:start + length]
        else:
            # Дополняем нулями (padding)
            pad = length - len(audio)
            return np.pad(audio, (0, pad), mode='constant')

    def mix_with_noise(self, clean, noise):
        """
        Смешивает чистую речь с шумом при случайном SNR.

        SNR (дБ) = 10 * log10(мощность_речи / мощность_шума)
        Чем выше SNR — тем лучше слышна речь.
        """
        snr_db = random.uniform(self.cfg['snr_min'], self.cfg['snr_max'])

        # Вычисляем RMS (среднеквадратичное значение = мера громкости)
        rms_clean = np.sqrt(np.mean(clean ** 2) + 1e-9)
        rms_noise = np.sqrt(np.mean(noise ** 2) + 1e-9)

        # Масштабируем шум под нужный SNR
        desired_rms_noise = rms_clean / (10 ** (snr_db / 20.0))
        noise_scaled = noise * (desired_rms_noise / rms_noise)

        noisy = clean + noise_scaled

        # Нормализация чтобы не было клиппинга
        max_val = np.max(np.abs(noisy)) + 1e-9
        if max_val > 1.0:
            noisy = noisy / max_val
            clean = clean / max_val

        return noisy, clean

    def compute_stft(self, audio):
        """
        Вычисляет STFT (Short-Time Fourier Transform).
        Результат: матрица комплексных чисел [частоты × время].
        Возвращает логарифмическую амплитуду (log-magnitude).
        """
        stft = librosa.stft(
            audio,
            n_fft=self.cfg['n_fft'],
            hop_length=self.cfg['hop_length'],
            win_length=self.cfg['win_length'],
            window='hann',
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        # Log-scale (в дБ): сжимает динамический диапазон
        log_mag = np.log1p(magnitude)  # log(1 + |X|) — числово стабильно
        return log_mag.astype(np.float32), phase, magnitude

    def __getitem__(self, idx):
        # 1. Загружаем чистую речь
        clean_audio = self.load_and_resample(self.clean_files[idx])
        clean_audio = self.crop_or_pad(clean_audio, self.seg_len)

        # 2. Загружаем случайный шум
        noise_path = random.choice(self.noise_files)
        noise_audio = self.load_and_resample(noise_path)
        noise_audio = self.crop_or_pad(noise_audio, self.seg_len)

        # 3. Смешиваем при случайном SNR
        noisy_audio, clean_audio = self.mix_with_noise(clean_audio, noise_audio)

        # 4. Вычисляем STFT
        noisy_log_mag, _, _ = self.compute_stft(noisy_audio)
        clean_log_mag, _, _ = self.compute_stft(clean_audio)

        # 5. Обрезаем до фиксированного числа фреймов
        n_frames = self.cfg['n_frames']
        noisy_log_mag = noisy_log_mag[:, :n_frames]
        clean_log_mag = clean_log_mag[:, :n_frames]

        # 6. Добавляем размерность канала [1, freq, time]
        noisy_tensor = torch.FloatTensor(noisy_log_mag).unsqueeze(0)
        clean_tensor = torch.FloatTensor(clean_log_mag).unsqueeze(0)

        return noisy_tensor, clean_tensor


def create_dataloaders(cfg):
    """Создаёт DataLoader для обучения и валидации."""
    # Собираем списки файлов
    clean_files = glob.glob(f"{cfg['clean_dir']}/*.flac") + \
                  glob.glob(f"{cfg['clean_dir']}/*.wav")
    noise_files = glob.glob(f"{cfg['noise_dir']}/*.wav") + \
                  glob.glob(f"{cfg['noise_dir']}/*.mp3")

    print(f'Найдено файлов чистой речи: {len(clean_files)}')
    print(f'Найдено файлов шума: {len(noise_files)}')

    if len(clean_files) == 0 or len(noise_files) == 0:
        raise ValueError('Нет данных! Сначала выполните загрузку датасета.')

    # Перемешиваем и делим на train/val
    random.shuffle(clean_files)
    split = int(len(clean_files) * cfg['train_split'])
    train_files = clean_files[:split]
    val_files = clean_files[split:]

    print(f'Обучающих: {len(train_files)}, Валидационных: {len(val_files)}')

    # Создаём датасеты
    train_ds = SpeechDenoiseDataset(train_files, noise_files, cfg, augment=True)
    val_ds   = SpeechDenoiseDataset(val_files,   noise_files, cfg, augment=False)

    # DataLoader — итератор по батчам
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,    # быстрая передача на GPU
        drop_last=True,     # отбрасываем последний неполный батч
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
    )

    return train_loader, val_loader
