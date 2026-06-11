"""
visualization.py — Визуализация результатов обучения и шумоподавления.

Содержит:
- plot_training_history()  — графики потерь, SI-SNR и LR по эпохам
- visualize_denoising()    — сравнение осциллограмм и спектрограмм
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display


def plot_training_history(history, save_path=None):
    """
    Строит три графика по истории обучения:
    1. Функция потерь (train / val)
    2. Метрика SI-SNR (train / val)
    3. Скорость обучения (Cosine Annealing)
    """
    epochs      = [h['epoch']      for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses   = [h['val_loss']   for h in history]
    train_snrs   = [h['train_snr']  for h in history]
    val_snrs     = [h['val_snr']    for h in history]
    lrs          = [h['lr']         for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('История обучения U-Net шумоподавителя', fontsize=14, fontweight='bold')

    # График 1: Потери
    axes[0].plot(epochs, train_losses, 'b-o', ms=3, label='Train Loss')
    axes[0].plot(epochs, val_losses,   'r-o', ms=3, label='Val Loss')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss (L1 + MSE)')
    axes[0].set_title('Функция потерь')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График 2: SI-SNR
    axes[1].plot(epochs, train_snrs, 'b-o', ms=3, label='Train SI-SNR')
    axes[1].plot(epochs, val_snrs,   'r-o', ms=3, label='Val SI-SNR')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('SI-SNR (дБ)')
    axes[1].set_title('Метрика SI-SNR (↑ лучше)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # График 3: Learning Rate
    axes[2].plot(epochs, lrs, 'g-o', ms=3)
    axes[2].set_xlabel('Эпоха')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Скорость обучения (Cosine Annealing)')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'График сохранён: {save_path}')
    plt.show()


def visualize_denoising(noisy_audio, clean_audio, denoised_audio, sr, save_path=None):
    """
    Строит сравнительную визуализацию (2 ряда × 3 столбца):
    - Верхний ряд: осциллограммы (временной сигнал)
    - Нижний ряд:  спектрограммы (частотно-временное представление)

    Столбцы: зашумлённый / очищенный (модель) / оригинал (эталон)
    """
    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    titles  = ['Зашумлённый сигнал', 'Очищенный (модель)', 'Оригинал (эталон)']
    signals = [noisy_audio, denoised_audio, clean_audio]
    colors  = ['#e74c3c', '#2ecc71', '#3498db']

    for col, (title, sig, color) in enumerate(zip(titles, signals, colors)):
        # Верхний ряд: осциллограмма
        ax_wave = fig.add_subplot(gs[0, col])
        t = np.linspace(0, len(sig) / sr, len(sig))
        ax_wave.plot(t, sig, color=color, linewidth=0.5, alpha=0.8)
        ax_wave.set_title(title, fontweight='bold', fontsize=10)
        ax_wave.set_xlabel('Время (с)')
        ax_wave.set_ylabel('Амплитуда')
        ax_wave.set_ylim(-1, 1)
        ax_wave.grid(True, alpha=0.3)

        # Нижний ряд: спектрограмма
        ax_spec = fig.add_subplot(gs[1, col])
        D    = librosa.stft(sig, n_fft=512, hop_length=128)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(
            D_db, sr=sr, hop_length=128, x_axis='time', y_axis='hz',
            ax=ax_spec, cmap='magma'
        )
        ax_spec.set_title(f'Спектрограмма ({title})', fontsize=9)
        ax_spec.set_ylim(0, 8000)  # 0-8 кГц — речевой диапазон

    plt.suptitle('Сравнение сигналов до и после шумоподавления',
                 fontsize=13, fontweight='bold', y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'График сохранён: {save_path}')
    plt.show()
