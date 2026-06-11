"""
inference.py — Инференс, оценка качества и экспорт модели.

Содержит:
- load_best_model()       — загружает лучший чекпоинт
- denoise_array()         — обрабатывает numpy-массив аудио
- denoise_file()          — обрабатывает произвольный WAV-файл (с overlap-add)
- evaluate_on_test_samples() — оценка SI-SNR на тестовых примерах
- compute_all_metrics()   — оценка при разных входных уровнях SNR
- export_model()          — экспорт в PyTorch и ONNX форматы
"""

import os
import glob
import random
import numpy as np
import torch
import librosa
import soundfile as sf

from model import UNetDenoiser
from train import si_snr


# ══════════════════════════════════════════════
#   Загрузка модели
# ══════════════════════════════════════════════

def load_best_model(cfg, device):
    """Загружает лучший чекпоинт."""
    ckpt_path = f"{cfg['checkpoint_dir']}/best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = UNetDenoiser(base_ch=32).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    epoch   = checkpoint['epoch']
    val_snr = checkpoint.get('val_snr', 'N/A')
    print(f'✅ Загружена модель (эпоха {epoch}, val SI-SNR={val_snr:.2f} дБ)')
    return model


# ══════════════════════════════════════════════
#   Инференс
# ══════════════════════════════════════════════

def denoise_array(audio_array, model, cfg, device):
    """Обрабатывает numpy-массив аудио через модель (один сегмент)."""
    seg_len = cfg['segment_samples']
    if len(audio_array) < seg_len:
        audio_array = np.pad(audio_array, (0, seg_len - len(audio_array)))
    audio_array = audio_array[:seg_len]

    stft      = librosa.stft(audio_array, n_fft=cfg['n_fft'], hop_length=cfg['hop_length'])
    magnitude = np.abs(stft)
    phase     = np.angle(stft)
    log_mag   = np.log1p(magnitude).astype(np.float32)

    n_frames = cfg['n_frames']
    if log_mag.shape[1] < n_frames:
        log_mag = np.pad(log_mag, ((0, 0), (0, n_frames - log_mag.shape[1])))
    log_mag = log_mag[:, :n_frames]

    model.eval()
    with torch.no_grad():
        inp  = torch.FloatTensor(log_mag).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(inp).squeeze().cpu().numpy()

    pred_magnitude = np.expm1(pred[:, :stft.shape[1]])
    pred_magnitude = np.maximum(pred_magnitude, 0)
    pred_stft      = pred_magnitude * np.exp(1j * phase)
    denoised       = librosa.istft(pred_stft, hop_length=cfg['hop_length'], length=seg_len)

    return denoised


def denoise_file(input_path, output_path, model, cfg, device):
    """
    Очищает от шума произвольный WAV-файл.

    Поддерживает файлы любой длины — обработка по сегментам с overlap-add.
    """
    sr      = cfg['sample_rate']
    seg_len = cfg['segment_samples']
    hop     = seg_len // 2  # перекрытие 50% для сглаживания границ

    # 1. Загружаем
    audio, orig_sr = librosa.load(input_path, sr=sr, mono=True)
    total_samples  = len(audio)
    print(f'Загружено: {input_path}')
    print(f'Длительность: {total_samples/sr:.2f} сек, частота: {orig_sr} → {sr} Гц')

    # Дополняем нулями чтобы разбить на целое число сегментов
    n_segments = max(1, (total_samples - seg_len) // hop + 2)
    padded_len  = (n_segments - 1) * hop + seg_len
    audio_padded = np.pad(audio, (0, padded_len - total_samples))

    output_audio = np.zeros(padded_len)
    weights      = np.zeros(padded_len)

    # Ханновское окно для плавного сшивания сегментов
    window = np.hanning(seg_len)

    model.eval()
    with torch.no_grad():
        for i in range(n_segments):
            start = i * hop
            end   = start + seg_len
            if end > padded_len:
                break

            segment   = audio_padded[start:end]
            stft      = librosa.stft(segment, n_fft=cfg['n_fft'], hop_length=cfg['hop_length'],
                                     win_length=cfg['win_length'])
            magnitude = np.abs(stft)
            phase     = np.angle(stft)
            log_mag   = np.log1p(magnitude).astype(np.float32)

            n_frames = cfg['n_frames']
            if log_mag.shape[1] < n_frames:
                log_mag = np.pad(log_mag, ((0, 0), (0, n_frames - log_mag.shape[1])))
            log_mag = log_mag[:, :n_frames]

            inp            = torch.FloatTensor(log_mag).unsqueeze(0).unsqueeze(0).to(device)
            pred_log_mag   = model(inp).squeeze().cpu().numpy()

            pred_magnitude = np.expm1(pred_log_mag[:, :stft.shape[1]])
            pred_magnitude = np.maximum(pred_magnitude, 0)
            pred_stft      = pred_magnitude * np.exp(1j * phase[:, :pred_magnitude.shape[1]])

            denoised_seg = librosa.istft(
                pred_stft,
                hop_length=cfg['hop_length'],
                win_length=cfg['win_length'],
                length=seg_len
            )

            # Overlap-add с весовым окном
            output_audio[start:end] += denoised_seg * window
            weights[start:end]      += window

    # Нормализуем перекрытия
    mask = weights > 1e-6
    output_audio[mask] /= weights[mask]

    # Обрезаем до исходной длины и нормализуем
    output_audio = output_audio[:total_samples]
    output_audio = output_audio / (np.max(np.abs(output_audio)) + 1e-9) * 0.95

    # Сохраняем
    sf.write(output_path, output_audio, sr, subtype='PCM_16')
    print(f'✅ Сохранено: {output_path}')

    return audio[:total_samples], output_audio


# ══════════════════════════════════════════════
#   Оценка качества
# ══════════════════════════════════════════════

def evaluate_on_test_samples(model, cfg, device, n_samples=5):
    """
    Оценка модели на тестовых примерах.
    Выводит SI-SNR до и после обработки.
    """
    clean_files = glob.glob(f"{cfg['clean_dir']}/*.flac") + \
                  glob.glob(f"{cfg['clean_dir']}/*.wav")
    noise_files = glob.glob(f"{cfg['noise_dir']}/*.wav")

    test_files = random.sample(clean_files, min(n_samples, len(clean_files)))

    print('\n📊 Оценка на тестовых примерах:')
    print(f'{"Файл":>30} | {"SNR до":>8} | {"SNR после":>9} | {"Улучшение":>10}')
    print('─' * 65)

    all_improvements = []

    for i, clean_path in enumerate(test_files):
        clean, sr = librosa.load(clean_path, sr=cfg['sample_rate'], mono=True)
        seg = cfg['segment_samples']
        if len(clean) < seg:
            clean = np.pad(clean, (0, seg - len(clean)))
        clean = clean[:seg]

        noise, _ = librosa.load(random.choice(noise_files), sr=cfg['sample_rate'], mono=True)
        if len(noise) < seg:
            noise = np.pad(noise, (0, seg - len(noise)))
        noise = noise[:seg]

        # SNR = 5 дБ (сложный случай)
        snr_db    = 5.0
        rms_c     = np.sqrt(np.mean(clean ** 2) + 1e-9)
        rms_n     = np.sqrt(np.mean(noise ** 2) + 1e-9)
        noise_sc  = noise * (rms_c / (rms_n * 10 ** (snr_db / 20)))
        noisy     = clean + noise_sc

        denoised = denoise_array(noisy, model, cfg, device)

        c_t = torch.FloatTensor(clean)
        n_t = torch.FloatTensor(noisy)
        d_t = torch.FloatTensor(denoised)

        snr_before  = si_snr(n_t.unsqueeze(0), c_t.unsqueeze(0))
        snr_after   = si_snr(d_t.unsqueeze(0), c_t.unsqueeze(0))
        improvement = snr_after - snr_before
        all_improvements.append(improvement)

        fname = os.path.basename(clean_path)[:25]
        print(f'{fname:>30} | {snr_before:>8.2f} | {snr_after:>9.2f} | {improvement:>+10.2f}')

        # Визуализируем первый пример
        if i == 0:
            from visualization import visualize_denoising
            visualize_denoising(
                noisy, clean, denoised, cfg['sample_rate'],
                save_path=f"{cfg['results_dir']}/comparison_plot.png"
            )

    print('─' * 65)
    print(f'Среднее улучшение SI-SNR: {np.mean(all_improvements):+.2f} дБ')


def compute_all_metrics(model, cfg, device, n_test=20):
    """
    Вычисляет метрики качества при разных входных уровнях SNR.
    Строит сводный график улучшения SI-SNR.
    """
    import matplotlib.pyplot as plt

    clean_files = glob.glob(f"{cfg['clean_dir']}/*.flac")[:n_test]
    noise_files = glob.glob(f"{cfg['noise_dir']}/*.wav")

    results    = {}
    snr_levels = [0, 5, 10, 15, 20]

    for test_snr in snr_levels:
        snr_improvements = []

        for clean_path in clean_files:
            clean, _ = librosa.load(clean_path, sr=cfg['sample_rate'])
            seg = cfg['segment_samples']
            if len(clean) < seg:
                clean = np.pad(clean, (0, seg - len(clean)))
            clean = clean[:seg]

            noise, _ = librosa.load(random.choice(noise_files), sr=cfg['sample_rate'])
            if len(noise) < seg:
                noise = np.pad(noise, (0, seg - len(noise)))
            noise = noise[:seg]

            rms_c    = np.sqrt(np.mean(clean ** 2) + 1e-9)
            rms_n    = np.sqrt(np.mean(noise ** 2) + 1e-9)
            noise_sc = noise * (rms_c / (rms_n * 10 ** (test_snr / 20)))
            noisy    = clean + noise_sc

            denoised = denoise_array(noisy, model, cfg, device)

            snr_in  = si_snr(torch.FloatTensor(noisy).unsqueeze(0),
                             torch.FloatTensor(clean).unsqueeze(0))
            snr_out = si_snr(torch.FloatTensor(denoised).unsqueeze(0),
                             torch.FloatTensor(clean).unsqueeze(0))
            snr_improvements.append(snr_out - snr_in)

        results[test_snr] = {
            'mean_improvement': np.mean(snr_improvements),
            'std':              np.std(snr_improvements),
        }
        print(f'  SNR вход {test_snr:>3} дБ → улучшение: '
              f'{np.mean(snr_improvements):+.2f} ± {np.std(snr_improvements):.2f} дБ')

    # График
    fig, ax = plt.subplots(figsize=(7, 4))
    snr_vals     = list(results.keys())
    improvements = [results[s]['mean_improvement'] for s in snr_vals]
    stds         = [results[s]['std'] for s in snr_vals]

    ax.bar(snr_vals, improvements, yerr=stds, capsize=5,
           color='#2ecc71', alpha=0.8, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Входное отношение сигнал/шум (дБ)')
    ax.set_ylabel('Улучшение SI-SNR (дБ)')
    ax.set_title('Эффективность шумоподавления при различных уровнях шума')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(snr_vals)

    save_path = f"{cfg['results_dir']}/snr_improvement_chart.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nГрафик сохранён: {save_path}')
    plt.show()

    return results


# ══════════════════════════════════════════════
#   Экспорт модели
# ══════════════════════════════════════════════

def export_model(model, cfg, device, base_dir):
    """Сохраняет финальную модель в PyTorch и ONNX форматах."""
    # 1. Полный чекпоинт PyTorch
    full_save_path = f'{base_dir}/final_model_full.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'cfg':              cfg,
        'architecture':     'UNetDenoiser',
        'base_ch':          32,
    }, full_save_path)
    print(f'✅ Полная модель: {full_save_path}')

    # 2. ONNX
    try:
        onnx_path   = f'{base_dir}/denoiser.onnx'
        dummy_input = torch.randn(1, 1, cfg['n_freq'], cfg['n_frames']).to(device)
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['noisy_spectrogram'],
            output_names=['denoised_spectrogram'],
            dynamic_axes={
                'noisy_spectrogram':   {0: 'batch_size'},
                'denoised_spectrogram': {0: 'batch_size'},
            },
            verbose=False
        )
        print(f'✅ ONNX модель: {onnx_path}')
    except Exception as e:
        print(f'⚠️  ONNX экспорт не удался: {e}')
