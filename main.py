"""
main.py — Точка входа. Связывает все модули воедино.

Порядок запуска:
1. python main.py --mode setup     — создаёт структуру папок на Google Drive
2. python main.py --mode download  — скачивает датасеты LibriSpeech и UrbanSound8K
3. python main.py --mode train     — обучает модель
4. python main.py --mode evaluate  — оценивает качество на тестовых примерах
5. python main.py --mode export    — экспортирует модель в PyTorch и ONNX
6. python main.py --mode denoise --input path/to/file.wav  — очищает файл

В Google Colab ячейки можно запускать напрямую, импортируя нужные функции.
"""

import os
import argparse
import warnings
import torch

warnings.filterwarnings('ignore')

from config import CFG, DEVICE, BASE_DIR, DIRS
from preprocessing import create_dataloaders
from model import UNetDenoiser
from train import train
from inference import (
    load_best_model,
    denoise_file,
    evaluate_on_test_samples,
    compute_all_metrics,
    export_model,
)
from visualization import plot_training_history


# ══════════════════════════════════════════════
#   Утилиты
# ══════════════════════════════════════════════

def setup_dirs():
    """Создаёт структуру папок на Google Drive."""
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
    print('✅ Структура папок создана:')
    for d in DIRS:
        print(f'  {d}')


def download_datasets():
    """Скачивает UrbanSound8K и LibriSpeech train-clean-100."""
    import tarfile
    import shutil

    CLEAN_DIR = CFG['clean_dir']
    NOISE_DIR = CFG['noise_dir']

    # ─── UrbanSound8K ───────────────────────────
    if len(os.listdir(NOISE_DIR)) < 100:
        print('Загрузка UrbanSound8K...')
        os.system(
            'wget -q --show-progress -O /content/UrbanSound8K.tar.gz '
            'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz'
        )
        print('Распаковка...')
        with tarfile.open('/content/UrbanSound8K.tar.gz', 'r:gz') as tar:
            tar.extractall('/content/')
        count = 0
        for fold in range(1, 11):
            fold_path = f'/content/UrbanSound8K/audio/fold{fold}'
            for fname in os.listdir(fold_path):
                if fname.endswith('.wav'):
                    shutil.copy(f'{fold_path}/{fname}', NOISE_DIR)
                    count += 1
        print(f'✅ Скопировано {count} шумовых файлов')
    else:
        count = len([f for f in os.listdir(NOISE_DIR) if f.endswith('.wav')])
        print(f'✅ Шумы уже загружены: {count} файлов')

    # ─── LibriSpeech train-clean-100 ────────────
    if len([f for f in os.listdir(CLEAN_DIR) if f.endswith(('.wav', '.flac'))]) < 1000:
        print('Загрузка LibriSpeech train-clean-100 (~6.3 GB)...')
        os.system(
            'wget -q --show-progress -O /content/train-clean-100.tar.gz '
            'https://www.openslr.org/resources/12/train-clean-100.tar.gz'
        )
        print('Распаковка (берём только первые 5000 файлов)...')
        os.makedirs('/content/librispeech_tmp', exist_ok=True)
        with tarfile.open('/content/train-clean-100.tar.gz', 'r:gz') as tar:
            members = [m for m in tar.getmembers() if m.name.endswith('.flac')]
            for i, member in enumerate(members[:5000]):
                tar.extract(member, '/content/librispeech_tmp')
                if i % 500 == 0:
                    print(f'  Распакован {i}/{min(5000, len(members))}...')
        count = 0
        for root, dirs, files in os.walk('/content/librispeech_tmp'):
            for fname in files:
                if fname.endswith('.flac'):
                    shutil.copy(os.path.join(root, fname), f'{CLEAN_DIR}/{fname}')
                    count += 1
        shutil.rmtree('/content/librispeech_tmp', ignore_errors=True)
        clean_archive = '/content/train-clean-100.tar.gz'
        if os.path.exists(clean_archive):
            os.remove(clean_archive)
        print(f'✅ Скопировано {count} файлов чистой речи')
    else:
        count = len(os.listdir(CLEAN_DIR))
        print(f'✅ Чистая речь уже загружена: {count} файлов')

    print(f'\n📊 Итого:')
    print(f'  Файлов чистой речи: {len(os.listdir(CLEAN_DIR))}')
    print(f'  Файлов шума:        {len(os.listdir(NOISE_DIR))}')


def print_model_info(model):
    """Выводит статистику по параметрам модели."""
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Всего параметров:    {total:,}')
    print(f'Обучаемых:           {trainable:,}')
    print(f'Примерный размер:    {total * 4 / 1e6:.1f} MB')


def print_final_report(model, cfg):
    """Итоговый отчёт об архитектуре и конфигурации."""
    print('\n' + '═' * 50)
    print('  ИТОГОВЫЙ ОТЧЁТ')
    print('═' * 50)
    print(f'  Архитектура:      U-Net (EncoderDecoder)')
    print(f'  Параметры:        {sum(p.numel() for p in model.parameters()):,}')
    print(f'  Входной формат:   WAV, 16кГц, моно')
    print(f'  STFT:             n_fft={cfg["n_fft"]}, hop={cfg["hop_length"]}')
    print(f'  Датасет речи:     LibriSpeech train-clean-100')
    print(f'  Датасет шумов:    UrbanSound8K (10 классов)')
    print(f'  Обучение:         AdamW + CosineAnnealing LR')
    print(f'  Функция потерь:   0.8×L1 + 0.2×MSE')
    print(f'  Метрика:          SI-SNR (дБ)')
    print('═' * 50)
    print(f'  Файлы сохранены в: {BASE_DIR}')


# ══════════════════════════════════════════════
#   CLI
# ══════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='U-Net Speech Denoiser — дипломная работа'
    )
    parser.add_argument(
        '--mode',
        choices=['setup', 'download', 'train', 'evaluate', 'export', 'denoise'],
        default='train',
        help='Режим запуска',
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Путь к входному WAV-файлу (только для режима denoise)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь к выходному WAV-файлу (только для режима denoise)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f'PyTorch версия: {torch.__version__}')
    print(f'CUDA доступна:  {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('⚠️  GPU не найден! Зайдите в Среда выполнения → Сменить среду → T4 GPU')

    if args.mode == 'setup':
        setup_dirs()

    elif args.mode == 'download':
        download_datasets()

    elif args.mode == 'train':
        setup_dirs()
        print('\n📦 Создаём DataLoader-ы...')
        train_loader, val_loader = create_dataloaders(CFG)

        # Проверка формы батча
        noisy_batch, clean_batch = next(iter(train_loader))
        print(f'Форма батча зашумлённых: {noisy_batch.shape}')
        print(f'Форма батча чистых:      {clean_batch.shape}')

        print('\n🧠 Создаём модель...')
        model = UNetDenoiser(base_ch=32).to(DEVICE)
        print_model_info(model)

        # Быстрый тест прямого прохода
        with torch.no_grad():
            dummy = torch.randn(2, 1, CFG['n_freq'], CFG['n_frames']).to(DEVICE)
            out   = model(dummy)
            print(f'Тест прямого прохода: {dummy.shape} → {out.shape} ✅')

        history = train(model, train_loader, val_loader, CFG, DEVICE)

        plot_training_history(
            history,
            save_path=f"{CFG['results_dir']}/training_history.png"
        )

    elif args.mode == 'evaluate':
        model = load_best_model(CFG, DEVICE)
        evaluate_on_test_samples(model, CFG, DEVICE, n_samples=5)
        print('\n📊 Итоговая оценка модели:')
        print('─' * 50)
        compute_all_metrics(model, CFG, DEVICE, n_test=20)

    elif args.mode == 'export':
        model = load_best_model(CFG, DEVICE)
        export_model(model, CFG, DEVICE, BASE_DIR)
        print_final_report(model, CFG)

    elif args.mode == 'denoise':
        if not args.input:
            print('⚠️  Укажите входной файл: --input path/to/file.wav')
            return
        output = args.output or args.input.replace('.wav', '_denoised.wav')
        model  = load_best_model(CFG, DEVICE)
        denoise_file(args.input, output, model, CFG, DEVICE)


if __name__ == '__main__':
    main()
