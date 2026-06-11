

import torch

BASE_DIR = '/content/drive/MyDrive/speech_denoising'

CLEAN_DIR = f'{BASE_DIR}/data/clean_speech'
NOISE_DIR = f'{BASE_DIR}/data/noise'

DIRS = [
    f'{BASE_DIR}/data/clean_speech',
    f'{BASE_DIR}/data/noise',
    f'{BASE_DIR}/checkpoints',
    f'{BASE_DIR}/results',
    f'{BASE_DIR}/logs',
]


CFG = {
    #Аудио
    'sample_rate': 16000,      # 16 кГц — стандарт для речи
    'segment_len': 2.0,        # длина сегмента для обучения

    #STFT
    'n_fft': 512,              
    'hop_length': 128,        
    'win_length': 512,         

    #Обучение
    'batch_size': 16,          
    'num_epochs': 50,          
    'learning_rate': 1e-3,     
    'weight_decay': 1e-4,      

    
    #SNR = 0 дБ: речь и шум одинаковой громкости
    'snr_min': 0,
    'snr_max': 20,

    'train_split': 0.9,        #90% на обучение, 10% на валидацию
    'num_workers': 2,          

    'clean_dir': CLEAN_DIR,
    'noise_dir': NOISE_DIR,
    'checkpoint_dir': f'{BASE_DIR}/checkpoints',
    'results_dir': f'{BASE_DIR}/results',
    'log_path': f'{BASE_DIR}/logs/training_log.json',
}

# Вычисляем производные параметры
CFG['segment_samples'] = int(CFG['sample_rate'] * CFG['segment_len'])
CFG['n_freq'] = CFG['n_fft'] // 2 + 1   # = 257
# Количество временных фреймов в сегменте
CFG['n_frames'] = CFG['segment_samples'] // CFG['hop_length'] + 1  # = 251

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
