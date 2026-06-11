"""
train.py — Функция потерь, метрики и цикл обучения.

Функция потерь: L = 0.8 × L1 + 0.2 × MSE
Метрика:        SI-SNR (Scale-Invariant Signal-to-Noise Ratio, дБ)

Особенности цикла обучения:
- Cosine Annealing LR — плавное снижение скорости обучения
- Early Stopping       — остановка если нет улучшений 10 эпох
- Gradient Clipping    — предотвращает взрывной рост градиентов
- Checkpoint saving    — сохраняет лучшую и периодические чекпоинты
- JSON-лог             — для последующего построения графиков
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ══════════════════════════════════════════════
#   Функция потерь
# ══════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    Комбинированная функция потерь:
    L = alpha * L1(спектр) + (1-alpha) * MSE(спектр)

    L1 устойчива к выбросам, MSE штрафует за большие ошибки сильнее.
    """

    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        l1_loss  = self.l1(pred, target)
        mse_loss = self.mse(pred, target)
        return self.alpha * l1_loss + (1 - self.alpha) * mse_loss


# ══════════════════════════════════════════════
#   Метрика SI-SNR
# ══════════════════════════════════════════════

def si_snr(pred, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio (дБ).
    Стандартная метрика для задач разделения источников звука.
    Чем выше — тем лучше. Хорошими считаются значения > 15 дБ.

    pred, target: тензоры любой формы [B, ...]
    """
    pred   = pred.flatten(1)    # [B, N]
    target = target.flatten(1)

    # Центрирование (вычитаем среднее)
    pred   = pred   - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    # Проекция pred на target
    dot            = (pred * target).sum(dim=1, keepdim=True)
    target_norm_sq = (target ** 2).sum(dim=1, keepdim=True) + eps
    s_target       = dot / target_norm_sq * target

    # Ошибка
    e_noise = pred - s_target

    # SI-SNR
    si_snr_val = 10 * torch.log10(
        (s_target ** 2).sum(dim=1) / ((e_noise ** 2).sum(dim=1) + eps) + eps
    )
    return si_snr_val.mean().item()


# ══════════════════════════════════════════════
#   Одна эпоха обучения / валидации
# ══════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    """Одна эпоха обучения — возвращает среднюю потерю и SI-SNR."""
    model.train()
    total_loss, total_sisnr, n_batches = 0, 0, 0

    for noisy, clean in tqdm(loader, desc='Обучение', leave=False):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(noisy)
        loss = criterion(pred, clean)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss   += loss.item()
        total_sisnr  += si_snr(pred.detach(), clean.detach())
        n_batches    += 1

    return total_loss / n_batches, total_sisnr / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Валидация — модель не обновляется, только считаем метрики."""
    model.eval()
    total_loss, total_sisnr, n_batches = 0, 0, 0

    for noisy, clean in tqdm(loader, desc='Валидация', leave=False):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        pred = model(noisy)
        loss = criterion(pred, clean)

        total_loss  += loss.item()
        total_sisnr += si_snr(pred, clean)
        n_batches   += 1

    return total_loss / n_batches, total_sisnr / n_batches


# ══════════════════════════════════════════════
#   Полный цикл обучения
# ══════════════════════════════════════════════

def train(model, train_loader, val_loader, cfg, device):
    """Полный цикл обучения с ранней остановкой и сохранением чекпоинтов."""
    criterion = CombinedLoss(alpha=0.8)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
        betas=(0.9, 0.999),
    )

    # Планировщик LR: начинаем с lr, плавно снижаем до lr/100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['num_epochs'],
        eta_min=cfg['learning_rate'] / 100,
    )

    best_val_loss      = float('inf')
    patience_counter   = 0
    early_stop_patience = 10
    history            = []

    best_ckpt = f"{cfg['checkpoint_dir']}/best_model.pt"

    print(f'\n🚀 Начинаем обучение на {cfg["num_epochs"]} эпох')
    print(f'   Устройство: {device}')
    print(f'   Батчей/эпоха (train): {len(train_loader)}')
    print(f'   Батчей/эпоха (val):   {len(val_loader)}')
    print('─' * 65)
    print(f'{"Эпоха":>6} | {"Train L":>9} | {"Val L":>9} | {"Train SNR":>10} | {"Val SNR":>9} | {"LR":>8}')
    print('─' * 65)

    for epoch in range(1, cfg['num_epochs'] + 1):
        train_loss, train_snr = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_snr   = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Логируем
        history.append({
            'epoch':      epoch,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_snr':  train_snr,  'val_snr':  val_snr,
            'lr':         current_lr,
        })

        print(f'{epoch:>6} | {train_loss:>9.4f} | {val_loss:>9.4f} | '
              f'{train_snr:>10.2f} | {val_snr:>9.2f} | {current_lr:>8.6f}')

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             val_loss,
                'val_snr':              val_snr,
                'cfg':                  cfg,
            }, best_ckpt)
            print(f'         ✅ Лучшая модель сохранена (val_loss={val_loss:.4f})')
        else:
            patience_counter += 1

        # Периодически сохраняем чекпоинт (каждые 10 эпох)
        if epoch % 10 == 0:
            ckpt_path = f"{cfg['checkpoint_dir']}/epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

        # Сохраняем лог
        with open(cfg['log_path'], 'w') as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f'\n⏹️  Early stopping на эпохе {epoch}')
            break

    print('─' * 65)
    print(f'\n🏆 Обучение завершено!')
    print(f'   Лучший val_loss: {best_val_loss:.4f}')
    print(f'   Чекпоинт: {best_ckpt}')
    return history
