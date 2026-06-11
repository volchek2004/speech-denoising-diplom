"""
model.py — Архитектура модели U-Net для шумоподавления речи.

U-Net — свёрточная сеть вида энкодер-декодер:
- Энкодер последовательно уменьшает пространственный размер и увеличивает
  количество каналов — выделяет признаки.
- Декодер восстанавливает размер обратно.
- Skip-connections соединяют соответствующие уровни — сохраняют мелкие детали.

Сеть предсказывает маску (0..1), которая говорит «насколько этот частотный
бин в этот момент времени является речью».
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Базовый блок: Conv → BatchNorm → LeakyReLU (дважды)."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Блок энкодера: уменьшение размера в 2 раза + ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),       # ↓2x по частоте и времени
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """Блок декодера: увеличение размера + конкатенация со skip + ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)  # ↑2x
        # Если размеры не совпадают — обрезаем (из-за нечётных размеров)
        if x.shape != skip.shape:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        x = torch.cat([x, skip], dim=1)  # конкатенация по каналам
        return self.conv(x)


class UNetDenoiser(nn.Module):
    """
    U-Net для предсказания маски шумоподавления.

    Вход:  [B, 1, 257, 251]  — лог-спектрограмма зашумлённой речи
    Выход: [B, 1, 257, 251]  — маска (значения 0..1)

    Очищенная спектрограмма = Вход × Маска
    """

    def __init__(self, base_ch=32):
        super().__init__()
        # ─── Энкодер ─────────────────────────────────
        self.enc0 = ConvBlock(1, base_ch)                       # 257×251 → 257×251
        self.enc1 = DownBlock(base_ch,   base_ch * 2)           # → 128×125
        self.enc2 = DownBlock(base_ch * 2, base_ch * 4)         # → 64×62
        self.enc3 = DownBlock(base_ch * 4, base_ch * 8)         # → 32×31

        # Дно (bottleneck) — самый маленький, самый абстрактный уровень
        self.bottleneck = DownBlock(base_ch * 8, base_ch * 16)  # → 16×15

        # ─── Декодер ─────────────────────────────────
        self.dec3 = UpBlock(base_ch * 16, base_ch * 8,  base_ch * 8)  # → 32×31
        self.dec2 = UpBlock(base_ch * 8,  base_ch * 4,  base_ch * 4)  # → 64×62
        self.dec1 = UpBlock(base_ch * 4,  base_ch * 2,  base_ch * 2)  # → 128×125
        self.dec0 = UpBlock(base_ch * 2,  base_ch,      base_ch)      # → 257×251

        # Финальный слой: 1×1 свёртка → маска 0..1
        self.final = nn.Sequential(
            nn.Conv2d(base_ch, 1, kernel_size=1),
            nn.Sigmoid(),   # ограничиваем [0, 1]
        )

    def forward(self, x):
        # Запоминаем входной размер для восстановления
        input_size = x.shape

        # Энкодер — сохраняем активации для skip-connections
        s0 = self.enc0(x)         # skip 0
        s1 = self.enc1(s0)        # skip 1
        s2 = self.enc2(s1)        # skip 2
        s3 = self.enc3(s2)        # skip 3
        bn = self.bottleneck(s3)  # bottleneck

        # Декодер — восстанавливаем с помощью skip-connections
        d3 = self.dec3(bn, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        # Маска
        mask = self.final(d0)

        # Обрезаем до точного входного размера
        mask = mask[:, :, :input_size[2], :input_size[3]]

        # Применяем маску: очищенный спектр = зашумлённый × маска
        return x * mask
