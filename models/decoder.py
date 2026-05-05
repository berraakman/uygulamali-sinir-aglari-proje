"""
Derinlik Decoder (U-Net tarzı)
Encoder'dan gelen çok ölçekli özellikleri birleştirerek derinlik haritası üretir.
Skip connection'lar ince detayları korur.

Referans: Ronneberger et al., "U-Net", 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ELU bloğu."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DepthDecoder(nn.Module):
    """
    U-Net tarzı decoder.
    
    Encoder'dan gelen 5 seviye feature map'i alır,
    skip connection ile birleştirerek yukarı örnekler,
    sonunda tek kanallı sigmoid çıktı (disparity) üretir.
    
    Args:
        num_ch_enc (np.ndarray): Encoder'ın her seviyedeki kanal sayıları [64, 64, 128, 256, 512]
    """

    def __init__(self, num_ch_enc):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # Decoder katmanları
        self.upconvs = nn.ModuleList()    # Upsampling öncesi convolution
        self.fuse_convs = nn.ModuleList()  # Skip birleştirme sonrası convolution

        for i in range(4, -1, -1):
            # Upconv: giriş kanalı → decoder kanalı
            if i == 4:
                in_ch = self.num_ch_enc[-1]       # En derin encoder çıktısı
            else:
                in_ch = self.num_ch_dec[i + 1]    # Bir önceki decoder seviyesi
            out_ch = self.num_ch_dec[i]
            self.upconvs.append(ConvBlock(in_ch, out_ch))

            # Fuse conv: skip bağlantısından sonra
            fuse_in_ch = self.num_ch_dec[i]
            if i > 0:
                fuse_in_ch += self.num_ch_enc[i - 1]  # + skip kanalları
            self.fuse_convs.append(ConvBlock(fuse_in_ch, self.num_ch_dec[i]))

        # Son katman: 1 kanallı derinlik çıktısı
        self.depth_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.num_ch_dec[0], 1, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, encoder_features):
        """
        Args:
            encoder_features (list[Tensor]): 5 seviye özellik haritası
        
        Returns:
            Tensor: Sigmoid çıktı [B, 1, H, W] (0-1 arası)
        """
        x = encoder_features[-1]  # En derin özellik (1/32)

        for i, (upconv, fuse) in enumerate(zip(self.upconvs, self.fuse_convs)):
            # 1) Upconv
            x = upconv(x)

            # 2) 2x upsample
            x = F.interpolate(x, scale_factor=2, mode="nearest")

            # 3) Skip connection (encoder'dan)
            skip_idx = 4 - i - 1  # [3, 2, 1, 0, -1]
            if 0 <= skip_idx < len(encoder_features):
                skip = encoder_features[skip_idx]
                # Boyut uyumsuzluğu olabilir, düzelt
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
                x = torch.cat([x, skip], dim=1)

            # 4) Fuse conv
            x = fuse(x)

        # Son sigmoid çıktı
        return self.depth_head(x)
