"""
DepthNet – Birleşik Derinlik Tahmin Modeli
Encoder (ResNet) + Decoder (U-Net) → RGB'den derinlik haritası.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResNetEncoder
from .decoder import DepthDecoder


class DepthNet(nn.Module):
    """
    Uçtan uca monoküler derinlik tahmin ağı.
    
    RGB görüntü girer, metre cinsinden derinlik haritası çıkar.
    Sigmoid çıktısı [min_depth, max_depth] aralığına dönüştürülür.
    
    Args:
        num_layers (int): ResNet encoder katman sayısı (18 veya 50)
        pretrained (bool): ImageNet pretrained ağırlıklar
        min_depth (float): Minimum derinlik (metre)
        max_depth (float): Maximum derinlik (metre)
    """

    def __init__(self, num_layers=18, pretrained=True,
                 min_depth=0.1, max_depth=10.0):
        super().__init__()

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.encoder = ResNetEncoder(num_layers, pretrained)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DepthNet] Toplam parametre: {total_params:,}")
        print(f"[DepthNet] Eğitilebilir: {trainable:,}")
        print(f"[DepthNet] Derinlik aralığı: [{min_depth}, {max_depth}] m")

    def forward(self, rgb):
        """
        Args:
            rgb (Tensor): [B, 3, H, W], 0-1 arası
        Returns:
            Tensor: [B, 1, H, W], metre cinsinden derinlik
        """
        features = self.encoder(rgb)
        sigmoid_output = self.decoder(features)

        depth = self.min_depth + (self.max_depth - self.min_depth) * sigmoid_output

        if depth.shape[2:] != rgb.shape[2:]:
            depth = F.interpolate(
                depth, size=rgb.shape[2:],
                mode="bilinear", align_corners=False
            )
        return depth
