"""
Derinlik Tahmin Loss Fonksiyonları

L_total = L1 + Scale-Invariant + λ × Edge-Aware Smoothness

Referanslar:
  - Eigen et al. 2014 (Scale-Invariant loss)
  - Godard et al. 2017 (Edge-aware smoothness)
"""

import torch
import torch.nn as nn


class DepthLoss(nn.Module):
    """
    Monoküler derinlik tahmini için birleşik loss fonksiyonu.
    
    Üç bileşen:
      1. L1 Loss: Piksel bazında mutlak hata
      2. Scale-Invariant Loss: Log uzayında ölçek-bağımsız hata  
      3. Edge-Aware Smoothness: Renk kenarlarına duyarlı pürüzsüzlük
    
    Args:
        si_weight (float): Scale-Invariant loss'taki α katsayısı
        smooth_weight (float): Smoothness loss ağırlığı (λ)
    """

    def __init__(self, si_weight=0.5, smooth_weight=0.001):
        super().__init__()
        self.si_weight = si_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred, target, rgb=None):
        """
        Args:
            pred (Tensor): Tahmin [B, 1, H, W]
            target (Tensor): Ground truth [B, 1, H, W]
            rgb (Tensor): RGB görüntü [B, 3, H, W] (smoothness için)
        
        Returns:
            Tensor: Toplam loss (skaler)
        """
        # Geçerli pikseller (derinlik > 0 olan)
        mask = target > 0.01
        p = pred[mask]
        t = target[mask]

        if p.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # ── L1 Loss ──
        l1_loss = torch.mean(torch.abs(p - t))

        # ── Scale-Invariant Loss (Eigen et al. 2014) ──
        log_diff = torch.log(p + 1e-6) - torch.log(t + 1e-6)
        si_loss = torch.mean(log_diff ** 2) - self.si_weight * (torch.mean(log_diff) ** 2)

        # Toplam
        total_loss = l1_loss + si_loss

        # ── Edge-Aware Smoothness Loss (opsiyonel) ──
        if rgb is not None and self.smooth_weight > 0:
            smooth = self._edge_aware_smoothness(pred, rgb)
            total_loss += self.smooth_weight * smooth

        return total_loss

    def _edge_aware_smoothness(self, disp, img):
        """
        Kenar-duyarlı pürüzsüzlük loss'u.
        Görüntüde kenar olan yerlerde derinlik süreksizliğine izin verir.
        
        Formül: |∂d/∂x| × exp(-|∂I/∂x|) + |∂d/∂y| × exp(-|∂I/∂y|)
        """
        # Derinlik gradyanları
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        # Görüntü gradyanları
        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

        # Kenar olan yerde cezayı azalt
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()
