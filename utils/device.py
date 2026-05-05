"""
Cross-Platform Device Seçimi
Mac (MPS), Masaüstü (CUDA) ve CPU otomatik algılama.
"""

import torch


def get_device():
    """
    Kullanılabilir en iyi cihazı otomatik seçer.
    
    Returns:
        torch.device: cuda, mps veya cpu
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] NVIDIA GPU kullanılıyor: {gpu_name}")
        return device
    
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS GPU kullanılıyor (M-serisi çip)")
        return device
    
    else:
        device = torch.device("cpu")
        print("[Device] GPU bulunamadı, CPU kullanılıyor (yavaş olacak)")
        return device
