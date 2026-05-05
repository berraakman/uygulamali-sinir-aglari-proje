"""
Proje Ayarları
Tüm hiperparametreler ve yol tanımları burada.
"""

import os


class Config:
    # ─────────────────────────────────────────────
    # Veri Yolları
    # ─────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "data", "nyu_depth_v2")
    
    # ─────────────────────────────────────────────
    # Görüntü Boyutları
    # ─────────────────────────────────────────────
    image_height = 256
    image_width = 320

    # ─────────────────────────────────────────────
    # Model Ayarları
    # ─────────────────────────────────────────────
    encoder_name = "resnet18"       # "resnet18" veya "resnet50"
    num_layers = 18                 # 18 veya 50
    pretrained = True               # ImageNet pretrained ağırlıklar

    # ─────────────────────────────────────────────
    # Derinlik Aralığı (metre)
    # ─────────────────────────────────────────────
    min_depth = 0.1                 # minimum derinlik
    max_depth = 10.0                # maximum derinlik (NYU iç mekan)

    # ─────────────────────────────────────────────
    # Eğitim Hiperparametreleri
    # ─────────────────────────────────────────────
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 30
    scheduler_step_size = 5         # Bu epoch'ta lr 10x küçülür
    scheduler_gamma = 0.1

    # ─────────────────────────────────────────────
    # Loss Ağırlıkları
    # ─────────────────────────────────────────────
    si_loss_weight = 0.5            # Scale-Invariant loss katsayısı
    smooth_loss_weight = 0.001      # Smoothness loss katsayısı

    # ─────────────────────────────────────────────
    # DataLoader
    # ─────────────────────────────────────────────
    num_workers = 0

    # ─────────────────────────────────────────────
    # Kayıt Yolları
    # ─────────────────────────────────────────────
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    results_dir = os.path.join(os.path.dirname(__file__), "results")

    # ─────────────────────────────────────────────
    # Loglama
    # ─────────────────────────────────────────────
    log_frequency = 100             # Her N batch'te bir log bas
    save_frequency = 5              # Her N epoch'ta bir checkpoint kaydet
