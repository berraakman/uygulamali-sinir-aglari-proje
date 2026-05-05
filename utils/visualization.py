"""
Derinlik Haritası Görselleştirme
RGB görüntü, ground truth ve tahmin yan yana karşılaştırma.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_depth(depth_map, cmap="magma", title="Depth Map"):
    """
    Tek bir derinlik haritasını görselleştirir.
    
    Args:
        depth_map (np.ndarray): 2D derinlik haritası
        cmap (str): Renk haritası
        title (str): Başlık
    """
    plt.figure(figsize=(8, 4))
    plt.imshow(depth_map, cmap=cmap)
    plt.colorbar(label="Derinlik (m)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_comparison(rgb, depth_gt, depth_pred, save_path, index=0):
    """
    RGB, ground truth ve tahmin derinlik haritalarını yan yana kaydeder.
    Rapor ve sunum için ideal görsel çıktı.
    
    Args:
        rgb (np.ndarray): RGB görüntü [H, W, 3] (0-1 arası)
        depth_gt (np.ndarray): Ground truth derinlik [H, W]
        depth_pred (np.ndarray): Tahmin edilen derinlik [H, W]
        save_path (str): Kaydedilecek dosya yolu
        index (int): Görüntü numarası
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Girdi", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Ground Truth
    im1 = axes[1].imshow(depth_gt, cmap="magma")
    axes[1].set_title("Ground Truth Derinlik", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Tahmin
    im2 = axes[2].imshow(depth_pred, cmap="magma",
                          vmin=depth_gt.min(), vmax=depth_gt.max())
    axes[2].set_title("Tahmin Edilen Derinlik", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Örnek #{index + 1}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Görsel kaydedildi: {save_path}")


def save_loss_plot(train_losses, val_losses, save_path):
    """
    Eğitim ve validation loss grafiklerini kaydeder.
    
    Args:
        train_losses (list): Epoch başına train loss
        val_losses (list): Epoch başına validation loss
        save_path (str): Kaydedilecek dosya yolu
    """
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Eğitim ve Validation Loss Eğrisi", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss grafiği kaydedildi: {save_path}")
