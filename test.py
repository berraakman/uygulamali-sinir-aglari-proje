"""
Test Scripti
Eğitilmiş modelle tek bir görüntüden derinlik tahmini yapar.

Kullanım:
    python test.py --checkpoint checkpoints/best_model.pth --image test_image.jpg
    python test.py --checkpoint checkpoints/best_model.pth --image_dir test_images/
"""

import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from models.depth_net import DepthNet
from utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Monoküler Derinlik Tahmini - Test")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint dosya yolu (.pth)")
    parser.add_argument("--image", type=str, default=None,
                        help="Tek görüntü yolu")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Görüntü klasörü yolu")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Sonuçların kaydedileceği klasör")
    parser.add_argument("--ext", type=str, default="jpg",
                        help="Klasördeki görüntü uzantısı")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Model checkpoint'ını yükler."""
    print(f"Model yükleniyor: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Config bilgilerini checkpoint'tan al
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        cfg = checkpoint["config"]
        num_layers = cfg.get("num_layers", 18)
        min_depth = cfg.get("min_depth", 0.1)
        max_depth = cfg.get("max_depth", 10.0)
        height = cfg.get("image_height", 256)
        width = cfg.get("image_width", 320)
        state_dict = checkpoint["model_state_dict"]
    else:
        # Düz state_dict
        num_layers = 18
        min_depth = 0.1
        max_depth = 10.0
        height = 256
        width = 320
        state_dict = checkpoint

    model = DepthNet(num_layers, pretrained=False,
                     min_depth=min_depth, max_depth=max_depth)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, height, width


def predict_image(model, image_path, height, width, device):
    """Tek bir görüntü için derinlik tahmini yapar."""
    # Yükle ve hazırla
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)
    img_resized = img.resize((width, height), Image.BILINEAR)
    img_tensor = transforms.ToTensor()(img_resized)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)

    # Tahmin
    with torch.no_grad():
        depth = model(img_tensor)

    depth_np = depth.squeeze().cpu().numpy()
    return img_resized, depth_np, original_size


def save_result(rgb_img, depth_np, save_path, image_name):
    """RGB ve derinlik haritasını yan yana kaydeder."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RGB
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Girdi", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Derinlik
    vmax = np.percentile(depth_np, 95)
    im = axes[1].imshow(depth_np, cmap="magma", vmin=depth_np.min(), vmax=vmax)
    axes[1].set_title("Tahmin Edilen Derinlik", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Derinlik (m)")

    plt.suptitle(image_name, fontsize=11, color="gray")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    device = get_device()

    # Model yükle
    model, height, width = load_model(args.checkpoint, device)

    # Görüntü yollarını topla
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, f"*.{args.ext}")))
    else:
        print("Hata: --image veya --image_dir belirtmelisiniz!")
        sys.exit(1)

    print(f"\n{len(image_paths)} görüntü işlenecek...\n")
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, image_path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(image_path))[0]
        rgb_img, depth_np, orig_size = predict_image(model, image_path, height, width, device)

        # Kaydet
        save_path = os.path.join(args.output_dir, f"{name}_depth.png")
        save_result(rgb_img, depth_np, save_path, name)

        # Numpy olarak da kaydet (sayısal analiz için)
        np.save(os.path.join(args.output_dir, f"{name}_depth.npy"), depth_np)

        print(f"  [{idx+1}/{len(image_paths)}] {name} → {save_path}")

    print(f"\n✅ Tamamlandı! Sonuçlar: {args.output_dir}/")


if __name__ == "__main__":
    main()
