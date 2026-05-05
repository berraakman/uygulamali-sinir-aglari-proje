"""
NYU Depth V2 Dataset Yükleyici
RGB ve derinlik haritalarını yükler, augmentation uygular.
"""

import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class NYUDepthDataset(Dataset):
    """
    NYU Depth V2 veri seti yükleyicisi.
    
    Beklenen klasör yapısı:
        data_path/
        ├── train/
        │   ├── rgb/      (*.png veya *.jpg)
        │   └── depth/    (*.png - 16 bit)
        ├── val/
        │   ├── rgb/
        │   └── depth/
        └── test/
            ├── rgb/
            └── depth/
    
    Args:
        data_path (str): Veri kök dizini
        split (str): "train", "val" veya "test"
        height (int): Çıktı yüksekliği
        width (int): Çıktı genişliği
        augment (bool): Data augmentation uygula
    """

    def __init__(self, data_path, split="train", height=256, width=320, augment=True):
        super().__init__()

        self.height = height
        self.width = width
        self.is_train = (split == "train")
        self.augment = augment and self.is_train

        # RGB ve derinlik dosyalarını bul
        rgb_dir = os.path.join(data_path, split, "rgb")
        depth_dir = os.path.join(data_path, split, "depth")

        # Hem .png hem .jpg destekle
        self.rgb_paths = sorted(
            glob.glob(os.path.join(rgb_dir, "*.png")) +
            glob.glob(os.path.join(rgb_dir, "*.jpg"))
        )
        self.depth_paths = sorted(
            glob.glob(os.path.join(depth_dir, "*.png")) +
            glob.glob(os.path.join(depth_dir, "*.npy"))
        )


        assert len(self.rgb_paths) > 0, \
            f"RGB görüntü bulunamadı: {rgb_dir}"
        assert len(self.rgb_paths) == len(self.depth_paths), \
            f"RGB ({len(self.rgb_paths)}) ve Depth ({len(self.depth_paths)}) sayısı eşleşmiyor!"

        # Dönüşümler
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

        print(f"[Dataset] {split}: {len(self.rgb_paths)} görüntü yüklendi")

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # ─── RGB Yükle ───
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        rgb = rgb.resize((self.width, self.height), Image.BILINEAR)

        # ─── Derinlik Yükle ───
        depth_path = self.depth_paths[idx]
        if depth_path.endswith(".npy"):
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = np.array(Image.open(depth_path)).astype(np.float32)
            # Eğer 8-bit (0-255) bir görsel ise metreye çevir (0 - 10.0m aralığı)
            depth = (depth / 255.0) * 10.0

        # ─── Augmentation ───
        do_flip = False
        if self.augment:
            # Rastgele yatay çevirme
            if random.random() > 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = np.fliplr(depth).copy()
                do_flip = True

            # Renk değişikliği (sadece RGB'ye, derinliğe değil)
            if random.random() > 0.5:
                rgb = self.color_jitter(rgb)

        # ─── Tensor'a Çevir ───
        rgb = self.to_tensor(rgb)  # [3, H, W], 0-1 arası
        rgb = self.normalize(rgb)  # ImageNet standartlarına çek

        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H_orig, W_orig]
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(self.height, self.width),
            mode="nearest"
        ).squeeze(0)  # [1, H, W]

        return {
            "rgb": rgb,         # [3, H, W]
            "depth": depth,     # [1, H, W]
        }
