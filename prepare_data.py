"""
NYU Depth V2 (DenseDepth formatı) → Proje formatına dönüştürücü.

İndirilen format:
    nyu_data/data/nyu2_train/bedroom_0001_out/1.jpg (RGB)
    nyu_data/data/nyu2_train/bedroom_0001_out/1.png (Depth)
    nyu_data/data/nyu2_test/00000_colors.png (RGB)
    nyu_data/data/nyu2_test/00000_depth.png  (Depth)

Hedef format:
    data/nyu_depth_v2/train/rgb/00001.jpg
    data/nyu_depth_v2/train/depth/00001.png
    data/nyu_depth_v2/val/rgb/00001.jpg
    data/nyu_depth_v2/val/depth/00001.png
    data/nyu_depth_v2/test/rgb/00001.png
    data/nyu_depth_v2/test/depth/00001.png

Kullanım:
    python prepare_data.py --source /Users/berra/Downloads/nyu_data/data
"""

import os
import glob
import shutil
import argparse
import random

random.seed(42)


def prepare_data(source_dir, target_dir, val_ratio=0.1):
    """DenseDepth NYU formatını proje formatına çevirir."""

    # Hedef klasörleri oluştur
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_dir, split, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, "depth"), exist_ok=True)

    # ─── TRAIN veriyi topla ───
    train_src = os.path.join(source_dir, "nyu2_train")
    scene_dirs = sorted(glob.glob(os.path.join(train_src, "*_out")))

    print(f"Train sahne sayısı: {len(scene_dirs)}")

    all_pairs = []
    for scene_dir in scene_dirs:
        # Her sahnede jpg (RGB) ve png (depth) çiftleri var
        jpgs = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
        for jpg_path in jpgs:
            # RGB: X.jpg, Depth: X.png
            base = os.path.splitext(os.path.basename(jpg_path))[0]
            png_path = os.path.join(scene_dir, f"{base}.png")
            if os.path.exists(png_path):
                all_pairs.append((jpg_path, png_path))

    print(f"Toplam train çifti: {len(all_pairs)}")

    # Train/Val ayır
    random.shuffle(all_pairs)
    val_count = int(len(all_pairs) * val_ratio)
    val_pairs = all_pairs[:val_count]
    train_pairs = all_pairs[val_count:]

    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # ─── Train kopyala ───
    print("\nTrain görüntüleri kopyalanıyor...")
    for idx, (rgb_path, depth_path) in enumerate(train_pairs):
        dst_rgb = os.path.join(target_dir, "train", "rgb", f"{idx:05d}.jpg")
        dst_depth = os.path.join(target_dir, "train", "depth", f"{idx:05d}.png")
        shutil.copy2(rgb_path, dst_rgb)
        shutil.copy2(depth_path, dst_depth)
        if (idx + 1) % 5000 == 0:
            print(f"  {idx+1}/{len(train_pairs)}")

    print(f"  ✅ Train: {len(train_pairs)} çift kopyalandı")

    # ─── Val kopyala ───
    print("Val görüntüleri kopyalanıyor...")
    for idx, (rgb_path, depth_path) in enumerate(val_pairs):
        dst_rgb = os.path.join(target_dir, "val", "rgb", f"{idx:05d}.jpg")
        dst_depth = os.path.join(target_dir, "val", "depth", f"{idx:05d}.png")
        shutil.copy2(rgb_path, dst_rgb)
        shutil.copy2(depth_path, dst_depth)

    print(f"  ✅ Val: {len(val_pairs)} çift kopyalandı")

    # ─── Test kopyala ───
    test_src = os.path.join(source_dir, "nyu2_test")
    color_files = sorted(glob.glob(os.path.join(test_src, "*_colors.png")))

    print("Test görüntüleri kopyalanıyor...")
    test_count = 0
    for idx, color_path in enumerate(color_files):
        base = os.path.basename(color_path).replace("_colors.png", "")
        depth_path = os.path.join(test_src, f"{base}_depth.png")
        if os.path.exists(depth_path):
            dst_rgb = os.path.join(target_dir, "test", "rgb", f"{idx:05d}.png")
            dst_depth = os.path.join(target_dir, "test", "depth", f"{idx:05d}.png")
            shutil.copy2(color_path, dst_rgb)
            shutil.copy2(depth_path, dst_depth)
            test_count += 1

    print(f"  ✅ Test: {test_count} çift kopyalandı")

    # ─── Özet ───
    print(f"\n{'='*40}")
    print(f"  Veri hazırlama tamamlandı!")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Val:   {len(val_pairs)}")
    print(f"  Test:  {test_count}")
    print(f"  Hedef: {target_dir}")
    print(f"{'='*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        default="/Users/berra/Downloads/nyu_data/data",
                        help="İndirilen nyu_data/data klasörü")
    parser.add_argument("--target", type=str,
                        default="./data/nyu_depth_v2",
                        help="Hedef klasör")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation oranı (varsayılan %10)")
    args = parser.parse_args()

    prepare_data(args.source, args.target, args.val_ratio)
