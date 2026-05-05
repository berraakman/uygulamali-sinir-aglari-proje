"""
Değerlendirme Scripti
Test seti üzerinde metrikleri hesaplar ve görsel sonuçları kaydeder.

Kullanım:
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from config import Config
from utils.device import get_device
from utils.metrics import compute_depth_metrics, print_metrics
from utils.visualization import save_comparison
from models.depth_net import DepthNet
from datasets.nyu_dataset import NYUDepthDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Derinlik Tahmini Değerlendirme")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint yolu")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"], help="Değerlendirme split'i")
    parser.add_argument("--num_visuals", type=int, default=10,
                        help="Kaydedilecek görsel örnek sayısı")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    cfg = Config()
    device = get_device()

    # ─── Model Yükle ───
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        model_cfg = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
    else:
        model_cfg = {"num_layers": 18, "min_depth": 0.1, "max_depth": 10.0}
        state_dict = checkpoint

    model = DepthNet(
        num_layers=model_cfg.get("num_layers", 18),
        pretrained=False,
        min_depth=model_cfg.get("min_depth", 0.1),
        max_depth=model_cfg.get("max_depth", 10.0)
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ─── Dataset ───
    dataset = NYUDepthDataset(
        cfg.data_path, args.split, cfg.image_height, cfg.image_width, augment=False)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2)

    print(f"\n{'='*50}")
    print(f"  Değerlendirme: {args.split} seti ({len(dataset)} görüntü)")
    print(f"{'='*50}\n")

    # ─── Değerlendirme ───
    all_metrics = []
    visual_count = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            rgb = batch["rgb"].to(device)
            depth_gt = batch["depth"].to(device)

            depth_pred = model(rgb)

            # Metrik hesapla
            pred_np = depth_pred.cpu().numpy()[0, 0]
            gt_np = depth_gt.cpu().numpy()[0, 0]
            metrics = compute_depth_metrics(gt_np, pred_np,
                                            cfg.min_depth, cfg.max_depth)
            all_metrics.append(metrics)

            # Görsel kaydet
            if visual_count < args.num_visuals:
                rgb_np = rgb.cpu().numpy()[0].transpose(1, 2, 0)  # [H,W,3]
                save_path = os.path.join(cfg.results_dir, f"eval_{idx:04d}.png")
                save_comparison(rgb_np, gt_np, pred_np, save_path, idx)
                visual_count += 1

            if (idx + 1) % 50 == 0:
                print(f"  İşlenen: {idx+1}/{len(dataset)}")

    # ─── Ortalama Metrikler ───
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    print(f"\n{'='*50}")
    print(f"  SONUÇLAR ({args.split} seti, {len(dataset)} görüntü)")
    print(f"{'='*50}")
    print_metrics(avg_metrics)

    # ─── Sonuçları Dosyaya Kaydet ───
    results_path = os.path.join(cfg.results_dir, f"metrics_{args.split}.txt")
    with open(results_path, "w") as f:
        f.write(f"Değerlendirme: {args.split} seti\n")
        f.write(f"Görüntü sayısı: {len(dataset)}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"\nMetrikler:\n")
        for key, val in avg_metrics.items():
            f.write(f"  {key}: {val:.6f}\n")

    print(f"  Sonuçlar kaydedildi: {results_path}")
    print(f"  Görseller kaydedildi: {cfg.results_dir}/")
    print()


if __name__ == "__main__":
    evaluate()
