"""
Eğitim Scripti
Monoküler derinlik tahmin modelini eğitir.

Kullanım:
    python train.py
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from utils.device import get_device
from utils.metrics import compute_depth_metrics, print_metrics
from utils.visualization import save_loss_plot
from models.depth_net import DepthNet
from datasets.nyu_dataset import NYUDepthDataset
from losses.depth_loss import DepthLoss


def validate(model, val_loader, criterion, device):
    """Validation seti üzerinde loss ve metrikleri hesaplar."""
    model.eval()
    total_loss = 0
    all_metrics = []

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device)
            depth_gt = batch["depth"].to(device)

            depth_pred = model(rgb)
            loss = criterion(depth_pred, depth_gt, rgb)
            total_loss += loss.item()

            # Metrikleri hesapla
            pred_np = depth_pred.cpu().numpy()
            gt_np = depth_gt.cpu().numpy()
            for b in range(pred_np.shape[0]):
                m = compute_depth_metrics(gt_np[b, 0], pred_np[b, 0])
                all_metrics.append(m)

    avg_loss = total_loss / max(len(val_loader), 1)

    # Ortalama metrikler
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return avg_loss, avg_metrics


def train():
    """Ana eğitim fonksiyonu."""
    cfg = Config()
    device = get_device()

    # Klasörleri oluştur
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("  Monoküler Derinlik Tahmini - Eğitim")
    print("=" * 50)

    # ─── Model ───
    model = DepthNet(
        num_layers=cfg.num_layers,
        pretrained=cfg.pretrained,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth
    ).to(device)

    # ─── Dataset ───
    train_set = NYUDepthDataset(
        cfg.data_path, "train", cfg.image_height, cfg.image_width, augment=True)
    val_set = NYUDepthDataset(
        cfg.data_path, "val", cfg.image_height, cfg.image_width, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=False)

    # ─── Optimizer + Scheduler ───
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)

    # ─── Loss ───
    criterion = DepthLoss(
        si_weight=cfg.si_loss_weight,
        smooth_weight=cfg.smooth_loss_weight)

    # ─── TensorBoard ───
    writer = SummaryWriter(cfg.log_dir)

    # ─── Eğitim Döngüsü ───
    best_val_loss = float("inf")
    best_val_metrics = {}
    train_losses = []
    val_losses = []
    start_time = time.time()

    print(f"\nEğitim başlıyor: {cfg.num_epochs} epoch, "
          f"batch_size={cfg.batch_size}, lr={cfg.learning_rate}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} görüntü\n")

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device)
            depth_gt = batch["depth"].to(device)

            # Forward
            depth_pred = model(rgb)
            loss = criterion(depth_pred, depth_gt, rgb)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Loglama
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % cfg.log_frequency == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch [{epoch+1}/{cfg.num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Süre: {elapsed:.0f}s")

            writer.add_scalar("train/batch_loss", loss.item(), global_step)

        # Epoch sonu
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)

        # ─── Validation ───
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        writer.add_scalar("val/loss", val_loss, epoch)

        for key, val in val_metrics.items():
            writer.add_scalar(f"val/{key}", val, epoch)

        epoch_time = time.time() - epoch_start
        print(f"\n{'─'*50}")
        print(f"  Epoch {epoch+1}/{cfg.num_epochs} tamamlandı ({epoch_time:.0f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_metrics:
            print(f"  abs_rel: {val_metrics.get('abs_rel', 0):.4f} | "
                  f"δ<1.25: {val_metrics.get('delta_1', 0):.4f}")
        print(f"{'─'*50}\n")

        # ─── Checkpoint Kaydet ───
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            save_path = os.path.join(cfg.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "config": {
                    "num_layers": cfg.num_layers,
                    "min_depth": cfg.min_depth,
                    "max_depth": cfg.max_depth,
                    "image_height": cfg.image_height,
                    "image_width": cfg.image_width,
                }
            }, save_path)
            print(f"  ✅ En iyi model kaydedildi (val_loss: {val_loss:.4f})\n")

        if (epoch + 1) % cfg.save_frequency == 0:
            save_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

    # ─── Eğitim Bitti ───
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  🎉 Eğitim tamamlandı!")
    print(f"  Toplam süre: {total_time/3600:.1f} saat")
    print(f"  En iyi Val Loss (Hata Oranı): {best_val_loss:.4f}")
    if best_val_metrics:
        print(f"  En iyi Doğruluk (δ < 1.25): % {best_val_metrics.get('delta_1', 0)*100:.1f}")
        print(f"  En iyi Sapma (abs_rel): {best_val_metrics.get('abs_rel', 0):.4f}")
    print(f"{'='*50}")

    # Loss grafiğini kaydet
    save_loss_plot(
        train_losses, val_losses,
        os.path.join(cfg.results_dir, "loss_curve.png"))

    writer.close()


if __name__ == "__main__":
    train()

# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# python train.py
