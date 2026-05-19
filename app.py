"""
Monoküler Derinlik Tahmini - Web Arayüzü
Sürükle-bırak ile fotoğraf yükleyip derinlik haritası oluşturur.
Flask tabanlı web uygulaması.
"""
# source venv/bin/activate && python app.py

import os
import io
import base64
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from flask import Flask, render_template, request, jsonify

from models.depth_net import DepthNet
from utils.device import get_device

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")

# ─── Global model (startup'ta bir kez yüklenir) ────────────────
MODEL = None
DEVICE = None
HEIGHT = 256
WIDTH = 320


def load_model_once():
    """Model'i uygulama başlatıldığında bir kez yükler."""
    global MODEL, DEVICE, HEIGHT, WIDTH

    DEVICE = get_device()
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint bulunamadı: {checkpoint_path}")
        return False

    print(f"🔄 Model yükleniyor: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        cfg = checkpoint["config"]
        num_layers = cfg.get("num_layers", 18)
        min_depth = cfg.get("min_depth", 0.1)
        max_depth = cfg.get("max_depth", 10.0)
        HEIGHT = cfg.get("image_height", 256)
        WIDTH = cfg.get("image_width", 320)
        state_dict = checkpoint["model_state_dict"]
    else:
        num_layers = 18
        min_depth = 0.1
        max_depth = 10.0
        state_dict = checkpoint

    MODEL = DepthNet(num_layers, pretrained=False,
                     min_depth=min_depth, max_depth=max_depth)
    MODEL.load_state_dict(state_dict)
    MODEL.to(DEVICE)
    MODEL.eval()
    print("✅ Model başarıyla yüklendi!")
    return True


def predict(image_bytes):
    """Yüklenen görselden derinlik tahmini yapar."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((WIDTH, HEIGHT), Image.BILINEAR)

    img_tensor = transforms.ToTensor()(img_resized)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_tensor = normalize(img_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = MODEL(img_tensor)

    depth_np = depth.squeeze().cpu().numpy()
    return img_resized, depth_np


def create_depth_image(depth_np):
    """Derinlik numpy array'ini renkli görsel olarak encode eder."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))
    vmax = np.percentile(depth_np, 95)
    ax.imshow(depth_np, cmap="magma", vmin=depth_np.min(), vmax=vmax)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def create_comparison_image(rgb_img, depth_np):
    """Yan yana karşılaştırma görseli oluşturur."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Girdi", fontsize=13, fontweight="bold", color="#e0e0e0")
    axes[0].axis("off")

    vmax = np.percentile(depth_np, 95)
    im = axes[1].imshow(depth_np, cmap="magma", vmin=depth_np.min(), vmax=vmax)
    axes[1].set_title("Tahmin Edilen Derinlik", fontsize=13, fontweight="bold", color="#e0e0e0")
    axes[1].axis("off")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Derinlik (m)", color="#e0e0e0")
    cbar.ax.yaxis.set_tick_params(color="#e0e0e0")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#e0e0e0")

    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if MODEL is None:
        return jsonify({"error": "Model henüz yüklenmedi!"}), 503

    if "image" not in request.files:
        return jsonify({"error": "Görüntü dosyası bulunamadı!"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi!"}), 400

    allowed_extensions = {"png", "jpg", "jpeg", "bmp", "webp", "tiff"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        return jsonify({"error": f"Desteklenmeyen dosya formatı: .{ext}"}), 400

    image_bytes = file.read()

    try:
        rgb_img, depth_np = predict(image_bytes)

        depth_b64 = create_depth_image(depth_np)
        comparison_b64 = create_comparison_image(rgb_img, depth_np)

        # İstatistikler
        stats = {
            "min_depth": f"{depth_np.min():.3f}",
            "max_depth": f"{depth_np.max():.3f}",
            "mean_depth": f"{depth_np.mean():.3f}",
            "std_depth": f"{depth_np.std():.3f}",
        }

        return jsonify({
            "depth_image": depth_b64,
            "comparison_image": comparison_b64,
            "stats": stats,
            "filename": file.filename,
        })

    except Exception as e:
        return jsonify({"error": f"Tahmin hatası: {str(e)}"}), 500


# ─── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model_once()
    print("\n🌐 Web arayüzü başlatılıyor: http://localhost:8080\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
