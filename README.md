# Monoküler Derinlik Tahmini (CNN)

> Evrişimsel Sinir Ağları ile Tek Görüntüden Derinlik Kestirimi

## 📌 Proje Açıklaması

Bu proje, tek bir RGB görüntüden piksel bazında derinlik haritası tahmin eden bir derin öğrenme modelidir. ResNet-18 encoder ve U-Net tarzı decoder mimarisi kullanılmaktadır.

## 🏗️ Mimari

```
RGB Görüntü (3×256×320) → ResNet-18 Encoder → U-Net Decoder → Derinlik Haritası (1×256×320)
```

## ⚙️ Kurulum

```bash
python3 -m venv depth_env
source depth_env/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

## 📦 Dataset

NYU Depth V2 veri setini indirin ve `data/nyu_depth_v2/` altına yerleştirin:

```
data/nyu_depth_v2/
├── train/
│   ├── rgb/
│   └── depth/
├── val/
│   ├── rgb/
│   └── depth/
└── test/
    ├── rgb/
    └── depth/
```

## 🚀 Kullanım

### Eğitim
```bash
python train.py
```

### Test (tek görüntü)
```bash
python test.py --checkpoint checkpoints/best_model.pth --image test_image.jpg
```

### Değerlendirme
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

### TensorBoard
```bash
tensorboard --logdir=logs/
```

## 📊 Sonuçlar

| Metrik | Değer |
|--------|-------|
| Abs Rel ↓ | 0.0772 |
| RMSE ↓ | - |
| δ < 1.25 ↑ | % 94.0 |

## 🛠️ Teknolojiler

- Python 3.10
- PyTorch
- ResNet-18 (ImageNet pretrained)
- Apple MPS / NVIDIA CUDA uyumlu
