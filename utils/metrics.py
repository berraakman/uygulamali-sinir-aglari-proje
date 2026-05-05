"""
Derinlik Tahmin Metrikleri
Standart monoküler derinlik değerlendirme metrikleri.
Referans: Eigen et al. 2014
"""

import numpy as np


def compute_depth_metrics(gt, pred, min_depth=0.1, max_depth=10.0):
    """
    Ground truth ve tahmin derinlik haritaları arasındaki hata metriklerini hesaplar.
    
    Args:
        gt (np.ndarray): Ground truth derinlik
        pred (np.ndarray): Tahmin edilen derinlik
        min_depth (float): Minimum geçerli derinlik
        max_depth (float): Maximum geçerli derinlik
    
    Returns:
        dict: 7 standart derinlik metriği
    """
    # Geçerli pikselleri maskele
    mask = (gt > min_depth) & (gt < max_depth) & (pred > min_depth)
    gt = gt[mask]
    pred = pred[mask]

    if len(gt) == 0:
        return {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log",
                                   "delta_1", "delta_2", "delta_3"]}

    # Threshold accuracy (delta metrikleri)
    thresh = np.maximum(gt / pred, pred / gt)
    delta_1 = (thresh < 1.25).mean()
    delta_2 = (thresh < 1.25 ** 2).mean()
    delta_3 = (thresh < 1.25 ** 3).mean()

    # Hata metrikleri
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    return {
        "abs_rel": abs_rel,       # Absolute Relative Error ↓
        "sq_rel": sq_rel,         # Squared Relative Error ↓
        "rmse": rmse,             # Root Mean Squared Error ↓
        "rmse_log": rmse_log,     # RMSE (log space) ↓
        "delta_1": delta_1,       # δ < 1.25 ↑
        "delta_2": delta_2,       # δ < 1.25² ↑
        "delta_3": delta_3,       # δ < 1.25³ ↑
    }


def print_metrics(metrics):
    """Metrikleri güzel formatla yazdırır."""
    print("\n" + "=" * 65)
    print(f"  {'Metrik':<12} | {'Değer':>10} | {'Yön':>6} | {'Açıklama'}")
    print("-" * 65)
    for key, val in metrics.items():
        direction = "↓ iyi" if key in ["abs_rel", "sq_rel", "rmse", "rmse_log"] else "↑ iyi"
        labels = {
            "abs_rel": "Absolute Relative",
            "sq_rel": "Squared Relative",
            "rmse": "RMSE",
            "rmse_log": "RMSE (log)",
            "delta_1": "δ < 1.25",
            "delta_2": "δ < 1.25²",
            "delta_3": "δ < 1.25³",
        }
        print(f"  {key:<12} | {val:>10.4f} | {direction:>6} | {labels.get(key, '')}")
    print("=" * 65 + "\n")
