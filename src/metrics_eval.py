# src/metrics_eval.py

import numpy as np
from skimage.metrics import (
    structural_similarity as ssim_metric,
    peak_signal_noise_ratio as psnr_metric,
    mean_squared_error as mse_metric,
)


def compare_images(target, ref):
    """
    Compute PSNR, MSE, and SSIM between two images (float in [0,1]).

    target: reconstructed / SR image
    ref: ground truth HR image

    Returns:
        dict with keys: 'psnr', 'mse', 'ssim'
    """
    target = np.clip(target.astype(np.float32), 0.0, 1.0)
    ref = np.clip(ref.astype(np.float32), 0.0, 1.0)

    psnr_val = psnr_metric(ref, target, data_range=1.0)
    mse_val = mse_metric(ref, target)
    ssim_val = ssim_metric(ref, target, data_range=1.0, channel_axis=-1)

    return {
        "psnr": psnr_val,
        "mse": mse_val,
        "ssim": ssim_val,
    }
