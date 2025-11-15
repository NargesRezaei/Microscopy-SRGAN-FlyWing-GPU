# Microscopy Image Super-Resolution using SRGAN
### TensorFlow 2.x • NVIDIA RTX 3060 • VGG19 Perceptual Loss • Custom Augmentation Pipeline

---

## 🔬 Overview

This repository implements a Super-Resolution Generative Adversarial Network (**SRGAN**) 
designed to enhance low-resolution bright-field microscopy images of fly wing (32×32) to high-resolution (128×128).  
The full workflow includes:

- Raw image patch extraction  
- Rotation & flip augmentations  
- LR/HR paired dataset generation  
- GPU-accelerated SRGAN training (Generator + Discriminator)  
- Perceptual loss using VGG19 features  
- Evaluation with PSNR / SSIM   

This work was partially presented in a conference poster (included below).

---

## 🐝 Why Fly Wing Images?

Fly wings were selected as the microscopy target because they naturally contain both 
**fine microstructures** (hair-like ridges, thin membrane textures) and **coarse macro-structures** (veins and large geometric patterns).  
This combination makes them an ideal testbed for super-resolution research.

Using fly wings allowed us to evaluate whether SRGAN can recover both:
- **High-frequency details** (texture, thin edges)
- **Large-scale structure** (vein geometry, global shape)

In practice, the fly wing dataset provides a rich balance of detail and structure, making it a strong candidate for validating microscopy super-resolution models.

---

## 🧬 Network Architecture (Farsi Diagram)

Below is the diagram of the SRGAN generator & discriminator architecture used in this project:

![Network Architecture](examples/network_arch_fa.png)

---

## 🧪 Example Results

A real microscopy example (32×32 → 128×128):

| Low-Resolution | SRGAN Output | Ground Truth |
|----------------|--------------|--------------|
| ![](examples/example_lr.png) | ![](examples/example_sr.png) | ![](examples/example_hr.png) |

The SRGAN successfully reconstructs edge continuity, fine texture, and structural details from heavily downsampled inputs.

---

## 📊 Quantitative Results

| Metric | Value |
|--------|-------|
| **PSNR** | 29.40 dB |
| **SSIM** | 0.85 |

---

## ⚡ GPU Acceleration

Training was performed on:

- **NVIDIA RTX 3060 (12 GB)**  
- TensorFlow 2.x  
- `tf.distribute.MirroredStrategy` for multi-GPU compatibility  

```python
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
strategy = tf.distribute.MirroredStrategy()
```

SRGAN training on CPU is not practical — GPU is strongly recommended.

---

## 🛠️ Preprocessing Pipeline

All preprocessing steps are implemented in:

```text
src/preprocess_patches.py
```

### Steps:
- Extract **128×128** HR patches from raw microscopy images  
- Apply augmentation:  
  - 90°, 180°, 270° rotations  
  - Vertical flips  
- Generate **32×32** LR patches by downsampling  
- Save paired LR/HR images for SRGAN training  

---

## 📁 Project Structure

```text
microscopy-super-resolution-tensorflow/
│
├── src/
│   ├── preprocess_patches.py      # Patch extraction + augmentation
│   ├── dataset.py                 # LR/HR loader
│   ├── models_srgan.py            # Generator + Discriminator + VGG extractor
│   ├── train_sr.py                # Full SRGAN training loop
│   └── metrics_eval.py            # PSNR, SSIM, MSE metrics
│
├── examples/
│   ├── example_lr.png             # LR example
│   ├── example_sr.png             # SRGAN output
│   ├── example_hr.png             # HR example
│   └── network_arch_fa.png        # Farsi architecture diagram
│
└── data/
    ├── raw/                       # Raw microscopy images (not included)
    ├── patches_hr/                # HR patches (generated locally)
    └── patches_lr32/              # LR patches (generated locally)
```

---

## 🚀 Training Instructions

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run preprocessing (patch extraction + augmentation):
```bash
python -m src.preprocess_patches
```

### Train SRGAN:
```bash
python -m src.train_sr
```

Trained generator models are saved in:

```text
checkpoints/generator_epoch_X.h5
```

---

## 📊 Evaluation

Evaluation metrics (PSNR, SSIM, MSE) are implemented in:

```text
src/metrics_eval.py
```

### Example usage:
```python
from src.metrics_eval import compare_images

scores = compare_images(sr_image, hr_image)
print(scores)   # {'psnr':..., 'mse':..., 'ssim':...}
```

---

## 📝 Conference Poster (Farsi)

A Farsi poster presentation of this work is included:

```text
sr_gan_poster.pdf
```
---

## 💡 Acknowledgements

This project uses TensorFlow, Keras, OpenCV, scikit-image, NumPy, and CUDA-enabled GPU computation.
