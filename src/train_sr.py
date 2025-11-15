# src/train_sr.py

import os
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from .models_srgan import create_generator, create_discriminator, build_vgg_feature_extractor
from .dataset import load_lr_hr_pairs


def show_random_pair(lr_images, hr_images):
    """
    Show a random LR/HR pair for sanity check.
    """
    idx = random.randint(0, len(lr_images) - 1)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Low-resolution")
    plt.imshow(np.clip(lr_images[idx], 0, 1))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("High-resolution")
    plt.imshow(np.clip(hr_images[idx], 0, 1))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main(
    lr_dir,
    hr_dir,
    epochs=100,
    batch_size=1,
    max_images=None,
    save_dir="checkpoints",
):
    """
    Main training loop for SRGAN on microscopy patches.
    """
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

    lr_images, hr_images = load_lr_hr_pairs(lr_dir, hr_dir, max_images=max_images)
    print(f"Loaded {len(lr_images)} LR/HR pairs.")

    if len(lr_images) == 0:
        print("No image pairs found. Check your data directories.")
        return

    hr_shape = hr_images.shape[1:]  # (H_hr, W_hr, C)
    lr_shape = lr_images.shape[1:]  # (H_lr, W_lr, C)

    lr_dataset = tf.data.Dataset.from_tensor_slices(lr_images).batch(batch_size)
    hr_dataset = tf.data.Dataset.from_tensor_slices(hr_images).batch(batch_size)

    os.makedirs(save_dir, exist_ok=True)

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    print("Number of replicas in strategy:", strategy.num_replicas_in_sync)

    with strategy.scope():
        generator = create_generator(lr_shape, num_res_blocks=16)
        discriminator = create_discriminator(hr_shape)
        vgg = build_vgg_feature_extractor(hr_shape)

        print(generator.summary())
        print(discriminator.summary())
        print(vgg.summary())

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        mse = tf.keras.losses.MeanSquaredError()

        def disc_loss(real_out, fake_out):
            """
            Discriminator loss: real->1, fake->0.
            """
            real_loss = bce(tf.ones_like(real_out), real_out)
            fake_loss = bce(tf.zeros_like(fake_out), fake_out)
            return real_loss + fake_loss

        def gen_loss(hr_img, gen_img, fake_out):
            """
            Generator loss = adversarial loss + perceptual (VGG) loss.
            """
            adv_loss = bce(tf.ones_like(fake_out), fake_out)
            content_loss = mse(vgg(hr_img), vgg(gen_img))
            total = 1e-3 * adv_loss + content_loss
            return total

        gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        disc_optimizer = tf.keras.optimizers.Adam(1e-4)

        @tf.function
        def train_step(lr_batch, hr_batch):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_img = generator(lr_batch, training=True)

                real_out = discriminator(hr_batch, training=True)
                fake_out = discriminator(gen_img, training=True)

                g_loss = gen_loss(hr_batch, gen_img, fake_out)
                d_loss = disc_loss(real_out, fake_out)

            gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
            disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

            return g_loss, d_loss

        for epoch in range(epochs):
            print(f"\nStart epoch {epoch + 1}/{epochs}")
            start_time = time.time()

            g_losses = []
            d_losses = []

            for lr_batch, hr_batch in zip(lr_dataset, hr_dataset):
                g_loss, d_loss = train_step(lr_batch, hr_batch)
                g_losses.append(g_loss.numpy())
                d_losses.append(d_loss.numpy())

            print(
                f"Epoch {epoch + 1} finished in {time.time() - start_time:.2f}s "
                f"- G loss: {np.mean(g_losses):.4f}, D loss: {np.mean(d_losses):.4f}"
            )

            gen_path = os.path.join(save_dir, f"generator_epoch_{epoch + 1}.h5")
            generator.save(gen_path)
            print(f"Saved generator to: {gen_path}")


if __name__ == "__main__":
    lr_folder = "data/patches_lr32"
    hr_folder = "data/patches_hr"

    main(
        lr_dir=lr_folder,
        hr_dir=hr_folder,
        epochs=100,
        batch_size=1,
        max_images=None,
        save_dir="checkpoints",
    )
