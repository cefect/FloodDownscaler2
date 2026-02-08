"""Model architectures for super-resolution experiments."""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    AveragePooling2D,
    Concatenate,
    Add,
)

RMSE_WET_THRESH = 0.5


 

def PSNR(high_resolution, super_resolution):
    """Compute PSNR for tensors in the [0,1] range."""
    return tf.image.psnr(high_resolution, super_resolution, max_val=1.0)


def SSIM(high_resolution, super_resolution, max_val=1.0):
    """Compute SSIM for tensors in the [0,1] range."""
    high_resolution = tf.cast(high_resolution, tf.float32)
    super_resolution = tf.cast(super_resolution, tf.float32)
    return tf.image.ssim(high_resolution, super_resolution, max_val=max_val)


def RMSE(high_resolution, super_resolution):
    """Compute RMSE for tensors in the [0,1] range."""
    high_resolution = tf.cast(high_resolution, tf.float32)
    super_resolution = tf.cast(super_resolution, tf.float32)
    diff = super_resolution - high_resolution
    return tf.sqrt(tf.reduce_mean(tf.square(diff)))


def RMSE_wet(high_resolution, super_resolution, threshold=RMSE_WET_THRESH):
    """Compute wet-pixel RMSE, excluding chips with zero wet pixels."""
    high_resolution = tf.cast(high_resolution, tf.float32)
    super_resolution = tf.cast(super_resolution, tf.float32)

    mask = tf.cast(high_resolution > threshold, tf.float32)
    diff_sq = tf.square(super_resolution - high_resolution) * mask

    wet_pixel_count = tf.reduce_sum(mask, axis=[1, 2, 3])
    mse_per_sample = tf.math.divide_no_nan(
        tf.reduce_sum(diff_sq, axis=[1, 2, 3]),
        wet_pixel_count,
    )
    rmse_per_sample = tf.sqrt(mse_per_sample)

    wet_samples = wet_pixel_count > 0.0
    rmse_sum = tf.reduce_sum(tf.where(wet_samples, rmse_per_sample, tf.zeros_like(rmse_per_sample)))
    wet_sample_total = tf.reduce_sum(tf.cast(wet_samples, tf.float32))
    return tf.math.divide_no_nan(rmse_sum, wet_sample_total)


def CSI(high_resolution, super_resolution, threshold=0.01):
    """Compute CSI at a given threshold (same scale as inputs)."""
    high_resolution = tf.cast(high_resolution, tf.float32)
    super_resolution = tf.cast(super_resolution, tf.float32)

    obs = tf.greater_equal(high_resolution, threshold)
    pred = tf.greater_equal(super_resolution, threshold)

    obs_f = tf.cast(obs, tf.float32)
    pred_f = tf.cast(pred, tf.float32)

    tp = tf.reduce_sum(pred_f * obs_f)
    fp = tf.reduce_sum(pred_f * (1.0 - obs_f))
    fn = tf.reduce_sum((1.0 - pred_f) * obs_f)
    denom = tp + fp + fn

    return tf.math.divide_no_nan(tp, denom)
