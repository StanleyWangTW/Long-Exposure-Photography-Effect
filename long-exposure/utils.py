import numpy as np


def srgb_to_linear(img_srgb):
    img_srgb = img_srgb / 255.0
    threshold = 0.04045
    below = img_srgb <= threshold
    above = img_srgb > threshold
    img_linear = np.zeros_like(img_srgb)
    img_linear[below] = img_srgb[below] / 12.92
    img_linear[above] = ((img_srgb[above] + 0.055) / 1.055) ** 2.4
    return img_linear


def linear_to_srgb(img_linear):
    threshold = 0.0031308
    below = img_linear <= threshold
    above = img_linear > threshold
    img_srgb = np.zeros_like(img_linear)
    img_srgb[below] = img_linear[below] * 12.92
    img_srgb[above] = 1.055 * (img_linear[above] ** (1 / 2.4)) - 0.055
    return np.clip(img_srgb * 255.0, 0, 255).astype(np.uint8)