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

def simulate_camera_response(image_list: list[np.ndarray], mode="average") -> np.ndarray:
    if mode == "max":
        return np.max(np.stack(image_list, axis=0), axis=0).astype(np.uint8)

    linear_sum = None
    for img in image_list:
        linear = srgb_to_linear(img.astype(np.float32))
        if linear_sum is None:
            linear_sum = linear
        else:
            linear_sum += linear

    if mode == "average":
        result_linear = linear_sum / len(image_list)
    elif mode == "sum":
        result_linear = np.clip(linear_sum, 0, 1.0)
    else:
        raise ValueError("mode must be 'average', 'sum', or 'max'")

    return linear_to_srgb(result_linear)
