import logging
import os
import numpy as np
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def srgb_to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB image (uint8 or float) to linear RGB (float32)."""
    img = img_srgb.astype(np.float32) / 255.0
    th = 0.04045
    below = img <= th
    lin = np.zeros_like(img)
    lin[below] = img[below] / 12.92
    lin[~below] = ((img[~below] + 0.055) / 1.055) ** 2.4
    return lin


def linear_to_srgb(img_linear: np.ndarray) -> np.ndarray:
    """Convert a linear RGB image to sRGB (uint8)."""
    th = 0.0031308
    below = img_linear <= th
    srgb = np.zeros_like(img_linear)
    srgb[below] = img_linear[below] * 12.92
    srgb[~below] = 1.055 * (img_linear[~below] ** (1 / 2.4)) - 0.055
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


class LongExposure:
    def __init__(self, video_path, output_image_path, step=1, method='max'):
        """
        method: 'max', 'avg', 'pnorm', or 'weighted'
        """
        self.video = video_path
        self.output = output_image_path
        self.step = step
        self.method = method

    def frame_stack(self):
        logging.info("Processing video %r with step %r, method=%r", 
                     self.video, self.step, self.method)
        cap = cv2.VideoCapture(self.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # buffers
        r_acc = g_acc = b_acc = None
        w_acc = None  # for weighted
        p = 15        # for pnorm

        count = 0
        for i in tqdm(range(total)):
            ret, frame = cap.read()
            if not ret:
                break
            if i % self.step != 0:
                continue

            # convert to linear domain
            lin = srgb_to_linear(frame)
            b_lin, g_lin, r_lin = cv2.split(lin)

            if r_acc is None:
                # init accumulators
                shape = r_lin.shape
                r_acc = np.zeros(shape, np.float32)
                g_acc = np.zeros(shape, np.float32)
                b_acc = np.zeros(shape, np.float32)
                if self.method == 'weighted':
                    w_acc = np.zeros(shape, np.float32)

            if self.method == 'max':
                r_acc = np.maximum(r_acc, r_lin)
                g_acc = np.maximum(g_acc, g_lin)
                b_acc = np.maximum(b_acc, b_lin)

            elif self.method == 'avg':
                # linear average
                r_acc += r_lin
                g_acc += g_lin
                b_acc += b_lin

            elif self.method == 'pnorm':
                # p-norm composite
                r_acc += np.power(r_lin, p)
                g_acc += np.power(g_lin, p)
                b_acc += np.power(b_lin, p)

            elif self.method == 'weighted':
                # brightness as weight
                L = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
                # amplify
                w = np.power(L, 3.0)
                r_acc += r_lin * w
                g_acc += g_lin * w
                b_acc += b_lin * w
                w_acc += w

            else:
                raise ValueError(f"Unknown method: {self.method}")

            count += 1

        cap.release()

        # finalize
        if self.method == 'avg':
            r_acc /= count
            g_acc /= count
            b_acc /= count

        elif self.method == 'pnorm':
            r_acc = np.power(r_acc / count, 1/p)
            g_acc = np.power(g_acc / count, 1/p)
            b_acc = np.power(b_acc / count, 1/p)

        elif self.method == 'weighted':
            # avoid division by zero
            mask = w_acc > 0
            r_acc[mask] /= w_acc[mask]
            g_acc[mask] /= w_acc[mask]
            b_acc[mask] /= w_acc[mask]
            # zero where no weight: remains black

        # merge back
        lin_out = cv2.merge([b_acc, g_acc, r_acc])
        self.linear_out = lin_out  # for debug if needed

        # gamma-correct back to sRGB
        srgb_out = linear_to_srgb(lin_out)
        logging.info("Saving result to %r", self.output)
        cv2.imwrite(self.output, srgb_out)

    def __call__(self):
        self.frame_stack()


if __name__ == "__main__":
    # Example usage: 修改路徑與 method
    video = r"C:\long_exposure\Long-Exposure-Photography-Effect\test_videos\road2_200_video.mp4"
    out = os.path.basename(video).replace(".mp4", "_weighted.png")
    le = LongExposure(video, out, step=1, method='weighted')
    le()