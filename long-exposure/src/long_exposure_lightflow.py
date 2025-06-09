import logging
import os
import numpy as np
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def srgb_to_linear(img_srgb):
    """Convert an sRGB image (uint8 or float) to linear RGB (float32)"""
    img_srgb = img_srgb.astype(np.float32) / 255.0
    threshold = 0.04045
    below = img_srgb <= threshold
    above = img_srgb > threshold

    img_linear = np.zeros_like(img_srgb)
    img_linear[below] = img_srgb[below] / 12.92
    img_linear[above] = ((img_srgb[above] + 0.055) / 1.055) ** 2.4

    return img_linear

def linear_to_srgb(img_linear):
    """Convert a linear RGB image to sRGB"""
    threshold = 0.0031308
    below = img_linear <= threshold
    above = img_linear > threshold

    img_srgb = np.zeros_like(img_linear)
    img_srgb[below] = img_linear[below] * 12.92
    img_srgb[above] = 1.055 * (img_linear[above] ** (1 / 2.2)) - 0.055

    return np.clip(img_srgb * 255.0, 0, 255).astype(np.uint8)


class LongExposure:
    def __init__(self, video, output_image_path, step=1, method='max'):
        self.video = video
        self.output_image_path = output_image_path
        self.step = step
        self.method = method

    def max_pool2d(self, img, size=2, stride=1):
        h, w = img.shape
        out_h = (h - size) // stride + 1
        out_w = (w - size) // stride + 1
        pooled = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                window = img[i*stride:i*stride+size, j*stride:j*stride+size]
                pooled[i, j] = np.max(window)
        return pooled

    def frame_stack(self):
        logging.info("Processing video %r with step %r", self.video, self.step)

        stream = cv2.VideoCapture(self.video)
        fps = stream.get(cv2.CAP_PROP_FPS)
        print(f'fps: {fps}')
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

        r, g, b = None, None, None
        p = 45

        os.makedirs("frames", exist_ok=True)
        for count in tqdm(range(total_frames)):
            success, frame = stream.read()
            if not success:
                break

            if count % self.step == 0:
                cv2.imwrite(f"frames/frame_{count:04d}.jpg", frame)
                frame_linear = srgb_to_linear(frame)
                b_curr, g_curr, r_curr = cv2.split(frame_linear)

                if r is None or g is None or b is None:
                    r = np.zeros_like(r_curr)
                    g = np.zeros_like(g_curr)
                    b = np.zeros_like(b_curr)

                if self.method == 'max':
                    r = np.maximum(r, r_curr)
                    g = np.maximum(g, g_curr)
                    b = np.maximum(b, b_curr)

                elif self.method == 'avg':
                    r += np.power(r_curr, p) / (total_frames // self.step)
                    g += np.power(g_curr, p) / (total_frames // self.step)
                    b += np.power(b_curr, p) / (total_frames // self.step)

        if self.method == 'avg':
            r = np.power(r, 1/p)
            g = np.power(g, 1/p)
            b = np.power(b, 1/p)

        long_exposure_linear = cv2.merge([b, g, r])
        long_exposure_srgb = linear_to_srgb(long_exposure_linear)

        logging.info("Saving image as %r", self.output_image_path)
        cv2.imwrite(self.output_image_path, long_exposure_srgb)
        stream.release()

    def img_post_process(self):
        img = cv2.imread(self.output_image_path)
        b, g, r = cv2.split(img.astype("float"))
        r = self.max_pool2d(r, size=7, stride=1)
        g = self.max_pool2d(g, size=7, stride=1)
        b = self.max_pool2d(b, size=7, stride=1)
        out_img = cv2.merge([b, g, r]).astype("uint8")
        cv2.imwrite(self.output_image_path, out_img)

    def __call__(self):
        self.frame_stack()
        # self.img_post_process()


if __name__ == "__main__":
    test_dir = r'C:\Users\tanks\OneDrive - NTHU\桌面\攝影學期末'
    method = 'avg'  # or 'max'
    video_path = os.path.join(test_dir, "aligned_video.mp4")
    output_image_path = os.path.basename(video_path).replace(".mp4", f"_{method}.png")

    long_exposure = LongExposure(video_path, output_image_path, step=1, method=method)
    long_exposure()

