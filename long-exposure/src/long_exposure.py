import os
from os.path import join, basename

import numpy as np
import cupy as cp
import cv2
from tqdm import tqdm


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


class LongExposure:
    def __init__(self, video, output_image_path, method='weighted', secs=None):
        self.video = video
        self.output_image_path = output_image_path
        self.method = method
        self.secs = secs

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
        ''' Process the video file to create a long exposure image by frames.'''
        print(f"Processing video {self.video}")

        cap = cv2.VideoCapture(self.video)
        FPS = round(cap.get(cv2.CAP_PROP_FPS))
        print(f'FPS: {FPS}')
        
        if self.secs is not None:
            total_frames = int(round(FPS * self.secs))
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for count in tqdm(range(total_frames)):
            _, frame = cap.read()
            frame = srgb_to_linear(frame.astype("float"))

            if frame is not None:
                # current frame RGB
                b_curr, g_curr, r_curr = cv2.split(frame.astype("float"))

                # If first frame, initialize arrays
                if count == 0:
                    r = cp.zeros_like(r_curr, dtype="float")
                    g = cp.zeros_like(g_curr, dtype="float")
                    b = cp.zeros_like(r_curr, dtype="float")

                    if self.method == 'weighted':
                        r_w = cp.zeros(r_curr.shape, dtype="float")
                        g_w = cp.zeros(g_curr.shape, dtype="float")
                        b_w = cp.zeros(r_curr.shape, dtype="float")

                b_curr, g_curr, r_curr = cp.asarray(b_curr), cp.asarray(g_curr), cp.asarray(r_curr)

                if self.method == 'max':
                    b = cp.maximum(b, b_curr)
                    g = cp.maximum(g, g_curr)
                    r = cp.maximum(r, r_curr)

                elif self.method == 'avg':
                    b += b_curr 
                    g += g_curr
                    r += r_curr

                elif self.method == 'weighted':
                    b += np.power(b_curr, 4)
                    g += np.power(g_curr, 4)
                    r += np.power(r_curr, 4)

                    r_w += np.power(r_curr, 3)
                    g_w += np.power(g_curr, 3)
                    b_w += np.power(b_curr, 3)

        cap.release()

        if self.method == 'avg':
            r /= total_frames
            g /= total_frames
            b /= total_frames

        elif self.method == 'weighted':
            r[r_w > 0] /= r_w[r_w > 0]
            g[g_w > 0] /= g_w[g_w > 0]
            b[b_w > 0] /= b_w[b_w > 0]

        r, g, b = cp.asnumpy(r), cp.asnumpy(g), cp.asnumpy(b)
        long_exposure_img = cv2.merge([b, g, r])
        long_exposure_img = linear_to_srgb(long_exposure_img).astype("uint8")

        print(f"Saving image as {self.output_image_path}")
        cv2.imwrite(self.output_image_path, long_exposure_img.astype("uint8"))

    def fg_mask_process(self):
        ''' Post-process the long exposure image'''
        fg_mask = cv2.imread(self.output_image_path)
        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        cv2.imwrite(self.output_image_path, fg_mask)

    def __call__(self):
        self.frame_stack()


if __name__ == "__main__":
    test_dir = r'E:\計算攝影學\Long-Exposure-Photography-Effect\test_videos'

    method = 'weighted'

    video_path = os.path.join(test_dir, 'road1_aligned.mp4')

    out_dir = join('results', basename(video_path).split('.')[0])
    os.makedirs(out_dir, exist_ok=True)

    output_image_path = join(out_dir, basename(video_path).replace(".mp4", f"_{method}.png"))

    long_exposure = LongExposure(video_path, output_image_path, method)
    long_exposure()
