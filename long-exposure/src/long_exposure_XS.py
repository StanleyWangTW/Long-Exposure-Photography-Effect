import logging
import os

import numpy as np
import cupy as cp
import cv2
from tqdm import tqdm

from camera_simulation import srgb_to_linear, linear_to_srgb


logging.basicConfig(level=logging.INFO)

class LongExposure:
    def __init__(self, video, output_image_path, step=1, p=15):
        self.video = video
        self.output_image_path = output_image_path
        self.step = step
        self.p = p

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
        logging.info("Processing video %r with step %r", self.video, self.step)

        # Open a pointer to the video file
        stream = cv2.VideoCapture(self.video)
        fps = stream.get(cv2.CAP_PROP_FPS)
        print(f'fps: {fps}')

        # Get the total frames to be used by the progress bar
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        secs = 4
        total_frames = int(round(fps * secs))

        for count in tqdm(range(total_frames)):
            # Split the frame into its respective channels
            _, frame = stream.read()
            frame = srgb_to_linear(frame.astype("float"))

            if count % self.step == 0 and frame is not None:
                # Get the current RGB
                b_curr, g_curr, r_curr = cv2.split(frame.astype("float"))

                # If the first frame, initialize the RGB arrays
                if count == 0:
                    r = cp.zeros_like(r_curr, dtype="float")
                    g = cp.zeros_like(g_curr, dtype="float")
                    b = cp.zeros_like(r_curr, dtype="float")

                b_curr, g_curr, r_curr = cp.asarray(b_curr), cp.asarray(g_curr), cp.asarray(r_curr)

                p = 15
                # Use average pooling to accumulate the values
                b += cp.power(b_curr, self.p) / (total_frames // self.step)
                g += cp.power(g_curr, self.p) / (total_frames // self.step)
                r += cp.power(r_curr, self.p) / (total_frames // self.step)

        r, g, b = cp.power(r, 1/self.p), cp.power(g, 1/self.p), cp.power(b, 1/self.p)

        r, g, b = cp.clip(r, 0, 255), cp.clip(g, 0, 255), cp.clip(b, 0, 255)
        r, g, b = cp.asnumpy(r), cp.asnumpy(g), cp.asnumpy(b)
        long_exposure_img = cv2.merge([b, g, r])
        long_exposure_img = linear_to_srgb(long_exposure_img)

        logging.info("Saving image as %r", self.output_image_path)
        cv2.imwrite(self.output_image_path, long_exposure_img.astype("uint8"))

        # Release the stream pointer
        stream.release()
    
    def img_post_process(self):
        ''' Post-process the long exposure image'''
        img = cv2.imread(self.output_image_path)

        b, g, r = cv2.split(img.astype("float"))

        r = self.max_pool2d(r, size=7, stride=1)
        g = self.max_pool2d(g, size=7, stride=1)
        b = self.max_pool2d(b, size=7, stride=1)

        out_img = cv2.merge([b, g, r]).astype("uint8")
        cv2.imwrite(output_image_path, out_img)

    def __call__(self):
        self.frame_stack()
        # self.img_post_process()


if __name__ == "__main__":
    test_dir = r'E:\計算攝影學\Long-Exposure-Photography-Effect\test_videos'

    video_path = os.path.join(test_dir, "cross_section_aligned.mp4")
    output_image_path = os.path.basename(video_path).replace(".mp4", f".png")
    
    long_exposure = LongExposure(video_path, output_image_path, step=1)
    long_exposure()
