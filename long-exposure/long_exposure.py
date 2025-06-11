import os
from os.path import join, basename

import numpy as np
import cupy as cp
import cv2
from tqdm import tqdm

from utils import srgb_to_linear, linear_to_srgb
from motion_correction import align_video
from ViBe import ViBe


class LongExposure:
    def __init__(self, video, output_image_path, method='weighted', secs=None):
        self.video = video
        self.output_image_path = output_image_path
        self.method = method
        self.secs = secs

    def frame_stack(self):

        ''' Process the video file to create a long exposure image by frames.'''

        cap = cv2.VideoCapture(self.video)
        FPS = round(cap.get(cv2.CAP_PROP_FPS))
        print(f"Creating long-exposure image using {self.video} (FPS: {FPS}) ...")

        if self.secs is not None:
            total_frames = int(round(FPS * self.secs))
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Frame stacking method: {self.method}")
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

        print(f"Long-exposure image saved as {self.output_image_path}")
        cv2.imwrite(self.output_image_path, long_exposure_img.astype("uint8"))

    def post_photoshop(self, fg_mask):
        print(self.output_image_path)
        long_exposure = cv2.imread(self.output_image_path).astype('float32')
        fg_mask = cv2.imread(fg_mask) / 255.0

        moving_part = long_exposure * fg_mask

        # Convert to HLS and adjust saturation (or other photoshop process you want)
        moving_part = cv2.cvtColor(moving_part.astype('uint8'), cv2.COLOR_BGR2HLS)
        H, L, S = cv2.split(moving_part.astype('float32'))

        S = np.power(S / 255, 1/1.5) * 255

        H = np.clip(H, 0, 180)
        L = np.clip(L, 0, 255)
        S = np.clip(S, 0, 255)
        moving_part = cv2.merge([H, L, S])
        moving_part = cv2.cvtColor(moving_part.astype('uint8'),
                                   cv2.COLOR_HLS2BGR).astype('float32')


        long_exposure_img = moving_part + long_exposure * (1 - fg_mask)

        cv2.imwrite(f"{self.output_image_path.replace('.png', '')}_long_exposure.png", long_exposure_img.astype('uint8'))
        print(f"Post-photoshop image saved as {self.output_image_path.replace('.png', '')}_long_exposure.png")


    def __call__(self):
        self.frame_stack()


if __name__ == "__main__":
    test_dir = r'test_videos'

    method = 'weighted'
    align = True # Set to True if you want to align the video frames (motion correction) before processing
    light_trail_photoshop = True

    video_path = os.path.join(test_dir, 'spot_light2.mp4') # path of the input video file

    # output image path
    out_dir = join('results', basename(video_path).split('.')[0])
    os.makedirs(out_dir, exist_ok=True)
    output_image_path = join(out_dir, basename(video_path).replace(".mp4", f"_{method}.png"))

    # video motion correction
    if align:
        align_video(video_path, video_path.replace('.mp4', '_aligned.mp4'))
        video_path = video_path.replace('.mp4', '_aligned.mp4')

    # # long exposure image synthesis using video frames
    long_exposure = LongExposure(video_path, output_image_path, method)
    long_exposure()

    if light_trail_photoshop:
        # creating foreground mask using ViBe
        vibe = ViBe(video_path)
        vibe.get_mask_video()
        vibe.get_mask()

        # post-photoshop using foreground mask
        long_exposure.post_photoshop(vibe.get_mask_path())