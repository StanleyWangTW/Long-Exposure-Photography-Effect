'''Output a video whose each frame is foreground mask of a video using ViBe (Video Backgound Extraction)).'''

import numpy as np
import cupy as cp
import cv2
from tqdm import tqdm

from utils import srgb_to_linear, linear_to_srgb


class ViBe:
    def __init__(self, input_video, num_samples=20, min_matches=2, radius=20, subsample_factor=16):
        self.input_video = input_video
        self.num_samples = num_samples
        self.min_matches = min_matches
        self.radius = radius
        self.subsample_factor = subsample_factor
        self.samples = None

    def initialize(self, first_frame):
        h, w = first_frame.shape
        self.samples = np.zeros((self.num_samples, h, w), dtype=np.uint8)
        for i in range(self.num_samples):
            self.samples[i] = np.roll(first_frame, np.random.randint(-1, 2), axis=0)
            self.samples[i] = np.roll(self.samples[i], np.random.randint(-1, 2), axis=1)

    def update(self, frame):
        h, w = frame.shape
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        matches = np.zeros((h, w), dtype=np.uint8)

        for i in range(self.num_samples):
            dist = np.abs(self.samples[i].astype(np.int16) - frame.astype(np.int16))
            matches += (dist < self.radius).astype(np.uint8)

        fg_mask[matches < self.min_matches] = 255

        rand_mask = np.random.randint(0, self.subsample_factor, (h, w))
        update_mask = (rand_mask == 0)

        for i in range(self.num_samples):
            rand_choice = np.random.randint(0, self.num_samples)
            self.samples[rand_choice][update_mask] = frame[update_mask]

        return fg_mask
    
    def get_mask_video(self):
        output_video = self.input_video.replace('.mp4', '_fg.mp4')

        cap = cv2.VideoCapture(self.input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.initialize(gray)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS), (gray.shape[1], gray.shape[0]), isColor=False)

        print(f"ViBe: Segmenting video {self.input_video}")
        for _ in tqdm(range(total_frames-1)):
            _, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = self.update(gray)

            out.write(fg_mask.astype('uint8'))

        print(f'ViBe processing complete. Output saved to: {output_video}')
        cap.release()
        out.release()

    def get_mask(self):
        output_image_path = self.input_video.replace('.mp4', '_fg.png')
        p = 2.5

        cap = cv2.VideoCapture(self.input_video.replace('.mp4', '_fg.mp4'))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('Creating final output mask from ViBe output video...')
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

                b_curr, g_curr, r_curr = cp.asarray(b_curr), cp.asarray(g_curr), cp.asarray(r_curr)

                b += cp.power(b_curr, p)
                g += cp.power(g_curr, p)
                r += cp.power(r_curr, p)

        cap.release()

        r = cp.power(r / total_frames, 1/p)
        g = cp.power(g / total_frames, 1/p)
        b = cp.power(b / total_frames, 1/p)

        r, g, b = cp.asnumpy(r), cp.asnumpy(g), cp.asnumpy(b)
        long_exposure_img = cv2.merge([b, g, r])
        fg_mask = linear_to_srgb(long_exposure_img).astype("uint8")

        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        cv2.imwrite(output_image_path, fg_mask)
        print(f"Foreground mask saved as {output_image_path}")

    def get_mask_path(self):
        return self.input_video.replace('.mp4', '_fg.png')


if __name__ == "__main__":
    vibe = ViBe(input_video=r'test_videos\cross_section_aligned.mp4')
    vibe.get_mask()