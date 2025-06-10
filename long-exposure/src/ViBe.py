import numpy as np
import cv2
from tqdm import tqdm

'''Output a video whose each frame is foreground mask of a video using ViBe (Video Backgound Extraction)).'''

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

        print(f"Processing video {self.input_video} with {total_frames} frames")
        for _ in tqdm(range(total_frames-1)):
            _, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = self.update(gray)

            out.write(fg_mask.astype('uint8'))

        print(f'Video processing complete. Output saved to: {output_video}')
        cap.release()
        out.release()


if __name__ == "__main__":
    vibe = ViBe(input_video=r'test_videos\cross_section.mp4')
    vibe.get_mask_video()