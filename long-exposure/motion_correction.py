'''Aligns video frames using SIFT feature matching and homography estimation.'''

import cv2
import numpy as np
from tqdm import tqdm


def align_video(input_video, output_video):
    # take frame at (# of frames // 2) as reference image
    cap = cv2.VideoCapture(input_video)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ref_gray = None
    for frame_idx in range(TOTAL_FRAMES):
        _, frame = cap.read()
        if frame_idx == TOTAL_FRAMES // 2:
            ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            break
    cap.release()

    # create video writer
    height, width = ref_gray.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, FPS, (width, height))

    # initialize SIFT
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)

    # read video and align frames
    print(f"Aligning video frames of {input_video}")
    cap = cv2.VideoCapture(input_video)
    for fname_idx in tqdm(range(TOTAL_FRAMES)):
        # skip reference image
        if fname_idx == TOTAL_FRAMES // 2:
            video_writer.write(frame)
            continue

        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des = sift.detectAndCompute(gray, None)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des, des_ref)

        if len(matches) < 4:
            print(f"Not enough matches found for frame {fname_idx}. Skipping alignment.")
            continue

        # create homography
        src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        aligned = cv2.warpPerspective(img, H, (ref_gray.shape[1], ref_gray.shape[0]))
        video_writer.write(aligned)

    cap.release()
    video_writer.release()

    print(f"Aligned video saved to {output_video}")

if __name__ == "__main__":
    input_video = r'test_videos\road1.mp4'
    align_video(input_video, input_video.replace('.mp4', '_aligned.mp4'))