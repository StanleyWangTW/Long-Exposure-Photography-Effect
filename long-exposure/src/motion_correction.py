import os
import cv2
import numpy as np
from tqdm import tqdm

def align_images(input_dir, output_dir, reference_img_name):
    os.makedirs(output_dir, exist_ok=True)

    # 讀取基準圖
    ref_path = os.path.join(input_dir, reference_img_name)
    ref_img = cv2.imread(ref_path)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # 初始化 SIFT
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)

    # 讀取所有圖片名稱
    filenames = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])

    for fname in tqdm(filenames):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        # 跳過基準圖（因為本身不需要轉換）
        if fname == reference_img_name:
            cv2.imwrite(output_path, ref_img)
            continue

        img = cv2.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des = sift.detectAndCompute(gray, None)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des, des_ref)

        if len(matches) < 4:
            print(f"[警告] {fname} 匹配點不足，跳過")
            continue

        # 建立 homography
        src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        aligned = cv2.warpPerspective(img, H, (ref_img.shape[1], ref_img.shape[0]))
        cv2.imwrite(output_path, aligned)

    print("所有圖片已對齊並儲存至：", output_dir)

# 使用範例
input_folder = "frames"
output_folder = "align_frames"
reference_image = "frame_0000.jpg"

align_images(input_folder, output_folder, reference_image)

def images_to_video(image_folder, output_video_path, fps=30):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        print("❌ 找不到圖片")
        return

    # 讀取第一張圖片取得尺寸
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape

    # 建立影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可改為 'XVID'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_name in images:
        img_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)
        else:
            print(f"⚠️ 無法讀取：{img_path}")

    video_writer.release()
    print(f"✅ 影片儲存至：{output_video_path}")

# 使用範例
input_image_dir = "align_frames"
output_video_file = "aligned_video.mp4"
images_to_video(input_image_dir, output_video_file, fps=30)

