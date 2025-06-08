import cv2
from src.camera_simulation import simulate_camera_response

# 打開影片
cap = cv2.VideoCapture("test_videos/road2.mp4")

frames = []
count = 0
MAX_FRAMES = 30  # 你可以調整要取幾幀

while count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    count += 1

cap.release()

# 執行 camera 模擬
#
output_img = simulate_camera_response(frames)

# 儲存結果
cv2.imwrite("output_long_exposure.png", output_img)
print("儲存結果至 output_long_exposure.png")
