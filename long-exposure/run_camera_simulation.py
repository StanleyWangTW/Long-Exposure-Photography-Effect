import cv2
from src.camera_simulation import simulate_camera_response

cap = cv2.VideoCapture("test_videos/road2.mp4")
frames = []
MAX_FRAMES = 30

for _ in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

for mode in ["average", "sum", "max"]:
    output = simulate_camera_response(frames, mode=mode)
    cv2.imwrite(f"output_{mode}.png", output)
    print(f"Saved output_{mode}.png")