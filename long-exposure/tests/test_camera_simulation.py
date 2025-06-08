import numpy as np
from src.camera_simulation import simulate_camera_response

def test_avg_mode_output_gray():
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 60
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 180
    result = simulate_camera_response([img1, img2], mode="average")
    pixel = result[0, 0, 0]
    assert 110 <= pixel <= 130, "Average mode pixel not in expected range"

def test_sum_mode_clip():
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 200
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 200
    result = simulate_camera_response([img1, img2], mode="sum")
    pixel = result[0, 0, 0]
    assert pixel == 255, "Sum mode did not clip correctly"

def test_max_mode_value():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 150
    result = simulate_camera_response([img1, img2], mode="max")
    pixel = result[0, 0, 0]
    assert pixel == 150, "Max mode failed to select max value"
