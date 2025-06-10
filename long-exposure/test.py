import cv2
import numpy as np


img = r'results\cross_section_aligned\cross_section_aligned_color_weighted.png'
long_exposure = cv2.imread(img).astype('float32')
fg_mask = cv2.imread(r'cross_section_aligned_fg.png') / 255.0

long_exposure_img = long_exposure * fg_mask

long_exposure_img = cv2.cvtColor(long_exposure_img.astype('uint8'), cv2.COLOR_BGR2HLS)
H, L, S = cv2.split(long_exposure_img.astype('float32'))
S = np.power(S / 255, 1/1.5) * 255
# L = np.power(L / 255, 1.1) * 255

H = np.clip(H, 0, 180)
L = np.clip(L, 0, 255)
S = np.clip(S, 0, 255)
long_exposure_img = cv2.merge([H, L, S])
long_exposure_img = cv2.cvtColor(long_exposure_img.astype('uint8'),
                                 cv2.COLOR_HLS2BGR).astype('float32')

long_exposure_img += long_exposure * (1 - fg_mask)

cv2.imwrite(f"{img.replace('.png', '')}_long_exposure.png", long_exposure_img.astype('uint8'))