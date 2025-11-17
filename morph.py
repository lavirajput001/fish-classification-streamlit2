import cv2
import numpy as np

def get_largest_mask(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return binary

    c = max(contours, key=cv2.contourArea)

    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask
