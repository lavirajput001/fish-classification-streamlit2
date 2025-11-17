import cv2
import numpy as np

def uwie_enhance(img):
    # LAB white balance + CLAHE on V channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    result = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return result
