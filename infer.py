import cv2
import numpy as np
from uwie import uwie_enhance
from morph import get_largest_mask
from firefly import firefly_optimize_threshold
from model import load_model_and_encoder

model, label_encoder = load_model_and_encoder()

def predict_image_streamlit(img_rgb):

    img_bgr = img_rgb[:, :, ::-1]

    # Step 1: UWIE enhancement
    enhanced = uwie_enhance(img_bgr)

    # Step 2: Convert to gray
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Step 3: Firefly optimize threshold
    th = firefly_optimize_threshold(gray)

    _, binary = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    # Step 4: Morphological segmentation (largest contour)
    mask = get_largest_mask(binary)

    # Apply mask
    segmented = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    # Step 5: Crop ROI
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return segmented[:, :, ::-1], "Unknown", 0.0

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    roi = segmented[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (128, 128))

    # Step 6: Model prediction
    inp = roi_resized.astype("float32") / 255.0
    inp = np.expand_dims(inp, axis=0)

    pred = model.predict(inp)[0]
    idx = np.argmax(pred)
    label = label_encoder[idx]
    confidence = pred[idx]

    return roi_resized, label, confidence
