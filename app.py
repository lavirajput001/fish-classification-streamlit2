import streamlit as st
import numpy as np
import cv2
from PIL import Image
from infer import predict_image_streamlit

st.set_page_config(page_title="Fish Classification", layout="centered")
st.title("🐟 Fish Species Classification (CNN + Firefly + UWIE)")

uploaded_file = st.file_uploader("Upload Underwater Fish Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing image..."):
        result, pred_class, confidence = predict_image_streamlit(image_np)

    st.subheader("🔍 Preprocessing Result")
    st.image(result, caption="Segmented ROI", use_column_width=True)

    st.subheader("🎯 Prediction")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.4f}")
else:
    st.info("Please upload a fish image.")
