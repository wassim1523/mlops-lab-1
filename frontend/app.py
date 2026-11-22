import os

import requests
import streamlit as st

url = os.getenv("INFERENCE_URL", "http://127.0.0.1:8000")
st.title("Food-11 classifier")
image = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
if image:
    st.image(image, width=300)
    try:
        response = requests.post(
            f"{url}/predict",
            files={"file": (image.name, image.getvalue(), image.type)},
            timeout=30,
        )
        response.raise_for_status()
        prediction = response.json()
        st.write(
            f"**Prediction:** {prediction['category']} ({prediction['confidence']:.1%})"
        )
    except requests.RequestException as exc:
        st.error(f"Prediction unavailable: {exc}")
