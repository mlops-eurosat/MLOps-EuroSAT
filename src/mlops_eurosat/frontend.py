import base64
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

IMAGE_SIZE = 64  # or whatever your model actually uses


def preprocess_for_display(image: Image.Image) -> Image.Image:
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return image


st.set_page_config(
    page_title="EuroSAT Land Cover Classifier",
    layout="wide",
)

st.title("EuroSAT Land Cover Classification")
st.markdown("Upload a satellite image and classify it into one of the EuroSAT land-use categories.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file)

        resized_image = preprocess_for_display(image)

        st.subheader(f"Model Input (resized to {IMAGE_SIZE}×{IMAGE_SIZE})")
        st.image(
            resized_image,
            caption="This is what the model actually sees",
            width=200,
        )

    with st.spinner("Running inference..."):
        uploaded_file.seek(0)
        image_b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")

        response = requests.post(
            API_URL,
            json={"instances": [{"image_b64": image_b64}]},
            timeout=30,
        )

    if response.status_code == 200:
        result = response.json()["predictions"][0]

        with col2:
            st.subheader("Prediction")

            st.metric(
                label="Predicted Class",
                value=result["class_name"],
            )

            probabilities = result["probabilities"]

            top_class = max(probabilities, key=probabilities.get)

            st.metric(
                label="Confidence",
                value=f"{probabilities[top_class] * 100:.2f}%",
            )

        st.divider()

        st.subheader("Class Probabilities")

        df = pd.DataFrame(
            probabilities.items(),
            columns=["Class", "Probability"],
        ).sort_values("Probability", ascending=False)

        st.dataframe(
            df.style.format({"Probability": "{:.2%}"}),
            width="stretch",
            hide_index=True,
        )

    else:
        st.error(f"API error: {response.text}")
