import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Fruit Recognition AI",
    page_icon="🍎",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #ff9966, #ff5e62);
}

.main-title {
    text-align: center;
    color: white;
    font-size: 55px;
    font-weight: bold;
    text-shadow: 2px 2px 8px black;
}

.upload-text {
    text-align: center;
    color: yellow;
    font-size: 22px;
    animation: blink 1.5s linear infinite;
}

@keyframes blink {
    50% {
        opacity: 0.4;
    }
}

.result-card {
    background-color: rgba(255,255,255,0.2);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 25px;
}

section[data-testid="stSidebar"] {
    background-color: #ff7067;
}

</style>
""", unsafe_allow_html=True)

# Load Model
model = tf.keras.models.load_model("fruit_model.h5")

# Class Names
class_names = [
    "Apple",
    "Banana",
    "Mango",
    "NotFruit",
    "Orange"
]

# Sidebar
st.sidebar.title("🍉 Fruit AI")
st.sidebar.success("CNN Based Fruit Recognition System")

st.sidebar.markdown("""
### Supported Classes

🍎 Apple

🍌 Banana

🥭 Mango

🍊 Orange

🚫 NotFruit
""")

# Title
st.markdown(
    "<h1 class='main-title'>🍎 Fruit Recognition AI 🍌</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 class='upload-text'>Upload an Image to Predict</h3>",
    unsafe_allow_html=True
)

# Upload
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image"
    )

    # Preprocessing
    img = image.resize((128, 128))
    img_array = np.array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    confidence = float(np.max(prediction) * 100)

    predicted_index = int(np.argmax(prediction))

    fruit_name = class_names[predicted_index]

    fruit_emoji = {
        "Apple": "🍎",
        "Banana": "🍌",
        "Mango": "🥭",
        "Orange": "🍊",
        "NotFruit": "🚫"
    }

    emoji = fruit_emoji.get(fruit_name, "❓")

    st.subheader("Prediction Result")

    # Not Fruit
    if fruit_name == "NotFruit":

        st.error(
            f"{emoji} This image is NOT a fruit."
        )

    # High Confidence
    elif confidence >= 90:

        st.balloons()

        st.success(
            f"{emoji} The given fruit is {fruit_name.upper()}"
        )

    # Medium Confidence
    elif confidence >= 50:

        st.warning(
            f"{emoji} Maybe this fruit is {fruit_name.upper()}"
        )

    # Low Confidence
    else:

        st.error(
            "❓ Cannot predict this image confidently."
        )

    # Confidence Meter
    st.subheader("Confidence Level")

    st.progress(confidence / 100)

    st.write(
        f"Confidence Score: {confidence:.2f}%"
    )