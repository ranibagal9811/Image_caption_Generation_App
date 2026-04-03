import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Image Caption Generator", layout="wide")

# ----------------------------
# Load Model & Tokenizer
# ----------------------------
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("imagecaptionmodel.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_resources()
max_length = 30

# ----------------------------
# Load MobileNetV2
# ----------------------------
cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# ----------------------------
# Feature Extraction
# ----------------------------
def extract_feature(image):
    image = image.resize((224, 224))
    image = np.array(image)

    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = cnn.predict(image, verbose=0)
    return feature

# ----------------------------
# Caption Generator
# ----------------------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    final_caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return final_caption

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Welcome", "About", "Working", "Generate Caption", "More Info"]
)

# ----------------------------
# Welcome Page
# ----------------------------
if page == "Welcome":
    st.title("🖼️ Image Caption Generator")
    st.write("""
    Upload an image and generate captions using Deep Learning.

    - CNN: MobileNetV2  
    - LSTM: Caption Generator  
    """)

# ----------------------------
# About Page
# ----------------------------
elif page == "About":
    st.title("📌 About Project")
    st.write("""
    This project uses deep learning to generate captions for images.

    - Feature extraction using MobileNetV2  
    - Sequence modeling using LSTM  
    """)

# ----------------------------
# Working Page
# ----------------------------
elif page == "Working":
    st.title("⚙️ How It Works")
    st.write("""
    1. Upload image  
    2. Extract features using CNN  
    3. Generate caption word-by-word  
    """)

# ----------------------------
# Generate Caption Page
# ----------------------------
elif page == "Generate Caption":
    st.title("🧠 Generate Caption")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                feature = extract_feature(image)
                caption = generate_caption(model, tokenizer, feature, max_length)

                st.success("Caption Generated!")
                st.markdown(f"### 📝 {caption}")

# ----------------------------
# More Info Page
# ----------------------------
elif page == "More Info":
    st.title("📊 More Information")

    st.write("""
    ### 📂 Dataset Categories

    This model is trained on images from the following 6 categories:

    - 🐾 Animal  
    - 🎉 Festival  
    - 🍔 Food  
    - 🍎 Fruits  
    - 🌿 Nature  
    - 🏝 Tourism  

    ---
             
    ### 📊 Dataset Size

    - Total Images: 1200 (training dataset)  
    - Captions per Image: 3  
    - Total Captions: ~ 3600  

    ---
    
    ### ⚠️ Important Note

    👉 For best results, please upload images related to the above categories only.

    Since the model is trained specifically on these categories:

    - Uploading unrelated images may result in inaccurate captions  
    - The model performs best within these domains  

    ---
    
    ### 🧠 Model Details

    - CNN: MobileNetV2 (Feature Extraction)  
    - LSTM: Caption Generation  
    - Max Caption Length: 30  

    ---
    
    ### 🚀 Future Improvements

    - Expand dataset with more categories  
    - Improve caption quality using Attention mechanism  
    - Increase model generalization  
    """)
