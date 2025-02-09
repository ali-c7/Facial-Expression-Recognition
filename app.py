import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io  
import pandas as pd
from preprocess import preprocess_image
import os
import gdown

# Load CSS from file
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load model
FILE_ID = st.secrets["FILE_ID"]
MODEL_PATH = "facial_expression_recognition.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

def predict_emotion(image):

    processed_image, pil_processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    return CATEGORIES[predicted_class], confidence, prediction[0], pil_processed_image

model = download_model()

CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("😊 Facial Expression Recognition")
with st.sidebar:
    st.markdown("## Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes))
   
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        st.markdown("### Analysis")
        if st.button('🔍 Predict Emotion', use_container_width=True):
            with st.spinner('Analyzing image...'):
                image_for_prediction = Image.open(io.BytesIO(file_bytes))
                emotion, confidence, prediction, processed_image = predict_emotion(image_for_prediction)
               
                st.markdown("#### Preprocessed Face")
                st.image(processed_image, use_container_width=True)
               
                # Enhanced emotion display
                st.markdown(f"""
                    <div class="emotion-box">
                        <h2 style='margin:0; font-size: 1.8rem;'>{emotion.upper()}</h2>
                        <p style='margin:0; font-size: 1.2rem;'>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
               
                st.markdown("#### Confidence Levels")
                results_df = pd.DataFrame({
                    'Emotion': CATEGORIES,
                    'Confidence': prediction
                }).sort_values('Confidence', ascending=False)
                
                # Format confidence values as percentages
                results_df['Confidence'] = results_df['Confidence'].apply(lambda x: f"{x:.1%}")
                st.table(results_df.set_index('Emotion'))
else:
    st.info("👈 Please upload an image from the sidebar to begin")

