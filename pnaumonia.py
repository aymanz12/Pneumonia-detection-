import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🫁",
    layout="wide"
)

# Enhanced CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 10px;
    font-weight: bold;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 30px;
}
.result-text {
    font-size: 2rem;
    font-weight: bold;
    margin-top: 20px;
    text-align: center;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.pneumonia-detected {
    background-color: #ffebee;
    color: #c62828;
    border: 2px solid #f44336;
}
.normal-detected {
    background-color: #e8f5e8;
    color: #2e7d32;
    border: 2px solid #4caf50;
}
.info-box {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid #1E88E5;
    margin: 20px 0;
}
.metric-container {
    display: flex;
    justify-content: space-around;
    margin: 20px 0;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    min-width: 120px;
}
.upload-section {
    border: 2px dashed #1E88E5;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>🫁 Pneumonia Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered chest X-ray analysis for pneumonia detection</p>", unsafe_allow_html=True)

# Model path configuration
MODEL_PATH = r"C:\Users\PC\Desktop\detection de le pnaumonia\vgg_model.h5"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.info("Please verify the model file exists at the specified path.")
    st.stop()

# Load model with caching
@st.cache_resource
def load_pneumonia_model():
    """Load the pneumonia detection model with error handling"""
    try:
        with st.spinner('Loading AI model...'):
            model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Image preprocessing function
def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224 for VGG)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Prediction function
def make_prediction(model, processed_image):
    """Make prediction using the loaded model"""
    try:
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Handle different output formats
        if len(prediction.shape) == 2:
            if prediction.shape[1] == 1:
                probability = float(prediction[0][0])
            elif prediction.shape[1] == 2:
                probability = float(prediction[0][1])
            else:
                probability = float(np.max(prediction[0]))
        else:
            probability = float(prediction[0])
        
        return probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Load the model
model = load_pneumonia_model()

if model is not None:
    # Create main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload X-Ray Image")
        
        # File uploader with enhanced styling
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...", 
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        # Display model information
        with st.expander("ℹ️ Model Information", expanded=False):
            st.markdown("""
            **Model Architecture:** VGG-based CNN  
            **Input Size:** 224x224 pixels  
            **Training Data:** Chest X-ray images  
            **Purpose:** Binary classification (Normal vs Pneumonia)
            """)
        
        # Instructions
        st.markdown("""
        <div class='info-box'>
        <h4>📋 Instructions:</h4>
        <ol>
        <li>Upload a clear chest X-ray image</li>
        <li>Ensure the image shows the full chest area</li>
        <li>Click 'Analyze X-Ray' to get results</li>
        <li>Review the confidence score and interpretation</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔍 Analysis Results")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
            
            # Analysis button
            if st.button("🔬 Analyze X-Ray", type="primary", use_container_width=True):
                with st.spinner('🤖 Analyzing image...'):
                    # Add a small delay for better UX
                    time.sleep(1)
                    
                    # Preprocess image
                    processed_img = preprocess_image(image)
                    
                    if processed_img is not None:
                        # Make prediction
                        probability = make_prediction(model, processed_img)
                        
                        if probability is not None:
                            # Determine result
                            if probability > 0.5:
                                result = "PNEUMONIA DETECTED"
                                confidence = probability * 100
                                result_class = "pneumonia-detected"
                                emoji = "🚨"
                            else:
                                result = "NORMAL"
                                confidence = (1 - probability) * 100
                                result_class = "normal-detected"
                                emoji = "✅"
                            
                            # Display main result
                            st.markdown(f"""
                            <div class='result-text {result_class}'>
                                {emoji} {result}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display metrics
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric(
                                    label="Confidence Level",
                                    value=f"{confidence:.1f}%"
                                )
                            with col_b:
                                risk_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
                                st.metric(
                                    label="Certainty",
                                    value=risk_level
                                )
                            
                            # Progress bar
                            st.progress(confidence / 100)
                            
                            # Interpretation
                            st.markdown("### 📊 Interpretation")
                            if probability > 0.5:
                                if confidence > 80:
                                    interpretation = "The model shows high confidence in detecting pneumonia patterns."
                                elif confidence > 60:
                                    interpretation = "The model detects pneumonia with moderate confidence."
                                else:
                                    interpretation = "The model suggests possible pneumonia but with lower confidence."
                            else:
                                if confidence > 80:
                                    interpretation = "The model shows high confidence that the X-ray appears normal."
                                elif confidence > 60:
                                    interpretation = "The model suggests the X-ray is likely normal."
                                else:
                                    interpretation = "The model is uncertain about the classification."
                            
                            st.info(interpretation)
                            
                        else:
                            st.error("Failed to analyze the image. Please try again.")
                    else:
                        st.error("Failed to process the image. Please check the image format.")
        else:
            st.info("👆 Please upload a chest X-ray image to begin analysis")
            
            # Sample images section
            st.markdown("### 🖼️ Sample Images")
            st.markdown("""
            For testing purposes, you can use sample chest X-ray images from medical databases 
            or educational resources. Make sure to use actual chest X-ray images for accurate results.
            """)

# Footer sections
st.markdown("---")

# How it works section
with st.expander("🔬 How It Works", expanded=False):
    st.markdown("""
    This AI system uses a deep learning model based on the VGG architecture to analyze chest X-ray images:
    
    1. **Image Processing**: The uploaded image is resized and normalized to match the model's requirements
    2. **Feature Extraction**: The neural network extracts relevant features from the X-ray image
    3. **Classification**: The model classifies the image as either normal or showing signs of pneumonia
    4. **Confidence Score**: A probability score indicates the model's confidence in its prediction
    
    The model has been trained on thousands of chest X-ray images to learn the patterns associated with pneumonia.
    """)

# Technical details
with st.expander("⚙️ Technical Details", expanded=False):
    st.markdown(f"""
    - **Framework**: TensorFlow {tf.__version__}
    - **Model Architecture**: VGG-based Convolutional Neural Network
    - **Input Resolution**: 224×224 pixels
    - **Color Channels**: RGB (3 channels)
    - **Output**: Binary classification probability
    - **Preprocessing**: Normalization to [0,1] range
    """)

# Disclaimer
st.markdown("---")
st.markdown("### ⚠️ Important Disclaimer")
st.error("""
**MEDICAL DISCLAIMER**: This application is for educational and research purposes only. 
It is NOT a substitute for professional medical diagnosis or treatment. 

- Always consult qualified healthcare professionals for medical advice
- Do not make medical decisions based solely on this tool's output
- This AI system may produce false positives or false negatives
- Proper medical diagnosis requires trained radiologists and clinical context
""")

# Additional info
st.markdown("---")
st.markdown("### 📚 Additional Resources")
st.markdown("""
- **World Health Organization**: [Pneumonia Information](https://www.who.int/news-room/fact-sheets/detail/pneumonia)
- **CDC**: [Pneumonia Prevention and Treatment](https://www.cdc.gov/pneumonia/)
- **Medical Imaging**: Always consult with qualified radiologists for proper interpretation
""")



