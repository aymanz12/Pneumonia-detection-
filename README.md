# Pneumonia Detection from Chest X-Rays

An AI-powered web application that detects pneumonia from chest X-ray images using deep learning.

## 🔍 Overview

This project uses a VGG-based convolutional neural network to analyze chest X-ray images and classify them as either normal or showing signs of pneumonia. The application is built with Streamlit for an interactive web interface.

![Application Interface](screenshots/app_interface.png)
*Main interface of the Pneumonia Detection Application*

## 🌟 Features

- **AI-Powered Detection**: Uses a trained VGG model for accurate pneumonia detection
- **User-Friendly Interface**: Clean, intuitive web interface built with Streamlit
- **Real-time Analysis**: Instant results with confidence scores
- **Medical Disclaimer**: Appropriate warnings for medical use
- **Detailed Interpretation**: Provides confidence levels and interpretation of results

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **Streamlit** - Web application framework
- **OpenCV** - Image processing
- **PIL (Pillow)** - Image handling
- **NumPy** - Numerical computations

## 📋 Requirements

```bash
streamlit>=1.28.0
tensorflow>=2.10.0
opencv-python>=4.8.0
Pillow>=9.0.0
numpy>=1.21.0
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Update model path**
   - Edit `pneumonia.py` line 37 to point to your model location:
   ```python
   MODEL_PATH = "vgg_model.h5"  # Update this path
   ```

## 🎯 Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run pneumonia.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload a chest X-ray image** (JPG, JPEG, or PNG)

4. **Click "Analyze X-Ray"** to get the prediction

5. **Review the results** with confidence scores and interpretation

### Application Screenshots

![Upload Interface](screenshots/upload_interface.png)
*Upload section where users can select X-ray images*

![Analysis Results](screenshots/analysis_results.png)
*Results section showing pneumonia detection with confidence scores and interpretation*

## 📊 Model Information

- **Architecture**: VGG-based CNN
- **Input Size**: 224×224 pixels
- **Training Data**: [Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Output**: Binary classification (Normal vs Pneumonia)
- **Framework**: TensorFlow/Keras

## 🧪 Model Training

The model training process is documented in `project_ML.ipynb`. The notebook includes:

- Data preprocessing and augmentation
- Model architecture design
- Training process and hyperparameter tuning
- Model evaluation and validation
- Performance metrics and visualization

## 📈 Performance

The model achieves competitive performance on chest X-ray classification:
- Training accuracy: [96%]
- Validation accuracy: [94%]
- Test accuracy: [90%]

## ⚠️ Important Disclaimer

**This application is for educational and research purposes only.**

- This tool is NOT a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals for medical advice
- Do not make medical decisions based solely on this tool's output
- The AI system may produce false positives or false negatives
- Proper medical diagnosis requires trained radiologists and clinical context


## 📚 References

- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) - Kaggle
- **Research Paper**: "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" - Cell 2018
- **VGG Architecture**: "Very Deep Convolutional Networks for Large-Scale Image Recognition" - ICLR 2015
- **Medical Imaging**: WHO Pneumonia Guidelines and Chest X-ray Interpretation Standards
