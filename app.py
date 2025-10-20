import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import requests  # For fetching image from URL
from tensorflow.keras.models import load_model

# --- 1. Configurations ---

# Set page config - This MUST be the first Streamlit command
st.set_page_config(
    page_title="Lung Disease Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide default footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Constants (MUST match your training)
CLASSES = ['normal', 'pnemo', 'tb']
IMG_SIZE = (150, 150) # The size you used in training

# Paths
SVM_MODEL_PATH = "svm_pipeline.joblib"
EXTRACTOR_PATH = "vgg16_feature_extractor.h5"

# --- 2. Model Loading ---

@st.cache_resource
def load_models():
    """
    Load the SVM model AND the VGG16 feature extractor.
    """
    models = {}
    
    # Load the VGG16 feature extractor
    try:
        models['feature_extractor'] = load_model(EXTRACTOR_PATH)
        print("VGG16 Feature Extractor loaded.")
    except Exception as e:
        st.error(f"Error loading VGG16 Feature Extractor from {EXTRACTOR_PATH}: {e}")
        st.error("Please make sure 'vgg16_feature_extractor.h5' is in the same directory.")
        return None
    
    # Load the SVM model
    try:
        models['svm'] = joblib.load(SVM_MODEL_PATH)
        print("SVM Model loaded.")
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {SVM_MODEL_PATH}.")
        return None
    except Exception as e:
        st.error(f"Error loading model {SVM_MODEL_PATH}: {e}")
        return None
            
    return models

# --- 3. Image Processing ---

def process_image(image_bytes, target_size):
    """
    Process the uploaded image: decode, resize, and normalize.
    """
    try:
        # Convert the uploaded file (in bytes) to an OpenCV image
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Could not decode the image. It might be corrupt or an invalid format.")
            return None
        
        # Resize the image to the target size
        image = cv2.resize(image, target_size)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --- 4. Page Functions ---

def show_prediction_page(models):
    """
    Displays the main prediction interface.
    """
    st.title("🫁 Lung Disease Classification")
    st.write("Upload a Chest X-Ray or paste an image URL to classify it as **Normal**, **Pneumonia (pnemo)**, or **Tuberculosis (tb)**.")
    
    # Create two columns for input methods
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("1. Upload an X-Ray Image", type=["png", "jpg", "jpeg"])

    with col2:
        image_url = st.text_input("2. Or Paste Image URL")

    image = None
    image_bytes = None

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = process_image(image_bytes, IMG_SIZE)
    
    elif image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status() # Check for bad responses
            image_bytes = response.content
            image = process_image(image_bytes, IMG_SIZE)
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching image from URL: {e}")
            
    if image is not None:
        # Create two columns for image and results
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.image(image, caption='Uploaded X-Ray', use_container_width=True) # Fixed parameter name

        with res_col2:
            st.write("") # Add a little space
            with st.spinner("Classifying..."):
                try:
                    # 1. Prepare image for VGG16
                    img_array = image.astype(np.float32) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0) # Add batch dimension -> (1, 150, 150, 3)

                    # 2. Extract features using VGG16
                    # This will output the (1, 512) features your SVM expects
                    features = models['feature_extractor'].predict(img_batch)
                    
                    # 3. Predict with SVM
                    probabilities = models['svm'].predict_proba(features)
                    
                    # 4. Display Results
                    st.write("### 🧠 Prediction Result")
                    
                    predicted_class_index = int(np.argmax(probabilities, axis=1)[0])
                    predicted_class_name = CLASSES[predicted_class_index]
                    confidence = float(probabilities[0, predicted_class_index])

                    if predicted_class_name == 'normal':
                        st.success(f"**Predicted Class:** {predicted_class_name.upper()} (Healthy)")
                    else:
                        st.error(f"**Predicted Class:** {predicted_class_name.upper()} (Disease Detected)")
                    
                    st.info(f"**Confidence:** {confidence*100:.2f}%")
                    
                    st.write("---")
                    st.write("#### Class Probabilities:")
                    prob_df_chart = pd.DataFrame({
                        "Probability": probabilities.flatten()
                    }, index=CLASSES)
                    st.bar_chart(prob_df_chart)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

def show_project_page():
    """
    Displays the 'About the Project' page content.
    """
    st.title("🤖 About the Project")

    st.markdown("""
    This web application is a tool for classifying lung conditions from chest X-ray images. It uses a combination of Deep Learning and traditional Machine Learning to predict whether an X-ray shows a **Normal** lung, one affected by **Pneumonia**, or one affected by **Tuberculosis**.
    """)
    
    st.subheader("Our Methodology")
    st.markdown("""
    Instead of building a giant Deep Learning model from scratch, which would require millions of images and huge computational power, we used a smarter, more efficient approach.

    #### 1. Transfer Learning (Feature Extraction)
    We started with a powerful, pre-trained model called **VGG16**. This model was trained by Google on millions of different images (cats, dogs, cars, etc.) and is already an expert at detecting shapes, textures, and patterns.

    We used this "expert" model as a **feature extractor**:
    * We fed it an X-ray image (resized to 150x150).
    * We stripped off its final classification layer.
    * We captured the output from its new final layer (`GlobalAveragePooling2D`). This output is a list of **512 numbers**—a "feature vector"—that numerically describes the most important patterns in the X-ray.

    #### 2. Machine Learning (Classification)
    This is where the **SVM (Support Vector Machine)** model comes in. Instead of training a model on 67,500+ pixels, we trained it on the much smaller and richer **512-feature vectors**.
    """)
    
    st.subheader("Model Comparison Results")
    st.markdown("""
    We trained and compared several machine learning models using this feature extraction method. The **Simple CNN** was trained from scratch as a baseline and performed the worst, highlighting the power of transfer learning.
    
    Here is the final summary of all models, ranked by their F1-Score:
    """)
    
    # --- Results Table ---
    results_data = {
        'Model': ['LightGBM', 'SVM (RBF Kernel)', 'Logistic Regression', 'Random Forest', 'KNN (n=5)', 'Simple CNN'],
        'Accuracy': [0.889, 0.878, 0.873, 0.860, 0.831, 0.752],
        'Macro Precision': [0.891, 0.850, 0.849, 0.900, 0.854, 0.772],
        'Macro Recall': [0.887, 0.897, 0.878, 0.787, 0.788, 0.687],
        'Macro F1-Score': [0.888, 0.870, 0.862, 0.828, 0.815, 0.717]
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df.style.format({
        'Accuracy': '{:.3f}',
        'Macro Precision': '{:.3f}',
        'Macro Recall': '{:.3f}',
        'Macro F1-Score': '{:.3f}'
    }).background_gradient(cmap='viridis', subset=['Macro F1-Score']), use_container_width=True)
    
    
    st.subheader("Final Model Selection")
    st.markdown("""
    As the results show, **LightGBM** had the best overall F1-Score.
    
    However, the **SVM (RBF Kernel)** was a very close second and had the **highest Macro Recall** (89.7%). In a medical context, high recall is critical as it means the model is excellent at "finding" all positive cases (i.e., not missing sick patients).
    
    Given its robust performance and excellent recall, the **SVM model** was chosen to power this application.
    """)

def show_dataset_page():
    """
    Displays the 'About the Dataset' page content.
    """
    st.title("📊 About the Dataset")

    st.markdown("""
    The lung images dataset is a comprehensive collection of images used for the training and evaluation of deep-learning models for the diagnosis of lung infections.

    The images are in **PNG format** with a resolution of **512x512**.
    """)

    st.info("""
    **Dataset at a Glance:**
    * **Total Images:** 17,275
    * **Normal:** 10,406 images
    * **Pneumonia:** 5,775 images
    * **Tuberculosis:** 1,094 images
    """)

    st.markdown("""
    The dataset has been divided into `train`, `test`, and `val` sets.

    The images were sourced from multiple locations, including:
    * RSNA
    * Montgomery County chest X-ray set
    * Shenzhen chest X-ray
    * Qatar University, Doha, Qatar
    * The University of Dhaka, Bangladesh, and their collaborators

    The original images were in the DCM format, and they were converted to the Png format to ensure compatibility with deep learning models. This dataset is an essential resource for researchers, clinicians, and data scientists working on lung infection diagnosis, and it provides a valuable tool for the development of advanced AI models for lung disease diagnosis.
    """)

# --- 5. Main Application ---

def main():
    # Load models once
    models = load_models()
    if models is None:
        st.error("Fatal Error: Models could not be loaded. The application cannot continue.")
        return

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Prediction", "About the Project", "About the Dataset"]
    )

    # Page Routing
    if page == "Prediction":
        show_prediction_page(models)
    elif page == "About the Project":
        show_project_page()
    elif page == "About the Dataset":
        show_dataset_page()

# Run the app
if __name__ == "__main__":
    main()