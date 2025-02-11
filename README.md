# Potato Leaf Disease Detection

## Overview
This project focuses on **Potato Leaf Disease Detection** using a **Deep Learning-based Convolutional Neural Network (CNN)**. The model classifies potato leaf images into three categories: **Potato Late Blight, Potato Early Blight, and Healthy Potato Leaves**. A **Streamlit-based web application** has been developed to allow users to upload images and get real-time disease classification results.

## Features
- **Deep Learning Model**: A CNN-based model trained using TensorFlow.
- **High Accuracy**: Achieves **90-95% accuracy** in classifying leaf diseases.
- **Automated Image Processing**: Uses `image_dataset_from_directory` for efficient image loading and preprocessing.
- **Streamlit Web App**: Provides a user-friendly interface for real-time disease classification.
- **Model Visualization**: Training and validation accuracy plots for performance evaluation.

## Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **Matplotlib**
- **NLTK (for chatbot in Streamlit app)**

## Dataset
The dataset is structured into three folders:
```
/dataset
    /Train  # Training images
    /Valid  # Validation images
    /Test   # Test images
```
Each folder contains subdirectories for different disease classes.<br>
<a href="https://github.com/JayRathod341997/AICTE-Internship-files">Dataset Link</a>

## Model Architecture
The CNN model consists of:
- Multiple **Convolutional Layers** with ReLU activation
- **Max Pooling Layers** for feature extraction
- **Dropout Layers** to prevent overfitting
- **Flatten and Dense Layers** for final classification
- **Softmax Activation** for multi-class classification

## Installation & Setup
### **1. Clone the Repository**
```sh
git clone https://github.com/your-username/potato-leaf-disease-detection.git
cd potato-leaf-disease-detection
```

### **2. Install Dependencies**
```sh
pip install tensorflow streamlit matplotlib nltk transformers
```

### **3. Download NLTK Data (for chatbot feature)**
```sh
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### **4. Train the Model (Optional)**
If you want to retrain the model, run:
```sh
python train_model.py
```

### **5. Run the Streamlit App**
```sh
streamlit run app.py
```
This will launch a **web application** where you can upload potato leaf images for classification.

## Model Training and Evaluation
The model is trained on **10 epochs** with **Adam optimizer** and `categorical_crossentropy` loss function.
```python
training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=10)
```
After training, the model is saved as:
```sh
trained_plant_disease_model.keras
```

## Accuracy Visualization
To visualize training progress:
```python
plt.plot(epochs, training_history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.title('Accuracy Results')
plt.legend()
plt.show()
```

## Future Improvements
- **Dataset Expansion**: Increase dataset size for better generalization.
- **Mobile-Based Real-Time Detection**: Deploy as a mobile app for practical field use.
- **Integration with IoT Sensors**: Combine AI with real-time environmental monitoring for better predictions.

## Contributing
Contributions are welcome! Feel free to submit **issues** or **pull requests**.



---
**Developed by Grandhe Rama Bhaktha Ravi Sri Ram** 🚀

