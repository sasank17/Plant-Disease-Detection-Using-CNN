# Plant Disease Detection Using CNN

This project involves building a deep learning model using a Convolutional Neural Network (CNN) to detect plant diseases from images of plant leaves. The goal is to assist farmers and agricultural researchers in identifying diseases early and accurately to ensure better crop health.

## Features
- Detects various plant diseases from leaf images.
- Uses a CNN model trained on a labeled dataset of healthy and diseased plant leaves.
- Achieves high accuracy through data augmentation and careful model tuning.
- Saves the trained model and label binarizer for future predictions.

## Dataset
The dataset used consists of labeled images of healthy and diseased plant leaves. Each image is resized to 256x256 pixels before being fed into the CNN.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-Learn
- Matplotlib

## Project Structure
- `network.py`: Contains the code for the CNN model.
- `dataset/`: Directory containing the plant leaf images.
- `label_transform.pkl`: Pickle file for label binarization.
- `cnn_model.pkl`: Trained CNN model.
- `networkinfo.log`: Log file to track training progress and accuracy.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd plant-disease-detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Prepare the dataset by placing images in the `dataset/` folder.
5. Run the script to train the model:
   ```bash
   python network.py
   ```
6. After training, use the saved model to predict diseases on new images.

## Example Output
During training, the accuracy and loss graphs will be displayed. After training, the model can predict plant diseases as follows:
```plaintext
Predicted Disease: Tomato_Late_Blight
```

## Results
- Training Accuracy: ~98%
- Validation Accuracy: ~95%

## Future Enhancements
- Increase the dataset size for better generalization.
- Fine-tune the model for different plant species and environments.
- Develop a mobile application for real-time plant disease detection in the field.


