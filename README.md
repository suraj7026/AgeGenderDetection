# Real-time Age & Gender Detection

A Streamlit application that uses your webcam to detect faces and predict their age and gender in real-time using pre-trained deep learning models.

## Features

- 🎥 **Real-time webcam detection** - Uses your camera to detect faces live
- 👤 **Age prediction** - Estimates age using a trained neural network
- ⚥ **Gender classification** - Predicts gender (Male/Female) with confidence
- 🎨 **Visual feedback** - Color-coded bounding boxes (Pink for Female, Blue for Male)
- 🚀 **Easy to use** - Simple checkbox interface to start/stop detection

## Requirements

- Python 3.8+
- Webcam
- Pre-trained model files (`agemodel.h5` and `gendermodel.h5`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd AgeGenderDetection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present:**
   - `agemodel.h5` - Age prediction model
   - `gendermodel.h5` - Gender classification model

## Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Start detection:**
   - Check the "Start Webcam" box to begin real-time detection
   - Position yourself in front of the camera
   - The app will detect faces and show age/gender predictions

4. **Stop detection:**
   - Uncheck the "Start Webcam" box to stop

## How it Works

1. **Face Detection**: Uses MTCNN (Multi-task CNN) for robust face detection
2. **Age Prediction**: Preprocesses face regions to 200x200 pixels for age model
3. **Gender Classification**: Preprocesses face regions to 128x128 pixels for gender model
4. **Real-time Processing**: Continuously processes video frames with confidence filtering (>0.9)

## Model Requirements

- **Age Model**: Expects 200x200 pixel RGB face images
- **Gender Model**: Expects 128x128 pixel RGB face images
- **Input Format**: RGB images (converted from BGR webcam feed)

## Troubleshooting

### Common Issues

1. **"MTCNN is not available" error:**
   ```bash
   pip install mtcnn
   ```

2. **Webcam not opening:**
   - Ensure no other applications are using the camera
   - Check camera permissions
   - Try different camera indices (0, 1, 2)

3. **Model loading errors:**
   - Ensure `agemodel.h5` and `gendermodel.h5` are in the same directory as `app.py`
   - Check file permissions

4. **Performance issues:**
   - Close other applications using the camera
   - Ensure good lighting conditions
   - Check system resources

## Technical Details

- **Face Detection**: MTCNN with confidence threshold 0.9
- **Age Range**: Clipped to 0-100 years for realistic predictions
- **Gender Threshold**: 0.5 probability threshold for Male/Female classification
- **Frame Processing**: Real-time BGR to RGB conversion for model compatibility

## License

This project is open source. Please ensure you have the necessary permissions for the model files you're using.

## Contributing

Feel free to submit issues and enhancement requests!
