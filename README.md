🌊 Marine Pollution Detection System
A comprehensive AI-powered application for detecting and classifying marine pollution objects using computer vision and deep learning.

🎯 Features
Real-time Detection: Detect marine pollution objects in images and videos
15 Object Classes: Identify masks, bottles, plastic bags, nets, electronics, and more
Web Interface: User-friendly Streamlit web application
Multiple Input Methods: Image upload, webcam, and sample image detection
Visualization: Bounding boxes with confidence scores and class labels
Dataset Statistics: Comprehensive analysis of the training data
🏷️ Supported Object Classes
Mask - Face masks and protective gear
Can - Metal cans and containers
Cellphone - Mobile phones and electronic devices
Electronics - Electronic waste and components
Gbottle - Glass bottles
Glove - Gloves and protective equipment
Metal - Metal objects and debris
Misc - Miscellaneous items
Net - Fishing nets and ropes
Pbag - Plastic bags
Pbottle - Plastic bottles
Plastic - Various plastic items
Rod - Rods and cylindrical objects
Sunglasses - Eyewear and accessories
Tire - Tires and rubber objects
🚀 Quick Start
1. Install Dependencies
pip install -r requirements.txt
2. Train the Model (Optional)
If you want to train your own model:

python train_model.py
3. Run the Web Application
streamlit run app.py
The application will open in your browser at http://localhost:8501

4. Command Line Detection
For command-line usage:

# Detect objects in an image
python inference.py --mode image --input path/to/image.jpg

# Detect objects in a video
python inference.py --mode video --input path/to/video.mp4

# Real-time webcam detection
python inference.py --mode webcam
📁 Project Structure
marine-pollution-detection/
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── inference.py           # Command-line inference script
├── utils.py               # Utility functions
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── data.yaml              # Dataset configuration
├── train/                 # Training images and labels
├── valid/                 # Validation images and labels
├── test/                  # Test images and labels
└── models/                # Trained model weights
🎮 Usage Guide
Web Application
Home Page: Overview and quick stats
Dataset Stats: View training data statistics and class distribution
Image Detection: Upload images for object detection
Webcam Detection: Real-time detection using your camera
Sample Detection: Test detection on sample images
Command Line Interface
The inference script supports three modes:

Image Mode: Detect objects in static images
Video Mode: Process video files frame by frame
Webcam Mode: Real-time detection using webcam
⚙️ Configuration
Key parameters can be adjusted in config.py:

CONFIDENCE_THRESHOLD: Minimum confidence for detections (default: 0.5)
IOU_THRESHOLD: Intersection over Union threshold (default: 0.45)
IMG_SIZE: Input image size for the model (default: 640)
EPOCHS: Number of training epochs (default: 100)
🔧 Technical Details
Model: YOLOv8 (You Only Look Once version 8)
Framework: Ultralytics YOLO
Backend: PyTorch
Frontend: Streamlit
Computer Vision: OpenCV
Visualization: Matplotlib, Plotly
📊 Dataset Information
The dataset contains:

Training: 3,628 images with annotations
Validation: 1,001 images with annotations
Test: 501 images with annotations
Total: 5,130 images across 15 object classes
🎨 Visualization Features
Color-coded bounding boxes for different object classes
Confidence scores for each detection
Interactive charts and statistics
Real-time detection visualization
Export capabilities for results
🚨 Important Notes
The model requires significant computational resources for training
GPU acceleration is recommended for faster inference
Webcam detection requires camera permissions
Large video files may take time to process
🤝 Contributing
Feel free to contribute to this project by:

Adding new object classes
Improving detection accuracy
Enhancing the user interface
Adding new features
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Dataset provided by Roboflow Universe
YOLOv8 implementation by Ultralytics
Streamlit for the web interface
OpenCV for computer vision capabilities
🌊 Help protect our oceans by detecting and preventing marine pollution! 🌊
