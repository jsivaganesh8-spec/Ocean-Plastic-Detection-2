Ocean Plastic Detection 2

A comprehensive AI-powered application for detecting and classifying marine-pollution objects using computer vision and deep learning.

Features

Real-time detection: Detect marine-pollution objects in images and videos

15 object classes: Masks, bottles, plastic bags, nets, electronics, etc.

Web interface: User-friendly front-end built using Streamlit

Multiple input methods: Image upload, webcam, sample images

Visualization: Bounding boxes with confidence scores and class labels

Dataset statistics: Includes analysis of the training data and class distribution

Supported object classes:

Mask – face masks / protective gear

Can – metal cans/containers

Cellphone – mobile phones & electronics

Electronics – electronic waste/components

Gbottle – glass bottles

Glove – gloves/protective equipment

Metal – metal objects/debris

Misc – miscellaneous items

Net – fishing nets/ropes

Pbag – plastic bags

Pbottle – plastic bottles

Plastic – various plastic items

Rod – rods/cylindrical objects

Sunglasses – eyewear & accessories

Tire – tires/rubber objects

Quick Start

Install dependencies:

pip install -r requirements.txt


(Optional) Train the model from scratch:

python train_model.py


Run the web application:

streamlit run app.py


The app will open in your browser at http://localhost:8501.

Command-line detection:

# Detect objects in an image  
python inference.py --mode image --input path/to/image.jpg  

# Detect objects in a video  
python inference.py --mode video --input path/to/video.mp4  

# Real-time webcam detection  
python inference.py --mode webcam  

Project Structure
marine-pollution-detection/
├── app.py                # Streamlit web application  
├── train_model.py        # Model training script  
├── inference.py          # Command-line inference script  
├── utils.py              # Utility functions  
├── config.py             # Configuration settings  
├── requirements.txt      # Python dependencies  
├── data.yaml             # Dataset configuration  
├── train/                # Training images & labels  
├── valid/                # Validation images & labels  
├── test/                 # Test images & labels  
└── models/               # Trained model weights  

Usage Guide
Web Application

Home Page: Overview and quick stats

Dataset Stats: View training-data statistics & class distributions

Image Detection: Upload an image to run detection

Webcam Detection: Real-time detection using your camera

Sample Detection: Try sample images included in the repo

Command-Line Interface

The inference script supports three modes:

Image mode: Static images

Video mode: Frame-by-frame detection in videos

Webcam mode: Real-time detection via webcam

Configuration

Key parameters you can adjust in config.py:

CONFIDENCE_THRESHOLD – minimum confidence for a detection (default ~0.5)

IOU_THRESHOLD – Intersection-Over-Union threshold (default ~0.45)

IMG_SIZE – Input image size for the model (e.g., 640)

EPOCHS – Number of training epochs, if training from scratch

Technical Details

Model: YOLOv8 (You Only Look Once version 8)

Framework: Ultralytics YOLO backend, PyTorch

Frontend: Streamlit for web interface

Computer Vision: OpenCV for image/video processing

Visualization: Matplotlib, Plotly for charts/statistics

Dataset Information

Training set: ~3,628 images with annotations

Validation set: ~1,001 images

Test set: ~501 images

Total: ~5,130 images across 15 object classes

Visualization Features

Color-coded bounding boxes for different classes

Confidence scores displayed for each detection

Interactive charts for dataset statistics

Real-time detection visualization and export capabilities

Important Notes

Training requires significant computational resources (GPU recommended)

Webcam detection requires camera permissions

Large video files may take time to process

Contributing

Contributions are welcome! You can help by:

Adding new object classes

Improving detection accuracy

Enhancing the user interface

Introducing new features

License

This project is licensed under the MIT License – see the LICENSE file for full details.

Acknowledgments

Dataset provided via Roboflow Universe

YOLOv8 implementation by Ultralytics

Web interface powered by Streamlit

Computer vision processing by OpenCV

Thanks to all contributors for helping protect our oceans by detecting and preventing marine pollution!
