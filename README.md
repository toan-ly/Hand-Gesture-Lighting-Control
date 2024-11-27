# Hand Gesture Lighting Control
This project combines MediaPipe, PyTorch, and real-time image processing to create a seamless, touchless light control experience.

# Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Demo](#demo)

## Introduction
This project implements a system for real-time light control through hand gestures. It leverages MediaPipe for gesture recognition, an MLP (Multi-Layer Perceptron) model for classification, and provides support for both simulated and hardware-based lighting control using Modbus protocol.

## Features
- **Gesture-Based Control**: Supports predefined hand gestures for turning lights on/off and controlling individual lights.
- **Simulation and Hardware Support**: Operates in a simulation mode or controls physical devices via Modbus protocol.
- **Real-Time Performance**: Achieves fast, accurate gesture recognition using a webcam and trained MLP model.
-**Customization Gestures**: Easily configure and expand the system with new gestures defined in YAML files.

## Requirements
- **Python**: Primary programming language.
- **MediaPipe**: For hand landmark detection and feature extraction.
- **PyTorch**: For training and inference of the MLP model.
- **OpenCV**: For real-time video capture and processing.
- **Modbus-Tk**: For hardware control (optional).

## Setup and Installation
### Prerequisites
- **Python 3.10+**
- **Anaconda** or **Miniconda** (recommended for environment management)

### Installation Steps
1. **Clone the repository**: 
```bash
git clone https://github.com/toan-ly/Hand-Gesture-Lighting-Control
cd Hand-Gesture-Lighting-Control
```

2. **Create and activate a virtual environment**:
```bash
conda create -n gesture_env python=3.10 -y
conda activate gesture_env
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. Download or prepare data:
- Use the provided data collection script `generate_landmark_data.py` in `src/data_preprocessing` to generate data.

5. Train the model: Run the training notebook in the `notebook` folder to train the custom model.

6. Place the trained model in the `models` directory.

## Usage
### Running the Application
For simulation mode:
```bash
python main.py
```
For hardware control, set `device=True` in `main.py` and ensure the device is connected.

### Collecting Gesture Data
Run the data collection script:
```bash
python src/data_preprocessing/generate_landmark_data.py
```

### Training a Custom Model
Open and execute the training notebook `notebooks/train_model.ipynb`
