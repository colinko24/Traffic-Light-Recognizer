

# Traffic Light Recognition System

An autonomous vehicle prototype designed to detect traffic lights in real-time and classify their state as **Red**, **Yellow**, or **Green**. This project implements a two-stage deep learning pipeline.


## Project Architecture
Following the methodology by Addison Sears-Collins, this system operates in two distinct phases:
1. **Object Detection**: Utilizing a pre-trained **SSD ResNet50 V1** model (trained on the COCO dataset) to locate traffic lights in a high-resolution frame.
2. **State Classification**: Running the detected crops through a custom **InceptionV3** neural network to identify the specific color or "no-light" state.

### Prerequisites
* Python 3.7+
* TensorFlow 2.3+
* OpenCV & NumPy

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/traffic-light-recognition.git](https://github.com/yourusername/traf

This project is based on the tutorial [](https://automaticaddison.com/how-to-detect-and-classify-traffic-lights/) by Addison Sears-Collins.

