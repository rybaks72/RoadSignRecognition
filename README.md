# Road Sign Recognition Application

This project is prepared for the Artificial Intelligence Fundamentals course.

The goal of the project is to build a road sign recognition system that can detect and classify traffic signs from images.

## Planned approach

The system follows the general computer vision pipeline:

1. Image acquisition
2. Preprocessing
3. Feature extraction
4. Classification

The preferred AI-based solution is based on convolutional neural networks.  
The project may use CNN-based classification and YOLO-based object detection for locating and recognizing road signs.

## Dataset

Planned dataset:

- GTSRB - German Traffic Sign Recognition Benchmark

## Project structure

```text
data/       - dataset files
notebooks/  - experiments and dataset previews
src/        - training and prediction scripts
models/     - saved trained models