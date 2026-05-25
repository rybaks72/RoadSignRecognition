# Road Sign Recognition Application

This project is a road sign recognition system that can detect and classify traffic signs from images. It was prepared for the Artificial Intelligence Fundamentals course.

## Authors

- Sylwia Rybak
- Wojciech Sendek
- Stanisław Zieliński
- Jakub Szostak

## Project Features

- **Multi-sign detection**: Uses color masks and contour detection to find multiple potential signs in a single image.
- **CNN Classification**: Classifies detected signs using a deep Convolutional Neural Network trained on the GTSRB dataset.
- **Visualization**: Draws bounding boxes and labels with confidence scores on the original image.
- **GUI and CLI**: Supports both a Graphical User Interface and a Command Line Interface.
- **Pre-trained model included**: The application can be used without retraining the model.

## Installation

Run all commands from the project root directory.

### 1. Install Python

Make sure Python 3.10 or newer is installed.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```text
models/              - Saved trained models and training history plots
src/                 - Source code: training, prediction, model and GUI
README.md            - Project documentation
requirements.txt     - Project dependencies
```

## Technical Details

- **Model**: Custom CNN with 6 convolutional layers, batch normalization and dropout for regularization.
- **Input Size**: Images are resized to 48x48 pixels before being processed by the model.
- **Normalization**: Input data is normalized using GTSRB dataset statistics:
  - Mean: `[0.3403, 0.3121, 0.3214]`
  - Std: `[0.1340, 0.1295, 0.1386]`
- **Detection method**: The application uses color-based masking and contour detection to locate possible traffic signs before classification.
- **Classification method**: Each detected sign region is cropped, resized and passed to the CNN classifier.

## Dataset

The trained model is already included in the `models/` directory, so downloading the dataset is **not required** for running prediction or using the GUI.

The dataset is only required if the user wants to retrain the model.

This project uses the **GTSRB dataset**:

German Traffic Sign Recognition Benchmark:  
https://benchmark.ini.rub.de/gtsrb_dataset.html

The dataset is not included in this repository because of its large size.

After downloading and extracting the dataset, place it in the following structure:

```text
data/
└── archive/
    └── Train/
        ├── 0/
        ├── 1/
        ├── 2/
        └── ...
```

The training script expects the following path:

```text
data/archive/Train
```

Prediction does not require the dataset because the trained model is already provided.

Training requires downloading the GTSRB dataset and placing it in:

```text
data/archive/Train
```

## Usage

Run all commands from the project root directory.

## GUI

Launch the graphical interface to upload and classify images:

```bash
python src/main.py gui
```

The GUI allows the user to select an image, run detection and classification, and display the annotated result.

## Command Line Interface

### 1. Detect and classify all signs in an image

```bash
python src/main.py predict path/to/image.jpg
```

### 2. Detect, classify and save an annotated image

```bash
python src/main.py predict path/to/image.jpg output/annotated.jpg
```

### 3. Classify the whole image as a single sign

This mode is provided for backward compatibility and assumes that the whole image contains one traffic sign:

```bash
python src/main.py predict-one path/to/image.jpg
```

### 4. Train the model

Training requires downloading the GTSRB dataset and placing it in:

```text
data/archive/Train
```

Then run:

```bash
python src/main.py train
```

Each training run saves a new version of the model in the `models/` directory, for example:

```text
models/gtsrb_model_3.pth
```

The corresponding training history plot is also saved in the `models/` directory.
