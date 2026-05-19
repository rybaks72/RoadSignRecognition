# Road Sign Recognition Application

This project is a road sign recognition system that can detect and classify traffic signs from images. It was prepared for the Artificial Intelligence Fundamentals course.

### Authors
- Sylwia Rybak
- Wojciech Sendek
- Stanisław Zieliński
- Jakub Szostak

## Project Features

- **Multi-sign detection**: Uses color masks and contour detection to find multiple potential signs in a single image.
- **CNN Classification**: Classifies detected signs using a deep Convolutional Neural Network trained on the GTSRB dataset.
- **Visualization**: Draws bounding boxes and labels with confidence scores on the original image.
- **GUI & CLI**: Supports both a Graphical User Interface and a Command Line Interface.

## Installation

1. Clone the repository.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```text
models/     - Saved trained models and training history
src/        - Source code (training, prediction, model, and GUI)
README.md   - Project documentation
requirements.txt - Project dependencies
```

## Technical Details

- **Model**: Custom CNN with 6 convolutional layers, batch normalization, and dropout for regularization.
- **Input Size**: Images are resized to **48x48 pixels** before being processed by the model.
- **Normalization**: Input data is normalized using GTSRB dataset statistics (Mean: [0.3403, 0.3121, 0.3214], Std: [0.1340, 0.1295, 0.1386]).

## Usage

Run all commands from the project root directory.

### GUI
Launch the graphical interface to upload and classify images:
```bash
python src/main.py gui
```

### Command Line Interface

**1. Detect and classify all signs in an image:**
```bash
python src/main.py predict path/to/image.jpg
```

**2. Detect, classify and save an annotated image:**
```bash
python src/main.py predict path/to/image.jpg output/annotated.jpg
```

**3. Classify the whole image as a single sign (Backward compatibility):**
```bash
python src/main.py predict-one path/to/image.jpg
```

**4. Train the model (Requires downloading the GTSRB dataset):**
```bash
python src/main.py train
```
Each training run saves a new version of the model named `models/gtsrb_model_{number}.pth` and its corresponding training history plot.

The prediction scripts default to the latest numbered model (`models/gtsrb_model_{n}.pth`). If no such file exists, they will default to looking for `models/gtsrb_model_1.pth`.
