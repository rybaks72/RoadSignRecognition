# Road Sign Recognition Application

This project is prepared for the Artificial Intelligence Fundamentals course.

The goal of the project is to build a road sign recognition system that can detect and classify traffic signs from images.

## What the app can do now

The first version classified the whole input image as a single sign.  
This version can also recognize more than one sign in the same image:

1. it detects several traffic-sign candidates using color masks and contours,
2. it crops each candidate,
3. it classifies every crop using the trained GTSRB CNN model,
4. it can optionally save an annotated output image with bounding boxes.

This is a lightweight detector, not a separately trained YOLO detector. It works best for typical red, blue and yellow road signs.

## Dataset

- GTSRB - German Traffic Sign Recognition Benchmark

## Project structure

```text
data/       - dataset files
notebooks/  - experiments and dataset previews
src/        - training, prediction and GUI scripts
models/     - saved trained models
```

## Usage

Run commands from the project root directory.

### GUI

```bash
python src/main.py gui
```

In the GUI, upload an image and click **Classify Signs**. The app will draw boxes around detected signs and list all predictions.

### Detect and classify all signs in an image

```bash
python src/main.py predict path/to/image.jpg
```

### Detect, classify and save an annotated image

```bash
python src/main.py predict path/to/image.jpg output/annotated.jpg
```

### Old one-sign behavior

This classifies the whole image as one cropped sign:

```bash
python src/main.py predict-one path/to/image.jpg
```

### Train the model

```bash
python src/main.py train
```

The default prediction code expects the trained model here:

```text
models/gtsrb_final_model.pth
```
