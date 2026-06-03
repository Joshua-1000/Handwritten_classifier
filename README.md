# Handwritten Digits Classifier

A CNN-based handwritten digit classification project using TensorFlow/Keras.

## Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from a custom dataset.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Matplotlib

## Dataset

The project uses a custom dataset located in `data/handwritten_digits_dataset/` with directories for each digit (0-9).

## Project Structure

- `hand_digit_classifier.py` - Main training and classification script
- `data/` - Dataset directory
- `cache/` - Cache files for joblib and model data

## Model Architecture

The CNN includes:
- Conv2D layers with ReLU activation
- MaxPooling2D layers for dimensionality reduction
- Dense layers for classification
- 80/20 train/validation split
- Data normalization (0-1 range)

## Usage

Run the classifier:
```bash
python hand_digit_classifier.py
```

## Author

Joshua
