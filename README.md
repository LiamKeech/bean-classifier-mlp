# Dry Bean Classifier - Feedforward Neural Network

A feedforward neural network with a single hidden layer, trained via
backpropagation and gradient descent, implemented entirely from scratch in
Java (no ML libraries) to classify dry bean varieties from geometric
features extracted via computer vision.

## Overview

The network takes 16 numeric features describing a bean's shape and classifies it into
one of 7 registered dry bean types:

`DERMASON`, `SIRA`, `SEKER`, `BARBUNYA`, `HOROZ`, `CALI`, `BOMBAY`

- **Input layer:** 16 neurons (one per feature)
- **Hidden layer:** 1 hidden layer
- **Output layer:** 7 neurons, one-hot encoded
- **Training:** backpropagation with gradient descent

## Dataset

Data from Koklu, M. and Ozkan, I.A. (2020), *"Multiclass Classification of
Dry Beans Using Computer Vision and Machine Learning Techniques."*
Computers and Electronics in Agriculture, 174, 105507.
[DOI: 10.1016/j.compag.2020.105507](https://doi.org/10.1016/j.compag.2020.105507)

Over 12,000 bean grain images were segmented and reduced to 16 numeric
features (area, perimeter, axis lengths, eccentricity, shape factors, etc.).

- `data/BeanData.csv` - labelled training data, split into train/validation/test sets
- `data/Demo.csv` - 50 unlabeled beans; the trained network predicts their class

## Techniques explored

To improve training time and generalisation, this project investigates the following:

- Overfitting prevention
- Weight initialisation strategy
- Dynamic learning rate

Findings live in `docs/experiment-log.md`; per-run curves and final metrics in
`results/`.