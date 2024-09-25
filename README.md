# Fashion MNIST Classification using Neural Networks and CNN

### Project Overview

This project involves the classification of the Fashion MNIST dataset using two different approaches:

Part 1: A fully connected (Dense) neural network.
Part 2: A Convolutional Neural Network (CNN) to improve the model's accuracy and handle image data more efficiently.
Dataset

The dataset used is the Fashion MNIST dataset, which consists of 60,000 training images and 10,000 testing images. The images are 28x28 grayscale, representing different fashion items such as shirts, shoes, and bags, and are labeled from 0-9, each representing a different class of fashion items.

### Project Structure

Part 1: Fully Connected Neural Network (Dense Layers)
The first part of the lab uses a basic neural network architecture with fully connected layers to classify the Fashion MNIST dataset.
We visualize a few images from the dataset, normalize the data, and use a Sequential model with a Flatten layer, a Dense layer with 128 units, and a softmax output layer for classification.
The model is trained for 5 epochs using the Adam optimizer and sparse categorical cross-entropy loss.
Part 2: Convolutional Neural Network (CNN)
In the second part of the lab, we enhance our model by introducing convolutional layers, which are better suited for image data.
We use two convolutional layers followed by max-pooling layers and then flatten the output to pass through fully connected layers.
The model is trained for 10 epochs and achieves higher accuracy than the fully connected network.
Code Description

## Part 1: Fully Connected Neural Network
```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Display sample images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create a Sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
```
## Part 2: Convolutional Neural Network (CNN)
```
import tensorflow as tf

# Load Fashion MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
```
# Results

* Dense Neural Network: After 5 epochs, the fully connected network achieved an accuracy of 89.12%.
* Convolutional Neural Network: After 10 epochs, the CNN achieved an accuracy of 97.86%, showing significant improvement over the fully connected model.
