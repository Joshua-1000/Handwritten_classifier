import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#IMPORTING LIBRARIES
import tensorflow as tf
from tensorflow.keras import layers, models
import  matplotlib.pyplot as plt

#LOADING DATASETS
dataset = tf.keras.utils.image_dataset_from_directory(
    "data/handwritten_digits_dataset",
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=1,
    image_size=(28, 28),
    shuffle=True,
    seed=123
)

# splitting
train_size = int(0.8 * len(dataset))
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)


#Normalization
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))

validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))


#model
model = models.Sequential([

    # First convolution layer
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    ),
    # Pooling layer
    layers.MaxPooling2D((2, 2)),
    # Second convolution
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),
    # Second pooling
    layers.MaxPooling2D((2, 2)),
    # Convert to 1D
    layers.Flatten(),
    # Dense hidden layer
    layers.Dense(64, activation='relu'),
    # Output layer (10 digits)
    layers.Dense(10, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

test_loss, test_acc = model.evaluate(validation_dataset)
print("Accuracy:   ", test_acc)



