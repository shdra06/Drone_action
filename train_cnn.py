import numpy as np
import tensorflow as tf
import os

IMG_SIZE = 128
FRAMES = 5
CLASSES = ["up", "down", "left", "right", "none"]

X, y = [], []

for idx, cls in enumerate(CLASSES):
    for file in os.listdir("dataset"):
        if file.startswith(cls):
            data = np.load("dataset/" + file)
            X.append(data)
            y.append(idx)

X = np.array(X, dtype="float32") / 255.0
y = tf.keras.utils.to_categorical(y, len(CLASSES))

print("Dataset shape:", X.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu",
                           input_shape=(IMG_SIZE, IMG_SIZE, FRAMES)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X, y,
    epochs=20,
    batch_size=8,
    shuffle=True
)

model.save("model.h5")
print("Model retrained and saved.")
