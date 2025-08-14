import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0   

# View an example image
# plt.imshow(train_images[0], cmap='gray')
# plt.title(f"Label: {train_labels[0]}")
# plt.axis('off')
# plt.show()

# Define the model architecture
model= keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=15)   

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')   

# Save the model in the same directory if accuracy is above 98%
if (test_acc > 0.98):
    model.save('back-end/my_model.keras')  # Save to parent directory (root of project)
    print("Model saved successfully!")
else:
    print("Model accuracy is below 98%, not saving the model.")