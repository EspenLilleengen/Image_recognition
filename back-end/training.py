import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape for data augmentation (add channel dimension)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=10,      # Random rotation up to 10 degrees
    width_shift_range=0.1,  # Random horizontal shift up to 10%
    height_shift_range=0.1, # Random vertical shift up to 10%
    zoom_range=0.1,         # Random zoom up to 10%
    fill_mode='nearest'     # Fill strategy for empty pixels
)

# Split training data into train and validation
from sklearn.model_selection import train_test_split
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

print(f"Training samples: {len(train_images)}")
print(f"Validation samples: {len(val_images)}")
print(f"Test samples: {len(test_images)}")

# Define a more robust model architecture
def create_model():
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Create and compile model
model = create_model()

# Use a lowe learning rate and add learning rate scheduling
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Callbacks for better training
callbacks = [
    # Early stopping to prevent overfitting
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when plateau is reached
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    # Model checkpoint to save best model
    keras.callbacks.ModelCheckpoint(
        'back-end/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train the model with data augmentation
print("Training with data augmentation...")
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    steps_per_epoch=len(train_images) // 32,
    epochs=10,  
    validation_data=(val_images, val_labels),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('back-end/training_history.png')
plt.show()

# Evaluate on validation set
val_loss, val_acc = model.evaluate(val_images, val_labels, verbose=0)
print(f'\nValidation accuracy: {val_acc:.4f}')

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')

# Check for overfitting
overfitting_threshold = 0.05  # 5% difference threshold
if val_acc - test_acc > overfitting_threshold:
    print(f"⚠️  Potential overfitting detected! Validation accuracy is {val_acc - test_acc:.4f} higher than test accuracy")
else:
    print("✅ No significant overfitting detected")

# Save the model if it meets quality criteria
if test_acc > 0.95:  
    model.save('back-end/my_model.keras')
    print("✅ Model saved successfully!")
    
    # Also save the best model from checkpoint
    print("✅ Best model from training also saved as 'best_model.keras'")
else:
    print(f"❌ Model accuracy ({test_acc:.4f}) is below 95%, not saving the model.")
    print("Consider adjusting hyperparameters or using the best model from training.")

# Additional evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Predict on test set
test_predictions = model.predict(test_images)
test_pred_classes = np.argmax(test_predictions, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_pred_classes))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, test_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('back-end/confusion_matrix.png')
plt.show()