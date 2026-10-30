import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# 1. Configuration
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
DATASET_DIR = "dataset/"  # Create subfolders here (e.g., 'chair', 'table', 'door')

print("Loading and preprocessing dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"  # Match the ESP32-CAM grayscale capture
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

class_names = train_ds.class_names
print(f"Categories detected: {class_names}")

# Optimization for training pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. Build Ultra-Lightweight CNN
base_model = MobileNetV2(
    input_shape=(96, 96, 1), 
    include_top=False, 
    weights=None, 
    alpha=0.25  # Shrinks model size drastically for Arduino RAM
)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(96, 96, 1)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Train Model
model.fit(train_ds, validation_data=val_ds, epochs=15)

# 4. INT8 Full Integer Quantization
print("Converting to INT8 for Arduino microcontrollers...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for x, _ in train_ds.take(10):
        yield [x]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTIN_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

# Save final micro model
with open("assistive_device_model.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Success! Compressed model saved as 'assistive_device_model.tflite'.")
