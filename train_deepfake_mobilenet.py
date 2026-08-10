import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

IMG_SIZE = (128, 128)
BATCH = 32

# 1. Aggressive Data Augmentation
# This helps the model find deepfake artifacts even if the face is tilted or zoomed
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # Helpful for tilted faces
    width_shift_range=0.1, 
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train = datagen.flow_from_directory(
    "dataset/deepfake/train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary',
    subset='training'
)

val = datagen.flow_from_directory(
    "dataset/deepfake/train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary',
    subset='validation'
)

print(f"Class Indices: {train.class_indices}") 
# Note: Usually {'fake': 0, 'real': 1}

# 2. Load Pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base layers so we don't destroy the ImageNet weights
for layer in base_model.layers:
    layer.trainable = False

# 3. Add Custom Head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x) # Prevents overfitting (memorizing the data)

# Ensure the last layer before prediction has a name for Grad-CAM
# In MobileNetV2, the very last conv layer is usually 'out_relu'
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# 4. Compile with a slower learning rate for better precision
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train
# Increased epochs to 20 to give the augmentation time to work
model.fit(
    train, 
    validation_data=val, 
    epochs=20
)

# 6. Save
os.makedirs("models", exist_ok=True)
model.save("models/deepfake_model.h5")

print("✅ MobileNet Deepfake model trained with Augmentation and Dropout")