
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# AGGRESSIVE AUGMENTATION
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2
)

train = datagen.flow_from_directory("dataset/photoshop_ela", target_size=(128,128), batch_size=32, class_mode="binary", subset="training")
val = datagen.flow_from_directory("dataset/photoshop_ela", target_size=(128,128), batch_size=32, class_mode="binary", subset="validation")

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128,(3,3),activation="relu", name="conv2d_2"), # Name this for Grad-CAM
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=25)
model.save("models/photoshop_model.h5")