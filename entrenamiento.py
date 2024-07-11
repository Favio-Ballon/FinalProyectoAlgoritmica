import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import tensorflow as tf

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# reducir logs innecesarios
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configuración del generador de datos
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_gen = datagen.flow_from_directory(
    'fotos',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    'fotos',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Construcción del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(train_gen.class_indices), activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(train_gen, epochs=10, validation_data=val_gen)

# Guardar el modelo
model.save('face_recognition_model.h5')
