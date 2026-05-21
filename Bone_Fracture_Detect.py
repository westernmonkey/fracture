import os
import ssl
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from PIL import Image
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import ImageFile

ssl._create_default_https_context = ssl._create_unverified_context
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_truncated_images(directory):
    deleted_images = repaired_images = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isdir(file_path):
            continue
        try:
            with Image.open(file_path) as img:
                img.load()
                img.save(file_path)
                repaired_images += 1
        except (IOError, SyntaxError):
            print(f"Deleting corrupted/truncated image: {filename}")
            os.remove(file_path)
            deleted_images += 1
    print(f"Total repaired images: {repaired_images}")
    print(f"Total deleted images: {deleted_images}")

for directory in ['/Users/stavansagala/Training_Dataset/Fractured_Dataset',
                  '/Users/stavansagala/Training_Dataset/NF_Dataset']:
    process_truncated_images(directory)

TRAINING_DIR = '/Users/stavansagala/Training_Dataset'
VALIDATION_DIR = '/Users/stavansagala/Validation_Dataset'
TESTING_DIR = '/Users/stavansagala/Testing_Dataset'
IMG_HEIGHT, IMG_WIDTH = 200, 200
BATCH_SIZE = 20
EPOCHS = 10

train_datagen = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, vertical_flip=True,
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, fill_mode='nearest'
)

common_args = dict(target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary', color_mode='rgb')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=BATCH_SIZE, **common_args)
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(VALIDATION_DIR, batch_size=BATCH_SIZE, **common_args)
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(TESTING_DIR, batch_size=1, class_mode=None, shuffle=False, color_mode='rgb', target_size=(IMG_HEIGHT, IMG_WIDTH))

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=EPOCHS,
          validation_data=validation_generator, validation_steps=len(validation_generator),
          class_weight=class_weights)
model.save('bone_fracture_model.h5')
print("Model saved as bone_fracture_model.h5")

def load_and_predict():
    loaded_model = load_model('bone_fracture_model.h5')
    while True:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select an X-ray image for prediction")
        if file_path:
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((IMG_HEIGHT, IMG_WIDTH))
            prediction = loaded_model.predict(np.expand_dims(img_to_array(img) / 255.0, axis=0))[0][0]
            label_text = "Fracture Detected [1]" if prediction >= 0.5 else "Not Fractured [0]"
            color = (0, 255, 0) if prediction >= 0.5 else (255, 0, 0)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if prediction >= 0.5:
                h, w = img_cv.shape[:2]
                cv2.circle(img_cv, (w // 2, h // 2), 12, color, 2)
            plt.figure()
            plt.text(0.5, 1.05, label_text, ha='center', va='top', fontsize=10, color='black', transform=plt.gca().transAxes)
            plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

load_and_predict()