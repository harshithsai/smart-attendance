import os
import cv2
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

data_path = 'D:/Downloads/project/dataset'

image_size = (224, 224)

def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    return img

training_data_generation = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,validation_split=0.2,preprocessing_function=preprocess_image)

training_data = training_data_generation.flow_from_directory(data_path,target_size=image_size,batch_size=32,class_mode='categorical',subset='training')
 
validation_data = training_data_generation.flow_from_directory(data_path,target_size=image_size,batch_size=32,class_mode='categorical',subset='validation')

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(training_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(training_data,validation_data=validation_data,epochs=25)

model.save('trained_model.h5')
