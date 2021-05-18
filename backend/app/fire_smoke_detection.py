import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16                         # pretrained CNN                        
from tensorflow.keras.callbacks import ModelCheckpoint                  # furter train the saved model 
from tensorflow.keras import models, layers, optimizers                 # building DNN is keras 
from tensorflow.keras.models import load_model               # load saved model 
from tensorflow.keras.preprocessing import image

class FireSmokeDetection:

    def __init__(self, data_path, model_path):
        self.model_path = model_path
        self.data_path = data_path
        self.batch_size = 100 # Number of training examples to process before updating our models variables
        self.img_shape = 100 # Our data consists of images with width of 150 pixels and height of 150 pixels
        print("Model path " + model_path)
        self.model = load_model(model_path)

        self.image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

        self.data_gen = self.image_gen_train.flow_from_directory(batch_size=self.batch_size,
                                                        directory=data_path,
                                                        shuffle=False,
                                                        target_size=(self.img_shape,self.img_shape), 
                                                        class_mode='binary')


    def get_all_img_label_preds(self):
        img, label = self.data_gen[0] 

        # Predicting the images from the first batch 
        pred = np.round(self.model.predict(img)).flatten()

        return img, label, pred

    def get_pred(self, image_file_path):
        img = self.read_image(image_file_path)

        # pred = self.model.predict(img)

        raw_pred = (self.model.predict(img)).flatten()

        pred = np.round(self.model.predict(img)).flatten()
        
        return pred[0], raw_pred[0]

    def read_image(self, img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(100, 100))
        image_array = tf.keras.preprocessing.image.img_to_array(img)
        img_batch = np.expand_dims(image_array, axis=0)
        return img_batch

    def label_dict(self):
        return {1.0: 'No fire', 0.0: 'Fire'}



