# -*- coding: utf-8 -*-

import cv2
import imutils
import numpy as np

import keras
from keras.layers import Dense
from keras.layers import Activation 
from keras.models import Sequential
from keras.layers.pooling import MaxPooling3D
from keras.layers.convolutional import Conv3D
from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional_recurrent import ConvLSTM2D


class TemporalModelFacade:
    
    def __init__(self, model_type="cnn3d"):
        
        self.batch = 1
        self.chanel = 3
        self.window_size = 8
        
        self.model_type = model_type
        
        if self.model_type == "cnn3d":
            self.size_w, self.size_h = 100, 100
            self.model = self.conv_cnn3d()
        elif self.model_type == "cnn2d_lstm":
            self.size_w, self.size_h = 102, 102
            self.model = self.conv_cnn2d_lstm()
        
        self.compile(self.model)
    
    def mask_resize(self, img, size):
        rh, rw, _ = img.shape 
        mask = np.ones((size, size, 3), np.uint8) - 255
        if rh > rw:
            img = imutils.resize(img, height=size)
            ih, iw, _ = img.shape
            mask[0:ih, 0:iw] = img
        else:
            img = imutils.resize(img, width=size)
            ih, iw, _ = img.shape
            mask[0:ih, 0:iw] = img
            
        return mask
    
    def image_to_feature_vector(self, images, size=()):
        images = images[:self.window_size]
        res_imagens = np.array([cv2.resize(imagem, (size[1], size[2])) for imagem in images]).flatten()
        return res_imagens
    
    def conv_cnn3d(self):
        print("[INFO] constructing training/testing split...")
        self.input_shape = (self.window_size, self.size_w, self.size_h, self.chanel)
        
        model = Sequential()
    
        model.add(Conv3D(64, kernel_size=(2, 2, 2), input_shape=self.input_shape))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(Conv3D(128, kernel_size=(2, 2, 2)))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
      
        model.add(Conv3D(128, kernel_size=(2, 2, 2)))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(Conv3D(512, kernel_size=(2, 2, 2)))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))
      
        model.add(Conv3D(512, kernel_size=(1, 2, 2)))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(Conv3D(512, kernel_size=(1, 2, 2)))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    
        model.add(Conv3D(1024, kernel_size=(1, 2, 2)))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    
        model.add(Flatten())
    
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='softmax'))
        
        print(model.summary())
        
        return model
     
    def conv_cnn2d_lstm(self):
        print("[INFO] constructing training/testing split...")
        self.input_shape = (self.window_size, self.size_w, self.size_h, self.chanel)
        
        model = Sequential()
    
        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), recurrent_activation='hard_sigmoid', return_sequences=True, input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(keras.layers.BatchNormalization())
        
        model.add(ConvLSTM2D(filters=128, kernel_size=(2, 2), return_sequences=True))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(keras.layers.BatchNormalization())
            
        model.add(Flatten())
    
        model.add(Dense(256))
        model.add(keras.layers.BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
    
        print(model.summary())
    
        return model
    
    def compile(self, model):
        print("[INFO] compiling model...")
        
        if self.model_type == "cnn3d":
            model.load_weights('weights/fall_cnn3d_5x3.hdf5', by_name=True)
        elif self.model_type == "cnn2d_lstm":
            model.load_weights('weights/fall_cnn2d_lstm_5x3.hdf5')
        
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    
    def pred(self, images, model):
        
        images = np.array(images)
        pred = self.image_to_feature_vector(images, (self.window_size, self.size_w, self.size_h))
        pred = pred.reshape(1, self.window_size, self.size_w, self.size_h, self.chanel)
        
        res = model.predict(pred, batch_size=self.batch, verbose=0)
    
        return (res[0][0]), (res[0][1])
