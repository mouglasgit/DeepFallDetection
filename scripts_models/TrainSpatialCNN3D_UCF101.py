# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

from imutils import paths
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation
from keras.preprocessing import image
from keras.layers.pooling import MaxPooling3D
from keras.layers.convolutional import Conv3D
from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

import imutils
import tensorflow as tf
import time
MEMORY_USED_GPU = 0.65
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_USED_GPU)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class App:
    def __init__(self):
        
        self.window_size = 8
        self.history = None
        # param netwking
        self.chanel = 3
        self.num_classes = 10
        self.size_w, self.size_h = 100, 100
        self.epochs = 30
        self.batch = 4
        self.path = "/media/oberon/actions_datasets/UCF101/process_ucf101"

        # initialize the data matrix and labels list
        self.data = []
        self.labels = []
        
        (self.trainData, self.testData, self.trainLabels, self.testLabels) = (None, None, None, None)
        self.input_shape = (self.window_size, self.size_w, self.size_h, self.chanel)
        
        # init net
        self.init_net()
        self.model = self.conv_3D()
        self.compile_and_train(self.model)
        self.test(self.model)
    
    def image_to_feature_vector(self, images, size=()):
        images = images[:self.window_size]
        res_imagens = np.array([cv2.resize(imagem, (size[1], size[2])) for imagem in images]).flatten()
        return res_imagens

    def init_net(self):
        size = len(list(paths.list_images(self.path)))
        idx = 0
        
        # dataset_path = os.listdir(self.path)
        dataset_path = os.listdir(self.path)[:self.num_classes]
        
        for name_label in dataset_path:
        
            path_samples = "{}/{}".format(self.path, name_label)
            samples = os.listdir(path_samples)
            
            for sample in samples:
                
                name_sample = "{}/{}/{}".format(self.path, name_label, sample)
                name_sample = os.listdir(name_sample)
                        
                item_imagem = []    
                for (i, subsample) in enumerate(name_sample):
                    name_sample = "{}/{}/{}/{}".format(self.path, name_label, sample, subsample)
                    
                    if self.chanel == 1:
                        image = cv2.imread(name_sample, 0)
                    else:
                        image = cv2.imread(name_sample, 1)
                    item_imagem.append(image)
                    
#                     if i >= self.window_size:
#                         continue
                
                    per = float((idx * 100) / size)
                    print("[INFO] processed {}/{:.2f}%".format(idx, per))
                    idx += 1
                
                features = self.image_to_feature_vector(item_imagem, (self.window_size, self.size_w, self.size_h))
                self.data.append(features)
                self.labels.append(name_label)
              
        # encode the labels, converting them from strings to integers
        le = LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        
        self.data = np.array(self.data) / 255.0
        self.labels = np_utils.to_categorical(self.labels, self.num_classes)
        
        self.data = self.data.reshape(self.data.shape[0], self.window_size, self.size_w, self.size_h, self.chanel)
            
    def conv_3D(self):
        
        print("[INFO] constructing training/testing split...")
        (self.trainData, self.testData, self.trainLabels, self.testLabels) = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42)
        
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
        model.add(Dense(self.num_classes, activation='softmax'))
        
        print(model.summary())
        
        return model
    
    def test(self, model):
        # show the accuracy on the testing set
        print("[INFO] evaluating on testing set...")
        (loss, accuracy) = model.evaluate(self.testData, self.testLabels,
            batch_size=self.batch, verbose=0)
        
        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
            accuracy * 100))
        
        
        print(model.summary())
        
        print("[INFO] history on testing set...")
        print(self.history.history.keys())
    
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Evolution of loss and accuracy')
    
        # summarize history for accuracy
        ax1.plot(self.history.history['acc'], label='Train')
        ax1.plot(self.history.history['val_acc'], label='Val')
        ax1.set(xlabel='Epoch', ylabel='Accuracy')
        ax1.legend(loc='upper right')
    
        # summarize history for loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Val')
        ax2.set(xlabel='Epoch', ylabel='Loss')
        ax2.legend(loc='upper right')
        
        plt.savefig('fig_cnn3d.png')
        
    def compile_and_train(self, model):
        # train the model using SGD
        print("[INFO] compiling model...")
        
        
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        
        self.history = model.fit(self.trainData, self.trainLabels,
                  # class_weight={0: 1., 1: 1.53},
                  batch_size=self.batch,
                  epochs=self.epochs,
                  verbose=2,
                  validation_data=(self.testData, self.testLabels))
                # )
        
        model.save_weights('ucf101_detection_cnn3d.hdf5')

          
App()

