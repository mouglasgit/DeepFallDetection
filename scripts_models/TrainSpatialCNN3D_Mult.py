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


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import imutils
import tensorflow as tf
import time
MEMORY_USED_GPU = 0.75
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
        self.num_classes = 2
        self.size_w, self.size_h = 112, 112
        self.epochs = 60
        self.batch = 1
        self.path = "/media/oberon/actions_datasets/MultCamFallDataset/dataset/mul_seq_dataset"

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
        
        #----------
        test_data = self.testData
        test_labels = self.testLabels
        
        # test_data=trainData[500:1200]
        # test_labels=trainLabels[500:1200]
        
        print(len(test_data))
        print(len(test_labels))
        
        fall, nofall = self.pred(test_data[0], self.model)
        
        print(fall)
        print(nofall)
        print(test_labels[0])
        
        y_pred = []
        y_scores = []
        y_true = []
        for i in range(0, len(test_labels)):
            y_pred.append(np.argmax(self.pred(test_data[i], self.model)))
            y_scores.append(np.max(self.pred(test_data[i], self.model)))
            y_true.append(np.argmax(test_labels[i]))
        
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        y_true = np.array(y_true)
        
        accu = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)
        (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred).ravel()
        
        print(accu)
        print(recall)
        print(roc_auc)
        print(((tn, fp, fn, tp)))
        
    
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
                        image = self.mask_resize(image, self.size_w)
                    else:
                        image = cv2.imread(name_sample, 1)
                        image = self.mask_resize(image, self.size_w)
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
        # loss=0.6713, accuracy: 73.3333%
        print("[INFO] constructing training/testing split...")
        (self.trainData, self.testData, self.trainLabels, self.testLabels) = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42)
        

        model = Sequential()
     
        model.add(Conv3D(64, kernel_size=(3, 3, 3), input_shape=self.input_shape))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(keras.layers.BatchNormalization())
       
        model.add(Conv3D(128, kernel_size=(2, 2, 2)))
        model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        model.add(keras.layers.BatchNormalization())
       
        
        model.add(Flatten())
     
        model.add(Dense(128))
        model.add(keras.layers.BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
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
    
    def pred(self, images, model):
        images = np.array(images)
        pred = self.image_to_feature_vector(images, (self.window_size, self.size_w, self.size_h))
        pred = pred.reshape(1, self.window_size, self.size_w, self.size_h, self.chanel)
    
        res = model.predict(pred, batch_size=self.batch, verbose=0)
    
        return (res[0][0]), (res[0][1])
    
    def compile_and_train(self, model):
        # train the model using SGD
        print("[INFO] compiling model...")
        
        
        
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam(),
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

