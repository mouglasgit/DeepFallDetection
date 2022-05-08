
window_size = 8
# -*- coding: utf-8 -*-

import os
import cv2
import imutils
import numpy as np

from imutils import paths
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



import keras
# from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation, ZeroPadding3D, GlobalAveragePooling3D
from keras.preprocessing import image
from keras.layers.pooling import MaxPooling3D
from keras.layers.convolutional import Conv3D
from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional_recurrent import ConvLSTM2D
# from keras.layers import BatchNormalization
from keras.layers import Deconvolution3D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


history = None
chanel = 1
size_w, size_h = 64, 64
epochs = 80
batch = 4

name_file = "/home/oberon/Downloads/dataset_test"
path = "{}".format(name_file)

data = []
labels = []

(trainData, testData, trainLabels, testLabels) = (None, None, None, None)
input_shape = (window_size, size_w, size_h, chanel)

def mask_resize(img, size):
    #rh, rw, _ = img.shape 
    rh, rw = img.shape
    #mask = np.ones((size, size, 3), np.uint8) - 255
    mask = np.ones((size, size), np.uint8) - 255
    if rh > rw:
        img = imutils.resize(img, height=size)
        #ih, iw, _ = img.shape
        print(img.shape)
        ih, iw = img.shape
        mask[0:ih, 0:iw] = img
    else:
        img = imutils.resize(img, width=size)
        #ih, iw, _ = img.shape
        ih, iw = img.shape
        mask[0:ih, 0:iw] = img
        
    return mask


def image_to_feature_vector(images, size=()):
    global window_size, history, chanel, size_w, size_h, epochs, batch, path, data, labels, trainData, testData, trainLabels, testLabels, input_shape
    
    images = images[:window_size]
    res_imagens = np.array([cv2.resize(imagem, (size[1], size[2])) for imagem in images]).flatten()
    return res_imagens


def init_net():
    global window_size, history, chanel, size_w, size_h, epochs, batch, path, data, labels, trainData, testData, trainLabels, testLabels, input_shape
    
    size = len(list(paths.list_images(path)))
    idx = 0
    
    dataset_path = os.listdir(path)
    
    for name_label in dataset_path:
    
        path_samples = "{}/{}".format(path, name_label)
        samples = os.listdir(path_samples)
        
        for sample in samples:
            
            name_sample = "{}/{}/{}".format(path, name_label, sample)
            name_sample = os.listdir(name_sample)
                    
            item_imagem = []    
            for (i, subsample) in enumerate(name_sample):
                name_sample = "{}/{}/{}/{}".format(path, name_label, sample, subsample)
                
                if chanel == 1:
                    image = cv2.imread(name_sample, 0)
                    image = mask_resize(image, size_w)
                else:
                    image = cv2.imread(name_sample, 1)
                    image = mask_resize(image, size_w)
                item_imagem.append(image)
            
                per = float((idx * 100) / size)
                print("[INFO] processed {}/{:.2f}%".format(idx, per))
                idx += 1
            
            features = image_to_feature_vector(item_imagem, (window_size, size_w, size_h))
            data.append(features)
            labels.append(name_label)
          
    # encode the labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    data = np.array(data) / 255.0
    labels = np_utils.to_categorical(labels, 2)
    
    data = data.reshape(data.shape[0], window_size, size_w, size_h, chanel)


#---------------------------------------
def DSTCAE_Deconv(img_width, img_height, win_length):
    """
    int win_length: Length of window of frames

    Replace Upsampling with Deconv
    """
    

    
    input_shape = (win_length, img_width, img_height, 1)


    temp_pool = 2
    temp_depth = 5
    
    
    model = Sequential()
    model.add(Conv3D(16, (temp_depth, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling3D((2, 2, 2), padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((temp_pool, 2, 2), padding='same'))
    
    model.add(Deconvolution3D(8, (temp_depth, 3, 3), strides=(2, 2, 2), activation='relu', padding='same'))
    model.add(Deconvolution3D(8, (temp_depth, 3, 3), strides=(2, 2, 2), activation='relu', padding='same'))
    
    model.add(Conv3D(1, (temp_depth, 3, 3), activation='tanh', padding='same'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    
    
    print(model.summary())
    

    model_type = 'conv'
    model_name = 'DSTCAE_Deconv-pooling-win_{}'.format(win_length)    

    return model, model_name, model_type
#---------------------------------------


def compile_and_train(model):
    global name_file, window_size, history, chanel, size_w, size_h, epochs, batch, path, data, data, labels, trainData, testData, trainLabels, testLabels, input_shape
    
    print("[INFO] constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        data, labels, test_size=0.2, random_state=42)
    
    print("[INFO] compiling model...")
        
    #autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

    model.compile(loss='mean_squared_error',
              optimizer='adadelta',
              #optimizer=keras.optimizers.Adagrad(lr=0.1),
              metrics=['accuracy'])
    
    
    history = model.fit(trainData, trainLabels,
          # class_weight={0: 1., 1: 1.53},
          batch_size=batch,
          epochs=epochs,
          verbose=2,
          validation_data=(testData, testLabels))
    
    #model.fit(trainData, trainLabels, epochs=50, batch_size=1, verbose=2,
    #             validation_data=(testData, testLabels))
    
    

init_net()
model, model_name, model_type = DSTCAE_Deconv(64, 64, 8)
compile_and_train(model)
