
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


from sklearn.svm import SVC

history = None
chanel = 3
size_w, size_h = 104, 104
epochs = 80
batch = 4

name_file = "/home/oberon/Downloads/dataset_test"
path = "{}".format(name_file)

data = []
labels = []

(trainData, testData, trainLabels, testLabels) = (None, None, None, None)
input_shape = (window_size, size_w, size_h, chanel)

def mask_resize(img, size):
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
            
            if name_label == "fall":
                labels.append(0)
            elif name_label == "nofall":
                labels.append(1)
            
          
    # encode the labels
    labels = np.array(labels)
    data = np.array(data) / 255.0
    
    data = data.reshape(data.shape[0], window_size, size_w, size_h, chanel)


def train_svm():
    global name_file, window_size, history, chanel, size_w, size_h, epochs, batch, path, data, data, labels, trainData, testData, trainLabels, testLabels, input_shape
    
    print("[INFO] constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        data, labels, test_size=0.5, random_state=42)
    

    print (trainLabels)
    
    model = SVC(kernel='linear', class_weight='balanced', probability=True)
    
    print(trainData.shape)
    m_samples = trainLabels.shape[0]
    
    trainData = trainData.reshape(m_samples, -1)
    model.fit(trainData, trainLabels)
    
    
    y_true = testLabels
    
    testData = testData.reshape(testLabels.shape[0], -1)
    
    y_pred = model.predict(testData)
    
    print(y_pred)
    y_scores = model.predict_proba(testData)
    y_scores=[i.max() for i in y_scores]
    print(y_scores)
    
    accu = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    
    roc_auc = roc_auc_score(y_true, y_scores)
    (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred).ravel()
    

    print(accu)
    print(recall)
    print(precision)
    # print(roc_auc)
    # print(((tn, fp, fn, tp)))
    
    

init_net()
train_svm()
