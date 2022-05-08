# coding: utf-8

from __future__ import division, print_function

import os
import cv2
import math
import numpy as np
import tensorflow as tf

from model import yolov3
from utils.nms_utils import gpu_nms
from utils.misc_utils import parse_anchors, read_class_names

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, tostring


def ge_file_xml(path_save_anotation, name_img, res_wid, res_hei, objetos_coo):
        annotation = Element('annotation')
        
        folder = SubElement(annotation, 'folder')
        folder.text = 'Airplane2019'
        
        filename = SubElement(annotation, 'filename')
        filename.text = str(name_img)
        
        # source--
        source = SubElement(annotation, 'source')
        
        database = SubElement(source, 'database')
        database.text = 'The Airplane2019 Database'
        
        annotation1 = SubElement(source, 'annotation')
        annotation1.text = 'Airplane2019'
        
        image = SubElement(source, 'image')
        image.text = 'flickr'
        
        flickrid = SubElement(source, 'flickrid')
        flickrid.text = '08021989'
        # source...
        
        #owner-----
        owner = SubElement(annotation, 'owner')
        
        flickrid1 = SubElement(owner, 'flickrid')
        flickrid1.text = 'me'
        
        name_own = SubElement(owner, 'name')
        name_own.text = 'Mouglas'
        # owner....
        
        #size-----
        size = SubElement(annotation, 'size')
        
        width = SubElement(size, 'width')
        width.text = str(res_wid)
        
        height = SubElement(size, 'height')
        height.text = str(res_hei)
        
        depth = SubElement(size, 'depth')
        depth.text = '3'
        # size....
        
        #segmented-----
        segmented = SubElement(annotation, 'segmented')
        segmented.text = '0'
        # segmented....
        
        #object-----
        for obj in objetos_coo:
            
            (type_class, name_index, x, y, w, h) = obj
            
            object = SubElement(annotation, 'object')
            
            class_name = SubElement(object, 'name')
            class_name.text = str(type_class)
            
            index = SubElement(object, 'index')
            index.text = str(name_index)
            
            pose = SubElement(object, 'pose')
            pose.text = 'Unspecified'
            
            truncated = SubElement(object, 'truncated')
            truncated.text = '0'
            
            difficult = SubElement(object, 'difficult')
            difficult.text = '0'
            
            bndbox = SubElement(object, 'bndbox')
            
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(x)
            
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(y)
            
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(int(w + x))
            
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(int(h + y))
        
        # object....
        
        # files
        #==============================================
        nome_save = name_img[:-4]
        
        fileXML = open(path_save_anotation + str(nome_save) + ".xml", "w")
        fileXML.write('')
        
        fileXML = open(path_save_anotation + str(nome_save) + ".xml", "a")
        fileXML.write(tostring(annotation, encoding="unicode"))
        fileXML.close()
        
        # print tostring(annotation)
    
    #==========================================================================================

#----------------------------------------------------


anchors = parse_anchors("./data_fall/yolo_anchors.txt")
classes = read_class_names("./data_fall/labels.names")
num_class = len(classes)
new_size = [416, 416]
restore_path = "./data_fall/darknet_weights/yolov3.ckpt"

#-----------------------------------------
lab_classes = classes
class_colors = {}
for i in range(0, len(lab_classes)):
    # This can probably be written in a more elegant manner
    hue = math.pow(i, 9) / len(lab_classes)
    col = np.zeros((1, 1, 3)).astype("uint8")
    col[0][0][0] = hue
    col[0][0][1] = 140  # Saturation
    col[0][0][2] = 220  # Value
    cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
    col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
    class_colors.update({str(lab_classes[i]):col}) 
#-----------------------------------------


    
with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
    yolo_model = yolov3(num_class, anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=30, score_thresh=0.15, nms_thresh=0.15)

    saver = tf.train.Saver()
    saver.restore(sess, restore_path)
    
    
    list_dirs = range(1, 41)
    for atual in list_dirs:
        print(list_dirs)
        
        #path_abs = '/media/oberon/MouglasHD2T/actions_datasets/URFallDetectionDataset/nofall/rgb/fall-{:02d}-cam0-rgb'.format(atual)
        path_abs = '/media/oberon/MouglasHD2T/actions_datasets/URFallDetectionDataset/nofall/rgb/adl-{:02d}-cam0-rgb'.format(atual)
        
        path_load_img = '{}/JPEGImages/'.format(path_abs)
        path_save_ano = '{}/Annotations/'.format(path_abs)
        
        files = os.listdir(path_load_img)
        
        size = len(files)
        print(size)
        idx = 0
    
        for f in files:
            path_src = "{}{}".format(path_load_img, str(f))
            
            print (path_src)
            
            img_ori = cv2.imread(path_src, 1)
    
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.
            
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            
            # rescale the coordinates to the original image
            boxes_[:, 0] *= (width_ori / float(new_size[0]))
            boxes_[:, 2] *= (width_ori / float(new_size[0]))
            boxes_[:, 1] *= (height_ori / float(new_size[1]))
            boxes_[:, 3] *= (height_ori / float(new_size[1]))
            
            objetos_coo = []
    
            if len(boxes_) > 0:
                #-------------
                i = 0
                max_score = 0
                for j in range(len(boxes_)):
                    if scores_[j] > max_score:
                        max_score = scores_[j]
                        i = j
                        
                x0, y0, x1, y1 = boxes_[i]
                #-----------
                
                x = int(x0)
                y = int(y0)
                w = int(x1 - x0)
                h = int(y1 - y0)
                
                # class_name = classes[labels_[i]]
                class_name = "nofall"
                
                
                
                display_txt = '{} {:0.2f}'.format(class_name, scores_[i])
                
                cv2.rectangle(img_ori, (x, y - 20), (x + 50 + len(class_name) * 6, y), class_colors[classes[labels_[i]]], -1)
                cv2.putText(img_ori, display_txt, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 0, 0), 1)
                cv2.rectangle(img_ori, (x, y), (x + w, y + h), class_colors[classes[labels_[i]]], 2)
                
                index = 0  
                #----------------------------------
                if x <= 0:
                    x = 1
                    
                if y <= 0:
                    y = 1
                
                if x + w >= width_ori:    
                    w = width_ori - x - 1
                
                if y + h >= height_ori:
                    h = height_ori - y - 1
                
                #----------------------------------
                
                objetos_coo.append([class_name, index, x, y, w, h])
            
            #-----------
            
            if len(objetos_coo) > 0:
                ge_file_xml(path_save_ano, f, width_ori, height_ori, objetos_coo)
                
            per = float((idx * 100) / size)
            print (("%0.2f%%") % (per))    
            idx += 1
            
            cv2.imshow('crop', img_ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
