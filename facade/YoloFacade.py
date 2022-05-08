# coding: utf-8

import cv2
import numpy as np
import tensorflow as tf
from facade.model import yolov3
from utils.nms_utils import gpu_nms
from utils.misc_utils import parse_anchors, read_class_names

class YoloFacade:
    
    def __init__(self):
        
        self.detected = []
    
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.graph = tf.get_default_graph()
        
        anchors = parse_anchors("./weights/fall_yolo/yolo_anchors.txt")
        self.classes = read_class_names("./weights/fall_yolo/labels.names")
        num_class = len(self.classes)
        self.new_size = [416, 416]
        restore_path = "./weights/fall_yolo/darknet_weights/yolov3.ckpt"
    
        self.input_data = tf.placeholder(tf.float32, [1, self.new_size[1], self.new_size[0], 3], name='input_data')
        yolo_model = yolov3(num_class, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(self.input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    
        pred_scores = pred_confs * pred_probs
        self.boxes, self.scores, self.labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=30, score_thresh=0.15, nms_thresh=0.15)
    
        saver = tf.train.Saver()
        saver.restore(self.sess, restore_path)
    
    def predict(self, img_ori):
        
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(self.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        
        boxes_, scores_, labels_ = self.sess.run([self.boxes, self.scores, self.labels], feed_dict={self.input_data: img})
        
        boxes_[:, 0] *= (width_ori / float(self.new_size[0]))
        boxes_[:, 2] *= (width_ori / float(self.new_size[0]))
        boxes_[:, 1] *= (height_ori / float(self.new_size[1]))
        boxes_[:, 3] *= (height_ori / float(self.new_size[1]))
        
        if len(boxes_) == 0:
            print('no detection!')
        else:
            tmp_detected = []
            for i in range(len(boxes_)):
                if self.classes[labels_[i]] == 'fall' or self.classes[labels_[i]] == 'nofall':
                    x0, y0, x1, y1 = boxes_[i]
                    x = int(x0)
                    y = int(y0)
                    w = int(x1)
                    h = int(y1)
                    
                    tmp_detected.append((np.array([y, x, h, w]), self.classes[labels_[i]]))
            self.detected = tmp_detected
        return self.detected
    
