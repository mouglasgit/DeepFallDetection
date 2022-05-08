# coding: utf-8

from __future__ import division, print_function

import cv2
import numpy as np
import tensorflow as tf

import datetime
from facade.model import yolov3
from utils.nms_utils import gpu_nms
from utils.misc_utils import parse_anchors, read_class_names

cap=cv2.VideoCapture('/home/oberon/videos_test/action_test.avi')
#cap.set(cv2.CAP_PROP_POS_FRAMES, 688)

fourcc = cv2.VideoWriter_fourcc(*'x264')
ret, imgcv = cap.read()
# imgcv = imutils.resize(imgcv, width=1980)

h, w, _ = imgcv.shape 
out = cv2.VideoWriter(str("video.avi"), fourcc, 14.0, (imgcv.shape[1], imgcv.shape[0]))

anchors = parse_anchors("./weights/fall_yolo/yolo_anchors.txt")
classes = read_class_names("./weights/fall_yolo/labels.names")
num_class = len(classes)
new_size = [416, 416]
restore_path = "./weights/fall_yolo/darknet_weights/yolov3.ckpt"

tmp_fall = False
READ = [0, 0, 255]
xl = 10
yl = 45


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

    while True:
        start = datetime.datetime.now().microsecond * 0.001
        
        ret, img_ori = cap.read()
#         img_ori = imutils.resize(img_ori, width=1980)

        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        
        boxes_[:, 0] *= (width_ori / float(new_size[0]))
        boxes_[:, 2] *= (width_ori / float(new_size[0]))
        boxes_[:, 1] *= (height_ori / float(new_size[1]))
        boxes_[:, 3] *= (height_ori / float(new_size[1]))

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            
            x = int(x0)
            y = int(y0)
            w = int(x1 - x0)
            h = int(y1 - y0)
            
            
            display_txt = '{} {:0.2f} W:{} H:{}'.format(classes[labels_[i]], scores_[i], w, h)
            
            print (display_txt)
            
            if classes[labels_[i]] == 'nofall':
                cv2.putText(img_ori, "Status: No Fall", (xl + 3 + 1, yl + 3 + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(img_ori, "Status: No Fall", (xl + 3, yl + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
                cv2.rectangle(img_ori, (x, y - 20), (x + 180, y), (0, 255, 255), -1)
                cv2.putText(img_ori, display_txt, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 0, 0), 1)
                cv2.rectangle(img_ori, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            elif classes[labels_[i]] == 'fall' and float(scores_[i]) > 0.99:
                
                img_ori = np.uint8(img_ori * READ)
                    
                cv2.putText(img_ori, "Status: Fall", (xl + 3 + 1, yl + 3 + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(img_ori, "Status: Fall", (xl + 3, yl + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
                cv2.rectangle(img_ori, (x, y - 20), (x + 180, y), (0, 255, 0), -1)
                cv2.putText(img_ori, display_txt, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 0, 0), 1)
                cv2.rectangle(img_ori, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        #FPS----
        end = datetime.datetime.now().microsecond * 0.001
        elapse = end - start
        fps = np.round(1000.0 / elapse, 3)
        cv2.putText(np.uint8(img_ori), 'FPS: %s' % fps, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        out.write(img_ori)
        cv2.imshow('crop', img_ori)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            out.release()
            break

    cap.release()
