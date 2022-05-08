# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import deque, Counter
from sklearn.utils.linear_assignment_ import linear_assignment

import imutils
import collections
from facade.YoloFacade import YoloFacade
from facade.KalmanFilterTrack import KalmanFilterTrack
from facade.TemporalModelFacade import TemporalModelFacade


class CombineModels:

    def __init__(self): 
        self.window = 8
        self.actions = dict()
        
        self.idx_frame = 0  
        self.lifetime = 5
        self.min_pattern_counter = 1   
        self.trackers = []  
        self.ids_track = deque(['1', '2', '3', '4', '5', '6', '7', '7', '8', '9', '10'])
        
        self.yolo_detect = YoloFacade()
        self.temporal_model = TemporalModelFacade(model_type="cnn3d")
        # temp_model = TemporalModelFacade(model_type="cnn2d_lstm")

    def calc_iou(self, a, b):
        w_intersect = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
        h_intersect = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
        r_intersect = w_intersect * h_intersect
        r_a = (a[2] - a[0]) * (a[3] - a[1])
        r_b = (b[2] - b[0]) * (b[3] - b[1])
      
        return float(r_intersect) / (r_a + r_b - r_intersect)
    
    def find_best_iou(self, trackers, detections):
        best_iou = np.zeros((len(trackers), len(detections)), dtype=np.float32)
        for t, track in enumerate(trackers):
            for d, detect in enumerate(detections):
                best_iou[t, d] = self.calc_iou(track, detect[0]) 
        
        compatible_idx = linear_assignment(-best_iou)
        
        return best_iou, compatible_idx 
    
    def check_compatibility_tracker_detector(self, trackers, detections, max_thresh_iou=0.5): 
        best_iou, compatible_idx = self.find_best_iou(trackers, detections)
        incompatible_trackers = [t for t, _ in enumerate(trackers) if(t not in compatible_idx[:, 0])]
        incompatible_detections = [d for d, _ in enumerate(detections) if(d not in compatible_idx[:, 1])]
        compatibles = []
       
        for m in compatible_idx:
            if(best_iou[m[0], m[1]] < max_thresh_iou):
                incompatible_trackers.append(m[0])
                incompatible_detections.append(m[1])
            else:
                compatibles.append(m.reshape(1, 2))
            
        compatibles = np.empty((0, 2), dtype=int) if len(compatibles) == 0 else np.concatenate(compatibles, axis=0) 
        
        return compatibles, np.array(incompatible_detections), np.array(incompatible_trackers)
    
    def new_measurement(self, aux_track):
        position = aux_track.current_state.T[0].tolist()
        position = [position[0], position[2], position[4], position[6]]
        aux_track.coordinates = position
        
        return position
        
    def handle_compatible_detections(self, detect_rec, compatibles, track_rec):
        if compatibles.size > 0:
            for track_idx, detect_idx in compatibles:
                detect = detect_rec[detect_idx][0]
                detect = np.expand_dims(detect, axis=0).T
                aux_track = self.trackers[track_idx][0]
                aux_track.gain_update_measurement(detect)
                position = self.new_measurement(aux_track)
                track_rec[track_idx] = position
                aux_track.pattern_counter += 1
    
    def handle_incompatible_detections(self, detect_rec, incompatible_detections, track_rec):
        for idx in incompatible_detections:
            detect, label = detect_rec[idx]
            detect = np.expand_dims(detect, axis=0).T
            aux_track = KalmanFilterTrack()  
            x = np.array([[detect[0], 0, detect[1], 0, detect[2], 0, detect[3], 0]]).T
            aux_track.current_state = x
            aux_track.inference()
            position = self.new_measurement(aux_track)
            
            aux_track.id = self.ids_track.popleft()  
            self.trackers.append((aux_track, label))
            track_rec.append(position)
    
    def handle_incompatible_track(self, detect_rec, incompatible_trackers, track_rec): 
        for track_idx in incompatible_trackers:
            aux_track = self.trackers[track_idx][0]
            aux_track.no_pattern_counter += 1
            aux_track.inference()
            position = self.new_measurement(aux_track)
            track_rec[track_idx] = position
    
    def create_tracker_for_detections(self, detect_rec):
        track_rec = []
        if len(self.trackers) > 0:
            for track, _ in self.trackers:
                track_rec.append(track.coordinates)
        
        compatibles, incompatible_detections, incompatible_trackers = self.check_compatibility_tracker_detector(track_rec, detect_rec)  
        self.handle_compatible_detections(detect_rec, compatibles, track_rec)
        self.handle_incompatible_detections(detect_rec, incompatible_detections, track_rec)
        self.handle_incompatible_track(detect_rec, incompatible_trackers, track_rec)
        
    def remove_old_trackers(self):
        remove_trackers = filter(lambda x: x[0].no_pattern_counter > self.lifetime, self.trackers)  
        for track, _ in remove_trackers:
                self.ids_track.append(track.id)
        self.trackers = [x for x in self.trackers if x[0].no_pattern_counter <= self.lifetime]
    
    def process_predict(self, img):
        self.idx_frame += 1
        detect_rec = self.yolo_detect.predict(img)
        self.create_tracker_for_detections(detect_rec)
       
        good_tracker_list = []
        for track, label in self.trackers:
            if ((track.pattern_counter >= self.min_pattern_counter) and (track.no_pattern_counter <= self.lifetime)):
                good_tracker_list.append(track)
                position = track.coordinates
                
                y1, x1, y2, x2 = position[0], position[1], position[2], position[3]
                roi = img[y1:y2, x1:x2]
                roi_h, roi_w, roi_c = roi.shape
                
                if roi_h >= 40 and roi_w > 40 and roi_c != 0:
                    roi = self.temporal_model.mask_resize(roi, self.temporal_model.size_w)
                    if not(track.id in self.actions):
                        self.actions[track.id] = dict(rois=collections.deque(maxlen=self.window), labels=collections.deque(maxlen=self.window))
                        
                    self.actions[track.id]['rois'].append(roi)
                    self.actions[track.id]['labels'].append(label)
                        
                if  track.id in self.actions:
                    if len(self.actions[track.id]['rois']) >= self.window:
                        
                        fall, nofall = self.temporal_model.pred(self.actions[track.id]['rois'], self.temporal_model.model)            
                        count = Counter(self.actions[track.id]['labels'])
                        count_fall = count['fall'] if 'fall' in count else 0
                        count_nofall = count['nofall'] if 'nofall' in count else 0
                        
                        txt_id = "Id:{}".format(track.id)
                        if (fall > nofall) and (count_fall >= 4 and count_nofall <= 4): 
                            color = (0, 0, 255)
                            txt_act = "Act:{}".format("Fall")
                        else: 
                            color = (0, 255, 0)
                            txt_act = "Act:{}".format("No Fall") 
                            
                        cv2.rectangle(img, (x1, y1 - 30), (x1 + 80, y1), color, -1)
                        cv2.putText(img, txt_id, (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 1)
                        cv2.putText(img, txt_act, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 1)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
        self.remove_old_trackers()
        
        cv2.imshow("frame", img)
        return img

    
if __name__ == "__main__": 
    
    cap = cv2.VideoCapture('/home/oberon/videos/y2mate.com - man_falls_on_wet_floor_SnMtp7pIPME_360p.mp4')
    model = CombineModels()
    
    while True: 
        ret, img = cap.read()
        img = imutils.resize(img, width=600)
        
        model.process_predict(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
