# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import scipy.io as sio

path_dataset = '/media/oberon/MouglasHD2T/actions_datasets/UCFSports/ucf_sports_actions/ucf action/'
save_dataset = '/media/oberon/MouglasHD2T/actions_datasets/UCFSports/process_ucfsports_boxs/'

count_files = 0
size_files = sum([len(files) for r, d, files in os.walk(path_dataset)])

action_dirs = os.listdir(path_dataset)
print (action_dirs)

action_dirs = sorted(action_dirs)
size = len(action_dirs)
idx = 0
window_size = 8

def load_annotations(path_puppet):
    files = os.listdir(path_puppet)
    boxs = []
    for f in files:
        path_ann = "{}{}".format(path_puppet, f)
        coor = open(path_ann).read()
        
        x, y, w, h, cla = coor.split("\t")

        boxs.append([int(x), int(y), int(w), int(h)])    
        
    return boxs

 
# sliding_window = {}
for act_dir in action_dirs:
    
    path_act_dir = "{}{}".format(path_dataset, act_dir)
    
    print(path_act_dir)
    samples_files = os.listdir(path_act_dir)
    
    action_save_dir = "{}{}".format(save_dataset, act_dir)
    if not os.path.exists(action_save_dir):
        os.makedirs(action_save_dir)
    
    for sample_dir in samples_files:
        
        print(10*"=", sample_dir)
        
        path_video_file = "{}/{}/{}".format(path_act_dir, sample_dir, "jpeg")
        
        imgs_files = [i for i in os.listdir(path_video_file) if i[-3:] == 'jpg']
        
        id_sample = 0
        imgs_count = len(imgs_files)
        if imgs_count >= window_size:
            
            path_mat = "{}/{}/{}/".format(path_act_dir, sample_dir, "gt")
            
            boxs = load_annotations(path_mat)
            
            action_save_video_sample = "{}/{}".format(action_save_dir, sample_dir)
            # print(action_save_video_sample)
            if not os.path.exists(action_save_video_sample):
                os.makedirs(action_save_video_sample)
            
            count = 0
            for img_name in imgs_files:
                img_path = "{}/{}/{}/{}".format(path_act_dir, sample_dir, "jpeg", img_name)
                frame = cv2.imread(img_path)
                
                if frame is None or id_sample >= 8:
                    break
                
                        
                x, y, w, h = boxs[count]
                crop = frame[y:y + h, x:x + w]
                
                cv2.imwrite("{}/{}.{}".format(action_save_video_sample, id_sample, "png"), crop)
                # cv2.imshow("frame", frame)
                # cv2.waitKey(1)
               
                id_sample += 1
                count += 1
            
            # print("Count:", count, frames_count)
        
#         else:
#             print("=========================")
#             print(frames_count, window_size)
#             exit()        
        
        per = float((count_files * 100) / size_files)
        print(("%0.2f%%") % (per))    
        count_files += 1
