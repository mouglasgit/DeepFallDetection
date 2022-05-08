# -*- coding: utf-8 -*-

import os
import cv2

path_dataset = '/media/oberon/MouglasHD2T/actions_datasets/UCF101/UCF-101/'
save_dataset = '/media/oberon/MouglasHD2T/actions_datasets/UCF101/process_ucf101/'

count_files = 0
size_files = sum([len(files) for r, d, files in os.walk(path_dataset)])

action_dirs = os.listdir(path_dataset)
print (action_dirs)

action_dirs = sorted(action_dirs)
size = len(action_dirs)
idx = 0
 
# sliding_window = {}
for act_dir in action_dirs:
    
    path_act_dir = "{}{}".format(path_dataset, act_dir)
    video_files = os.listdir(path_act_dir)
    
    action_save_dir = "{}{}".format(save_dataset, act_dir)
    if not os.path.exists(action_save_dir):
        os.makedirs(action_save_dir)
    
    for video_name in video_files:
        path_video_file = "{}/{}".format(path_act_dir, video_name)
        # print(path_video_file) 
        
        action_save_video_sample = "{}/{}".format(action_save_dir, video_name[:-4])
        # print(action_save_video_sample)
        if not os.path.exists(action_save_video_sample):
            os.makedirs(action_save_video_sample)
    
        id_fps = 3
        id_sample = 0
        cap = cv2.VideoCapture(path_video_file)
        while True:
            ret, frame = cap.read()
            if ret == False or frame is None or id_sample >= 8:
                break
            
            if id_fps <= 0:
                # print("{}/{}.{}".format(action_save_video_sample, id_sample, "png"))
                cv2.imwrite("{}/{}.{}".format(action_save_video_sample, id_sample, "png"), frame)
                # cv2.imshow("frame", frame)
                # cv2.waitKey(1)
                id_fps = 3
                id_sample += 1
            
            id_fps -= 1
        
        
        per = float((count_files * 100) / size_files)
        print(("%0.2f%%") % (per))    
        count_files += 1
            
        
