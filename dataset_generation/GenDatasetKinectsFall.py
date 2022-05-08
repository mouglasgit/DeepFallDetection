# -*- coding: utf-8 -*-

import os
import cv2

path_dataset = '/media/oberon/MouglasHD2T/actions_datasets/kinects600/train/'
save_dataset = '/media/oberon/MouglasHD2T/actions_datasets/kinects600/process_kinectsFall/'

count_files = 0
# size_files = sum([len(files) for r, d, files in os.walk(path_dataset)])

# action_dirs = os.listdir(path_dataset)
# print (action_dirs)
# 
# action_dirs = sorted(action_dirs)

action_dirs = ['falling off bike', 'falling off chair']


size_files = sum(len(os.listdir("{}{}".format(path_dataset, a))) for a in action_dirs)

size = len(action_dirs)
idx = 0
window_size = 8
 
# sliding_window = {}
for act_dir in action_dirs:
    
    path_act_dir = "{}{}".format(path_dataset, act_dir)
    video_files = os.listdir(path_act_dir)
    
    action_save_dir = "{}{}".format(save_dataset, act_dir)
    if not os.path.exists(action_save_dir):
        os.makedirs(action_save_dir)
    
    for video_name in video_files:
        path_video_file = "{}/{}".format(path_act_dir, video_name)
    
        id_sample = 0
        cap = cv2.VideoCapture(path_video_file)
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frames_count >= window_size:
            
            action_save_video_sample = "{}/{}".format(action_save_dir, video_name[:-4])
            # print(action_save_video_sample)
            if not os.path.exists(action_save_video_sample):
                os.makedirs(action_save_video_sample)
            
            if int(frames_count / window_size) - 1 >= 3:
                init_fps = 3
            else:
                init_fps = 0
            
            id_fps = init_fps
            count = 0
            while True:
                ret, frame = cap.read()
                if ret == False or frame is None or id_sample >= 8:
                    break
                
                if id_fps <= 0:
                    # print("{}/{}.{}".format(action_save_video_sample, id_sample, "png"))
                    cv2.imwrite("{}/{}.{}".format(action_save_video_sample, id_sample, "png"), frame)
                    # cv2.imshow("frame", frame)
                    # cv2.waitKey(1)
                    id_fps = init_fps
                    id_sample += 1
                    count += 1
                    
                id_fps -= 1
            
            # print("Count:", count, frames_count)
        
#         else:
#             print("=========================")
#             print(frames_count, window_size)
#             print(path_video_file)
#             exit()        
        
        per = float((count_files * 100) / size_files)
        print(("%0.2f%%") % (per))    
        count_files += 1
