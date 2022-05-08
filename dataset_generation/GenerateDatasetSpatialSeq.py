
import os
import cv2
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET

path_source = '/home/avantia/Imagens/fall_dataset/videos/foi/qualidade/all/Annotations/'
path_imgs = '/home/avantia/Imagens/fall_dataset/videos/foi/qualidade/all/JPEGImages/'

# path_source = '/home/avantia/python-workspace/LabelImgVoc/generate_index/Annotations/'
# path_imgs = '/home/avantia/python-workspace/LabelImgVoc/generate_index/JPEGImages/'

files = [f for f in listdir(path_source) if isfile(join(path_source, f))]
files = sorted(files)
size = len(files)
idx = 0

sliding_window = {}
for f in files:
    path_file = "{}{}".format(path_source, str(f))
    
    path_img = "{}{}{}".format(path_imgs, str(f)[:-3], 'jpg')
    img = cv2.imread(path_img, 1)
    
    sh, sw, _ = img.shape
    
    tree = ET.parse(path_file)
    root = tree.getroot()
    
    for object in root.findall('object'):
        name = object.find('name').text
        
        class_index = object.find('index').text
        
        # if name == 'person':
        bndbox = object.find('bndbox')
            
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        x = xmin
        y = ymin
        w = xmax - x
        h = ymax - y
    
        #-----------------------------------------------
        try:
            sliding_window[class_index].append([f, class_index, name, x, y, w, h])
        except:
            sliding_window[class_index] = []
        #-----------------------------------------------
        
        display_txt = '{} {}'.format(name, class_index)
        
        cv2.rectangle(img, (x, y - 20), (x + 50 + len(display_txt) * 6, y), (0, 255, 0), -1)
        cv2.putText(img, display_txt, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 0, 0), 1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    
#     cv2.imshow('Img', img)
#     k = cv2.waitKey(1) & 0xFF        

    
    per = float((idx * 100) / size)
    print ("%0.2f%%") % (per)    
    idx += 1
    
    
print "Load end!===================="


print "Generate data===================================="
 
print sliding_window

win_size = 5
sample_num__fall = 3


subdata_id = 0
path_save_fall = '/home/avantia/Imagens/fall_dataset/videos/foi/qualidade/save_imgs/fall/'
path_save_nofall = '/home/avantia/Imagens/fall_dataset/videos/foi/qualidade/save_imgs/nofall/'

for item in sliding_window:
    seq_size = len(sliding_window[item])
    
    for i in range(seq_size - win_size + 1):
        sub_seq = sliding_window[item][i:i + win_size]
        
        num__fall = [j[2] for j in sub_seq].count('fall')
        if num__fall >= sample_num__fall:
            path_save = path_save_fall
        else:
            path_save = path_save_nofall
        
        dir_item = "{}{}-{}".format(path_save, subdata_id, item)
        if os.path.isdir(dir_item) == False:
            os.mkdir(dir_item)
        
        img_id = 0
        for sample in sub_seq:
            
            img = cv2.imread("{}{}.jpg".format(path_imgs, sample[0][:-4]), 1)
            x, y, w, h = sample[3], sample[4], sample[5], sample[6]
            crop = img[y:y + h, x:x + w]
               
            cv2.imwrite("{}/{}-{}.png".format(dir_item, img_id, sample[2]), crop)
               
            img_id += 1
    
        subdata_id += 1
        
