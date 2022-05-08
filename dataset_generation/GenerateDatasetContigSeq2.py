
import os
import cv2
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import time

path_source = '/media/oberon/MouglasHD2T/actions_datasets/MultCamFallDataset/dataset/Annotations/'
path_imgs = '/media/oberon/MouglasHD2T/actions_datasets/MultCamFallDataset/dataset/JPEGImages/'

files = [f for f in listdir(path_source) if isfile(join(path_source, f))]
files = sorted(files)
size = len(files)

idx = 0

sliding_window = {}

def get_index(f, i):
    path_file = "{}{}".format(path_source, str(f))
    tree = ET.parse(path_file)
    root = tree.getroot()    
    find_count = [obj.find('index').text for obj in root.findall('object')].count(str(i))

    return find_count

def returns_last(sliding_window, current, index):
    c = 0
    for i in range(current, -1, -1):
        joinid = '{}_{}'.format(i, index)
        try:
            if sliding_window[joinid]:
                return joinid
        except:
            pass
        c += 1
        
    return None

for i, f in enumerate(files):
    
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
        
        if i < size:
            
            # na primeira imagem cria id todos para todos
            if i == 0:
                joinid = '{}_{}'.format(idx, class_index)
                sliding_window[joinid] = []
                sliding_window[joinid].append([f, class_index, name, x, y, w, h])
                continue
            
            # se o objeto n esta presenta na imagem anterior, salva e cria um novo id
            count_index = get_index(files[i - 1], class_index)
            if count_index == 0:
                joinid = '{}_{}'.format(idx, class_index)
                sliding_window[joinid] = []        
                sliding_window[joinid].append([f, class_index, name, x, y, w, h])
            
            # se o objeto esta presente na imagem anterior adiciona ao conjunto corrente
            else:
                joinid = returns_last(sliding_window, i, class_index)
                if joinid != None:
                    sliding_window[joinid].append([f, class_index, name, x, y, w, h])
                else:
                    joinid = '{}_{}'.format(idx, class_index)
                    sliding_window[joinid] = []
                    sliding_window[joinid].append([f, class_index, name, x, y, w, h])
                    
#         print sliding_window
#         time.sleep(0.5)
        
        #-----------------------------------------------
        
        display_txt = '{} {}'.format(name, class_index)
        cv2.rectangle(img, (x, y - 20), (x + 50 + len(display_txt) * 6, y), (0, 255, 0), -1)
        cv2.putText(img, display_txt, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 0, 0), 1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#     cv2.imshow('Img', img)
#     k = cv2.waitKey(1000) & 0xFF        

    
    per = float((idx * 100) / size)
    print ("%0.2f%%") % (per)    
    idx += 1
    
    
print "Load end!===================="

print "Quantity per class:"
print ["{}:{}".format(i, len(sliding_window[i])) for i in sliding_window]
 
print "Generate data===================================="
 
print sliding_window


for item in sliding_window:
    id_pop = 0
    for  i, wind in enumerate(sliding_window[item]):
        if id_pop <= 2:
            sliding_window[item].pop(i)
            id_pop += 1
        else:
            id_pop = 0
        

print("___________________________")
print sliding_window


print "Quantity per class:"
print ["{}:{}".format(i, len(sliding_window[i])) for i in sliding_window]

win_size = 8
sample_num__fall = 5

subdata_id = 0
path_save_fall = '/media/oberon/MouglasHD2T/actions_datasets/MultCamFallDataset/dataset/mul_seq_dataset/fall/'
path_save_nofall = '/media/oberon/MouglasHD2T/actions_datasets/MultCamFallDataset/dataset/mul_seq_dataset/nofall/'

for item in sliding_window:
    seq_size = len(sliding_window[item])
    
    for i in range(seq_size - win_size + 1):
        sub_seq = sliding_window[item][i:i + win_size]
        
        num__fall = [j[2] for j in sub_seq].count('fall')
        if num__fall >= sample_num__fall:
            path_save = path_save_fall
        elif num__fall <= win_size - sample_num__fall:
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
        
