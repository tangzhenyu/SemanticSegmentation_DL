from PIL import Image
import cv2
import numpy as np
import os
import json
import pickle
root = '/media/D/DataSet/IS/VOCdevkit/VOC2012/SegmentationClass/'
map_dict=dict()

classes=range(0,21)
classes.append(255)

for file in os.listdir(root):
    png = np.array(Image.open(os.path.join(root,file)))
    rbg = cv2.imread(os.path.join(root,file))
    for i in classes:
        idx=np.where(png==i)
        if len(idx[0])==0:
            continue
        else:
            color=rbg[idx[0][0],idx[1][0],:]
            map_dict[str(i)]=color
    if len(map_dict)==22:
        break
with open('data/color_map','w') as f:
    pickle.dump(map_dict,f)


'''
with open('data/color_map','rb') as f:
    d = pickle.load(f)

img=np.array(Image.open('2007_000129.png'))
color_img=np.zeros((img.shape[0],img.shape[1],3))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        color_img[i,j,:]=d[str(img[i][j])]
cv2.imwrite('tmp.png', color_img)
'''