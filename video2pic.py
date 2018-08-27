# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:06:28 2018

@author: ZK
"""
import pandas as pd
import cv2
import os

f = pd.read_csv('./youtube_BB/youtube_boundingboxes_detection_validation.csv', header=None)
f.columns = ['youtube_id','timestamp_ms','class_id','class_name','object_id','object_presence','xmin','xmax','ymin','ymax']
for mp4 in os.listdir('./YTBB_mp4/'):
    mp4 = mp4.split('.')[0]
    mp4='ABQJpBm9hP8'
    if not os.path.exists('./YTBB_jpg/' + mp4 + '/'):
        os.makedirs('./YTBB_jpg/' + mp4 + '/')
        print(mp4)
    else:
        continue
#mp4 = 'AAQmL_BlrRs'

    id0 = f.loc[f['youtube_id'] == mp4]
    id0 = id0.loc[id0['object_presence'] == 'present']
    id0 = id0[['timestamp_ms']]

    vc = cv2.VideoCapture('./YTBB_mp4/' + mp4 + '.mp4')
    c = 0
    i = 0
    if vc.isOpened():
        rval,frame=vc.read()
    else:
        rval=False
    while rval:
        rval,frame=vc.read()
        if c == int(id0['timestamp_ms'].iloc[i]*30/1000):
            cv2.imwrite('./YTBB_jpg/' + mp4 + '@'+str(int(id0['timestamp_ms'].iloc[i]/1000))+'.jpg',frame)
            i += 1
        c += 1
        if i == len(id0):
            break
        cv2.waitKey(1)
    vc.release()

#f = f[['youtube_id']]
#f = f['youtube_id'].unique()
#f = list(f)
#with open('./youtube_BB/detection_train.txt', 'w') as file:
#    file.write(str(f))
#%%
import os
list1 = os.listdir('./OTB2015/')
for item in list1:
    print(item)
    with open('./OTB2015/'+item+'/groundtruth_rect.txt') as f:
        a = f.read().split('\n')
    if '\t' in a[0]:
        print('...')
        a = [','.join(i.split('\t')) for i in a]
    l = os.listdir('./OTB2015/'+item+'/img/')
    if not os.path.exists('./OTB2015/'+item+'/label/'):
        os.makedirs('./OTB2015/'+item+'/label/')
    for j,k in zip(a, l):
        with open('./OTB2015/'+item+'/label/'+k.split('.')[0]+'.txt', 'w') as f:
            f.write(j)
#%%
import os
l = os.listdir('./OTB2015/')
for item in l:
    print(item)
    d = './OTB2015/'+item+'/img/'
    for j in os.listdir(d):
        if j.split('.')[-1] != 'jpg':
            print('...')
            os.remove(d+j)
#%%
import os
import cv2
from PIL import Image
l = os.listdir('./OTB2015/')
for item in l:
    item = 'Car1'
#    item = 'Basketball'
    print(item)
    d = './OTB2015/'+item+'/img/'
    f = os.listdir(d)[0]
    img = Image.open(d+f)
    try:
        r,g,b = img.split()
    except ValueError:
        print ('...')
#%%
import os
l = os.listdir('./OTB2015/')
for item in l:
    img = './OTB2015/'+item+'/img/'
    label = './OTB2015/'+item+'/label/'
    if len(os.listdir(img)) != len(os.listdir(label)):
        print(item)
#%%
import os
l = os.listdir('./lq/label/')
for item in l:
    with open('./lq/label/'+item, 'r') as f:
        a = f.read().split()[2:]
    with open('./lq/label/'+item, 'w') as f:
        f.write(','.join(a))