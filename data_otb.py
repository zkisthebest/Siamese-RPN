# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:36:51 2018
@author: ZK
"""
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
import os
from axis import x1y1x2y2_to_xywh, xywh_to_x1y1x2y2, x1y1wh_to_xywh, x1y1wh_to_x1y1x2y2, point_center_crop, resize
import random

#%%
data_dir = './OTB2015/'
interval = 20

list1 = os.listdir(data_dir)
number = []
for item in list1:
    number.append(len(os.listdir(data_dir+item+'/img/')))
    
number = [i-interval for i in number]

sum1 = [0]
for a in range(len(number)):
    sum1.append(sum1[a]+number[a])

#%%
class MyDataset(Dataset):

    def __init__(self, root_dir, anchor_scale = 64, k = 5):
        self.root_dir = root_dir
        self.anchor_shape = self._get_anchor_shape(anchor_scale)
        self.k = k    
        
    """根据anchor_scale获得5个anchor的宽度和高度
    """
    def _get_anchor_shape(self, a):
        s = a**2
        r = [[3*math.sqrt(s/3.),math.sqrt(s/3.)], [2*math.sqrt(s/2.),math.sqrt(s/2.)], 
                 [a,a], [math.sqrt(s/2.),2*math.sqrt(s/2.)], [math.sqrt(s/3.),3*math.sqrt(s/3.)]]
        return [list(map(round, i)) for i in r]

    def __len__(self):
        return sum1[-1]
    
    def _which(self, index, sum1):
        low = 0
        high = len(sum1) - 1
        while(high - low > 1):
            mid = (high+low) // 2
            if sum1[mid] <= index:
                low = mid
            elif sum1[mid] > index:
                high = mid
        return low
    
    """读取数据集时，将会调用下面这个方法来获取数据
    """
    def __getitem__(self, index):
#        print(index)
        low = self._which(index, sum1)
        index -= sum1[low]
        folder = list1[low]
        
        img = os.listdir(self.root_dir + folder + '/img/')[index]
        img = Image.open(self.root_dir + folder + '/img/' + img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        gtbox = os.listdir(self.root_dir + folder + '/label/')[index]
        with open(self.root_dir + folder + '/label/' + gtbox) as f:
            gtbox = f.read().split(',')
        gtbox = [round(float(i)) for i in gtbox]
        gtbox = x1y1wh_to_xywh(gtbox)
        template, _, _ = self._transform(img, gtbox, 1, 127)
        
        rand = random.randrange(1,interval)
        
        img = os.listdir(self.root_dir + folder + '/img/')[index + rand]
        img = Image.open(self.root_dir + folder + '/img/' + img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        gtbox = os.listdir(self.root_dir + folder + '/label/')[index + rand]
        with open(self.root_dir + folder + '/label/' + gtbox) as f:
            gtbox = f.read().split(',')
        gtbox = [round(float(i)) for i in gtbox]
        gtbox = x1y1wh_to_xywh(gtbox)
        detection, pcc, ratio = self._transform(img, gtbox, 2, 255)


        a = (gtbox[2]+gtbox[3]) / 2.
        a = math.sqrt((gtbox[2]+a)*(gtbox[3]+a)) * 2
        gtbox = [127, 127, round(255*gtbox[2]/a), round(255*gtbox[3]/a)]

        clabel, rlabel = self._gtbox_to_label(gtbox)
        return template, detection, clabel, rlabel, torch.from_numpy(np.array(pcc).reshape((1,4))), torch.from_numpy(np.array(ratio).reshape((1,1)))
    
    '''数据转换，包括裁剪、变形、转换为tensor、归一化
    '''
    def _transform(self, img, gtbox, area, size):
        img, pcc = point_center_crop(img, gtbox, area)
        img, ratio = resize(img, size)
        img = F.to_tensor(img)
#        img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img, pcc, ratio
        
    """根据ground truth box构造class label和reg label
    """
    def _gtbox_to_label(self, gtbox):
        clabel = np.zeros([5, 17, 17]) - 100
        rlabel = np.zeros([20, 17, 17], dtype = np.float32)
        pos, neg = self._get_64_anchors(gtbox)
        for i in range(len(pos)):
            clabel[pos[i, 2], pos[i, 0], pos[i, 1]] = 1
        for i in range(len(neg)):
            clabel[neg[i, 2], neg[i, 0], neg[i, 1]] = 0
        pos_coord = self._anchor_coord(pos)
        channel0 = (gtbox[0] - pos_coord[:, 0]) / pos_coord[:, 2]
        channel1 = (gtbox[1] - pos_coord[:, 1]) / pos_coord[:, 3]
        channel2 = np.array([math.log(i) for i in (gtbox[2] / pos_coord[:, 2]).tolist()])
        channel3 = np.array([math.log(i) for i in (gtbox[3] / pos_coord[:, 3]).tolist()])
        for i in range(len(pos)):
            rlabel[pos[i][2]*4, pos[i][0], pos[i][1]] = channel0[i]
            rlabel[pos[i][2]*4 + 1, pos[i][0], pos[i][1]] = channel1[i]
            rlabel[pos[i][2]*4 + 2, pos[i][0], pos[i][1]] = channel2[i]
            rlabel[pos[i][2]*4 + 3, pos[i][0], pos[i][1]] = channel3[i]
        return torch.Tensor(clabel).long(), torch.Tensor(rlabel).float()
    
    """根据anchor在label中的位置来获取anchor在detection frame中的坐标
    """
    def _anchor_coord(self, pos):
        result = np.ndarray([0, 4])
        for i in pos:
            tmp = [7+15*i[0], 7+15*i[1], self.anchor_shape[i[2]][0], self.anchor_shape[i[2]][1]]
            result = np.concatenate([result, np.array(tmp).reshape([1,4])], axis = 0)
        return result

    def _get_64_anchors(self, gtbox):
        pos = {}
        neg = {}
        for a in range(17):
            for b in range(17):
                for c in range(5):
                    anchor = [7+15*a, 7+15*b, self.anchor_shape[c][0], self.anchor_shape[c][1]]
                    anchor = xywh_to_x1y1x2y2(anchor)
                    if anchor[0]>=0 and anchor[1]>=0 and anchor[2]<=255 and anchor[3]<=255:
                        iou = self._IOU(anchor, gtbox)
                        if iou >= 0.5:
                            pos['%d,%d,%d' % (a,b,c)] = iou
                        elif iou <= 0.2:
                            neg['%d,%d,%d' % (a,b,c)] = iou
        pos = sorted(pos.items(),key = lambda x:x[1],reverse = True)
        pos = [list(map(int, i[0].split(','))) for i in pos[:16]]
        neg = sorted(neg.items(),key = lambda x:x[1],reverse = True)
        neg = [list(map(int, i[0].split(','))) for i in neg[:(64-len(pos))]]
        return np.array(pos), np.array(neg)

#    def _f(self, x):
#        if x <= 0:      return 0
#        elif x >= 254:  return 254
#        else:           return x

    def _IOU(self, a, b):
#        a = xywh_to_x1y1x2y2(a)
        b = xywh_to_x1y1x2y2(b)
        sa = (a[2] - a[0]) * (a[3] - a[1]) 
        sb = (b[2] - b[0]) * (b[3] - b[1])
        w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area = w * h 
        return area / (sa + sb - area)

#%%
transformed_dataset_train = MyDataset(root_dir = data_dir)
train_dataloader = DataLoader(transformed_dataset_train, batch_size=1, shuffle=True, num_workers=0)
dataloader = {'train':train_dataloader, 'valid':train_dataloader}

#transformed_dataset_test = MyDataset(detection_root_dir = './lq/JPEGImages/', 
#                                gtbox_root_dir = './lq/label/')
#test_dataloader = DataLoader(transformed_dataset_train, batch_size=1, shuffle=False, num_workers=0)
#%%
#with open('./vot2015/bag/groundtruth.txt') as f:
#    a = f.read().split()
#b = [float(i) for i in a[0].split(',')]
#x = (b[0]+b[4]) / 2.
#y = (b[1]+b[5]) / 2.
#w = 1.1 * math.sqrt(math.sqrt(((b[0]-b[2])**2+(b[1]-b[3])**2) * ((b[2]-b[4])**2+(b[3]-b[5])**2)))
#h = w
#if (b[0]-b[2]) >= (b[1]-b[3]):
#    w = (b[0]-b[2])
#
#list1 = xywh_to_x1y1x2y2([x, y, w, h])
#list1 = os.listdir('./vot2015')
#number = [len(os.listdir('./vot2015/'+item)) for item in os.listdir('./vot2015')]
