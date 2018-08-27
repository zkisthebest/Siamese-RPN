# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:42:36 2018
@author: ZK
"""

def x1y1x2y2_to_xywh(gtbox):
    return list(map(round, [(gtbox[0]+gtbox[2])/2., (gtbox[1]+gtbox[3])/2., gtbox[2]-gtbox[0], gtbox[3]-gtbox[1]]))

def xywh_to_x1y1x2y2(gtbox):
    return list(map(round, [gtbox[0]-gtbox[2]/2., gtbox[1]-gtbox[3]/2., gtbox[0]+gtbox[2]/2., gtbox[1]+gtbox[3]/2.]))

def x1y1wh_to_xywh(gtbox):
    x1, y1, w, h = gtbox
    return [round(x1 + w/2.), round(y1 + h/2.), w, h]

def x1y1wh_to_x1y1x2y2(gtbox):
    x1, y1, w, h = gtbox
    return [x1, y1, x1+w, y1+h]
#%%
import torch
from torch.nn import Module
from torch.nn import functional as F
#%%
class SmoothL1Loss(Module):
    def __init__(self, use_gpu):
        super (SmoothL1Loss, self).__init__()
        self.use_gpu = use_gpu
        return
    
    def forward(self, clabel, target, routput, rlabel):
        
#        rloss = F.smooth_l1_loss(routput, rlabel)
        rloss = F.smooth_l1_loss(routput, rlabel, size_average=False, reduce=False)
        
            
        e = torch.eq(clabel.float(), target) 
        e = e.squeeze()
        e0,e1,e2,e3,e4 = e[0].unsqueeze(0),e[1].unsqueeze(0),e[2].unsqueeze(0),e[3].unsqueeze(0),e[4].unsqueeze(0)
        eq = torch.cat([e0,e0,e0,e0,e1,e1,e1,e1,e2,e2,e2,e2,e3,e3,e3,e3,e4,e4,e4,e4], dim=0).float()
        
        rloss = rloss.squeeze()
        rloss = torch.mul(eq, rloss)
        rloss = torch.sum(rloss)
        rloss = torch.div(rloss, eq.nonzero().shape[0]+1e-4)
        return rloss
#%%
class Myloss(Module):
    def __init__(self):
        super (Myloss, self).__init__()
        return 
    
    def forward(self, coutput, clabel, target, routput, rlabel, lmbda):
        closs = F.cross_entropy(coutput, clabel)

#        rloss = F.smooth_l1_loss(routput, rlabel)
        rloss = F.smooth_l1_loss(routput, rlabel, size_average=False, reduce=False)
        
            
        e = torch.eq(clabel.float(), target) 
        e = e.squeeze()
        e0,e1,e2,e3,e4 = e[0].unsqueeze(0),e[1].unsqueeze(0),e[2].unsqueeze(0),e[3].unsqueeze(0),e[4].unsqueeze(0)
        eq = torch.cat([e0,e0,e0,e0,e1,e1,e1,e1,e2,e2,e2,e2,e3,e3,e3,e3,e4,e4,e4,e4], dim=0).float()
        
        rloss = rloss.squeeze()
        rloss = torch.mul(eq, rloss)
        rloss = torch.sum(rloss)
        rloss = torch.div(rloss, eq.nonzero().shape[0]+1e-4)
        
        loss = torch.add(closs, lmbda, rloss)
        return loss
#%%
import math
from PIL import ImageStat, Image
from torchvision.transforms import functional as F2
#%%
def resize(img, size, interpolation=Image.BILINEAR):
    assert img.size[0] == img.size[1]
    return img.resize((size, size), interpolation), img.size[0] / size
#%%
def point_center_crop(img, gtbox, area):
    x, y, dw, dh = gtbox
    p = (dw + dh) / 2.
    a = math.sqrt((dw + p) * (dh + p))
    a *= area
    i = round(x - a/2.)
    j = round(y - a/2.)
    mean = tuple(map(round, ImageStat.Stat(img).mean))
    if i < 0:
        left = -i
        i = 0
    else: 
        left = 0
    if j < 0:
        top = -j
        j = 0
    else: 
        top = 0
    if x+a/2. > img.size[0]:
        right = round(x+a/2.-img.size[0])
    else:
        right = 0
    if y+a/2. > img.size[1]:
        bottom = round(y+a/2.-img.size[1])
    else:
        bottom = 0
        
    img = F2.pad(img, padding=(left, top, right, bottom), fill=mean, padding_mode='constant')   
    img = img.crop((i, j, i+round(a), j+round(a)))
    
    return img, [left, top, i, j]
#%%
def cosine_window(coutput1):
    math.cos()
    
    
    return



#%%
#class PointCenterCrop(object):
#    def __init__( gtbox, area):
#        gtbox = gtbox
#        area = area
#
#    def __call__( img):
#        return point_center_crop(img, gtbox, area)
#
#    def __repr__():
#        return __class__.__name__ + '(gtbox={0})'.format(gtbox)
    
#%%
'''
import torch.nn as nn

        features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),
        )
        
        k = 5
        conv1 = nn.Conv2d(256, 2*k*256, kernel_size=3)
         conv2 = nn.Conv2d(256, 4*k*256, kernel_size=3)
         conv3 = nn.Conv2d(256, 256, kernel_size=3)
         conv4 = nn.Conv2d(256, 256, kernel_size=3)

         cconv = nn.Conv2d(256, 2* k, kernel_size = 4, bias = False)
         rconv = nn.Conv2d(256, 4* k, kernel_size = 4, bias = False)
#         cconv.train(False)
#         rconv.train(False)
        
#         reset_params()
#         freeze_layers(8)
        
#    def reset_params():
#        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
#        model_dict =  state_dict()
#        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#        model_dict.update(pretrained_dict)
#         load_state_dict(model_dict)
        
#    def freeze_layers( number):
#        for i in range(number):
#             features[i].train(False)
            
#    def forward( template, detection):
        template =  features(template)
        detection =  features(detection)
        
        
        ckernal =  conv1(template)
        ckernal = ckernal.view(2* k, 256, 4, 4)
         cconv.weight = nn.Parameter(ckernal.data)
        cinput =  conv3(detection)
        coutput =  cconv(cinput)
        
        rkernal =  conv2(template)
        rkernal = rkernal.view(4* k, 256, 4, 4)
         rconv.weight = nn.Parameter(rkernal.data)
        rinput =  conv4(detection)
        routput =  rconv(rinput)
        
#        return template, detection
        return coutput, routput
'''
'''
#%%
import numpy as np
import math
import torch
from PIL import Image
from torchvision import transforms
import os
#from torch.utils.data import Dataset
'''
#%%


'''
#%%
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
import os
from axis import x1y1x2y2_to_xywh, xywh_to_x1y1x2y2, point_center_crop, resize
detection_root_dir = './lq/JPEGImages/'
gtbox_root_dir = './lq/label/'
def _get_anchor_shape( a):
        s = a**2
        r = [[3*math.sqrt(s/3.),math.sqrt(s/3.)], [2*math.sqrt(s/2.),math.sqrt(s/2.)], 
                 [a,a], [math.sqrt(s/2.),2*math.sqrt(s/2.)], [math.sqrt(s/3.),3*math.sqrt(s/3.)]]
        return [list(map(round, i)) for i in r]

def __len__():
        return len(os.listdir(detection_root_dir))
    
"""读取数据集时，将会调用下面这个方法来获取数据
"""
def __getitem__( index):
    
        img = os.listdir(detection_root_dir)[0]
        img = Image.open(detection_root_dir + img)
        gtbox = os.listdir(gtbox_root_dir)[0]
        with open(gtbox_root_dir + gtbox) as f:
            gtbox = f.read().split(' ')[1:]
        gtbox = [int(i) for i in gtbox]
        gtbox = x1y1x2y2_to_xywh(gtbox)
        template, _, _ = _transform(img, gtbox, 1, 127)
    for index in range(100):
#        index=80

        img = os.listdir(detection_root_dir)[index]
        img = Image.open(detection_root_dir + img)
        gtbox = os.listdir(gtbox_root_dir)[index]
        with open(gtbox_root_dir + gtbox) as f:
            gtbox = f.read().split(' ')[1:]
        gtbox = [int(i) for i in gtbox]
        gtbox = x1y1x2y2_to_xywh(gtbox)
#        template = _transform(img, gtbox, 1, 127)
        detection, pcc, ratio = _transform(img, gtbox, 2, 255)
        

        a = (gtbox[2]+gtbox[3]) / 2.
        a = math.sqrt((gtbox[2]+a)*(gtbox[3]+a)) * 2
        gtbox = [127, 127, round(255*gtbox[2]/a), round(255*gtbox[3]/a)]
        list1 = xywh_to_x1y1x2y2(gtbox)
        import cv2
        detection = cv2.cvtColor(np.asarray(detection),cv2.COLOR_RGB2BGR)

        cv2.rectangle(detection, (list1[0],list1[1]), (list1[2],list1[3]), (0,255,0), 1)
        detection = Image.fromarray(cv2.cvtColor(detection,cv2.COLOR_BGR2RGB))
        detection.save('./tmp/'+str(index)+'.jpg')
#detection = Image.fromarray(np.array(detection))
#detection.show()


        clabel, rlabel = _gtbox_to_label(gtbox)
        return template, detection, clabel, rlabel, pcc, ratio
    
#数据转换，包括裁剪、变形、转换为tensor、归一化
#
def _transform( img, gtbox, area, size):
        img, pcc = point_center_crop(img, gtbox, area)
        img, ratio = resize(img, size)
#        img = F.to_tensor(img)
#        img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img, pcc, ratio
    
    
#    def _transform( img, gtbox, area, scale):
#        trans = transforms.Compose([
#                PointCenterCrop(gtbox, area = area),
#                transforms.Resize(scale),
#                transforms.ToTensor(),
#                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                ])
#        return trans(img)
    
"""根据ground truth box构造class label和reg label
"""
def _gtbox_to_label( gtbox):
        clabel = np.zeros([5, 17, 17]) - 100
        rlabel = np.zeros([20, 17, 17], dtype = np.float32)
        pos, neg = _get_64_anchors(gtbox)
        for i in range(len(pos)):
            clabel[pos[i, 2], pos[i, 0], pos[i, 1]] = 1
        for i in range(len(neg)):
            clabel[neg[i, 2], neg[i, 0], neg[i, 1]] = 0
        pos_coord = _anchor_coord(pos)
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
def _anchor_coord( pos):
        result = np.ndarray([0, 4])
        for i in pos:
            tmp = [7+15*i[0], 7+15*i[1], anchor_shape[i[2]][0], anchor_shape[i[2]][1]]
            result = np.concatenate([result, np.array(tmp).reshape([1,4])], axis = 0)
        return result

def _get_64_anchors( gtbox):
        pos = {}
        neg = {}
        for a in range(17):
            for b in range(17):
                for c in range(5):
                    anchor = [7+15*a, 7+15*b, anchor_shape[c][0], anchor_shape[c][1]]
                    anchor = xywh_to_x1y1x2y2(anchor)
                    if anchor[0]>0 and anchor[1]>0 and anchor[2]<255 and anchor[3]<255:
                        iou = _IOU(anchor, gtbox)
                        if iou >= 0.6:
                            pos['%d,%d,%d' % (a,b,c)] = iou
                        elif iou <= 0.3:
                            neg['%d,%d,%d' % (a,b,c)] = iou
        pos = sorted(pos.items(),key = lambda x:x[1],reverse = True)
        pos = [list(map(int, i[0].split(','))) for i in pos[:16]]
        neg = sorted(neg.items(),key = lambda x:x[1],reverse = True)
        neg = [list(map(int, i[0].split(','))) for i in neg[:(64-len(pos))]]
        return np.array(pos), np.array(neg)

#    def _f( x):
#        if x <= 0:      return 0
#        elif x >= 254:  return 254
#        else:           return x

def _IOU( a, b):
#        a = xywh_to_x1y1x2y2(a)
        b = xywh_to_x1y1x2y2(b)
        sa = (a[2] - a[0]) * (a[3] - a[1]) 
        sb = (b[2] - b[0]) * (b[3] - b[1])
        w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area = w * h 
        return area / (sa + sb - area)
'''