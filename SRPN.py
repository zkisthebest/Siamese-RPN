# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:34:57 2018
@author: ZK
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
#%%    
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}

class SiameseRPN(nn.Module):
    def __init__(self):
        super(SiameseRPN, self).__init__()
        self.features = nn.Sequential(
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
        
        self.k = 5
        self.conv1 = nn.Conv2d(256, 2*self.k*256, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4*self.k*256, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu4 = nn.ReLU(inplace=True)

        self.cconv = nn.Conv2d(256, 2* self.k, kernel_size = 4, bias = False)
        self.rconv = nn.Conv2d(256, 4* self.k, kernel_size = 4, bias = False)
        
        self.reset_params()
        
    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
            
    def forward(self, template, detection):
        template = self.features(template)
        detection = self.features(detection)
        
        ckernal = self.conv1(template)
#        ckernal = self.relu1(ckernal)
        ckernal = ckernal.view(2* self.k, 256, 4, 4)
        self.cconv.weight = nn.Parameter(ckernal)
        cinput = self.conv3(detection)
#        cinput = self.relu3(cinput)
        coutput = self.cconv(cinput)
        
        rkernal = self.conv2(template)
#        rkernal = self.relu2(rkernal)
        rkernal = rkernal.view(4* self.k, 256, 4, 4)
        self.rconv.weight = nn.Parameter(rkernal)
        rinput = self.conv4(detection)
#        rinput = self.relu4(rinput)
        routput = self.rconv(rinput)
        
        return coutput, routput

#%%
if __name__ == '__main__':
    print('1')
    model = SiameseRPN()
    #y1, y2 = model(template, detection)

#    model2 = RPN()
    #z1, z2 = model(y1, y2)
#    model3 = SRPN()