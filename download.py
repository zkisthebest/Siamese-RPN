# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:45:30 2018

@author: ZK
"""

#!/usr/bin/python
import re 
import urllib
def getHtml(url):
    page=urllib.request.urlopen(url)
    html=page.read()
    html=html.decode('utf-8')
    return html
def getMp4(html):
	r=r"href='(http.*\.mp4)'"
	re_mp4=re.compile(r)
	mp4List=re.findall(re_mp4,html)
	filename=1
	for mp4url in mp4List:
		urllib.urlretrieve(mp4url,"%s.mp4" %filename)
		print  ('file "%s.mp4" done' %filename)
		filename+=1
url = 'https://v.youku.com/v_show/id_XMzcxMDc1MjYwMA==.html?spm=a2hww.11359951.m_26659.5~5!2~5~5~5~5~5~A'
#url="http://youtu.be/AAxYohQXjmY"
html=getHtml(url)
getMp4(html)
#%%
from selenium import webdriver
from time import sleep
with open('./youtube_BB/detection_train.txt') as f:
    a = f.read().split(', ')
browser = webdriver.Chrome()
#%%
for i in range(4, 10000):
#    i = 3
    tmp = a[i][1:len(a[i])-1]
    browser.get('https://www.clipconverter.cc/')
    elem = browser.find_element_by_name('mediaurl')
    elem.clear()
    elem.send_keys('http://youtu.be/' + tmp)

    browser.find_element_by_id('submiturl').click()
#    browser.switch_to_window(browser.window_handles[-1])
#    browser.close()
#    browser.switch_to_window(browser.window_handles[-1])
    sleep(4)
    elem = browser.find_element_by_name('filename')
    elem.clear()
    elem.send_keys(tmp)
    browser.find_element_by_xpath('//*[@id="submitconvert"]/input').click()
    sleep(30)
    browser.find_element_by_xpath('//*[@id="downloadbutton"]').click()

#%%
from pytube import YouTube
import os 

if not os.path.exists('./webm/'):
    os.makedirs('./webm/')
with open('./youtube_BB/detection_validation.txt') as f: a = f.read().split(', ')

for i in range(200, len(a)):
    i = 0
    name = a[i]
    name = name[1:-1]
    url = 'http://youtu.be/' + name
    print(name)
    try:
        yt = YouTube(url)
    except:
        continue
    print(name)
    stream = yt.streams.first()
    try:
        stream.download('./webm/', name)
    except:
        continue
    'https://youtu.be/AACebVo-JXY'
#%%
import os
for item in os.listdir('./YTBB_jpg/'):
    newname = item.split('_')
    os.rename('./YTBB_jpg/'+item, './YTBB_jpg/'+''.join(newname[:-1]) +'@'+ newname[-1])
os.rename('./127_.jpg', './abc@127.jpg')
