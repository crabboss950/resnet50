import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2
import math
import shutil
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)



  # 원본 이미지 넘버링하기
# path = "C:/Users/admin/Desktop/resnet/Data/original"
# OK = glob.glob(path+"/OK" + '/*')
# NG = glob.glob(path+"/NG"+'/*')



def rename(files):

  if 'OK' in files[0]:
     for i,f in enumerate(files):
         os.rename(f, os.path.join(path+"/OK", 'OK_' + '{0:03d}.png'.format(i)))
     dolphin = glob.glob(path+"/OK" + '/*')    
     print("OK {}번째 이미지까지 성공".format(i+1))

  elif 'NG' in files[0]:
     for i,f in enumerate(files):
         os.rename(f, os.path.join(path+"/NG", 'NG_' + '{0:03d}.png'.format(i)))
     shark = glob.glob(path+"/NG"+'/*')
     print("NG {}번째 이미지까지 성공".format(i+1))

# rename(OK)
# rename(NG)
# cv2를 이용해 이미지를 읽는 함수 정의
def read_img(file_path):
    img_arr = cv2.imread(file_path)
    return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB) # cvtColor로 BGR을 RGB로 바꿔줌

# 비율이 작은 test로 빠르게 split

# dolphin_test_count = round(len(OK)*0.2)
# shark_test_count = round(len(NG)*0.2)


def split( img_list, test_count, train_path, test_path):
  
  test_files=[]
  for i in random.sample( img_list, test_count ):
    test_files.append(i)

  # 차집합으로 train/test 리스트 생성하기
  train_files = [x for x in img_list if x not in test_files]

  for k in train_files:
    shutil.copy(k, train_path)
  
  for c in test_files:
    shutil.copy(c, test_path)

  print('train 폴더 이미지 개수 : {}\ntest 폴더 이미지 개수 : {}'.format(len(glob.glob(train_path+'/*')),len(glob.glob(test_path+'/*'))))


dolphin_train_path='C:/Users/admin/Desktop/resnet/Data/train/OK'
dolphin_test_path='C:/Users/admin/Desktop/resnet/Data/test/OK'

shark_train_path='C:/Users/admin/Desktop/resnet/Data/train/NG'
shark_test_path='C:/Users/admin/Desktop/resnet/Data/test/NG'

# split(OK, dolphin_test_count, dolphin_train_path, dolphin_test_path)
# split(NG, shark_test_count, shark_train_path, shark_test_path)

# path ='C:/Users/admin/Desktop/resnet/Data/train'
# OK = glob.glob(path+"/OK" + '/*')
# NG = glob.glob(path+"/NG"+ '/*')

# path ='C:/Users/admin/Desktop/resnet/Data/test'
# OK = glob.glob(path+"/OK" + '/*')
# NG = glob.glob(path+"/NG"+ '/*')


# rename(OK)
# rename(NG)