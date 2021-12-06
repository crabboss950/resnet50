import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms # 이미지 데이터 transform
from torchvision import models
import os
import glob
from torch.utils.data import Dataset, DataLoader # 데이터 커스터마이징, 이미지 데이터 로더
from PIL import Image # PIL = Python Image Library
import cv2 # albumentation transform을 쓰려면 꼭 이 라이브러리를 이용
import tensorflow as tf
import albumentations
import albumentations.pytorch
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np




device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)


# 경로 지정
train_path = 'C:/coding/python_project/resnet50/resnet50/Data/train'
test_path = 'C:/coding/python_project/resnet50/resnet50/Data/test'

Dataset_path = 'C:/coding/python_project/resnet50/resnet50/Data'

OK_dir = '/OK/'
NG_dir = '/NG/'


class inhovation_Dataset(Dataset):

  def __init__(self, file_path, mode, transform=None):
    self.all_data = sorted(glob.glob(os.path.join(file_path, mode, '*', '*')))
    self.transform = transform

  def __getitem__(self, index):

    if torch.is_tensor(index):        # 인덱스가 tensor 형태일 수 있으니 리스트 형태로 바꿔준다.
       index = index.tolist()

    data_path = self.all_data[index]
    #img = np.array(Image.open(data_path).convert("RGB")) 
    # albumenatation transform을 쓰려면 cv2 라이브러리로 이미지를 읽어야 함
    image=cv2.imread(data_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환

    # transform 적용
    if self.transform is not None:    
       augmented = self.transform(image=image)
       image = augmented['image'] 
    
    # 이미지 이름을 활용해 label 부여
    label=[]                                
    if os.path.basename(data_path).startswith("OK") == True:
        label = 0
    elif os.path.basename(data_path).startswith("NG") == True:
        label = 1
    return image, label

  def __len__(self):
    length = len(self.all_data)
    return length


  #  # 모델 저장 함수 정의
def save_model(model, saved_dir):
  os.makedirs(saved_dir, exist_ok=True)  # 폴더가 존재하지 않으면, 디렉토리를 생성함.
  check_point = {
  'net': model.state_dict(),
  # 'optim' : optimizer.state_dict(),
  # 'loss' : loss.state_dict(),
  # 'epoch' : epoch.state_dict()
  }
  output_path = os.path.join(saved_dir) # 옵션들을 합쳐 경로 지정
  torch.save(check_point, saved_dir+'/best_model_weight.pt') # 인수로 '모델의 매개 변수, 경로'를 넣어주면 된다.
  torch.save(model, saved_dir+'/best_model.pt')

# config 모델 파라미터 인자를 만들기위한 클래스
class Config:
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)


# 트레인 클래스
class train_test():
      def __init__(self, config):
        # 파라미터 인자
        self.trainloader = config.trainloader
        self.testloader = config.testloader
        self.model = config.model
        self.device = config.device
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.globaliter = config.globaliter
        print(len(self.trainloader))

      def train(self, epochs, log_interval):
          self.model.train()
          for epoch in range(1, epochs + 1 ):  # epochs 루프
              running_loss = 0.0
              #lr_sche.step()
              for i, data in enumerate(self.trainloader, 0): # batch 루프
                  # get the inputs
                  self.globaliter += 1
                  inputs, labels = data # input data, label 분리
                  inputs = inputs.float().to(self.device)
                  labels = labels.to(self.device)



                  # 가중치 초기화 -> 이전 batch에서 계산되었던 가중치를 0으로 만들고 최적화 진행
                  self.optimizer.zero_grad() 

                  # forward + backward + optimize
                  outputs = self.model(inputs)
                  loss = self.criterion(outputs, labels)
                  loss.backward()
                  self.optimizer.step()
                  running_loss += loss.item()

                  # 30 iteration마다 acc & loss 출력
                  if i % log_interval == log_interval -1 : # i는 1에포크의 iteration
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlearningLoss: {:.6f}\twhole_loss: {:.6f} '.format(
                        epoch, i*len(inputs), len(self.trainloader.dataset),
                        100. * i*len(inputs) / len(self.trainloader.dataset), 
                        running_loss / log_interval,
                        loss.item()))
                    running_loss = 0.0

                    
              with torch.no_grad():
                  self.model.eval()
                  correct = 0
                  total = 0
                  test_loss = 0
                  acc = []
                  for k, data in enumerate(self.testloader, 0):
                    images, labels = data
                    images = images.float().to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_loss += self.criterion(outputs, labels).item()
                    acc.append(100 * correct/total)

                  print('\nTest set : Average loss:{:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
                      test_loss, correct, total, 100 * correct/total
                  ))
                  

                    # 모델 저장
                  if acc[k] > 85:
                      save_model( self.model, Dataset_path)
                      print('Succeed save the model')

      print('Finished Training')

# def get_mean(dataset):
#   meanRGB = [np.mean(image.numpy(), axis=(1,2)) for image,_ in dataset]

#   meanR = np.mean([m[0] for m in meanRGB])
#   meanG = np.mean([m[1] for m in meanRGB])
#   meanB = np.mean([m[2] for m in meanRGB])

#   return meanR,meanG,meanB

# def get_std(dataset):
#   stdRGB = [np.std(image.numpy(), axis=(1,2)) for image,_ in dataset]

#   stdR = np.mean([s[0] for s in stdRGB])
#   stdG = np.mean([s[1] for s in stdRGB])
#   stdB = np.mean([s[2] for s in stdRGB])

#   return stdR,stdG,stdB


# albumentations_resize = albumentations.Compose([
#     albumentations.Resize(224, 224),
#     albumentations.pytorch.ToTensorV2()
# ])

# resize_train=inhovation_Dataset(Dataset_path, 'train', transform=albumentations_resize)
# resize_test=inhovation_Dataset(Dataset_path, 'test', transform=albumentations_resize)

# resize_train_mean = get_mean(resize_train)
# resize_train_std = get_std(resize_train)
# resize_test_mean = get_mean(resize_test)
# resize_test_std = get_std(resize_test)



albumentations_train = albumentations.Compose([
                                               
    albumentations.Resize(224, 224),
    albumentations.OneOf([
                          albumentations.HorizontalFlip(p=0.8), # p확률로 이미지 좌우 반전
                          albumentations.RandomRotate90(p=0.8), # p확률로 90도 회전
                          albumentations.VerticalFlip(p=0.8) # p확률로 이미지 상하 반전
    ], p=1),

    albumentations.OneOf([
                          albumentations.MotionBlur(p=0.8), # p확률로 이미지를 흐리게(?) 만들어 줌
                          albumentations.OpticalDistortion(p=0.8), # p확률로 이미지 왜곡
                          albumentations.GaussNoise(p=0.8) # 임의의 noise를 삽입
    ], p=1),
    # albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    albumentations.pytorch.ToTensorV2()
])

albumentations_test = albumentations.Compose([
    albumentations.Resize(224, 224),
    # albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    albumentations.pytorch.ToTensorV2()
])


trainset=inhovation_Dataset(Dataset_path, 'train', transform=albumentations_train)
testset=inhovation_Dataset(Dataset_path, 'test', transform=albumentations_test)

albumentations_train_loader = torch.utils.data.DataLoader(trainset, batch_size=16,shuffle=True, num_workers=0)

albumentations_test_loader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=True, num_workers=0)


resnet50 = models.resnet50(pretrained=True).to(device)
resnet50.fc = nn.Linear(resnet50.fc.in_features, 3).to(device)

# # 프리트레인된 모델이 있을시 주석 해제
# state_dict_path='C:/Users/admin/Desktop/resnet/Data/best_model_weight.pt'
# weights = torch.load(state_dict_path)
# # definc model : weights are randomly initiated
# resnet50.load_state_dict(weights['net'])


lr = 0.00005
optimizer = 'Adam'
dataset = 'train_loader'

# 파라미터 클래스
config = Config(
    trainloader = albumentations_train_loader,
    testloader = albumentations_test_loader,
    model = resnet50,
    device = device,
    optimizer = torch.optim.Adam(resnet50.parameters(), lr=lr),
    criterion= nn.CrossEntropyLoss().to(device),
    globaliter = 0
)

ready_to_train=train_test(config)
lr_sche = optim.lr_scheduler.StepLR(config.optimizer, step_size=10000, gamma=0.5) # 20 step마다 lr조정
epochs = 30
log_interval = 22

ready_to_train.train(epochs, log_interval)

