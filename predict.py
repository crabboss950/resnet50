import albumentations
import albumentations.pytorch
import torch
from torchvision import transforms,datasets # 이미지 데이터 transform
from torch.utils.data import DataLoader # 이미지 데이터 로더
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)


# def imshow(input, title):
#     # torch.Tensor를 numpy 객체로 변환
#     input = input.numpy().transpose((1, 2, 0))
#     # 이미지 정규화 해제하기
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     input = std * input + mean
#     input = np.clip(input, 0, 1)
#     # 이미지 출력
#     plt.imshow(input)
#     plt.title(title)
#     plt.show()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.int().numpy().transpose((1, 2, 0))
    # print(inp)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = np.clip(inp, 0, 1)
    # print(inp)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.
    plt.show()


class_names = ['OK','NG']

albumentations_pred = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.pytorch.ToTensorV2()
])

transforms_pred = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.ToTensor()
])

weights = torch.load('C:/Users/admin/Desktop/projec(1)/resnet/Data/best_model.pt')
# C:\Users\admin\Desktop\projec(1)\resnet\Data\best_model.pt
# image = Image.open('/content/drive/MyDrive/NG_000.jpg')
# 경로는 비전캠세이브 이미지로 바꿀거
image = cv2.imread('C:/Users/admin/Desktop/project/Vision/Vision/bin/x64/Debug/ImageSave/CAM1_20211108124501.bmp')
# C:/Users/admin/Desktop/project/Vision/Vision/bin/x64/Debug/ImageSave/CAM1_20211108124501.bmp
# 되는거 : CAM1_20211103174930 CAM1_20211103174937 CAM1_20211103175004
# 마지막에 찍은 사진들이 a4때고찍은건지 붙이고 찍은건지 반사되는 빛이 좀 다른듯


images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# images = Image.fromarray(images)
# imaged = transforms_pred(images).float().unsqueeze(0).to(device)
# augmented = albumentations_pred(image=images)
# imaged = augmented['image']
# imaged = imaged.float().to(device)

with torch.no_grad():
    weights.eval()
    augmented = albumentations_pred(image=images)
    imaged = augmented['image']
    imaged = imaged.unsqueeze(0).float().to(device)
    outputs = weights(imaged)
    _, preds = torch.max(outputs.data, 1)
    # print(preds)
    pred = class_names[preds[0]]
    imshow(imaged.cpu().data[0], title='predict : ' + class_names[preds[0]])
    # print((imaged.cpu().data[0]).size())
    print(pred)


