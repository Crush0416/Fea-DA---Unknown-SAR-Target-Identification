import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


root = "D:/DeepLearning_In_SAR/Dataset/mstar/"

def default_loader(path):
    return Image.open(path).convert('L')    #彩色图片转化为 RGB 三通道模式，灰度图转化为 L 模式
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset,self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)           
        return img,label

    def __len__(self):
        return len(self.imgs)

train_dataset=MyDataset(txt=root+'train1.txt', transform=transforms.ToTensor())
test_dataset=MyDataset(txt=root+'test1.txt', transform=transforms.ToTensor())
print('train data size: {}，\n data[0]: {}'.format(len(train_dataset), train_dataset[300]))
print('test data size: {}'.format(len(test_dataset)))

## 保存原始数据---> .mat
traindata = []
trainlabel = []
for i, (img, label) in enumerate (train_dataset):
    img = np.asarray(img).reshape(128,128)
    traindata.append(img)
    trainlabel.append(label)

traindata = np.array(traindata)
trainlabel = np.array(trainlabel)
# scipy.io.savemat('traindata.mat',{'data':traindata,'label':trainlabel})

testdata = []
testlabel = []
for i, (img, label) in enumerate (test_dataset):
    img = np.asarray(img).reshape(128,128)
    testdata.append(img)
    testlabel.append(label)

testdata = np.array(testdata)
testlabel = np.array(testlabel)
# scipy.io.savemat('testdata.mat',{'data':testdata,'label':testlabel})