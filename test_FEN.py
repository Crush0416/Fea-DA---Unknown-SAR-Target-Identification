import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print ('GPU is true')
else:
    print('CPU is true')

#超参数
PATH = './data/unknown5_8_10/models/fullmodel_class7_279Ep_lr0.0003.pkl'      ## 模型路径
num_classes = 7
batch_size = 100

##  导入数据
##  导入mat数据
train_dataset = scipy.io.loadmat('./data/unknown5_8_10/traindata_unknown5_8_10.mat')
test_dataset_T = scipy.io.loadmat('./data/unknown5_8_10/testdata_class7.mat')
test_dataset = scipy.io.loadmat('./data/unknown5_8_10/testdata_unknown5_8_10.mat')
traindata = train_dataset['data']
trainlabel = train_dataset['label'].squeeze()    ##  label必须是一维向量
testdata_T = test_dataset_T['data']
testlabel_T = test_dataset_T['label'].squeeze()
testdata = test_dataset['data']
testlabel = test_dataset['label'].squeeze()

class MyDataset(Dataset):
    def __init__(self, img, label, transform=None):
        super(MyDataset,self).__init__()       
        self.img = torch.from_numpy(img).float()
        self.label = torch.from_numpy(label).long()
        self.transform = transform

    def __getitem__(self, index): 
        img = self.img[index]
        label = self.label[index]
        return img,label

    def __len__(self):
        return self.img.shape[0]

train_dataset = MyDataset(img=traindata,label=trainlabel, transform=transforms.ToTensor())
test_dataset_T = MyDataset(img=testdata_T,label=testlabel_T, transform=transforms.ToTensor())
test_dataset = MyDataset(img=testdata,label=testlabel, transform=transforms.ToTensor())

print('train data size: {}'.format(train_dataset.img.shape[0]))
print('test data size1: {} \ntest data size2: {}'.format(len(test_dataset_T),len(test_dataset)))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)
                                           ##shuffle=True 将训练样本随机打乱防止过拟合

test_loader_T = torch.utils.data.DataLoader(dataset=test_dataset_T,
                                          batch_size=batch_size, 
                                          shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
##  创建一个一模一样的模型
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),   ## (128-5)/1+1=124
            nn.BatchNorm2d(20),            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                  ## 62*62
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=5, stride=1, padding=0),  ## 58*58
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                  ## 29*29
        self.layer3 = nn.Sequential(
            nn.Conv2d(40, 60, kernel_size=6, stride=1, padding=0),  ## 24*24
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                  ## 12*12
        self.layer4 = nn.Sequential(
            nn.Conv2d(60, 120, kernel_size=5, stride=1, padding=0),  ## 8*8
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                   ## 4*4
        self.layer5 = nn.Sequential(
            nn.Conv2d(180, 128, kernel_size=4, stride=1, padding=0), ## 1*1
            nn.BatchNorm2d(128),           
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1))
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1))
        
        self.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=3) 
        
        self.fc1 = nn.Linear(1*1*128, num_classes)
        self.fc2 = nn.Linear(512, num_classes)                
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out1 = self.layer3(out)
        out = self.layer4(out1)
        out1 = self.MaxPool2d(out1)
        out = self.layer5(torch.cat((out,out1),dim=1))
        #out = self.layer6(out)      
        conv5_fea = out.reshape(out.size(0), -1)        
        out = self.fc1(conv5_fea)
        # out = self.fc2(out)
        return out,conv5_fea
##  导入模型
model = torch.load(PATH)  


## training accuarcy && save conv_layer features 
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    conv5_fea = []
    label = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs, temp = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        conv5_fea.extend(temp.data.cpu().numpy())    ### 将数据从GPU转化为CPU
        label.extend(labels.data.cpu().numpy())
    print('correct number : {}, train data number : {}'.format(correct, total))
    print('Test Accuracy of the model on the train images: {} %'.format(100 * correct / total))
    conv5_fea = np.asarray(conv5_fea)
    label = np.asarray(label)
    # scipy.io.savemat('./data/unknown5_8_10/features/train_Conv5_fea_unknown5_8_10_1983_279Ep.mat',{'data':conv5_fea,'label':label})

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    conv5_fea = []
    label = []
    label_pre = []
    for images, labels in test_loader_T:
        images = images.to(device)
        labels = labels.to(device)
        outputs, temp = model(images)                 ## temp.data.cpu.numpy()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        conv5_fea.extend(temp.data.cpu().numpy())     ###  将数据从GPU转化为CPU
        label.extend(labels.data.cpu().numpy())
        label_pre.extend(predicted.data.cpu().numpy())
    print('correct number : {}, test data number : {}'.format(correct, total))
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    conv5_fea = np.asarray(conv5_fea)
    label = np.asarray(label)
    label_pre = np.asarray(label_pre)
    matrix = confusion_matrix(label,label_pre)
    print('############ confusion matrix ########### \n', matrix)
    # scipy.io.savemat('./data/unknown5_8_10/features/test_Conv5_fea_class7_1759_279Ep.mat',{'data':conv5_fea,'label':label})

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    conv5_fea = []
    label = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs, temp = model(images)   ## temp.data.cpu.numpy()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        conv5_fea.extend(temp.data.cpu().numpy())     ###  将数据从GPU转化为CPU
        label.extend(labels.data.cpu().numpy())
    print('correct number : {}, test data number : {}'.format(correct, total))
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    conv5_fea = np.asarray(conv5_fea)
    label = np.asarray(label)
    # scipy.io.savemat('./data/unknown5_8_10/features/test_Conv5_fea_unknown5_8_10_2059_279Ep.mat',{'data':conv5_fea,'label':label})
                                            