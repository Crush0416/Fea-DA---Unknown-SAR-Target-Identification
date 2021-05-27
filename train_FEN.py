import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print ('GPU is true')
    print('cuda version: {}'.format(torch.version.cuda))
else:
    print('CPU is true')

## 设置随机种子
seed = 10
# random.seed(seed)
# np.random.seed(seed)
torch.manual_seed(seed)    ##  CPU
# torch.cuda.manual_seed(seed) # GPU: 设置一个随机种子，使得每次训练的随机参数一样，保证结果的一致性。
    
# Hyper parameters
num_epochs = 279       ##  80-99.2577
num_classes = 7
batch_size = 32
learning_rate = 3e-4

##自有数据集准备
##  导入mat数据
# train_dataset = scipy.io.loadmat('traindata.mat')
# test_dataset = scipy.io.loadmat('testdata.mat')
# traindata = train_dataset['data'][0:2448,:,:].reshape(2448,1,128,128)     ### 训练：1-9类   测试：10
# trainlabel = train_dataset['label'][0,0:2448]
# testdata = test_dataset['data'][0:2151,:,:].reshape(2151,1,128,128)       ### 训练：1-9类   测试：10
# testlabel = test_dataset['label'][0,0:2151]
# train_dataset = scipy.io.loadmat('./data/unknown10/traindata_unknown10.mat')
# test_dataset = scipy.io.loadmat('./data/unknown10/testdata_class9.mat')
train_dataset = scipy.io.loadmat('./data/unknown5_8_10/traindata_unknown5_8_10.mat')
test_dataset = scipy.io.loadmat('./data/unknown5_8_10/testdata_class7.mat')
traindata = train_dataset['data']
trainlabel = train_dataset['label'].squeeze()    ##  label必须是一维向量
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

train_dataset=MyDataset(img=traindata,label=trainlabel, transform=transforms.ToTensor())
test_dataset=MyDataset(img=testdata,label=testlabel, transform=transforms.ToTensor())
print('train data size: {}'.format(train_dataset.img.shape[0]))
print('test data size: {}'.format(len(test_dataset)))


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
                                           ##shuffle=True 将训练样本随机打乱防止过拟合

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


# Convolutional neural network (5 convolutional layers)
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

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1, last_epoch = -1)  #动态调整学习率， eg：step_size = 100, gamma = 0.1 每隔100个epoch 学习率减小10倍
# Train the model
print('training...')

total_step = len(train_loader)
train_loss = []
test_loss = []
train_acc = []
test_acc = []
for epoch in range(num_epochs):   
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):       
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs, _ = model(images)                  #输入数据
        loss = criterion(outputs, labels)        #计算误差
        
        # Backward and optimize
        optimizer.zero_grad()                    #清空上一次梯度
        loss.backward()                          #误差反向传播
        optimizer.step()                         #优化器参数更新
        total_loss += loss.item()
        
        if (i+1) % 20 == 0:

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    scheduler.step()    # 学习率规划开始执行
    # model.train() 
    ## 验证
    model.eval()
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
        conv5_fea.extend(temp.data.cpu().numpy())  ### 将数据从GPU转化为CPU
        label.extend(labels.data.cpu().numpy())  ## append():以每一个单元往后添加；extend（）：直接在数组后面拼接
    print('correct number : {}, train data number : {}, train ACC :{:.8f} %'
          .format(correct, total,100 * correct / total))
    train_acc.append(100 * correct / total)     ##  训练精度
    conv5_fea = np.asarray(conv5_fea)
    label = np.asarray(label)
    # scipy.io.savemat('./models/features/train_Conv5_fea.mat',{'data':conv5_fea,'label':label})

    correct = 0
    total = 0
    temp_loss = 0
    for imgT, label in test_loader:
        imgT = imgT.to(device)
        label = label.to(device)
        outputs, _ = model(imgT)
        loss = criterion(outputs,label)
        _, predicted = torch.max(outputs.data, 1)
        temp_loss += loss.item()
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print('correct number : {},  test data number : {},  test ACC :{:.8f} %'.format(correct, total,100*correct/total))
    print ('########  Epoch [{}/{}], training Loss: {:.8f}, testing Loss: {:.8f}  ##########'
           .format(epoch+1, num_epochs, total_loss, temp_loss))
   
    ##  保存训练、测试误差、精度
    test_acc.append(100 * correct / total)    ## 测试精度
    train_loss.append(total_loss)
    test_loss.append(temp_loss)
    model.train()
train_acc = np.asarray(train_acc)
test_acc = np.asarray(test_acc)
train_loss = np.asarray(train_loss)
test_loss = np.asarray(test_loss)
# scipy.io.savemat('./data/unknown5_8_10/acc_loss_279Ep_lr0.0003.mat',
                # {'train_acc':train_acc,'test_acc':test_acc,
                 # 'train_loss':train_loss,'test_loss':test_loss})
print('finished training ')

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
        label.extend(labels.data.cpu().numpy())      ## append():以每一个单元往后添加；extend（）：直接在数组后面拼接
    print('correct number : {}, train data number : {}'.format(correct, total))
    print('Test Accuracy of the model on the train images: {} %'.format(100 * correct / total))
    conv5_fea = np.asarray(conv5_fea)
    label = np.asarray(label)
    # scipy.io.savemat('./models/features/train_Conv5_fea.mat',{'data':conv5_fea,'label':label})
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    conv5_fea = []
    label = []
    label_predict = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs, temp = model(images)   ## temp.data.cpu.numpy()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        conv5_fea.extend(temp.data.cpu().numpy())     ###  将数据从GPU转化为CPU
        label.extend(labels.data.cpu().numpy())
        label_predict.extend(predicted.data.cpu().numpy())
    print('correct number : {}, test data number : {}'.format(correct, total))
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    conv5_fea = np.asarray(conv5_fea)
    label = np.asarray(label)
    label_predict = np.asarray(label_predict)
    matrix = confusion_matrix(label,label_predict)
    print('############ confusion matrix ########### \n', matrix)
    # scipy.io.savemat('./data/class10/confusion_matrix.mat',{'matrix':matrix})

# Save the model checkpoint  保存模型：1.只保存模型参数：
                                        # 保存
                                        # torch.save(the_model.state_dict(), PATH)
                                        # 读取
                                        # the_model = TheModelClass(*args, **kwargs)
                                        # the_model.load_state_dict(torch.load(PATH))
#                                       2.保存整个模型：
                                        #保存
                                        # torch.save(the_model, PATH)
                                        #读取
                                        # the_model = torch.load(PATH)
#                                       PATH的格式：'./model_file_name/the_model_name.tar'

# torch.save(model.state_dict(), './models/model_1to9_143L.ckpt')
# torch.save(model,'./data/unknown5_8_10/models/fullmodel_class7_279Ep_lr0.0003.pkl')




