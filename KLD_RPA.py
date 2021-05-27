from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import svm
import scipy.io as sio
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import scipy.stats
import math
from sklearn.manifold import TSNE

# Device configuration
   
# Hyper parameters
seed = 10
np.random.seed(seed)

## 计算Arc cos 角度
def arc_cos(x,y):
    cos = np.dot(x,y) / (np.linalg.norm(x)*np.linalg.norm(y))
    arc_dist = math.acos(cos)
    return arc_dist
    
## load data 
train_dataset = sio.loadmat('./data/unknown5_8_10/features/train_Conv5_fea_unknown5_8_10_1983_279Ep.mat')
traindata = train_dataset['data']
trainlabel = train_dataset['label'].T

test_dataset = sio.loadmat('./data/unknown5_8_10/features/test_Conv5_fea_unknown5_8_10_2059_279Ep.mat')
testdata = test_dataset['data']
testlabel = test_dataset['label'].T

test_dataset_T = sio.loadmat('./data/unknown5_8_10/features/test_Conv5_fea_class7_1759_279Ep.mat')
testdata_T = test_dataset_T['data']
testlabel_T = test_dataset_T['label'].T

##        数据归一化
# scaler = MinMaxScaler()
# traindata = scaler.fit_transform(traindata.T).T    ##  按列进行归一化
# testdata = scaler.fit_transform(testdata.T).T
# testdata_T = scaler.fit_transform(testdata_T.T).T

##     打乱特征顺序给SVM训练
index = [i for i in range(len(traindata))]
np.random.shuffle(index)
traindata1 = traindata[index]
trainlabel1 = trainlabel[index]

##       计算特征中心点
feature_center = []
index_sort = []
bb = set(trainlabel[:,0])
for i in set(trainlabel[:,0]):
    index = np.where(trainlabel[:,0] == i)
    data = traindata[index,:].reshape(len(np.array(index).T),128)
    clf = KMeans(n_clusters=1)     ## n_clusters=1,max_iter=500,n_init=15
    clf.fit(data)
    score = clf.score(data)
    center = clf.cluster_centers_
    feature_center.extend(center)
    index_sort.append(len(np.array(index).T))
feature_center = np.array(feature_center) 
    
##      t-SNE对特征中心进行降维   128d ---> 2d
fea_center_all = feature_center
print('Computing t-SNE embedding')
tsnef = TSNE(n_components=2, init='pca', random_state=0)
result_fea_center = tsnef.fit_transform(fea_center_all)
print(' t-SNE embedding completed...')
# sio.savemat('./data/unknown8/features/feature_center.mat', {'data':fea_center_class9})   ##  保存中心点

## 计算训练数据KL散度 确定门限值大小
KL_train = []
for i, fea in enumerate(traindata):
    KL_temp = []
    for fea_center in feature_center:
        KL1 = scipy.stats.entropy(fea, fea_center + 0.001)   ##添加一个小的偏置，防止计算KL散度时出现分母为0的情况
        # KL2 = scipy.stats.entropy(fea_center, fea + 0.001)
        # KL_temp.append(KL1 + KL2)
        KL_temp.append(KL1)
    KL_train.append(KL_temp)
KL_train = np.array(KL_train)

##      计算训练数据KL门限值   
element = np.unique(trainlabel)
data_num = []
num1 = 0
for i in element:
    index = np.where(trainlabel[:, 0] == i)
    num = len(np.asarray(index).T)
    num1 = num1 + num
    data_num.append([i,num,num1])
data_num = np.asarray(data_num)
data_sum = np.r_[0,data_num[:,2]]
th = []
for i in data_num[:,0]:
    th_temp = max(KL_train[data_sum[i]:data_sum[i+1],i])
    th.append(th_temp)
threshold = max(th)
print('KL training max threshlod : {}'.format(threshold))    

##       training SVM
print('------- training and testing svm -------')
clf = svm.SVC(C=1, kernel='rbf', gamma=0.01, max_iter=200, decision_function_shape='ovr')
clf.fit(traindata1, trainlabel1.ravel())

##      Test on training data
T = clf.predict(traindata)
train_result = T.reshape(len(trainlabel),1)
precision = sum(train_result == trainlabel)/trainlabel.shape[0]
print('Training Precision : {} %'.format(float(precision)*100))

##         svm决策函数
decision1 = clf.decision_function(traindata)
decision2 = clf.decision_function(testdata)

##         Test on test data
T = clf.predict(testdata_T)
test_result = T.reshape(len(testlabel_T),1)
precision = sum(test_result == testlabel_T)/testlabel_T.shape[0]
print('Test Precision: {} %'.format(float(precision)*100))
print('target recognizing, please wait few seconds...')
###  判断是否过门限：是--->标记为新目标；  否--->进入SVM判断类别

##################---RPA_KL测试流程说明---###########################################
##  1. 将RPA门限值设定为-3.14-3.14。默认所有满足KL门限值的目标都是新目标。         ##
##  2. 调节KL门限值，使得'已知目标精度*0.5+未知目标精度*0.5’识别精度达到最高。     ##
##  3. 固定KL门限值，调节RPA门限值，使得总体识别精度达到最高，确定新目标分布范围。 ##
##  4. 固定并保存KL门限值，RPA门限值                                               ##
#####################################################################################
##               KL散度判别
New_target = []
target_KL = []
target_predict = []
KL_test = []
arc_dist_test = []
for i, fea in enumerate(testdata):
    KL_temp = []
    arc_cos_temp = []
    for fea_center in feature_center:
        KL1 = scipy.stats.entropy(fea,fea_center + 0.001)
        # KL2 = scipy.stats.entropy(fea_center,fea + 0.001)
        arcos = arc_cos(fea,fea_center)
        arc_cos_temp.append(arcos)
        KL_temp.append(KL1)
    KL_test.append(KL_temp)
    arc_dist_test.append(arc_cos_temp)
    th1 = min(KL_temp)
    index = KL_temp.index(th1)
    th3 = min(arc_cos_temp)
    if th1 > threshold*4.12:                   #  设置KL判决门限
        New_target.append([i, index])      ## 将目标索引i，所属最近类别index 保存下来，后续判断
    else:
        fea = fea.reshape(1, 128)
        T = clf.predict(fea)
        target_predict.append(int(T))
        target_KL.append(i)
KL_test = np.array(KL_test)        ## 查看测试样本KL散度
arc_dist_test = np.array(arc_dist_test)
New_target = np.asarray(New_target)

##          RPA（相对位置角）鉴别
## 设置角度门限范围
th2a = -3.09      
th2b = -0.3
th2c = -0.7
th2d = -0.3
th2e = -3.14
th2f = -2.9
wrong_target_index = New_target[:,0].squeeze()
target_category = New_target[:,1].squeeze()
wrong_data = testdata[wrong_target_index, :]
result_target = tsnef.fit_transform(wrong_data)
cos_angle = []
New_target_index = []
target_index = []
for i in range(len(result_target)):
    a1 = result_fea_center[target_category[i],0]
    b1 = result_fea_center[target_category[i],1]     ##  所属类别中心点坐标（a1, b1）
    a2 = result_target[i,0]
    b2 = result_target[i,1]                          ##  测试目标坐标（a2, b2）
    cosine = (a2 - a1) / math.sqrt((a2-a1)**2 + (b2-b1)**2)  # 计算余弦
    cos_temp = math.acos(cosine)            # 求反余弦角度
    if b2 >= b1:                            # 判断位置，确定角大小 angle: [-3.14 - +3.14]
        cos_temp = cos_temp
    else:
        cos_temp = - cos_temp
    cos_angle.append(cos_temp)
    if th2a < cos_temp < th2b:                                      ##  ---判别式1
    # if th2a < cos_temp < th2b or (th2c <= cos_temp <= th2d):      ##  ---判别式2
    # if (th2a <= cos_temp <= th2b) or \                           
       # (th2c <= cos_temp <= th2d) or \
       # (th2e <= cos_temp <= th2f):        # 判断门限2 RPA  [下限， 上限]  ---判别式3
        New_target_index.append(i)          # 未知目标索引
    else:
        target_index.append(i)              # 错判目标索引
cos_angle = np.asarray(cos_angle).reshape(len(cos_angle),1)
New_target_Final = wrong_target_index[New_target_index]    # 未知目标索引-final
target_RPA_index = wrong_target_index[target_index]        # 错判数据索引-final
target_misjudge = testdata[target_RPA_index,:]             # 错判数据-final
T1 = clf.predict(target_misjudge)                          # 预测标签

##  合并测试结果
test_result_all = np.ones((len(testlabel),1)) * 10086
test_result_all[New_target_Final] = 7                      ## 设置新样本标签
test_result_all[target_KL] = np.array(target_predict).reshape(len(target_predict),1)   
                                                           ## KLD预测标签
test_result_all[target_RPA_index] = np.array(T1).reshape(len(T1),1)    
                                                           ## RPA预测标签
## show the confusion matrix
matrix = confusion_matrix(testlabel_T,test_result)
print('############ confusion matrix ########### \n', matrix)

matrix = confusion_matrix(testlabel,test_result_all)
print('############ confusion matrix ########### \n', matrix)

m = testlabel_T.shape[0]    ##  已知目标数量
n = testlabel.shape[0]      ##  总目标数量
precision1 = sum(test_result_all[0:m] == testlabel_T)/testlabel_T.shape[0]     
                                                                    ## 已知目标精度
precision2 = sum(test_result_all[m:n] == testlabel[m:n])/testlabel[m:n].shape[0]   
                                                                    ## 新目标精度
precision3 = sum(test_result_all == testlabel)/testlabel.shape[0]   ## 总体目标精度
precision4 = precision1*0.5 + precision2*0.5                        ## 调参指示精度
print('Known Target Precision : {} % \nUnknown Target Precision : {} % \nTotal Target Precision: {} % \nPrecision Metric: {}'.format(float(precision1)*100,float(precision2)*100,float(precision3)*100,float(precision4)*100))

# sio.savemat('./data/unknown5_8_10/test_unknown5_8_10_confusion_matrix_279Ep.mat',{'matrix': matrix})



