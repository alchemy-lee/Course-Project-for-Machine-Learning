# In[0]
# 导入模块
from scipy.misc     import imsave
from keras          import metrics
from PIL            import Image
import keras.backend     as K
import numpy             as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from sklearn.svm import SVC
import os

# 使用 keras 搭建的 CNN 网络
from my_CNN import *
# 用于 FGSM 算法的函数
from FGSM import *

# In[1]
# 数据读取，使用 csvPreprocess 生成的 npy 数据

# 将数据归一化并复制为 3 通道
def data_preprocess(data):
    data = data / 255
    data = data.reshape(data.shape[0], 48, 48)
    pred_data = np.empty([data.shape[0], 48, 48, 3])
    for index, item in enumerate(pred_data):
        item[:, :, 0] = data[index]
        item[:, :, 1] = data[index]
        item[:, :, 2] = data[index]
    return pred_data

# 训练集数据
train_data = np.load("data/train_data.npy")
train_data = data_preprocess(train_data)
train_label = np.load("data/train_label.npy")
train_label = to_categorical(train_label, 7)

# 验证集数据
val_data = np.load("data/val_data.npy")
val_data = data_preprocess(val_data)
val_label = np.load("data/val_label.npy")
val_label = to_categorical(val_label, 7)

# 测试集数据
test_data = np.load("data/test_data.npy")
test_data = data_preprocess(test_data)
test_label = np.load("data/test_label.npy")
test_label = to_categorical(test_label, 7)


# In[2]
# 使用 CNN 网络进行训练并预测
model = CNN()
model.fit(train_data, train_label, validation_data=(val_data, val_label), epochs = 50)
print(model.evaluate(test_data, test_label))


# In[3]
# 选择预测正确的 200 个图片生成对抗样本
pre = model.predict(test_data)
cate = []
for i in pre:
    cate.append(np.argmax(i))
Yt = []
for i in range(len(cate)):
    Yt.append(np.argmax(test_label[i]))
test_data_original = []
test_label_original = []
equal_samples =[]
for i in range(len(cate)):
    if cate[i] == np.argmax(test_label[i]):
        equal_samples.append(i)
# 随机选择 200 张图片
choice = np.random.choice(equal_samples, 200, replace=False)
for i in choice:
    if cate[i] == np.argmax(test_label[i]):
        test_data_original.append(test_data[i])
        test_label_original.append(test_label[i])

test_data_original = np.array(test_data_original)
test_label_original = np.array(test_label_original)
print(test_data_original.shape)
print(test_label_original.shape)

# 生成对抗样本并进行预测
test_data_adversarial = generate_adversarial_examples(test_data_original, model)
print(model.evaluate(test_data_adversarial, test_label_original))

# In[4]
# 从训练集选择 300 个图片生成对抗样本并加入训练集
train_data_original = []
train_label_original = []
choice = np.random.choice(train_data.shape[0], 300, replace=False)
for i in choice:
    train_data_original.append(train_data[i])
    train_label_original.append(train_label[i])
train_data_original = np.array(train_data_original)
train_label_original = np.array(train_label_original)
print(train_data_original.shape)
print(train_label_original.shape)

# 生成训练集对抗样本
train_data_adversarial = generate_adversarial_examples(train_data_original, model)

# 原始训练集和对抗样本共同组成的训练集
train_data_con = np.concatenate((train_data, train_data_adversarial))
print(train_data_con.shape)
train_label_con = np.concatenate((train_label, train_label_original))
print(train_label_con.shape)


# In[5]
# 使用新的训练集进行训练
model_adv = CNN()
model_adv.fit(train_data_con, train_label_con, validation_data=(val_data, val_label), epochs = 30)

# 对抗训练网络在 200 个测试集对抗样本上的准确率
print(model_adv.evaluate(test_data_adversarial, test_label_original))
# 对抗训练网络在完整测试集上的准确率
print(model.evaluate(test_data, test_label))


# In[6]
# 生成对抗样本图片样例
plot_adversarial_examples('images/', 'Perturbation and classification', perturbation_model = model, display_model = model)


# In[7]
# SVM
# 归一化处理
scaler = StandardScaler()
Training_data = scaler.fit_transform(train_data)
PrivateTest_data = scaler.fit_transform(test_data)

# SVM 分类
clf = SVC(gamma='auto',kernel='rbf',max_iter=30000)
print("fit start")
clf.fit(Training_data, train_label) 
print("fit finished")
score = clf.score(PrivateTest_data, test_label)
print(score)

