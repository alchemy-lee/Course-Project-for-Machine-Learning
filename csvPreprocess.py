import numpy as np
import scipy.io
import sklearn.metrics
import pandas
import time

# 预处理数据并存储为 npy 文件，方便后续使用

# 读取原始 csv 文件
start = time.clock()
csv_data = pandas.read_csv('data/fer2013.csv')
print(csv_data.shape)
csv_data = np.asarray(csv_data)
end = time.clock()
print("Read data: %f" % (end - start))

# 处理为 np 数组并进行存储
train_data = []
train_label = []
val_data = []
val_label = []
test_data = []
test_label = []
for data in csv_data:
    t = []
    for i in data[1].split(' '):
        t.append(int(i))
    if data[2] == "Training":
        train_data.append(t)
        train_label.append(data[0])
    elif data[2] == "PublicTest":
        val_data.append(t)
        val_label.append(data[0])
    elif data[2] == "PrivateTest":
        test_data.append(t)
        test_label.append(data[0])
train_data = np.asarray(train_data)
train_label = np.asarray(train_label)
val_data = np.asarray(val_data)
val_label = np.asarray(val_label)
test_data = np.asarray(test_data)
test_label = np.asarray(test_label)

np.save("data/train_data", train_data)
np.save("data/train_label", train_label)
np.save("data/val_data", val_data)
np.save("data/val_label", val_label)
np.save("data/test_data", test_data)
np.save("data/test_label", test_label)
