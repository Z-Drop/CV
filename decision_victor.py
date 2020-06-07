import cv2
import os
from skimage import feature as ft
import numpy as np
from sklearn.model_selection._split import train_test_split
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from sklearn.decomposition._pca import PCA
# from builtins import ints

pos_im_path = 'images/CarData/TrainImages/positive'
neg_im_path = 'images/CarData/TrainImages/negative'
# read the image files:
# read all the files in the positive image path (so all the required images)
pos_im_listing = os.listdir(pos_im_path)
neg_im_listing = os.listdir(neg_im_path)

data = []
labels = []
for file in pos_im_listing:
    img = cv2.imread(pos_im_path + '/' + file, 0)  # open the file
    img = cv2.resize(img, (64, 128))
    # calculate HOG for positive features
    fd = ft.hog(img, 9, (8, 8), (2, 2), block_norm='L2',
                feature_vector=True)  # fd= feature descriptor
    # print(fd)
    data.append(fd)
    labels.append(1)

for file in neg_im_listing:
    img = cv2.imread(neg_im_path + '/' + file, 0)
    img = cv2.resize(img, (64, 128))
    # Now we calculate the HOG for negative features
    fd = ft.hog(img, 9, (8, 8), (2, 2), block_norm='L2', feature_vector=True)
    print(fd)
    data.append(fd)
    labels.append(0)
# encode the labels, converting them from strings to integers

data = np.array(data)
labels = np.array(labels)
data = PCA(n_components=2).fit_transform(data)

# 拆分训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size=0.5)
# 构建线性SVM对象并训练
clf = LinearSVC(C=1, loss="hinge").fit(x_train, y_train)
# 训练数据预测正确率
print(clf.score(x_train, y_train))
# 测试数据预测正确率
print('Accuracy Rate:', clf.score(x_test, y_test))
# 画训练数据散点图
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
# 画测试数据散点图
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolors='b')

plt.show()
