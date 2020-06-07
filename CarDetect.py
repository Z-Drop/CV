import cv2
from skimage import feature as ft
import numpy as np
from sklearn.model_selection._split import train_test_split
from sklearn.svm import LinearSVC
# from matplotlib import pyplot as plt
from sklearn.decomposition._pca import PCA
from imutils import paths
from PIL import Image


def get_HoG_ft(image):  # 提取图像HoG特征
    return ft.hog(
        image,
        9,
        (8, 8),
        (2, 2),  # bin,cell,block
        block_norm='L2',  # block内部norm类型
        feature_vector=True)  # 将输出转化为一维向量


def get_images(im_path):  # 获取图像并预处理
    image = np.array(Image.open(im_path))
    image = cv2.resize(image, (128, 64))
    return image


def get_train_datas(pos_im_paths, neg_im_paths):
    train_datas = []  # 存放训练数据
    train_lables = []  # 存放训练数据标签
    # 提取HoG特征存入训练数据数组
    for pos_im_path in pos_im_paths:
        train_datas.append(get_HoG_ft(get_images(pos_im_path)))
        train_lables.append(1)
    for neg_im_path in neg_im_paths:
        train_datas.append(get_HoG_ft(get_images(neg_im_path)))
        train_lables.append(0)
    train_datas = np.array(train_datas)  # 将图片数组转化为n维矩阵
    train_lables = np.array(train_lables)
    print("train_datas:%d" % train_datas.shape[0])  # 输出训练图片数量
    # print(train_datas.shape)
    # np.savetxt('out.txt', train_datas)
    train_datas = PCA(n_components=2).fit_transform(train_datas)
    return train_datas, train_lables


def get_SVM_classifier(datas, labels, split_size):
    # 拆分训练数据和测试数据
    x_train, x_test, y_train, y_test = train_test_split(datas,
                                                        labels,
                                                        test_size=split_size)
    # 构建先修SVM对象并训练
    clf = LinearSVC(C=1, loss="hinge").fit(datas, labels)
    print("Score of train datas:{0:.2%}".format(clf.score(x_train, y_train)))
    print("Score of train datas(split_size:{0}):{1:.2%}".format(
        split_size, clf.score(x_test, y_test)))
    return clf


if __name__ == "__main__":
    pos_im_path = './images/CarData/TrainImages/positive'
    neg_im_path = './images/CarData/TrainImages/negative'
    test_im_path = './images/CarData/testImages'
    test_image = ''
    # read the image files:
    # read all the files in the positive image path (so all the required images)
    pos_paths = list(paths.list_files(pos_im_path,
                                      validExts='.pgm'))  # 提取目录中所有图片
    neg_paths = list(paths.list_files(neg_im_path, validExts='.pgm'))
    test_paths = list(paths.list_files(test_im_path, validExts='.pgm'))

    datas, labels = get_train_datas(pos_paths, neg_paths)
    clf = get_SVM_classifier(datas, labels, 0.5)
