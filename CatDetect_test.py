import CarDetect as cd
from imutils import paths
import cv2
from PIL import Image
import numpy as np
from sklearn.decomposition._pca import PCA
from matplotlib import pyplot as plt

pos_im_path = 'images/CarData/TrainImages/positive'
neg_im_path = 'images/CarData/TrainImages/negative'
test_im_path = 'images/CarData/testImage'
# test_image = ''
# read the image files:
# read all the files in the positive image path (so all the required images)
pos_paths = list(paths.list_files(pos_im_path, validExts='.pgm'))  # 提取目录中所有图片
neg_paths = list(paths.list_files(neg_im_path, validExts='.pgm'))
test_paths = list(paths.list_files(test_im_path, validExts='.jpg'))

datas, labels = cd.get_train_datas(pos_paths, neg_paths)
clf = cd.get_SVM_classifier(datas, labels, 0.5)

test_data = []
for test_path in test_paths:
    test_data.append(cd.get_HoG_ft(cd.get_images(test_path)))
test_data = PCA(n_components=2).fit_transform(test_data)
test_lables = clf.predict(test_data)
print(test_lables)
plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.axis('off')
    plt.imshow(cd.get_images(test_paths[i]))
plt.show()
plt.figure()
for i in range(12):
    if (test_lables[i] == 1):
        plt.subplot(3, 4, i + 1)
        plt.axis('off')
        plt.imshow(cd.get_images(test_paths[i]))
plt.show()

# # 拿一张图像进行分割实验
# test_im_path = 'images/CarData/testImages'
# test_paths = list(paths.list_files(test_im_path, validExts='.pgm'))
# test_image = np.array(Image.open(test_paths[12]))
# test_datas = []
# temp_labels = []
# test_labels = []
# imgs = []
# # for h in range(20, 65, 15):  # 遍历矩形框大小初始值为40*20,增大步长为15，最大值为100*50
# #     for i in range(0, test_image.shape[0] - h, h):  # 实现遍历固定框中的图像
# #         for j in range(0, test_image.shape[1] - h * 2, h * 2):
# #             img = test_image[i:i + h, j:j + h * 2]
# #             imgs.append(img)
# #     print("Size:({1},{0}),Conuts:{2}".format(h, h*2, len(imgs)))
# for h in range(40, 100, 40):  # 遍历矩形框大小初始值为40*20,增大步长为15，最大值为100*50
#     for i in range(0, test_image.shape[0] - h, h):  # 实现遍历固定框中的图像
#         for j in range(0, test_image.shape[1] - h * 2, h * 2):
#             img = test_image[i:i + h, j:j + h * 2]
#             imgs.append(img)
#     print("Size:({1},{0}),Conuts:{2}".format(h, h*2, len(imgs)))
# print(len(imgs))
# for img in imgs:
#     # print(img)
#     # print(img.shape)
#     # cv2.imshow('image', img)
#     # cv2.waitKey(0)
#     img = cv2.resize(img, (128, 64))
#     test_datas.append(cd.get_HoG_ft(img))  # 将分割开的图像分别求HoG特征
# test_datas = np.array(test_datas)
# print(test_datas)
# test_datas = PCA(n_components=2).fit_transform(test_datas)
# temp_labels = clf.predict(test_datas)  # 使用分类器预测标签
# print(temp_labels)
# plt.figure()
# cars = []
# for i in range(temp_labels.shape[0]):
#     if temp_labels[i] == 1:
#         test_labels.append(1)
#         break
# for i in range(temp_labels.shape[0]):
#     if temp_labels[i] == 1:
#         cars.append(imgs[i])
# cars.append(test_image)
# print(len(cars))
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.imshow(cars[i], cmap='gray')
# print(test_labels)
# plt.show()

# # 遍历标签检查师是否有已经识别的汽车,如果有就框出
# # 没有就变化遍历矩形
# # if temp_labels[i] == 1:
# #     image = cv2.rectangle(test_image, (i * h, i * 2 * h),
# #                           (i * h + h, i * 2 * h + 2 * h), (255, 0, 0))
