from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from imutils import paths
# import cv2

pos_im_path = 'images/CarData/TrainImages/positive'
neg_im_path = 'images/CarData/TrainImages/negative'
test_im_path = 'images/CarData/TestImages'
# read the image files:
# read all the files in the positive image path (so all the required images)
pos_paths = list(paths.list_files(pos_im_path, validExts='.pgm'))  # 提取目录中所有图片
neg_paths = list(paths.list_files(neg_im_path, validExts='.pgm'))
test_paths = list(paths.list_files(test_im_path, validExts='.pgm'))


def read_img(FilePath):
    image = Image.open(FilePath)  # 读取文件
    image = np.array(image)
    imgs = []
    plt.figure()
    for i in range(0, image.shape[0] - 64, 20):  # 实现遍历固定框中的图像
        for j in range(0, image.shape[1] - 128, 30):
            img = image[i:i + 64, j:j + 128]
            imgs.append(img)
    # image = cv2.resize(image, (128, 64))
    print(image.size)  # 输出图片大小
    print(image.shape)
    print(type(image))
    print(image)
    imgs = np.array(imgs)
    print(imgs)
    print(imgs.shape)
    nums = imgs.shape[0]
    for i in range(nums):
        plt.subplot(nums/3, 3, i + 1)
        plt.axis('off')
        plt.imshow(imgs[i], cmap='gray')
    plt.show()


def imageWri(FilePaths):
    i = 0
    for filePath in FilePaths:
        image = Image.open(filePath)
        image.save(test_im_path + '/PNG/' + 'test-{0}.png'.format(i))
        i = i + 1


def imageSize(FilePaths):
    for filePath in FilePaths:
        image = np.array(Image.open(filePath))
        print(image.shape)


read_img(test_paths[0])
