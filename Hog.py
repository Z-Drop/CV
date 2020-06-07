import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import cv2
from PIL import Image
import numpy as np
from imutils import paths
# image = cv2.imread("./images/Harry.jpg")
pos_im_path = 'images/CarData/TrainImages/positive'
neg_im_path = 'images/CarData/TrainImages/negative'
# read the image files:
# read all the files in the positive image path (so all the required images)
pos_paths = list(paths.list_files(pos_im_path, validExts='.pgm'))  # 提取目录中所有图片
neg_paths = list(paths.list_files(neg_im_path, validExts='.pgm'))
image = Image.open(pos_paths[0])
image = np.array(image)
image = cv2.resize(image, (64, 128))  # 将图像大小变为64*128
fd, hog_image = hog(
    image,
    orientations=9,  # 每一个cell向量的维数
    pixels_per_cell=(8, 8),  # 每一个cell的大小
    cells_per_block=(2, 2),  # 每一个block包含4个cell
    visualize=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
print(fd.shape)
print(type(fd))
plt.show()
