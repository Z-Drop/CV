from imutils import paths
from PIL import Image
import numpy as np
import CarDetect as cd
from sklearn.decomposition._pca import PCA


test_im_path = 'images/CarData/testImages'
test_paths = list(paths.list_files(test_im_path, validExts='.pgm'))
test_image = np.array(Image.open(test_paths[3]))
test_datas = []
test_labels = []
imgs = []
for h in range(20, 50, 15):  # 遍历矩形框大小初始值为40*20,增大步长为15，最大值为100*50
    for i in range(0, test_image.shape[0] - h, h):  # 实现遍历固定框中的图像
        for j in range(0, test_image.shape[1] - h*2, h*2):
            img = test_image[i:i + h, j:j + h*2]
            imgs.append(img)
    for img in imgs:
        test_datas.append(cd.get_HoG_ft(img))  # 将分割开的图像分别求HoG特征
    test_datas = np.array(test_datas)
    test_datas = PCA(n_components=2).fit_transform(test_datas)
    

