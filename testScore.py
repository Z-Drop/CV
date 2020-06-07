import CarDetect as cd
from imutils import paths
from sklearn.decomposition._pca import PCA

pos_im_path = 'images/CarData/TrainImages/positive'
neg_im_path = 'images/CarData/TrainImages/negative'
test_im_path = 'images/CarData/TestImages'
pos_paths = list(paths.list_files(pos_im_path, validExts='.pgm'))  # 提取目录中所有图片
neg_paths = list(paths.list_files(neg_im_path, validExts='.pgm'))
test_paths = list(paths.list_files(test_im_path, validExts='.pgm'))

datas, labels = cd.get_train_datas(pos_paths, neg_paths)
clf = cd.get_SVM_classifier(datas, labels, 0.5)

test_data = []
for test_path in test_paths:
    test_data.append(cd.get_HoG_ft(cd.get_images(test_path)))
test_data = PCA(n_components=2).fit_transform(test_data)
test_labels = clf.predict(test_data)
print(test_labels)
print(len(test_labels))
flag = 0
for i in range(len(test_labels)):
    if(test_labels[i] == 1):
        flag = flag + 1
print(flag)
print("test score :{0:.2%}".format(flag/170))
