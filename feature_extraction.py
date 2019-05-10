import cv2 as cv
import numpy as np
import pandas
from main_data import read_data
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC

print("thug1")

train_images , train_labels = read_data(500)

print("thug2")



lbp = np.zeros((train_images.shape[0] , train_images.shape[1] , train_images.shape[2]))
lbp_flattened = np.zeros((train_images.shape[0], train_images.shape[1]*train_images.shape[2]))

print("thug3")

for i in range(train_images.shape[0]):
    lbp_image = local_binary_pattern(train_images[i].T, 60 , 1, method = 'default')
    lbp[i] = lbp_image
    lbp_flattened[i] = np.reshape(lbp_image,(2304,))


print("thug4")



clf = SVC(gamma='auto')
print("thug5")

k = clf.fit(lbp_flattened , train_labels)

print("thug6")

test,labels = read_data(20000)
test = test[19900:20000]
labels = labels[1900:2000]
test_lbp = np.zeros((test.shape[0] , test.shape[1] , test.shape[2]))
test_flat = np.zeros((test.shape[0] , test.shape[1] * test.shape[2]))

print("thug7")

for i in range(test.shape[0]):
    a = local_binary_pattern(test[i].T, 60 , 1, method = 'default')
    test_lbp[i] = a
    test_flat[i] = np.reshape(a,(2304,))


print(test_flat.shape)
predict = clf.predict(test_flat)
count1 = count2 = 0
for i in range(predict.shape[0]):
    if(predict[i] == labels[i]):
        count1 = count1 + 1
    else:
        count2 = count2 + 1


print("Accuracy = " + str(count1/(count1 + count2))+"%")
