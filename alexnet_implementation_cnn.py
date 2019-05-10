import numpy as np
import pandas
import cv2 as cv

from main_rgb_data import read_data

import torchvision.models as models
import torch
from torchvision import transforms, utils

from sklearn.svm import SVC


from sklearn.neural_network import MLPClassifier

print("thug1")

sample = 2000

train , labels = read_data(sample)

print("thug2")

delimit = int(sample/10)

#sample = sample - 100
#train = train[:sample]
#labels = labels[:sample]

#test = train[:]


alexnet = models.alexnet(pretrained = False)
train_alexnet = np.zeros((sample,227,227,3))

print("thug3")


for i in range(sample):
    train_alexnet[i] = cv.resize(train[i] , (0,0) , fx = float(227/48) , fy = float(227/48))

trainfeature  = np.zeros((sample , 1000))

print("thug4")


for i in range(sample):
    temp = train_alexnet[i]
    tt = np.zeros((3,227,227))
    for i1 in range(227):
        for j1 in range(227):
            for k1 in range(3):
                tt[k1][i1][j1] = temp[i1][j1][k1]

    if(i%100 == 0):
        print(i)
    tt = np.float32(tt)
    trainfeature[i] = alexnet(torch.from_numpy(tt).unsqueeze(0)).detach().numpy()

train_feature = trainfeature[:sample - delimit]
test_feature = trainfeature[sample - delimit:sample]
train_labels = labels[:sample - delimit]
test_labels = labels[sample - delimit:sample]

print("thug5")

print("Neural Network")
clf = MLPClassifier(hidden_layer_sizes =(400,400))#, solver='sgd', alpha=1e-5,activation='relu', learning_rate = 'adaptive' , shuffle=True , random_state=1)
clf.fit(train_feature , train_labels)
p = clf.predict(test_feature)

count1 = count2 = 0
print(count1 , count2)

for i in range(p.shape[0]):
    if(p[i] == test_labels[i]):
        count1 = count1 + 1
    else:
        count2 = count2 + 1

print("Accuracy = " + str(count1/(count1 + count2)) + "%")
print(count1 , count2)

print("SVM")
svm = SVC(C = 1.0 , kernel = 'linear' , gamma = 'auto')
t = svm.fit(train_feature , train_labels)

print("thug6")


predict = svm.predict(train_feature)

print("thug7")


count1 = count2 = 0
for i in range(predict.shape[0]):
    if(predict[i] == train_labels[i]):
        count1 = count1 + 1
    else:
        count2 = count2 + 1

print("Accuracy = " + str(count1/(count1 + count2)) + "%")
print(count1 , count2)


