from __future__ import print_function
import numpy as np
import pandas
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

df = pandas.read_csv('fer2013.csv')
print((df['pixels'][0]))
Sample = 35887
X = np.zeros((Sample, 48, 48,1))
y = np.zeros(Sample)

for i in range(Sample):
    img = df['pixels'][i]
    img = img.split()
    temp = np.zeros(2304)
    for j in range(2304):
        # print(img[j])
        temp[j] = float(img[j]) / 255
    X[i] = (temp.reshape((48, 48,1), order='F'))

    y[i] = df['emotion'][i]
print(np.shape(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
y_train = (np.arange(7) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(7) == y_test[:, None]).astype(np.float32)
print(np.shape(X_train))




from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json

from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

batch_size = 128

from CNN_Model import baseline_model
from CNN_Save import baseline_model_saved


flag = 1 #0 for not saved and 1 for saved

# If model is not saved train the CNN model otherwise just load the weights
if(flag==0):
    # Train model
    model = baseline_model()
    # Note : 3259 samples is used as validation data &   28,709  as training samples

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=100,
              verbose=2,
              validation_split=0.2)
    model_json = model.to_json()
    with open("model_2layer_2_2_pool.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_2layer_2_2_pool.h5")
    print("Saved model to disk")
else:
    # Load the trained model
    print("Load model from disk")
    model = baseline_model_saved()


# Model will predict the probability values for 7 labels for a test image
score = model.predict(X_test)
# print(score[0])
pred = []
out = []
# print((y_test[0]))
for i in range(len(score)):
    max = 0
    maxInd = -1
    temp = score[i]
    # print(temp)
    for j in range(len(temp)):
        if(max<score[i][j]):
            max = temp[j]
            maxInd = j
    pred.append(maxInd)
    max = 0
    maxInd = -1

    for k in range(len(temp)):
        if(max<y_test[i][k]):
            max = y_test[i][k]
            maxInd = k
    out.append(maxInd)


print(pred)
print(out)

print(accuracy_score(out, pred))