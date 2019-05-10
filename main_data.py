import pandas
import numpy as np
import cv2 as cv


def read_data(Sample):
    df = pandas.read_csv('fer2013.csv')
    X_train = np.zeros((Sample,48,48))
    Y_train = np.zeros(Sample)

    for i in range(Sample):

        img = df['pixels'][i]
        img = img.split()
        temp = np.zeros(2304)

        for j in range(2304):
            temp[j] = float(img[j])/255
        temp = temp.reshape((48,48),order = 'F')
        
        for j in range(48):
            for k in range(48):
                X_train[i][j][k] = temp[j][k]
                
        Y_train[i] = df['emotion'][i]
    print("done reading data")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return(X_train,Y_train)
