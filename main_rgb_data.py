import numpy as np
import pandas
import cv2 as cv

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def read_data(Sample):
    df = pandas.read_csv('fer2013.csv')
    X_train = np.zeros((Sample, 48, 48,3))
    Y_train = np.zeros(Sample)

    for i in range(Sample):
        if (i%100 == 0):
            print(i)
        img = df['pixels'][i]
        img = img.split()
        temp = np.zeros(2304)

        for j in range(2304):
            temp[j] = float(img[j]) / 255
        temp = temp.reshape((48, 48), order='F')
        temp = temp.T
        for j in range(48):
            for k in range(48):
                X_train[i][j][k][0] = temp[j][k]
                X_train[i][j][k][1] = temp[j][k]
                X_train[i][j][k][2] = temp[j][k]

        Y_train[i] = df['emotion'][i]
    print("done reading data")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return (X_train, Y_train)



#X,y = read_data(10)
#img = X[4]
#print(np.shape(img))
#cv.imshow("Image",img)
#cv.waitKey(0)
#cv.destroyAllWindows()
