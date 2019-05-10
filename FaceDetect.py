import numpy as np
import cv2



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('images.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)


(x,y,w,h) = faces[0]
extractedImage = np.zeros((h+1,w+1))
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
roi_gray = gray[y:y + h, x:x + w]

print(h)
print(w)
for i in range(h):
    for j in range(w):
        temp = gray[i+y][j+x]
        extractedImage[i][j] = temp/255

CorrImage = cv2.resize(extractedImage,(48,48),interpolation=cv2.INTER_CUBIC)
cv2.imshow('img',img)

some = cv2.waitKey(0) & 0xFF
if some == 27:
    cv2.destroyAllWindows()