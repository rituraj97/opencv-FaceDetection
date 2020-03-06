import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('family.jpg')
imS = cv2.resize(img, (960, 540)) 
gray = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.6, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(imS,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = imS[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',imS)
cv2.waitKey(0)
cv2.destroyAllWindows()
