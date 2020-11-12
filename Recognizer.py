# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 20:57:21 2019

@author: ASUS
"""

import numpy as np
import cv2
import math
from keras.models import load_model
model=load_model('Expression_recognition.h5')
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
capture=cv2.VideoCapture(0)
facecascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

value=[]
labels=['anger','disgust','fear','happy','sad','surprise','neutral']
while True:
    ret,frame=capture.read()
    img=frame
    height,width,channel=frame.shape
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detect=facecascade.detectMultiScale(frame,1.3,5)
    
    
    value=[]
    for x,y,w,h in detect:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        face_image = gray[y:y+h,x:x+w] 
        # print(face_image.shape)
        test=cv2.resize(face_image,(48,48))
        test1=np.reshape(test,(1,48,48,1))
        result=model.predict_classes(test1)
        value.append(result)
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, labels[int(result)], (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
    cv2.imshow("Face Recognizer",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()