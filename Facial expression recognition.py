# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 00:35:15 2018

@author: LENOVO
"""
import numpy as np
import cv2
labels=['anger','disgust','fear','happy','sad','surprise','neutral']
def get_data(filename):
    Y=[]
    X=[]
    first=True
    for line in open(filename):
        if first:
            first=False
        else:
            row=line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    X=np.array(X)/255
    Y=np.array(Y)
    return X,Y
X,Y=get_data("train.csv")
X=X.reshape(4178,48,48,1)
Y=Y.reshape(4178,1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=0)
import keras as k
model=k.models.Sequential()
model.add(k.layers.Conv2D(32,(3,3),strides=(1,1),activation='relu',name="conv1",kernel_regularizer=k.regularizers.l2(0.01)))
model.add(k.layers.BatchNormalization())
model.add(k.layers.MaxPool2D())
model.add(k.layers.Conv2D(64,(3,3),strides=(1,1),activation='relu',name="conv2",kernel_regularizer=k.regularizers.l2(0.01)))
model.add(k.layers.BatchNormalization())
model.add(k.layers.MaxPool2D())
model.add(k.layers.Dropout(rate=0.25))
model.add(k.layers.Conv2D(128,(3,3),strides=(1,1),activation='relu',name='conv3',kernel_regularizer=k.regularizers.l2(l=0.01)))
model.add(k.layers.BatchNormalization())
model.add(k.layers.Conv2D(256,(3,3),strides=(1,1),activation='relu',padding='same',name='conv4'))
model.add(k.layers.MaxPool2D())
model.add(k.layers.Flatten())
model.add(k.layers.BatchNormalization())
model.add(k.layers.Dense(7,activation='softmax'))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=256,epochs=19)

val_loss,val_accuracy=model.evaluate(x_test,y_test)
predictions=model.predict_classes(x_test)
model.save('Expression_recognition.h5')

import matplotlib.pyplot as plt
import time as t
for i in range(15):
    res=predictions[i]
    res1=int(y_test[i])
    plt.imshow(x_test[i].reshape((48,48)),cmap='gray')
    plt.show()
    print(labels[res])
    print(labels[res1])
    t.sleep(5)
def checker(gray):
    check=np.reshape(gray,(1,48,48,1))
    print(check)
    predict=model.predict_classes(check)
    return predict
res=checker(x_test[0])


    
    
    
    
        
