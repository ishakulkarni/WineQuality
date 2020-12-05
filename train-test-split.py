import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

from sklearn.model_selection import train_test_split

rawdata = pd.read_csv('data/winequality-white.csv',sep=';')

data = rawdata.iloc[:,0:11]

rawlabel = rawdata.iloc[:,11]
lable=rawlabel
for i in range(0,len(rawlabel)):
    if(rawlabel[i]<5):
        lable[i]= 1
    elif(rawlabel[i]<8 and rawlabel[i] >4):
        lable[i]=2
    elif(rawlabel[i]>7):
        lable[i]=3

#print(lable.head())
#print("null values:",data.isnull().sum()) ##no null values found in red wine

X_tr, X_test, y_tr, y_test = train_test_split(data,lable,test_size=0.15,shuffle = True)


joblib.dump(X_test,'x_test_wh.joblib')
joblib.dump(y_test,'y_test_wh.joblib')

X_train, X_val, y_train, y_val = train_test_split(X_tr,y_tr,test_size=0.10,shuffle = True)


joblib.dump(X_train,'x_train_wh.joblib')
joblib.dump(y_train,'y_train_wh.joblib')

joblib.dump(X_val,'x_val_wh.joblib')
joblib.dump(y_val,'y_val_wh.joblib')