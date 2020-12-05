import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

X_train = joblib.load('data/x_train_w.joblib')
y_train = joblib.load('data/y_train_w.joblib')
x_val = joblib.load('data/x_val_w.joblib')
y_val = joblib.load('data/y_val_w.joblib')

x_test = joblib.load('data/x_test_w.joblib')
y_test = joblib.load('data/y_test_w.joblib')


print('train shape=',np.shape(X_train))
print('val shape=',np.shape(x_val))

## Random forest model was selected having maximum accuracy and F1 score
cvmodel = joblib.load('3-rf1.joblib')

## since we have less no. of samples, training this model again with entire training and validation data
xtr = pd.concat([X_train,x_val])
ytr = pd.concat([y_train,y_val])
print('combined train shape=',np.shape(xtr))
cvmodel.fit(xtr,ytr)
joblib.dump(cvmodel,'white-final-trained-model.joblib')
loaded = joblib.load('white-final-trained-model.joblib')
pred = loaded.predict(x_test)

print('test acc= ',accuracy_score(pred,y_test))
print('balanced test acc= ',balanced_accuracy_score(pred,y_test))
cm = confusion_matrix(y_test,pred)
display=ConfusionMatrixDisplay(cm).plot()
plt.figure(1)
plt.title('Test white wine')
classNames = ['poor','average','excellent']
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)
plt.yticks(tick_marks, classNames)
plt.show(display)
