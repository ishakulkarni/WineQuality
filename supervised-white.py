#from collections import Counter
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipe

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# For model evaluation
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


X_train = joblib.load('data/x_train_w.joblib')
y_train = joblib.load('data/y_train_w.joblib')
x_val = joblib.load('data/x_val_w.joblib')
y_val = joblib.load('data/y_val_w.joblib')

print(X_train.head())
print(x_val.head())

acc= []
f1 = []

###  1 Decision trees
dectree_param_grid = [{ 'clf__class_weight': ['balanced', None]
      ,'clf__criterion': ['gini', 'entropy']
      ,'clf__splitter': ['best']
      ,'clf__max_depth': [9, 10, 11]
      ,'clf__min_samples_leaf': [2, 3, 5]
}]

model1 = imbpipe([  ('scl', StandardScaler()),
                      ('clf',  DecisionTreeClassifier(random_state=101))])


dgs1 = GridSearchCV(estimator=model1, param_grid=dectree_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
dgs1.fit(X_train,y_train)
cv1 = dgs1.predict(x_val)
print('dec tree acc =:  ',accuracy_score(y_val,cv1))
print('dec tree f1 score= ' , f1_score(y_val,cv1,average='micro'))
print('parameters=',dgs1.best_params_)
joblib.dump(dgs1, "1-dectree1.joblib")
acc.append(accuracy_score(y_val,cv1))
f1.append(f1_score(y_val,cv1,average='micro'))

model2 = imbpipe([
        ('sampling', SMOTE()),
        ('scl', StandardScaler()),
        ('clf',  DecisionTreeClassifier(random_state=101))
        ])

dgs2 = GridSearchCV(estimator=model2, param_grid=dectree_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
dgs2.fit(X_train,y_train)
cv2 = dgs2.predict(x_val)

print('dec  tree SMOTE acc =:  ',accuracy_score(y_val,cv2))
print('dec tree f1 score= ' , f1_score(y_val,cv2,average='micro'))
print('parameters=',dgs2.best_params_)
joblib.dump(dgs2, "2-dectree2.joblib")
acc.append(accuracy_score(y_val,cv2))
f1.append(f1_score(y_val,cv2,average='micro'))

##   2 - random forest

rf_param_grid = [{ 'clf__class_weight': ['balanced', None]
      ,'clf__criterion': ['gini', 'entropy']
}]


rfmodel1 = imbpipe([
        ('scl', StandardScaler()),
        ('clf',  RandomForestClassifier(random_state=101))
        ])

RFgs1 = GridSearchCV(estimator=rfmodel1, param_grid=rf_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
RFgs1.fit(X_train,y_train)

cv1 = RFgs1.predict(x_val)
print('RF acc =:  ',accuracy_score(y_val,cv1))
print('RF f1 score= ' , f1_score(y_val,cv1,average='micro'))
joblib.dump(rfmodel1, "3-rf1.joblib")
print('parameters=',RFgs1.best_params_)
acc.append(accuracy_score(y_val,cv1))
f1.append(f1_score(y_val,cv1,average='micro'))

rfmodel2 = imbpipe([
        ('sampling', SMOTE()),
        ('scl', StandardScaler()),
        ('clf',  RandomForestClassifier(random_state=101))
        ])

RFgs2 = GridSearchCV(estimator=rfmodel2, param_grid=rf_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
RFgs2.fit(X_train,y_train)
cv2 = RFgs2.predict(x_val)
print('RF SMOTE acc =:  ',accuracy_score(y_val,cv2))
print('RF CV f1 score= ' , f1_score(y_val,cv2,average='micro'))
print('parameters=',RFgs2.best_params_)
joblib.dump(rfmodel2, "4-rf2.joblib")
acc.append(accuracy_score(y_val,cv2))
f1.append(f1_score(y_val,cv2,average='micro'))

###  3 Adaboost
ada_param_grid = [{ 'clf__learning_rate': [1, 2, 2e-1,  2e-3, 2e-5, 2e-7]}]

model1 = imbpipe([ ('scl', StandardScaler()),
                   ('clf',  AdaBoostClassifier(base_estimator=RandomForestClassifier(class_weight='balanced'),random_state=101))])

adgs1 = GridSearchCV(estimator=model1, param_grid=ada_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
adgs1.fit(X_train,y_train)

cv1 = adgs1.predict(x_val)
print('adaboost acc =:  ',accuracy_score(y_val,cv1))
print('adaboost f1 score= ' , f1_score(y_val,cv1,average='micro'))
print('parameters=',adgs1.best_params_)
joblib.dump(adgs1, "5-ada1.joblib")
acc.append(accuracy_score(y_val,cv1))
f1.append(f1_score(y_val,cv1,average='micro'))


model2 = imbpipe([
        ('sampling', SMOTE()),
        ('scl', StandardScaler()),
        ('clf', AdaBoostClassifier(base_estimator=RandomForestClassifier(class_weight='balanced'),random_state=101) )])

adgs2 = GridSearchCV(estimator=model2, param_grid=ada_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
adgs2.fit(X_train,y_train)
cv2 = adgs2.predict(x_val)
print('adaboost SMOTE acc =:  ',accuracy_score(y_val,cv2))
print('adaboost f1 score= ' , f1_score(y_val,cv2,average='micro'))
print('parameters=',adgs2.best_params_)
joblib.dump(adgs2, "6-ada2.joblib")
acc.append(accuracy_score(y_val,cv2))
f1.append(f1_score(y_val,cv2,average='micro'))

### 4  Gradient Boosting

gb_param_grid = [{ 'clf__learning_rate': [1, 2, 2e-1,  2e-3, 2e-5, 2e-7]}]

model1 = imbpipe([ ('scl', StandardScaler()),
                      ('clf',  GradientBoostingClassifier(random_state=101))])


gbgs1 = GridSearchCV(estimator=model1, param_grid=gb_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gbgs1.fit(X_train,y_train)

cv1 = gbgs1.predict(x_val)
print('gb acc =:  ',accuracy_score(y_val,cv1))
print('gb f1 score= ' , f1_score(y_val,cv1,average='micro'))
print('parameters=',gbgs1.best_params_)
joblib.dump(gbgs1, "7-gb1.joblib")

model2 = imbpipe([
        ('sampling', SMOTE()),
        ('scl', StandardScaler()),
        ('clf',  GradientBoostingClassifier(random_state=101))
        ])

gbgs2 = GridSearchCV(estimator=model2, param_grid=gb_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gbgs2.fit(X_train,y_train)
cv2 = gbgs2.predict(x_val)
print('gb SMOTE acc =:  ',accuracy_score(y_val,cv2))
print('gb f1 score= ' , f1_score(y_val,cv2,average='micro'))
print('parameters=',gbgs2.best_params_)
joblib.dump(gbgs2, "8-gb2.joblib")

acc.append(accuracy_score(y_val,cv1))
f1.append(f1_score(y_val,cv1,average='micro'))

acc.append(accuracy_score(y_val,cv2))
f1.append(f1_score(y_val,cv2,average='micro'))

### 5  K Nearest Neighbors

knn_param_grid = [{ }]
model1 = imbpipe([ ('scl', StandardScaler()),
                      ('clf',  KNeighborsClassifier(weights='distance',n_neighbors=5))])


knngs1 = GridSearchCV(estimator=model1, param_grid=knn_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
knngs1.fit(X_train,y_train)

cv1 = knngs1.predict(x_val)
print('knn cv acc 1 =:  ',accuracy_score(y_val,cv1))
print('knn CV1 f1 score= ' , f1_score(y_val,cv1,average='micro'))
print('parameters=',knngs1.best_params_)
joblib.dump(knngs1, "9-knn1.joblib")

model2 = imbpipe([
        ('sampling', SMOTE()),
        ('scl', StandardScaler()),
        ('clf',  KNeighborsClassifier(weights='distance',n_neighbors=5))
        ])

knngs2 = GridSearchCV(estimator=model2, param_grid=knn_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
knngs2.fit(X_train,y_train)
cv2 = knngs2.predict(x_val)
print('knn SMOTE cv acc2 =:  ',accuracy_score(y_val,cv2))
print('knn CV2 f1 score= ' , f1_score(y_val,cv2,average='micro'))
print('parameters=',knngs2.best_params_)
joblib.dump(knngs2, "10-knn2.joblib")

acc.append(accuracy_score(y_val,cv1))
f1.append(f1_score(y_val,cv1,average='micro'))

acc.append(accuracy_score(y_val,cv2))
f1.append(f1_score(y_val,cv2,average='micro'))

###  6 SVM

svm_param_grid = [{ 'kernel':['rbf']
                  , 'C':[0.001, 0.01,0.1, 1, 10, 100]
                  ,'gamma': [1e-3, 1e-1, 1, 1e1, 1e3]  }]


clf =  svm.SVC(decision_function_shape='ovo')

sgs1 = GridSearchCV(estimator=clf, param_grid=svm_param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
sgs1.fit(X_train,y_train)

cv1 = sgs1.predict(x_val)
print('svm cv acc 1 =:  ',accuracy_score(y_val,cv1))
print('svm CV1 f1 score= ' , f1_score(y_val,cv1,average='micro'))
print('parameters=',sgs1.best_params_)
joblib.dump(sgs1, "11-svm1.joblib")

acc.append(accuracy_score(y_val,cv1))
f1.append(f1_score(y_val,cv1,average='micro'))

print('model by accuracy = ',(np.argmax(np.array(acc))+1))  ##prints 1st occurance of max ; index starts from 0
print('max accuracy = ',np.max(np.array(acc)))
print('model by f1 score = ', (np.argmax(np.array(f1)))+1)
print('max f1 score = ', np.max(np.array(f1)))

