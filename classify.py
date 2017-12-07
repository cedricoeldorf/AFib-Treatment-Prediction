from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model, Sequential
from load_data import load_data
from keras.optimizers import RMSprop, Adam
#from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from keras import backend as K
from keras import metrics
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
with_rfe = input('Has RFE already been performed? (y/n) ')

if with_rfe == 'n':
    X = pd.read_csv('../AE_X.csv')
    y = pd.read_csv('../AE_y.csv')
    X = X.values
    y = y.values
    estimator = XGBClassifier()
    selector = RFECV(estimator, step=10, cv=5,scoring='accuracy', verbose = 2)

    selector.fit(X,y)
    print("Optimal number of features : %d" % selector.n_features_)
    X_new = selector.transform(X)
    X_new = pd.DataFrame(X_new)
    X_new.to_csv('../RFE_X.csv', index = False)
    X = X_new.copy()
else:
    X = pd.read_csv('../RFE_X.csv')
    y = pd.read_csv('../AE_y.csv')
    X = X.values
    y = y.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))




###########################################
"""
x_train = X[0:600]
x_test = X[600:]
y_train = y.iloc[0:600]
y_test = y.iloc[600:]
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
y_train = np.asarray(y_train).ravel()
y_test = np.asarray(y_test).ravel()

lasso = Lasso(alpha=0.1)

y_pred_lasso = lasso.fit(x_train, y_train).predict(x_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

#elastic net

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(x_train, y_train).predict(x_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()
"""
