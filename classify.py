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

from sklearn.linear_model import ElasticNet

#reg = linear_model.Lasso(alpha = 0.1)

X, y = load_data(with_demographics = False,from_source = True, stacked = True)

#y = pd.concat([y]*3)
#y.index = range(0,len(y))

large_list = []
for x in X:
    for i in range(0,2250,750):
        large_list.append(x[i:i+750])
        #large_list.append(small_list)
X = np.asarray(large_list)

new_list = []
large_list = []
for j in range(0,len(X)):
    new_list = []
    for i in range(0,len(X[j])):
        new_list.append(np.delete(X[j][i], [1,3,4,5,6,7,8,9,10,11]))
    large_list.append(new_list)

X = np.asarray(large_list)
###########################################
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
