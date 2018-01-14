###################################
## merge ae and traditional features
## Hyperparam search xgb

import pandas as pd
from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

traditional = pd.read_excel('../data/Merged features.xlsx')
#traditional = traditional.drop(['index'], axis = 1)
#traditional = traditional.drop(traditional.index[[76,151,245]]).reset_index()
traditional = pd.concat([traditional]*3)
traditional = traditional.sort_index()
traditional.index = range(0,len(traditional))

X = pd.read_csv('../RFE_X_nodemog.csv')


y = pd.read_csv('../AE_y.csv')
#columns = X.columns
X = X.values
y = y.values

"""
estimator = XGBClassifier()
selector = RFE(estimator, step=1, verbose = 2)

selector.fit(X,y)
print("Optimal number of features : %d" % selector.n_features_)
X_new = selector.transform(X)
X_new = pd.DataFrame(X_new)
X = X_new.copy()
X = X.values
"""


####################################
## XGB parameters
params = {
    # Parameters that we are going to tune.
    'max_depth': 9,
    'min_child_weight': 1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'n_estimators': 500,
    # Other parameters
    'objective':'binary:logistic',
}
"""
model = XGBClassifier()
x_train = X[0:600]
x_test = X[600:]
y_train = y[0:600]
y_test = y[600:]
best_score = 0
for g in ParameterGrid(params):
    model.set_params(**g)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    # save if best
    if accuracy > best_score:
        best_score = accuracy
        best_grid = g

print("OOB: %0.5f" % best_score)
print("Grid:", best_grid)
"""
#i = 329

accuracies = []
for b in range(0,957,3):
    print("###########################################")
    print("###########################################")
    print("Model " + str(b) + " out of 927: " + str((b/957)*100) + "%")

    X_test = X[b:b+30]
    X_train = np.delete(X, [i for i in range(b,b+30)],axis=0)

    y_test = y[b:b+30]
    y_train = np.delete(y, [i for i in range(b,b+30)],axis=0)
    y_test = y_test.ravel()
    y_train = y_train.ravel()

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("###########################################")
    print("###########################################")
    accuracies.append(accuracy)


from sklearn.metrics import roc_curve, auc
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.ion()
plt.show()
