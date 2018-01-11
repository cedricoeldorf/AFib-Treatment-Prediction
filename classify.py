from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from keras import backend as K
from keras import metrics
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
with_rfe = input('Has RFE already been performed? (y/n) ')
demogr = input("include demographics? (y/n) ")
if with_rfe == 'n':
    if demogr == 'y':
        X = pd.read_csv('../AE_X.csv')
        y = pd.read_csv('../AE_y.csv')
    else:
        X = pd.read_csv('../AE_X_no_demogr.csv')
        y = pd.read_csv('../AE_y_no_demogr.csv')
    X = X.values
    y = y.values
    estimator = XGBClassifier()
    selector = RFE(estimator, step=2, verbose = 2)

    selector.fit(X,y)
    print("Optimal number of features : %d" % selector.n_features_)
    X_new = selector.transform(X)
    X_new = pd.DataFrame(X_new)
    X_new.to_csv('../RFE_X.csv', index = False)
    X = X_new.copy()
    X = X.values
else:
    X = pd.read_csv('../RFE_X.csv')
    y = pd.read_csv('../AE_y.csv')
    X = X.values
    y = y.values

#################################
## Cross validation leave k out approach for exhaustive testing
#################################

#i = 329
accuracies = []
for b in range(0,927,3):
    print("###########################################")
    print("###########################################")
    print("Model " + str(b) + " out of 927: " + str((b/927)*100) + "%")

    X_test = X[b:b+60]
    X_train = np.delete(X, [i for i in range(b,b+60)],axis=0)

    y_test = y[b:b+60]
    y_train = np.delete(y, [i for i in range(b,b+60)],axis=0)
    y_test = y_test.ravel()
    y_train = y_train.ravel()

    model = XGBClassifier()
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
