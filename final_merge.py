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

hps = input("Do you want to do a Hyperparameter search? (y/n)")
set_train = input("which set would you like to train on? (ext, demog, ae, all)?")

if set_train == 'ext':
    traditional = pd.read_excel('../data/Merged features.xlsx')

    #traditional = traditional.drop(['index'], axis = 1)
    #traditional = traditional.drop(traditional.index[[76,151,245]]).reset_index()
    traditional = pd.concat([traditional]*3)
    traditional = traditional.sort_index()
    traditional.index = range(0,len(traditional))
    traditional = traditional.drop(['Dominant frequency','k(0.95)'], axis = 1)
    column = traditional.columns
    #columns = X.columns

    X = traditional.values
    y = pd.read_csv('../AE_y.csv')
    y = y.values
    params = {
        # Parameters that we are going to tune.
        'max_depth': 4,
        'min_child_weight': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'n_estimators': 10,
        # Other parameters
        'objective':'binary:logistic',
    }
if set_train == 'demog':
    demog = pd.read_excel('../data/demographics.xls')
    demog = demog.drop(['RecurrenceWithin1yr', 'Recurrence_early','Total_AF_dur','AF_episode','Weight', 'Med_IsoptinTild', 'Med_sotalol', 'Med_Flecainide',
'ULHD_PM','PM_ICD', 'Stroke', 'OSAS','CHADSVASc_5','Med_Digoxine','Med_Statine','ULHD_ICD','PVI_surg_abl','ULHD_CAD','COPD','ULHD_DM','ACE_ATII','Med_ACE'], axis = 1)
    column = demog.columns
    demog = demog.drop(demog.index[[76,151,245]]).reset_index()
    demog = demog.drop(['index'], axis = 1)
    ## missing cells, interpolate
    demog = demog.interpolate()
    demog = pd.concat([demog]*3)
    demog = demog.sort_index()
    demog.index = range(0,len(demog))
    X = demog.values
    y = pd.read_csv('../AE_y.csv')
    y = y.values
    params = {
        # Parameters that we are going to tune.
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'n_estimators': 10,
        # Other parameters
        'objective':'binary:logistic',
    }
#X = pd.read_csv('../RFE_X_nodemog.csv')
if set_train == 'ae':

    X = pd.read_csv('../RFE_X_nodemog.csv')

    X = X.iloc[:,[0, 3, 6, 7, 19, 27, 43, 45, 50, 52, 53, 57, 61, 102, 103, 104, 113, 152, 157, 176, 177, 181, 195, 208, 213, 220, 221, 234, 251, 252, 259, 272, 299, 305, 309, 316, 326, 327, 328, 332, 340, 345, 348, 364, 376, 377, 405, 422, 436, 449, 490, 498, 544, 574, 581, 592, 608, 632, 635, 642, 644, 649, 685, 712, 730, 737, 742, 745]
]


    #X = pd.concat([X,traditional], axis = 1)
    y = pd.read_csv('../AE_y.csv')
    #columns = X.columns
    X = X.values
    y = y.values
    params = {
        # Parameters that we are going to tune.
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'n_estimators': 5,
        # Other parameters
        'objective':'binary:logistic',
    }

if set_train == 'all':
    X = pd.read_csv('../RFE_X_nodemog.csv')
    X = X.iloc[:,[0, 3, 6, 7, 19, 27, 43, 45, 50, 52, 53, 57, 61, 102, 103, 104, 113, 152, 157, 176, 177, 181, 195, 208, 213, 220, 221, 234, 251, 252, 259, 272, 299, 305, 309, 316, 326, 327, 328, 332, 340, 345, 348, 364, 376, 377, 405, 422, 436, 449, 490, 498, 544, 574, 581, 592, 608, 632, 635, 642, 644, 649, 685, 712, 730, 737, 742, 745]
]
    demog = pd.read_excel('../data/demographics.xls')
    demog = demog.drop(['RecurrenceWithin1yr', 'Recurrence_early','Total_AF_dur','AF_episode','Weight', 'Med_IsoptinTild', 'Med_sotalol', 'Med_Flecainide',
    'ULHD_PM','PM_ICD', 'Stroke', 'OSAS','CHADSVASc_5','Med_Digoxine','Med_Statine','ULHD_ICD','PVI_surg_abl','ULHD_CAD','COPD','ULHD_DM','ACE_ATII','Med_ACE'], axis = 1)
    column = demog.columns
    demog = demog.drop(demog.index[[76,151,245]]).reset_index()
    demog = demog.drop(['index'], axis = 1)
    ## missing cells, interpolate
    demog = demog.interpolate()
    demog = pd.concat([demog]*3)
    demog = demog.sort_index()
    demog.index = range(0,len(demog))
    traditional = pd.read_excel('../data/Merged features.xlsx')

    #traditional = traditional.drop(['index'], axis = 1)
    #traditional = traditional.drop(traditional.index[[76,151,245]]).reset_index()
    traditional = pd.concat([traditional]*3)
    traditional = traditional.sort_index()
    traditional.index = range(0,len(traditional))
    traditional = traditional.drop(['Dominant frequency','k(0.95)'], axis = 1)

    X = pd.concat([X,traditional, demog], axis = 1)
    column = X.columns
    y = pd.read_csv('../AE_y.csv')
    #columns = X.columns
    X = X.values
    y = y.values
    params = {
        # Parameters that we are going to tune.
        'max_depth': 3,
        'min_child_weight': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'n_estimators': 5,
        'learning_rate':0.1,
        # Other parameters
        'objective':'binary:logistic',
    }
    estimator = XGBClassifier(**params)
    selector = RFE(estimator, step=1,n_features_to_select = 35, verbose = 2)

    selector.fit(X,y)
    print("Optimal number of features : %d" % selector.n_features_)
    X = selector.transform(X)
####################################
## XGB parameters (if doing hyperparameter search, put for example max_depth: [7,8,9] for searchable parameters)

if hps == 'y':
    params = {
        # Parameters that we are going to tune.
        'max_depth': [3,4,5,6],
        'min_child_weight': [1],
        'subsample':[0.7,0.8,0.9,1],
        'colsample_bytree': [0.7,0.8,0.9,1],
        'n_estimators': [5,7,10],
        'learning_rate':[0.08,0.1],
        'max_delta_step':[0,1],

        # Other parameters
        'objective':['binary:logistic'],
    }
    ########################################################
    ## Hyperparameter search
    ###########################################################3
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
else:
    #########################################################
    #i = 329

    accuracies = []
    k = int(input('How many observations in test? (multiples of 3)'))
    for b in range(0,987-k,3):
        print("###########################################")
        print("###########################################")
        print("Model " + str(b) + " out of 987: " + str((b/987-k)*100) + "%")

        X_test = X[b:b+k]
        X_train = np.delete(X, [i for i in range(b,b+k)],axis=0)

        y_test = y[b:b+k]
        y_train = np.delete(y, [i for i in range(b,b+k)],axis=0)
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
