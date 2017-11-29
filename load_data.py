import numpy as np
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def load_data(with_demographics = True, from_source = True, stacked = True):
    if (from_source == True) & (with_demographics == True):
        with open ('../data/training/X.pkl', 'rb') as fp:
            X = pickle.load(fp)
    if (from_source == False) & (with_demographics == True):
        X = prep_data()
        with open('../data/training/X.pkl', 'wb') as fp:
            pickle.dump(X, fp)
    if (from_source == True) & (with_demographics == False):
        with open ('../data/training/x.pkl', 'rb') as fp:
            X = pickle.load(fp)
            #X = np.delete(X, [76,151,245])
    X = np.array(X)
    if stacked == True:
        x = []
        for i in range(0,len(X)):
            x.append(np.ravel(X[i]))
    #X = np.array(x)

    y = pd.read_excel('../data/training/y.xlsx')
    y = y.drop(y.index[[76,151,245]])
    y.index = range(0,len(y))
    return X, y


def prep_data():
    with open ('../data/training/x.pkl', 'rb') as fp:
        X = pickle.load(fp)
    demog = pd.read_excel('../data/demographics.xls')

    ## PATIENTS THAT HAD TO BE DROPPED
    ## manually removed from ecg to retain 3 dimensional shape
    #X = np.delete(X, [76,151,245])
    # drop 'RecurrenceWithin1yr', 'Recurrence_early'
    demog = demog.drop(['RecurrenceWithin1yr', 'Recurrence_early'], axis = 1)

    demog = demog.drop(demog.index[[76,151,245]]).reset_index()

    ## missing cells, interpolate
    demog = demog.interpolate()

    ## Data needs same variance
    ## dont apply to demographics, but on ECG.
    ## PCA affected by feature with largest variance.
    ## Filter based methods, wrapper based methods
    ## must work with hybrid data
    ## feature significance
    ## lasso, sklearn. sequential forward floating search

    demog_matrix = []
    for i in range(0,len(demog)):
        print("Patient " + str(i) + ' out of ' + str(len(demog)))
        patient = pd.DataFrame()
        patient = patient.append([demog.iloc[i]]*2500,ignore_index=True)
        patient =  np.asarray(patient)
        #demog_matrix.append(patient)
        X[i] = np.hstack((patient,X[i]))

    return X

def smote_os(X, y):
    sm = SMOTE()
    X_res, y_res = sm.fit_sample(X, y)
    return X_res, y_res
