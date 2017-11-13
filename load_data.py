import numpy as np
import pickle
import pandas as pd

def load_data(with_demographics = True, from_source = True):
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
    y = pd.read_excel('../data/training/y.xlsx')
    y = y.drop(y.index[[76,151,245]]).reset_index()
    return X, y


def prep_data():
    with open ('../data/training/x.pkl', 'rb') as fp:
        X = pickle.load(fp)
    demog = pd.read_excel('../data/demographics.xls')

    ## PATIENTS THAT HAD TO BE DROPPED
    ## manually removed from ecg to retain 3 dimensional shape
    #X = np.delete(X, [76,151,245])
    demog = demog.drop(demog.index[[76,151,245]]).reset_index()

    ## missing cells, interpolate
    demog = demog.interpolate()

    demog_matrix = []
    for i in range(0,len(demog)):
        print("Patient " + str(i) + ' out of ' + str(len(demog)))
        patient = pd.DataFrame()
        patient = patient.append([demog.loc[i]]*2500,ignore_index=True)
        patient =  np.asarray(patient)
        #demog_matrix.append(patient)
        X[i] = np.hstack((patient,X[i]))

    return X
