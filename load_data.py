import numpy as np
import pickle
import pandas as pd
from sklearn.decomposition import PCA

def load_data(with_demographics = True, from_source = True, pca_demog = False):
    if (from_source == True) & (with_demographics == True):
        with open ('../data/training/X.pkl', 'rb') as fp:
            X = pickle.load(fp)
    if (from_source == False) & (with_demographics == True) & (pca_demog == False):
        X = prep_data()
        with open('../data/training/X.pkl', 'wb') as fp:
            pickle.dump(X, fp)
    if (from_source == True) & (with_demographics == False):
        with open ('../data/training/x.pkl', 'rb') as fp:
            X = pickle.load(fp)
            #X = np.delete(X, [76,151,245])
    if (from_source == False) & (with_demographics == True) & (pca_demog == True):
        #X = prep_data(pca_demog = True)
        #with open('../data/training/X_d.pkl', 'wb') as fp:
            #pickle.dump(X, fp)
        with open ('../data/training/X_d.pkl', 'rb') as fp:
            X = pickle.load(fp)
    X = np.array(X)
    y = pd.read_excel('../data/training/y.xlsx')
    y = y.drop(y.index[[76,151,245]]).reset_index()
    return X, y


def prep_data(pca_demog = False):
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

    if pca_demog == True:
        demog_tr = demog.loc[:290]
        demog_te = demog.loc[291:]
        pca = PCA(n_components=5)
        demog_tr = pca.fit_transform(demog_tr)
        demog_te = pca.transform(demog_te)
        demog_tr = pd.DataFrame(demog_tr)
        demog_te = pd.DataFrame(demog_te)
        demog = demog_tr.append(demog_te)

    demog_matrix = []
    for i in range(0,len(demog)):
        print("Patient " + str(i) + ' out of ' + str(len(demog)))
        patient = pd.DataFrame()
        patient = patient.append([demog.iloc[i]]*2500,ignore_index=True)
        patient =  np.asarray(patient)
        #demog_matrix.append(patient)
        X[i] = np.hstack((patient,X[i]))

    return X
