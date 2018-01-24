###################################################
## Run test data through preprocessing
###################################################

## TO DO
## 1. Create windows??????? oMG WHAT
## 2. Save original autoencoder model for predicitng on this set. (include normalization)
## 3. Add option for demographics vs none
## 4. Save RFE models and apply appropriate one (dependig on dataset chosen above)

from keras.models import load_model
import pickle
import pandas as pd
###################################################
## Step 1: Load the data in
##################################################
import scipy.io as sio
import pickle
import numpy as np
import glob, os
from xgboost import XGBClassifier

data_merged = input('have files been merge? (y/n) ')
if data_merged == 'n':
    os.chdir("../data/test")

    ######### Details #################
    ## 1. in the parent directory of where you cloned the repository
    ## 2. create a folder named 'data'.
    ## 3. inside data create a folder named 'training'
    ## 4. into training, paste the patient matrices ONLY
    ## 5. RUN merge_data (creates x.pkl)
    ## 6. paste target variable into training as y.xlsx

    names = []
    for file in glob.glob("*.mat"):
        print(file[15:-4])
        names.append(file[15:-4])
        names = [int(x) for x in names]
        names.sort()
        print(names)
        names = [str(x) for x in names]

    x = []
    for num in names:
        print(num)
        mat_contents = sio.loadmat('./patient_testing' + num + '.mat')
        mat_contents = mat_contents['ecg']
        x.append(mat_contents)
        with open('x.pkl', 'wb') as fp:
            pickle.dump(x, fp)
else:
    #x = np.array([np.array(xi) for xi in x])
    #x = [xi for xi in x]
    #x = np.ndarray(x)
    ## write list of matrices to pickle
    with open ('../data/test/x.pkl', 'rb') as fp:
        X = pickle.load(fp)
    X = np.array(X)


################
## set up y
##############
y = pd.read_excel('../data/testy/y.xlsx')
#y = y.drop(y.index[[76,151,245]])
y.index = range(0,len(y))
y = pd.concat([y]*3)
y = y.sort_index()
y.index = range(0,len(y))


################
## set up AE dataset
###############
large_list = []
for x in X:
    for i in range(0,2250,750):
        large_list.append(x[i:i+750])
X = np.asarray(large_list)

new_list = []
large_list = []
for j in range(0,len(X)):
    new_list = []
    for i in range(0,len(X[j])):
        new_list.append(np.delete(X[j][i], [0,2,3,4,5,7,8,9,10,11]))
    large_list.append(new_list)

X = np.asarray(large_list)

###########################################
## Run AE
###########################################

ae = load_model('../autoencoder.h5')
X = X.astype('float32') / np.linalg.norm(X)
X = X.reshape((len(X), np.prod(X.shape[1:])))
X = ae.predict(X, batch_size=1)
X = pd.DataFrame(X)
#############################
## demographics
##########################
demog = pd.read_csv('../data/testy/demogr.csv')
demog = demog.drop(['RecurrenceWithin1yr', 'Recurrence_early','Total_AF_dur','AF_episode'], axis = 1)
#demog = demog.drop(['RecurrenceWithin1yr', 'Recurrence_early','Total_AF_dur','AF_episode','Weight', 'Med_IsoptinTild', 'Med_sotalol', 'Med_Flecainide',
#'ULHD_PM','PM_ICD', 'Stroke', 'OSAS','CHADSVASc_5','Med_Digoxine','Med_Statine','ULHD_ICD','PVI_surg_abl','ULHD_CAD','COPD','ULHD_DM','ACE_ATII','Med_ACE'], axis = 1)

#demog = demog.drop(['index'], axis = 1)
## missing cells, interpolate
demog = demog.interpolate()
demog = pd.concat([demog]*3)
demog = demog.sort_index()
demog.index = range(0,len(demog))


###################################
## Extracted features
###################################
traditional = pd.read_csv('../data/testy/exttest.csv')

#traditional = traditional.drop(['index'], axis = 1)
#traditional = traditional.drop(traditional.index[[76,151,245]]).reset_index()
X = pd.concat([X,traditional, demog], axis = 1)


###################################
## RFE
#####################################
#RFE = load_model('../RFE_50.pkl')
RFE = pickle.load(open('../RFE_50.pkl', 'rb'))
params = {
    # Parameters that we are going to tune.
    'max_depth': 4,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 0.9,
    'n_estimators': 500,
    'learning_rate':0.1,
    # Other parameters
    'objective':'binary:logistic',
}


#estimator = XGBClassifier(**params)
X = RFE.transform(X)
X = pd.DataFrame(X)
X.to_csv('../data/testingx.csv', index = False)
