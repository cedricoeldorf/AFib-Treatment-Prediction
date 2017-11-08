import scipy.io as sio
import pickle

import glob, os
os.chdir("../data/training")

######### Details #################
## matrix key to use: 'ecg'
## 12 parts
names = []
for file in glob.glob("*.mat"):
    print(file[16:-4])
    names.append(file[16:-4])
    names = [int(x) for x in names]
    names.sort()
    print(names)
    names = [str(x) for x in names]

x = []
for num in names:
    print(num)
    mat_contents = sio.loadmat('./patient_training' + num + '.mat')
    mat_contents = mat_contents['ecg']
    x.append(mat_contents)

## write list of matrices to pickle
with open('x.pkl', 'wb') as fp:
    pickle.dump(x, fp)
#change back to original directory
os.chdir('../../AFib-Treatment-Prediction')

#####################
## read data in by doing:
##with open ('x.pkl', 'rb') as fp:
##    itemlist = pickle.load(fp)
######################

#############################
## Y
## keys: 'LeadsLabels'
#############################
mat_contents = sio.loadmat('../data/LeadsLabels.mat')
