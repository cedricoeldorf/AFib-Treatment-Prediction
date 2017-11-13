import scipy.io as sio
import pickle
import numpy as np
import glob, os
os.chdir("../data/training")

######### Details #################
## 1. in the parent directory of where you cloned the repository
## 2. create a folder named 'data'.
## 3. inside data create a folder named 'training'
## 4. into training, paste the patient matrices ONLY
## 5. RUN merge_data (creates x.pkl)
## 6. paste target variable into training as y.xlsx

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

#x = np.array([np.array(xi) for xi in x])
#x = [xi for xi in x]
#x = np.ndarray(x)
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
