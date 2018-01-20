################
## see https://blog.keras.io/building-autoencoders-in-keras.html
############


from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model, Sequential
from load_data import load_data
from keras.optimizers import RMSprop, Adam
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from keras import backend as K
from keras import metrics
import pandas as pd
from keras.models import load_model
######
# Call load data script
X, y = load_data(with_demographics = False,from_source = True, stacked = False)

# duplicate the y observatoins in place so that they match X
y = pd.concat([y]*3)
y = y.sort_index()
y.index = range(0,len(y))

#################################
## Create NEW 3 SECOND DATA WINDOWS
#################################

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

x_train = X[0:600]
x_test = X[600:]
y_train = y.iloc[0:600]
y_test = y.iloc[600:]
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
y_train = np.asarray(y_train).ravel()
y_test = np.asarray(y_test).ravel()

ae_exist = input("Does AE exist?")
if ae_exist == 'y':
    x_train = x_train.astype('float32') / np.linalg.norm(x_train)
    x_test = x_test.astype('float32') / np.linalg.norm(x_test)

    # flatten matrix
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    ae = load_model('../autoencoder.h5')
else:
    encoding_dim = 256
    ae = Sequential()
    inputLayer = Dense(1500, input_shape=(1500,),kernel_initializer='random_normal')
    ae.add(inputLayer)
    middle = Dense(encoding_dim, activation='relu',kernel_initializer='random_normal')
    ae.add(middle)
    #middle3 = Dense(256, activation='relu')
    #ae.add(middle3)
    middle2 = Dense(encoding_dim, activation='relu',kernel_initializer='random_normal')
    ae.add(middle2)
    output = Dense(1500, activation='tanh',kernel_initializer='random_normal')
    ae.add(output)

    opt = Adam(lr = 0.0000001)
    ae.compile(optimizer=opt, loss='mse')
    # Normalize
    x_train = x_train.astype('float32') / np.linalg.norm(x_train)
    x_test = x_test.astype('float32') / np.linalg.norm(x_test)

    # flatten matrix
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    start = time.time()
    print("> Training the model...")
    history = ae.fit(x_train, x_train,
           nb_epoch=2,
           batch_size=2,
           verbose=1,
           shuffle=False,  # whether to shuffle the training data before each epoch
           validation_data=(x_test, x_test))

    print("> Training is done in %.2f seconds." % (time.time() - start))
    ae.save('../autoencoder.h5')
    # how well does it work?
    print("> Scoring:")
    scoring = ae.evaluate(x_test, x_test, verbose=0)
    print(scoring)

x_test_encoded = ae.predict(x_test, batch_size=1)
"""
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.ion()
plt.show()
"""

##########################################
## transform original data and combine
############################
demogr = input("Include Demographics? (y/n) ")
if demogr == 'y':
    X = X.astype('float32') / np.linalg.norm(X)
    # flatten matrix
    X = X.reshape((len(X), np.prod(X.shape[1:])))
    X = ae.predict(X, batch_size=1)
    demog = pd.read_excel('../data/demographics.xls')
    demog = demog.drop(['RecurrenceWithin1yr', 'Recurrence_early','Total_AF_dur','AF_episode'], axis = 1)
    demog = demog.drop(demog.index[[76,151,245]]).reset_index()
    demog = demog.drop(['index'], axis = 1)
    ## missing cells, interpolate
    demog = demog.interpolate()
    demog = pd.concat([demog]*3)
    demog = demog.sort_index()
    demog.index = range(0,len(demog))
    df = pd.DataFrame(X)
    new_training = pd.concat([df, demog], axis=1)
    new_training.to_csv('../AE_X.csv', index = False)
    y.to_csv('../AE_y.csv', index = False)
else:
    X = X.astype('float32') / np.linalg.norm(X)
    # flatten matrix
    X = X.reshape((len(X), np.prod(X.shape[1:])))
    X = ae.predict(X, batch_size=1)
    df = pd.DataFrame(X)
    df.to_csv('../AE_X_no_demogr.csv', index = False)
    y.to_csv('../AE_y_no_demogr.csv', index = False)
##############################################################
## lasso
## stagewise approach
## xgboost feature importance. lasso has built in sparsity.
###############################################################
"""
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.ion()
plt.show()
"""
#####################
## classify, check our xgboost and scikit learn to see performance
#####################
