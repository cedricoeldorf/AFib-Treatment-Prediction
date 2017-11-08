#!/usr/bin/env python2

# -*- coding: utf-8 -*-

"""

Created on Fri Mar 17 11:26:13 2017



@author: kurtd

"""



import numpy as np

#np.set_printoptions(suppress=True, precision=4)



from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D

from keras.models import Model



#Building the model. 6 layer 1D convolutional autoencoder.  Filter sizes and numbers are random guesses



input_seq = Input(shape=(900,1, ))  # 3 sec of 300Hz measurements

x = Conv1D(32, 5, activation='relu', padding='same')(input_seq)

x = MaxPooling1D(5, padding='same')(x)

x = Conv1D(32, 5, activation='relu', padding='same')(x)

x = MaxPooling1D(5, padding='same')(x)

x = Conv1D(32, 5, activation='relu', padding='same')(x)

encoded = MaxPooling1D(3, padding='same')(x)



x = Conv1D(32, 5, activation='relu', padding='same')(encoded)

x = UpSampling1D(3)(x)

x = Conv1D(32, 5, activation='relu', padding='same')(x)

x = UpSampling1D(5)(x)

x = Conv1D(32, 5, activation='relu', padding='same')(x)

x = UpSampling1D(5)(x)

decoded = Conv1D(1, 5, activation='tanh', padding='same')(x)



autoencoder = Model(input_seq, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print autoencoder.summary()



# Building the data

ecgpath = '/home/kd758652/CinC/signals_for_Kurt'

onspath = '/home/kd758652/CinC/Q_onsets_for_Kurt'



from os import listdir

from os.path import isfile, join

onlyfiles = [f for f in listdir(ecgpath) if isfile(join(ecgpath, f))]



import scipy.io as sio

#import matplotlib.pyplot as plt

list_X = [];

for filename in onlyfiles:

    mat = sio.loadmat(ecgpath+'/'+filename)

    ecg = mat['ecg'][0]

    ons = sio.loadmat(onspath+'/'+filename+'_Qonsetseries.mat')

    splitpoints = ons['Qonset'][0]

    for i in splitpoints:

        if i+900 < ecg.size:

            example = ecg[i:i+900]

            list_X.append(example)

            #plt.plot(range(0,900),example)

            #plt.show()

train_X = np.array(list_X)

train_X.shape = (len(list_X),900,1)

#print train_X



encoder = Model(input_seq, encoded)



autoencoder.fit(train_X,train_X,epochs=100,validation_split=0.33)

encoded_X = encoder.predict(train_X)

sio.savemat('encoded_ecgs_tf.mat', {'encoded_windows':encoded_X})

decoded_X = autoencoder.predict(train_X)

sio.savemat('decoded_ecg_tfs.mat', {'decoded_windows':decoded_X})

autoencoder.save('autoencoder_tf.keras')
