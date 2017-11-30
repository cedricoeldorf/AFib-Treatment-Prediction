################
## see https://blog.keras.io/building-autoencoders-in-keras.html
############
AE = 'vanilla'

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
X, y = load_data(with_demographics = False,from_source = True, stacked = False)

y = pd.concat([y]*3)
y.index = range(0,len(y))
#################################
## NEW 3 SECOND DATA WINDOWS
#################################


large_list = []
for x in X:
    for i in range(0,2250,750):
        large_list.append(x[i:i+750])
        #large_list.append(small_list)
X = np.asarray(large_list)

new_list = []
large_list = []
for j in range(0,len(X)):
    new_list = []
    for i in range(0,len(X[j])):
        new_list.append(np.delete(X[j][i], [1,3,4,5,6,7,8,9,10,11]))
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


if AE == 'vanilla':
    encoding_dim = 256
    ae = Sequential()
    inputLayer = Dense(1500, input_shape=(1500,))
    ae.add(inputLayer)
    middle = Dense(encoding_dim, activation='relu')
    ae.add(middle)
    middle3 = Dense(64, activation='relu')
    ae.add(middle3)
    middle2 = Dense(encoding_dim, activation='relu')
    ae.add(middle2)
    output = Dense(1500, activation='tanh')
    ae.add(output)

    opt = RMSprop(lr = 0.000001)
    ae.compile(optimizer=opt, loss='mse')
    # Normalize
    x_train = x_train.astype('float32') / np.linalg.norm(x_train)
    x_test = x_test.astype('float32') / np.linalg.norm(x_test)

    # flatten matrix
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


if AE == 'conv':
    x_train = x_train.reshape(x_train.shape[0],750, 2, 1)
    x_test = x_test.reshape(x_test.shape[0],750, 2, 1)
    inputs = Input(shape=(750, 2, 1))
    h = Conv2D(4, 3, 3, activation='relu', border_mode='same')(inputs)
    encoded = MaxPooling2D((2, 2))(h)
    h = Conv2D(4, 3, 3, activation='relu', border_mode='same')(encoded)
    h = UpSampling2D((2, 2))(h)
    outputs = Conv2D(1, 3, 3, activation='relu', border_mode='same')(h)

    ae = Model(input=inputs, output=outputs)
    opt = RMSprop(lr = 0.00001)
    ae.compile(optimizer=opt, loss='mse')
    x_train = x_train.astype('float32') / np.linalg.norm(x_train)
    x_test = x_test.astype('float32') / np.linalg.norm(x_test)


start = time.time()
print("> Training the model...")
ae.fit(x_train, x_train,
       nb_epoch=50,
       batch_size=2,
       verbose=1,
       shuffle=False,  # whether to shuffle the training data before each epoch
       validation_data=(x_test, x_test))

print("> Training is done in %.2f seconds." % (time.time() - start))

# how well does it work?
print("> Scoring:")
scoring = ae.evaluate(x_test, x_test, verbose=0)
print(scoring)

x_test_encoded = ae.predict(x_test, batch_size=1)

if AE == 'conv':
    x_test_encoded = x_test_encoded.reshape(x_test_encoded.shape[0], 750,2)

plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.ion()
plt.show()

##########################################
## transform original data and combine
############################
X = X.astype('float32') / np.linalg.norm(X)
# flatten matrix
X = X.reshape((len(X), np.prod(X.shape[1:])))
X = ae.predict(X, batch_size=1)
demog = pd.read_excel('../data/demographics.xls')
demog = demog.drop(['RecurrenceWithin1yr', 'Recurrence_early'], axis = 1)
demog = demog.drop(demog.index[[76,151,245]]).reset_index()
## missing cells, interpolate
demog = demog.interpolate()
demog = pd.concat([demog]*3)
demog.index = range(0,len(demog))
df = pd.DataFrame(X)
new_training = pd.concat([df, demog], axis=1)
new_training.to_csv('../trainthis.csv', index = False)
