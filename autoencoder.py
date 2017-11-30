################
## see https://blog.keras.io/building-autoencoders-in-keras.html
############


from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model, Sequential
from load_data import load_data
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
x_train = X[0:600]
x_test = X[600:]
y_train = y.iloc[0:600]
y_test = y.iloc[600:]
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
y_train = np.asarray(y_train).ravel()
y_test = np.asarray(y_test).ravel()
encoding_dim = 64  # from 784 to 32; serious compression
ae = Sequential()

inputLayer = Dense(1500, input_shape=(1500,))
ae.add(inputLayer)

middle = Dense(encoding_dim, activation='relu')
ae.add(middle)

output = Dense(1500, activation='sigmoid')
ae.add(output)

ae.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# x_train is a 28x28 matrix with [0,255] color values
# this remaps things to the [0,1] range
x_train = x_train.astype('float32') / np.linalg.norm(x_train)
x_test = x_test.astype('float32') / np.linalg.norm(x_train)

# flattening the 28x28 matrix to a vector
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

start = time.time()
print("> Training the model...")
ae.fit(x_train, x_train,
       nb_epoch=10,
       batch_size=1,
       verbose=1,
       shuffle=False,  # whether to shuffle the training data before each epoch
       validation_data=(x_test, x_test))

print("> Training is done in %.2f seconds." % (time.time() - start))

# how well does it work?
print("> Scoring:")
scoring = ae.evaluate(x_test, x_test, verbose=0)
for i in range(len(ae.metrics_names)):
    print("   %s: %.2f%%" % (ae.metrics_names[i], 100 - scoring[i] * 100))

x_test_encoded = ae.predict(x_test, batch_size=1)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.ion()
plt.show()
"""
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(750,12))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='relu')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
"""
