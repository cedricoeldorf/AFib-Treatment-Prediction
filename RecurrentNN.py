#####################################
## RNN script
#####################################
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras import backend as K
from sklearn.decomposition import TruncatedSVD
from load_data import load_data, smote_os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
X, y = load_data(with_demographics = False,from_source = True, stacked = True)

batch_size = 1
num_classes = 2
epochs = 100

img_rows, img_cols = 2500, 17

##############################
## Balance data
##############################
X, y = shuffle(X, y)
X, y = smote_os(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## One hot encode target
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(LSTM(
    input_dim=1,
    output_dim=30000,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    8,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=2))
model.add(Activation("sigmoid"))

start = time.time()
#opt = RMSprop(lr = 0.00001)
opt = adam(lr = 0.000001)
model.compile(loss="binary_crossentropy", optimizer=opt)
hist = model.fit(
	    X_train,
	    y_train,
	    batch_size=1,nb_epoch=2,validation_split=0.05)
