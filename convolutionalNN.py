import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.decomposition import TruncatedSVD
from load_data import load_data
import numpy as np
X, y = load_data(with_demographics = True,from_source = False, stacked = True)

batch_size = 1
num_classes = 2
epochs = 100

img_rows, img_cols = 2500, 17

## Create training and testing set
#########################
## TO DO:
## Cross validation
## Dimensionality reduction to avoid exhaustion of machine resources
#########################

# test drop leads some
##### TRY DROP LEADS IN THIRD DIMENSION
"""
X2 = X.copy()
for i in range(0,len(X)):
    for j in range(0,len(X[i])):
        X2[i][j] = np.delete(X[i][j], [1,3,4,5,6,7,10,11])
"""

x_train = X[0:290]

x_test = X[290:]
y_train = y.RecurrenceWithin1yrTr[0:290]
y_test = y.RecurrenceWithin1yrTr[290:]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

## One hot encode target
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
###########
## dimensionality reduction
##########
x_train = [np.matrix(xi) for xi in x_train]
x_test = [np.matrix(xi) for xi in x_test]
pca = TruncatedSVD(n_components=10).fit(x_train)
x_train = pca.transform(X_train)
x_test = pca.transform(X_test)
"""
############
## Construct model
#############
## check if first part can be trained using unsupervised set.
###############################
# 1. first learn representation
## get physionet data
# 2. then supervised learning.
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(2))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dropout(2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr = 0.000001),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
