###################################################
## Run test data through preprocessing
###################################################

## TO DO
## 1. Create windows??????? oMG WHAT
## 2. Save original autoencoder model for predicitng on this set. (include normalization)
## 3. Add option for demographics vs none
## 4. Save RFE models and apply appropriate one (dependig on dataset chosen above)

from keras.models import load_model



AE = load_model('../autoencoder.h5')
