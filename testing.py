##################
## Final train and test
####################
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model

X_train = pd.read_csv('../rfe_100_window.csv')
y_train = pd.read_csv('../AE_y.csv')
params = {
    # Parameters that we are going to tune.
    'max_depth': 7,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'n_estimators': 1000,
    'learning_rate':0.05,
    # Other parameters
    'objective':'binary:logistic',
}

X_test = pd.read_csv('../data/testingx.csv')
y_test = pd.read_excel('../data/testy/y.xlsx')
y_test.index = range(0,len(y_test))
y_test = pd.concat([y_test]*3)
y_test = y_test.sort_index()
y_test.index = range(0,len(y_test))

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

model = XGBClassifier(**params)
#model = linear_model.LogisticRegression(C=1e5)
model.fit(X_train, y_train)

pred_type = input('Predict on avr or ind? ')
if pred_type == 'ind':

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("###########################################")
    print("###########################################")


    from sklearn.metrics import roc_curve, auc
    probs = model.predict_proba(X_test)
    preds = probs[:,0]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.ion()
    plt.show()
if pred_type == 'avr':

    y_pred = model.predict_proba(X_test)
    #predictions = [round(value) for value in y_pred]
    predictions = y_pred[:,1]
    patient_number = []
    for i in range(0,int(len(predictions)/3)):
        patient_number.append(i)
    patient_number = pd.DataFrame(patient_number)
    patient_number = pd.concat([patient_number]*3)
    patient_number = patient_number.sort_index()
    patient_number.index = range(0,len(patient_number))

    averaging = pd.DataFrame({'prediction':predictions.tolist(),'patient':patient_number.iloc[:,0]})
    averaging = averaging.groupby('patient',as_index=True)['prediction'].mean()
    averaging = [round(value) for value in averaging]
    # evaluate predictions

    accuracy = accuracy_score(y_test[0::3], averaging)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("###########################################")
    print("###########################################")
    from sklearn.metrics import roc_curve, auc
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.ion()
    plt.show()
