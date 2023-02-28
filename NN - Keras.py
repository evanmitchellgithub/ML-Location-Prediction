# https://visualstudiomagazine.com/articles/2020/12/15/pytorch-network.aspx
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import string
import re
from collections import Counter
import matplotlib.pyplot as plt
import random
from sklearn import svm
import torch
import numpy as np

# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


import warnings
warnings.filterwarnings('ignore') #UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.

from keras import backend as K
numClass = 6

#load dataset
location = []
bedroom = []
bathroom = []
propertyType = []
price = []
predictions = []
results = pd.read_csv('neuralData.csv')
print('Number of lines in neuralData CSV: ', len(results))

print('Converting CSV columns into lists:')
address = results['Address'].tolist()
location = results['Location']
bedroom = results['Bedroom'].tolist()
bathroom = results['Bathroom'].tolist()
propertyType = results['PropertyType'].tolist() #Need to convert the three different types into numbers
price = results['Price'].tolist()
predictions = results['Predictions'].tolist()
print('...Done.')

LabelList = [[] for _ in range(len(results))]
DataList = [[] for _ in range(len(results))]
for i in range(len(results)):
    DataList[i].append(bedroom[i])
    DataList[i].append(bathroom[i])
    DataList[i].append(propertyType[i])
    DataList[i].append(price[i])
    DataList[i].append(predictions[i])
dataset = results.values
LabelList = dataset[:,2]

X = np.array(DataList)
Y = LabelList
# print('y:', Y)
# print('x:', X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print('dummy_y:', dummy_y)

x_train, x_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2)
# define baseline model
model = Sequential()
# create model
model.add(Dense(8, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(numClass, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
K.set_value(model.optimizer.learning_rate, 0.001)

print('Starting training')
print('x_train[1]: ', x_train[1])
print('y_train[1]: ', y_train[1])
history = model.fit(x_train, y_train, verbose=1, batch_size=32, epochs=500, validation_split=0.1)
history_dict = history.history
print('\n\nhistory keys')
print(history_dict.keys())

# Neural network information #===================================================================
print('\n\nNeural network classifier')

y_pred = model.predict(x_test)
y_predROC = y_pred
y_pred_class = np.argmax(y_pred, axis=1) #turn into array of classes
y_test_class = np.argmax(y_test, axis=1)
print('y_test_class:', y_test_class)
print('y_pred_class:', y_pred_class)


#print('Confusion matrix:\n ', confusion_matrix(y_test_class, y_pred_class, labels=[0,1,2,3,4,5]))
print('Confusion matrix:\n ', confusion_matrix(y_test_class, y_pred_class))
print('Classification report:\n ', classification_report(y_test_class, y_pred_class))


loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('loss:', loss)
print('accuracy:', accuracy)
#print('accuracy score:', accuracy_score(y_test_class, y_pred_class))
print('y_test_class',y_test_class)
print('y_pred_class',y_pred_class)
confusion_matrix2 = confusion_matrix(y_test_class, y_pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix2, display_labels = [1,2,3,4,5,6])
cm_display.plot()
plt.show()

f1score = f1_score(y_test_class, y_pred_class, labels=None, pos_label=1, average='weighted')#, sample_weight=None, zero_division='warn')
print('F1 score:', f1score)
    

# DUMMY CLASSIFIER INFORMATION #=====================================================
print('\n\nDummy classifier')
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train)
y_pred2 = dummy_clf.predict(x_test)
y_preddummyROC = y_pred2
y_pred_class2 = np.argmax(y_pred2, axis=1)
y_test_class2 = np.argmax(y_test, axis=1)
print('y_test_class:', y_test_class2)
print('y_pred_class:', y_pred_class2)
print('Confusion matrix:\n ', confusion_matrix(y_test_class2, y_pred_class2))
print('Classification report:\n ', classification_report(y_test_class2, y_pred_class2))
print('accuracy score:', accuracy_score(y_test_class2, y_pred_class2))

# F1 score #=========================================================================
f1score = f1_score(y_test_class2, y_pred_class2, labels=None, pos_label=1, average='weighted')
print('F1 score:', f1score)

# Neural Net PLOT #===================================================================
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.subplot(211)
plt.rc('font' , size = BIGGER_SIZE ) # controls default text sizes
plt.rc('axes' , titlesize = BIGGER_SIZE ) # fontsize of the axes title
plt.rc('axes', labelsize = BIGGER_SIZE ) # fontsize of the x and y labels
plt.rc('xtick' , labelsize = SMALL_SIZE ) # fontsize of the tick labels
plt.rc('ytick' , labelsize = SMALL_SIZE ) # fontsize of the tick labels
plt.rc('legend' , fontsize = SMALL_SIZE ) # legend fontsize
plt.rc('figure' , titlesize = BIGGER_SIZE ) # fontsize of the figure title'

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss'); plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# ROC CURVE  #=====================================================
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import preprocessing

#used x_test
print('y_predROC: ', y_predROC)
y_predROC = y_predROC.argmax(axis=-1)
print('y_predROC argmas: ', y_predROC)
print('y_test: ', y_test)
y_testROC = y_test.argmax(axis=-1)
print('y_testROC argmas: ', y_testROC)
y_preddummyROC = y_preddummyROC.argmax(axis=-1)
print('y_preddummyROC argmas: ', y_preddummyROC)
target= [0,1,2,3,4,5]


# function for scoring roc auc score for multi-class
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)

# set plot figure size
fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
plt.rc('font' , size = BIGGER_SIZE ) # controls default text sizes
plt.rc('axes' , titlesize = BIGGER_SIZE ) # fontsize of the axes title
plt.rc('axes', labelsize = BIGGER_SIZE ) # fontsize of the x and y labels
plt.rc('xtick' , labelsize = SMALL_SIZE ) # fontsize of the tick labels
plt.rc('ytick' , labelsize = SMALL_SIZE ) # fontsize of the tick labels
plt.rc('legend' , fontsize = SMALL_SIZE ) # legend fontsize
plt.rc('figure' , titlesize = BIGGER_SIZE ) # fontsize of the figure title'
print('ROC AUC score:', multiclass_roc_auc_score(y_testROC, y_predROC))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.title('Neural Network classifier')
plt.show()

# set plot figure size
fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
plt.rc('font' , size = BIGGER_SIZE ) # controls default text sizes
plt.rc('axes' , titlesize = BIGGER_SIZE ) # fontsize of the axes title
plt.rc('axes', labelsize = BIGGER_SIZE ) # fontsize of the x and y labels
plt.rc('xtick' , labelsize = SMALL_SIZE ) # fontsize of the tick labels
plt.rc('ytick' , labelsize = SMALL_SIZE ) # fontsize of the tick labels
plt.rc('legend' , fontsize = SMALL_SIZE ) # legend fontsize
plt.rc('figure' , titlesize = BIGGER_SIZE ) # fontsize of the figure title'
print('ROC AUC score - Dummy classifier:', multiclass_roc_auc_score(y_testROC, y_preddummyROC))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.title('Dummy classifier')
plt.show()

#one hot encoding
# row_maxes = y_pred.max(axis=1).reshape(-1, 1)
# y_pred[:] = np.where(y_pred == row_maxes, 1, 0)
# print('y_pred: ', y_pred[1])
# print('y_test: ', y_test[1])
# y_pred = label_binarize(y_pred, classes=[1,2,3,4,5,6])
# print('ypred: ', y_pred[1])
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
# auc_keras = auc(fpr_keras, tpr_keras)
