# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:30:55 2018

@author: Madhan
"""

from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt

'''load your data here'''

class TrainLoadDataModule(object):
    def __init__(self):
        self.DIR = 'C:/Users/Madhan/Documents/Third Semester/Deep Learning/Assignment 2/'
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels
    
class TestLoadDataModule(object):
    def __init__(self):
        self.DIR = 'C:/Users/Madhan/Documents/Third Semester/Deep Learning/Assignment 2/'
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load(self, mode = 'test'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

ld1 = TrainLoadDataModule()
ld2 = TestLoadDataModule()

Train,Train_Test = ld1.load('train')
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(Train, Train_Test, test_size=0.20, random_state=100)

Test, Test_Test = ld2.load('test')

XTrain[0,:]
XTest[0:]
XTrain.shape
XTest.shape
YTrain.shape
YTest.shape
YTrain[0]
YTest[0]

plt.imshow(np.reshape(XTrain[0,:],(28,28)))
plt.imshow(np.reshape(XTest[0,:],(28,28)))

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

seed = 108726043
np.random.seed(seed)
#XTrain[0] = np.reshape(XTrain[0,:],(28,28))
num_of_pixels = XTrain.shape[1]
XTrain = XTrain.reshape(XTrain.shape[0], 1, 28, 28).astype('float32')
XTest = XTest.reshape(XTest.shape[0], 1, 28, 28).astype('float32')

print(num_of_pixels)

XTrain = (XTrain-np.min(XTrain))/(np.max(XTrain)-np.min(XTrain))
XTest = (XTest-np.min(XTest))/(np.max(XTest)-np.min(XTest))

class_names = np.unique(YTrain)
print(class_names)
YTrain = np_utils.to_categorical(YTrain)
YTest = np_utils.to_categorical(YTest)

num_of_classes = YTest.shape[1]
print(num_of_classes)

YTrain

num_of_classes


from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Flatten
from keras import backend as K
K.set_image_dim_ordering('th')


def baselineCNN():
    model = Sequential()
    
    #First hidden layer
    model.add(
        Conv2D(filters=48,        #number of filters
               kernel_size=(3,3),     #size of each filter
               input_shape=(1,28,28),
               activation='relu'
              )
    )
    
    #Pooling layer
    model.add(
        MaxPooling2D(pool_size=(2,2))
    )
    
    #Second hideen layer
    model.add(
        Conv2D(filters=96,        #number of filters
               kernel_size=(3,3),     #size of each filter
               input_shape=(1,28,28),
               activation='relu'
              )
    )
    
    #Pooling layer    
    model.add(
        MaxPooling2D(pool_size=(2,2))
    )    
    
    #Flattening layer
    model.add(
        Flatten()
    )
    
    #Fully connected layer
    model.add(
        Dense(
            units=100,
            activation='relu'
        )
    )
    
    #Output layer
    model.add(
        Dense(
            units=num_of_classes,
            activation='softmax'
        )
    )
    
    #compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    #return the model
    return model

model = baselineCNN()
print(model.summary())

import time
t = time.localtime(time.time())
timeStamp = str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) + '--' + str(t.tm_hour) + '-'+str(t.tm_min) + '-'+str(t.tm_sec)
timeStamp

from keras.callbacks import TensorBoard
tBoard = TensorBoard(log_dir='C:/Users/Madhan/Documents/Task3CNNlogs'.format(timeStamp))
num_epochs = 50
batch_size = 200

history = model.fit(
    XTrain,
    YTrain,
    validation_data=(XTest,YTest),
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=2,
    callbacks=[tBoard]
)

scores = model.evaluate(XTest,YTest,verbose=0)
print('Baseline error: %.2f' % (1-scores[1]))

print("Accuracy: %.2f" % scores[1])


import numpy as np
import matplotlib.pyplot as plt

class MiscFunctions:
    #Plot the model fit history
    @staticmethod
    def plot_history(history):
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
    
        ## As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
        ## Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    
        ## Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

mf = MiscFunctions()

mf.plot_history(history)

Test, Test_Test = ld2.load('test')
Test = Test.reshape(Test.shape[0], 1, 28, 28).astype('float32')
Test = (Test-np.min(Test))/(np.max(Test)-np.min(Test))

yPred = model.predict_classes(Test)
yPred_probabilities = model.predict(Test)

yPred

yPred_probabilities
type(yPred)
yPred.shape
type(Test_Test)
Test_Test.shape
Test_Test

class_names = np.unique(Test_Test)
print(class_names)
Test_Test = np_utils.to_categorical(Test_Test)

num_of_classes = Test_Test.shape[1]
print(num_of_classes)
Test_Test
num_of_classes
Test_Test_original=np.argmax(Test_Test,axis=1)
Test_Test_original

from sklearn.metrics import classification_report,confusion_matrix
print("Classification report \n=======================")
print(classification_report(y_true=Test_Test_original, y_pred=yPred))
print("Confusion matrix \n=======================")
print(confusion_matrix(y_true=Test_Test_original, y_pred=yPred))

import itertools
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix\n============================")
        else:
            print('Confusion matrix, without normalization\n============================')

        print(cm)
        print("\n")

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true=Test_Test_original, y_pred=yPred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
